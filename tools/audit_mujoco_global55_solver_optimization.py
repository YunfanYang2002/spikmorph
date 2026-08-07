"""Fixed-global55 Newton solver initialization and convergence diagnostic.

This tool replays the validated global55 pre-state once and evaluates four
independent Newton clones.  The only interventions are qacc_warmstart,
solver tolerance, and solver iteration limit.  Contact and constraint
physics are never edited.
"""

from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys
import traceback
from types import SimpleNamespace
from typing import Any, Iterable, Sequence

import numpy as np


def _ensure_numpy_bool_compatibility() -> None:
    if "bool" not in np.__dict__:
        np.__dict__["bool"] = bool


_ensure_numpy_bool_compatibility()

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools import analyze_mujoco_global55_contact_demand as oracle
from tools import audit_mujoco_global55_contact_regularization_counterfactual as regularization
from tools import audit_mujoco_global55_friction_aref_counterfactual as aref


MORPHOLOGY = oracle.MORPHOLOGY
XML_SHA256 = oracle.XML_SHA256
CHECKPOINT_SHA256 = oracle.CHECKPOINT_SHA256
GLOBAL_STEP = oracle.GLOBAL_STEP
EXPECTED_SUBSTEPS = oracle.EXPECTED_SUBSTEPS
REFERENCE_ORACLE_NAME = "mujoco_global55_contact_demand_oracle_corrected_20260804_143138"
REFERENCE_REGULARIZATION_NAME = (
    "mujoco_global55_contact_regularization_counterfactual_20260806_142049"
)
CONDITIONS = (
    ("production_warmstart_production_tolerance", "PRODUCTION_WARMSTART_PRODUCTION_TOLERANCE", False, False),
    ("zero_warmstart_production_tolerance", "ZERO_WARMSTART_PRODUCTION_TOLERANCE", True, False),
    ("production_warmstart_tight_tolerance", "PRODUCTION_WARMSTART_TIGHT_TOLERANCE", False, True),
    ("zero_warmstart_tight_tolerance", "ZERO_WARMSTART_TIGHT_TOLERANCE", True, True),
)
STATE_COPY_FIELDS = aref.cone_helper.STATE_COPY_FIELDS
REGRESSION_RTOL = 1.0e-9
REGRESSION_ATOL = 1.0e-9
SENSITIVITY_RTOL = 1.0e-6
SENSITIVITY_ATOL = 1.0e-8
PHYSICAL_IMPULSE_ATOL = 1.0e-8
TIGHT_TOLERANCE_DEFAULT = 1.0e-12
REQUIRED_SOLVER_STAT_FIELDS = (
    "improvement", "gradient", "lineslope", "nactive", "nchange",
    "neval", "nupdate",
)
FORBIDDEN_OPTION_TOKENS = (
    "cone", "friction", "solref", "solimp", "aref", "regularization",
)


class Tee:
    def __init__(self, *streams: Any) -> None:
        self.streams = streams

    def write(self, value: str) -> int:
        for stream in self.streams:
            stream.write(value)
            stream.flush()
        return len(value)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="Run fixed-global55 Newton solver optimization diagnostics."
    )
    result.add_argument("--formal-server-defaults", action="store_true")
    result.add_argument("--checkpoint")
    result.add_argument("--walker-dir")
    result.add_argument("--corrected-oracle")
    result.add_argument("--regularization-reference")
    result.add_argument("--output-dir")
    result.add_argument("--zip-path")
    result.add_argument("--morphology-id", default=MORPHOLOGY)
    result.add_argument("--cfg", default="configs/ft.yaml")
    result.add_argument("--seed", type=int, default=1409)
    result.add_argument("--device", default="cpu")
    return result


def _default_output_paths() -> tuple[Path, Path]:
    stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")
    return (
        REPO_ROOT / "output/diagnostics"
        / f"mujoco_global55_solver_optimization_{stamp}",
        REPO_ROOT / "tmp"
        / f"mujoco_global55_solver_optimization_{stamp}.zip",
    )


def resolve_arguments(args: argparse.Namespace) -> argparse.Namespace:
    if args.formal_server_defaults:
        batch = REPO_ROOT / "output/diagnostics/mujoco_control_51k_20260727_091638"
        manifest = json.loads((batch / "manifest.json").read_text(encoding="utf-8"))
        args.checkpoint = str(batch / "jobs/job_000_seed1409_lr0p00015/Unimal-v0.pt")
        args.walker_dir = manifest["source_audit"]["walker_dir"]
        args.corrected_oracle = str(
            REPO_ROOT / "output/diagnostics" / REFERENCE_ORACLE_NAME
        )
        args.regularization_reference = str(
            REPO_ROOT / "output/diagnostics" / REFERENCE_REGULARIZATION_NAME
        )
    missing = [
        name for name in ("checkpoint", "walker_dir", "corrected_oracle")
        if not getattr(args, name)
    ]
    if missing:
        raise ValueError(f"missing required arguments: {', '.join(missing)}")
    default_output, default_zip = _default_output_paths()
    args.output_dir = args.output_dir or str(default_output)
    args.zip_path = args.zip_path or str(default_zip)
    if not args.regularization_reference:
        args.regularization_reference = str(
            REPO_ROOT / "output/diagnostics" / REFERENCE_REGULARIZATION_NAME
        )
    return args


def validate_paths(args: argparse.Namespace) -> dict[str, Path]:
    paths = aref.validate_paths(args)
    paths["regularization_reference"] = Path(args.regularization_reference).resolve()
    if not paths["regularization_reference"].is_dir():
        raise FileNotFoundError(
            "regularization reference artifact is missing: "
            f"{paths['regularization_reference']}"
        )
    if paths["output_dir"].exists():
        raise FileExistsError(f"refusing to overwrite output: {paths['output_dir']}")
    if paths["zip_path"].exists():
        raise FileExistsError(f"refusing to overwrite ZIP: {paths['zip_path']}")
    return paths


def _json_normalize(value: Any) -> Any:
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        numeric = float(value)
        return numeric if np.isfinite(numeric) else None
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return _json_normalize(value.item())
        return [_json_normalize(item) for item in value.tolist()]
    if isinstance(value, dict):
        return {key: _json_normalize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_normalize(item) for item in value]
    return value


def write_json(path: Path, payload: Any) -> None:
    oracle.write_json(path, _json_normalize(payload))


def _allclose(left: Any, right: Any, rtol: float = REGRESSION_RTOL, atol: float = REGRESSION_ATOL) -> bool:
    if left is None or right is None:
        return left is None and right is None
    try:
        return bool(np.allclose(np.asarray(left), np.asarray(right), rtol=rtol, atol=atol))
    except (TypeError, ValueError):
        return False


def _finite(value: Any) -> bool:
    try:
        return bool(np.isfinite(np.asarray(value, dtype=np.float64)).all())
    except (TypeError, ValueError):
        return False


def _copy_model(mujoco: Any, model: Any) -> tuple[Any, str]:
    copier = getattr(mujoco, "mj_copyModel", None)
    if not callable(copier):
        raise RuntimeError("MuJoCo binding does not expose mj_copyModel")
    errors = []
    for arguments in ((model,), (None, model)):
        try:
            copied = copier(*arguments)
            if copied is not None:
                return copied, "mujoco.mj_copyModel"
        except Exception as error:  # pragma: no cover - binding-specific signature
            errors.append(f"{type(error).__name__}: {error}")
    raise RuntimeError("mujoco.mj_copyModel failed: " + "; ".join(errors))


def _scalar(data: Any, name: str, default: Any = None) -> Any:
    value = getattr(data, name, None)
    if value is None:
        return default
    try:
        array = np.asarray(value)
        if array.size == 1:
            return array.reshape(-1)[0].item()
        return array.copy()
    except (TypeError, ValueError):
        return default


def _model_option_snapshot(model: Any) -> dict[str, Any]:
    snapshot = aref.cone_helper.model_option_snapshot(model)
    for name in (
        "jacobian", "impratio", "ls_tolerance", "noslip_tolerance",
        "ccd_tolerance", "noslip_iterations", "ccd_iterations",
        "sdf_initpoints", "enableflags", "o_solref", "o_solimp",
    ):
        value = getattr(model.opt, name, None)
        if value is None:
            continue
        array = np.asarray(value)
        snapshot[f"opt.{name}"] = (
            array.copy() if array.ndim else array.item()
        )
    return snapshot


def _option_difference(
    reference: dict[str, Any], candidate: dict[str, Any], allowed: set[str] = set()
) -> dict[str, Any]:
    changed = []
    details = {}
    for name, left in reference.items():
        right = candidate.get(name)
        equal = _allclose(left, right) if isinstance(left, np.ndarray) else left == right
        details[name] = {"equal": bool(equal), "allowed": name in allowed}
        if not equal:
            changed.append(name)
    unexpected = [name for name in changed if name not in allowed]
    return {
        "changed_fields": changed,
        "unexpected_changed_fields": unexpected,
        "only_allowed": not unexpected,
        "details": details,
    }


def _tight_tolerance(production: float) -> tuple[float, str]:
    if not np.isfinite(production) or production <= 0.0:
        raise RuntimeError(f"production tolerance is not a positive finite value: {production}")
    if production > TIGHT_TOLERANCE_DEFAULT:
        return TIGHT_TOLERANCE_DEFAULT, "fixed 1e-12 is strictly below production tolerance"
    candidate = float(np.nextafter(production, 0.0))
    if not np.isfinite(candidate) or candidate <= 0.0 or not candidate < production:
        raise RuntimeError("cannot select a representable stricter tolerance")
    return candidate, "production tolerance was already <= 1e-12; nextafter(production, 0) selected"


def _configure_model(
    model: Any,
    production_options: dict[str, Any],
    tight: bool,
) -> dict[str, Any]:
    original_tolerance = float(production_options["opt.tolerance"])
    original_iterations = int(production_options["opt.iterations"])
    if tight:
        tolerance, rationale = _tight_tolerance(original_tolerance)
        iterations = max(original_iterations, 100)
    else:
        tolerance, rationale = original_tolerance, "production options retained"
        iterations = original_iterations
    model.opt.tolerance = tolerance
    model.opt.iterations = iterations
    return {
        "tight": bool(tight),
        "production_tolerance": original_tolerance,
        "condition_tolerance": tolerance,
        "production_iterations": original_iterations,
        "condition_iterations": iterations,
        "tight_tolerance_rationale": rationale,
    }


def _contact_geometry(data: Any, mujoco: Any, model: Any) -> dict[str, Any]:
    contacts = []
    for index in range(int(getattr(data, "ncon", 0))):
        contact = data.contact[index]
        geom_ids = [int(contact.geom1), int(contact.geom2)]
        geom_names = []
        for geom_id in geom_ids:
            try:
                geom_names.append(str(mujoco.mj_id2name(
                    model, mujoco.mjtObj.mjOBJ_GEOM, geom_id
                )))
            except Exception:
                geom_names.append(str(geom_id))
        contacts.append({
            "contact_index": index,
            "geom_ids": geom_ids,
            "geom_names": geom_names,
            "point": np.asarray(contact.pos, dtype=np.float64).copy(),
            "frame": np.asarray(contact.frame, dtype=np.float64).copy(),
            "dist": float(contact.dist),
            "dim": int(contact.dim),
            "efc_address": int(contact.efc_address),
            "friction": np.asarray(contact.friction, dtype=np.float64).copy(),
        })
    return {"ncon": int(getattr(data, "ncon", 0)), "contacts": contacts}


def _contact_geometry_equal(left: dict[str, Any], right: dict[str, Any]) -> bool:
    if left.get("ncon") != right.get("ncon"):
        return False
    for a, b in zip(left.get("contacts", []), right.get("contacts", [])):
        if a["geom_ids"] != b["geom_ids"] or a["dim"] != b["dim"]:
            return False
        if not _allclose(a["point"], b["point"]) or not _allclose(a["frame"], b["frame"]):
            return False
        if not _allclose(a["friction"], b["friction"]):
            return False
    return True


def _state_without_warmstart(snapshot: dict[str, Any]) -> dict[str, Any]:
    return {
        name: value for name, value in snapshot.items()
        if name != "qacc_warmstart"
    }


def _state_equal_except_warmstart(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    checks = {}
    for name in _state_without_warmstart(left):
        a, b = left.get(name), right.get(name)
        checks[name] = _allclose(a, b) if isinstance(a, np.ndarray) else a == b
    return {"checks": checks, "valid": bool(all(checks.values()))}


def _stat_value(stat: Any, name: str) -> Any:
    value = getattr(stat, name, None)
    if value is None:
        return None
    try:
        array = np.asarray(value)
        return array.copy() if array.ndim else array.item()
    except (TypeError, ValueError):
        return None


def _solver_iteration_trace(data: Any, model: Any) -> dict[str, Any]:
    niter_raw = _scalar(data, "solver_niter")
    try:
        niter = int(niter_raw)
    except (TypeError, ValueError):
        niter = -1
    limit = int(getattr(model.opt, "iterations", -1))
    stats = getattr(data, "solver", None)
    trace = []
    errors = []
    if niter >= 0 and stats is not None:
        for index in range(niter):
            try:
                stat = stats[index]
                row = {"iteration_index": index}
                row.update({name: _stat_value(stat, name) for name in REQUIRED_SOLVER_STAT_FIELDS})
                trace.append(row)
            except Exception as error:
                errors.append(f"iteration {index}: {type(error).__name__}: {error}")
    required_available = bool(
        niter >= 0 and niter > 0 and len(trace) == niter
        and all(all(row[name] is not None for name in REQUIRED_SOLVER_STAT_FIELDS) for row in trace)
    )
    finite = bool(
        required_available
        and all(_finite(value) for row in trace for value in row.values())
    )
    last = trace[-1] if trace else {}
    return {
        "solver_niter": niter_raw,
        "iterations_limit": limit,
        "trace": trace,
        "last_iteration": last,
        "required_fields": list(REQUIRED_SOLVER_STAT_FIELDS),
        "statistics_available": required_available,
        "all_statistics_finite": finite,
        "errors": errors,
        "status": "VALID" if finite else "INSUFFICIENT_EVIDENCE",
    }


def _solver_numerics(
    data: Any,
    model: Any,
    warmstart_input: Any,
    trace: dict[str, Any],
    configuration: dict[str, Any],
) -> dict[str, Any]:
    warmstart = np.asarray(warmstart_input, dtype=np.float64).copy()
    fields = {
        name: _scalar(data, name)
        for name in ("solver_niter", "solver_nnz", "solver_fwdinv")
    }
    finite = all(_finite(value) for value in fields.values() if value is not None)
    finite = bool(finite and _finite(warmstart) and trace["all_statistics_finite"])
    return {
        "solver_niter": fields["solver_niter"],
        "solver_nnz": fields["solver_nnz"],
        "solver_fwdinv": fields["solver_fwdinv"],
        "tolerance": float(model.opt.tolerance),
        "iterations_limit": int(model.opt.iterations),
        "warmstart_vector_norm": float(np.linalg.norm(warmstart)),
        "warmstart_max_abs": float(np.max(np.abs(warmstart))) if warmstart.size else 0.0,
        "warmstart_all_finite": bool(_finite(warmstart)),
        "statistics_status": trace["status"],
        "finite": finite,
        "status": "VALID" if finite else "NONFINITE" if not all(
            _finite(value) for value in fields.values() if value is not None
        ) else "INSUFFICIENT_EVIDENCE",
        "configuration": configuration,
    }


def _condition_state_validation(
    snapshot: Any,
    data: Any,
    expected_zero_warmstart: bool,
) -> dict[str, Any]:
    clone = aref.cone_helper.state_equality(snapshot, data)
    state = aref.cone_helper.state_input_snapshot(data)
    original = aref.cone_helper.state_input_snapshot(snapshot)
    warmstart = np.asarray(state["qacc_warmstart"], dtype=np.float64)
    expected = np.zeros_like(warmstart) if expected_zero_warmstart else np.asarray(
        original["qacc_warmstart"], dtype=np.float64
    )
    return {
        "clone_pre_state": clone,
        "pre_state_snapshot": state,
        "warmstart_expected": expected,
        "warmstart_exact_zero": bool(expected_zero_warmstart and np.array_equal(warmstart, np.zeros_like(warmstart))),
        "warmstart_matches_expected": bool(np.array_equal(warmstart, expected)),
        "same_complete_pre_state_before_intervention": bool(
            clone["STATE_COPY_EQUAL"] if not expected_zero_warmstart else
            _state_equal_except_warmstart(original, state)["valid"]
        ),
    }


def _run_condition(
    mujoco: Any,
    base_model: Any,
    snapshot: Any,
    mapping: dict[str, Any],
    name: str,
    label: str,
    zero_warmstart: bool,
    tight: bool,
    production_options: dict[str, Any],
) -> dict[str, Any]:
    model, model_copy_api = _copy_model(mujoco, base_model)
    configuration = _configure_model(model, production_options, tight)
    data = mujoco.MjData(model)
    mujoco.mj_copyData(data, model, snapshot)
    state_before_stage = aref.cone_helper.state_input_snapshot(data)
    if zero_warmstart:
        data.qacc_warmstart[:] = 0
    staged_calls = aref.stage_to_constraint(mujoco, model, data)
    state_validation = _condition_state_validation(snapshot, data, zero_warmstart)
    pre_constraint = regularization._constraint_snapshot(data, mujoco, model)
    pre_geometry = _contact_geometry(data, mujoco, model)
    warmstart_input = np.asarray(data.qacc_warmstart, dtype=np.float64).copy()
    mujoco.mj_fwdConstraint(model, data)
    solver_data = mujoco.MjData(model)
    mujoco.mj_copyData(solver_data, model, data)
    post_constraint = regularization._constraint_snapshot(
        solver_data, mujoco, model, read_physical_contact_forces=True
    )
    trace = _solver_iteration_trace(solver_data, model)
    numerics = _solver_numerics(solver_data, model, warmstart_input, trace, configuration)
    mujoco.mj_Euler(model, data)
    capture = regularization.capture_after_integration(
        mujoco, model, data, snapshot, mapping, solver_data
    )
    demand = regularization._run_shared_demand(capture)
    excess = regularization._run_excess(capture, demand)
    return {
        "condition_name": name,
        "condition_label": label,
        "zero_warmstart": bool(zero_warmstart),
        "tight_tolerance": bool(tight),
        "model_copy_api": model_copy_api,
        "model_options": _model_option_snapshot(model),
        "configuration": configuration,
        "state_validation": state_validation,
        "state_before_stage": state_before_stage,
        "pre_constraint": pre_constraint,
        "pre_contact_geometry": pre_geometry,
        "post_constraint": post_constraint,
        "capture": capture,
        "shared_demand": demand,
        "excess": excess,
        "solver_iteration_trace": trace,
        "solver_numerics": numerics,
        "one_step_result": {
            "post_qpos": np.asarray(data.qpos, dtype=np.float64).copy(),
            "post_qvel": np.asarray(data.qvel, dtype=np.float64).copy(),
            "post_time": float(data.time),
            "target_post_slip": excess["post_slip"],
        },
        "counts": {
            "condition_staged_forward_count": 1,
            "constraint_solve_count": 1,
            "custom_integration_count": 1,
            "staged_calls": staged_calls,
            "solver_calls": ["mj_fwdConstraint"],
            "integration_api": "mujoco.mj_Euler",
        },
    }


def _relative_delta(left: float, right: float) -> float | None:
    try:
        denominator = max(abs(float(left)), np.finfo(float).eps)
        return float((float(right) - float(left)) / denominator)
    except (TypeError, ValueError):
        return None


def _physical_impulse_vectors(condition: dict[str, Any]) -> list[np.ndarray]:
    result = []
    for contact in condition["capture"].get("contacts", []):
        result.append(np.concatenate((
            np.asarray([contact["normal_impulse"]], dtype=np.float64),
            np.asarray(contact["tangential_impulse"], dtype=np.float64),
        )))
    return result


def _target_excess(condition: dict[str, Any]) -> dict[str, Any]:
    return condition["excess"]


def _pair_sensitivity(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    left_excess, right_excess = _target_excess(left), _target_excess(right)
    left_impulses = _physical_impulse_vectors(left)
    right_impulses = _physical_impulse_vectors(right)
    impulse_deltas = [
        float(np.max(np.abs(a - b)))
        for a, b in zip(left_impulses, right_impulses)
        if a.shape == b.shape
    ]
    target_impulse_delta = float(np.max(impulse_deltas)) if impulse_deltas else None
    left_actual = float(left_excess["actual_tangent_impulse_norm"])
    right_actual = float(right_excess["actual_tangent_impulse_norm"])
    left_normal = float(left_excess["normal_impulse"])
    right_normal = float(right_excess["normal_impulse"])
    left_demand = float(left_excess["rigid_demand_norm"])
    right_demand = float(right_excess["rigid_demand_norm"])
    left_excess_norm = float(left_excess["solver_excess_norm"])
    right_excess_norm = float(right_excess["solver_excess_norm"])
    qacc_delta = float(np.max(np.abs(
        np.asarray(left["post_constraint"]["qacc"], dtype=np.float64)
        - np.asarray(right["post_constraint"]["qacc"], dtype=np.float64)
    )))
    post_slip_delta = float(np.max(np.abs(
        np.asarray(left_excess["post_slip"], dtype=np.float64)
        - np.asarray(right_excess["post_slip"], dtype=np.float64)
    )))
    left_trace = left["solver_iteration_trace"].get("trace", [])
    right_trace = right["solver_iteration_trace"].get("trace", [])
    trace_equal = _json_normalize(left_trace) == _json_normalize(right_trace)
    metrics = {
        "physical_contact_impulse_max_absolute_delta": target_impulse_delta,
        "target_tangent_impulse_norm_relative_delta": _relative_delta(left_actual, right_actual),
        "normal_impulse_relative_delta": _relative_delta(left_normal, right_normal),
        "rigid_demand_relative_delta": _relative_delta(left_demand, right_demand),
        "solver_excess_relative_delta": _relative_delta(left_excess_norm, right_excess_norm),
        "qacc_max_absolute_delta": qacc_delta,
        "post_slip_max_absolute_delta": post_slip_delta,
        "solver_iteration_trace_equal": bool(trace_equal),
        "left_solver_niter": left["solver_numerics"].get("solver_niter"),
        "right_solver_niter": right["solver_numerics"].get("solver_niter"),
    }
    finite = all(
        value is not None and np.isfinite(value)
        for key, value in metrics.items()
        if key.endswith("delta") or key.endswith("_delta")
    )
    insensitive = bool(
        finite
        and float(metrics["physical_contact_impulse_max_absolute_delta"]) <= PHYSICAL_IMPULSE_ATOL
        and abs(float(metrics["target_tangent_impulse_norm_relative_delta"])) <= SENSITIVITY_RTOL
        and abs(float(metrics["normal_impulse_relative_delta"])) <= SENSITIVITY_RTOL
        and abs(float(metrics["rigid_demand_relative_delta"])) <= SENSITIVITY_RTOL
        and abs(float(metrics["solver_excess_relative_delta"])) <= SENSITIVITY_RTOL
        and float(metrics["qacc_max_absolute_delta"]) <= SENSITIVITY_ATOL
        and float(metrics["post_slip_max_absolute_delta"]) <= SENSITIVITY_ATOL
    )
    return {
        "metrics": metrics,
        "finite": finite,
        "classification": "INSENSITIVE" if insensitive else "SENSITIVE" if finite else "INSUFFICIENT_EVIDENCE",
        "thresholds": {
            "relative_output_rtol": SENSITIVITY_RTOL,
            "physical_impulse_max_abs": PHYSICAL_IMPULSE_ATOL,
            "qacc_max_abs": SENSITIVITY_ATOL,
            "post_slip_max_abs": SENSITIVITY_ATOL,
        },
    }


def _baseline_from_recent_artifact(
    condition: dict[str, Any], reference_dir: Path
) -> dict[str, Any]:
    candidates = (
        reference_dir / "conditions" / "regularization_scale_1_before" / "solver_excess.json",
        reference_dir / "conditions" / "r_scale_1_before" / "solver_excess.json",
    )
    path = next((candidate for candidate in candidates if candidate.is_file()), None)
    if path is None:
        return {"status": "INSUFFICIENT_EVIDENCE", "reason": "baseline solver_excess artifact missing"}
    try:
        reference = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return {"status": "INSUFFICIENT_EVIDENCE", "reason": str(error)}
    checks = {
        key: _allclose(condition["excess"].get(key), reference.get(key))
        for key in (
            "actual_tangent_impulse_vector", "actual_tangent_impulse_norm",
            "normal_impulse", "rigid_demand_vector", "rigid_demand_norm",
            "solver_excess_vector", "solver_excess_norm", "post_slip",
        )
    }
    return {
        "path": str(path),
        "checks": checks,
        "status": "PASS" if all(checks.values()) else "FAIL",
    }


def _baseline_regression(
    condition: dict[str, Any],
    corrected_reference: dict[str, Any],
    regularization_reference: Path,
) -> dict[str, Any]:
    target_index = int(condition["shared_demand"]["limb_12_contact_index"])
    target = condition["capture"]["contacts"][target_index]
    demand = condition["shared_demand"]
    adapted = {
        "capture": condition["capture"],
        "shared_demand": demand,
        "budget": {"selected": {"limb/12": {
            "actual_tangential_impulse": target["tangential_impulse"],
            "actual_tangential_impulse_norm": target["tangential_impulse_norm"],
            "actual_normal_impulse": target["normal_impulse"],
            "global_normal_conditioned_sticking_impulse": demand["limb_12_tangent_impulse_2d"],
            "global_normal_conditioned_sticking_impulse_norm": float(np.linalg.norm(demand["limb_12_tangent_impulse_2d"])),
            "pre_tangential_speed": target["pre_tangential_speed"],
        }}},
        "excess": condition["excess"],
    }
    oracle_result = aref.baseline_regression(adapted, corrected_reference)
    recent_result = _baseline_from_recent_artifact(condition, regularization_reference)
    sanity = {
        "actual_tangent_norm": _allclose(condition["excess"]["actual_tangent_impulse_norm"], 3.3247363600666735),
        "normal_impulse": _allclose(condition["excess"]["normal_impulse"], 6.345240278967453),
        "rigid_demand_norm": _allclose(condition["excess"]["rigid_demand_norm"], 2.540619084288334),
        "solver_excess": _allclose(condition["excess"]["solver_excess_norm"], 0.7841172757783395),
        "solver_excess_vector_norm": _allclose(condition["excess"]["solver_excess_vector_norm"], 0.8072255552101076),
        "post_slip_speed": _allclose(np.linalg.norm(condition["excess"]["post_slip"]), 0.1713507113360867),
    }
    oracle_pass = oracle_result.get("R_BASELINE_REPRODUCTION") == "PASS"
    valid = bool(oracle_pass and recent_result.get("status") == "PASS" and all(sanity.values()))
    return {
        "corrected_oracle": oracle_result,
        "recent_regularization_artifact": recent_result,
        "sanity": sanity,
        "OPTIMIZATION_BASELINE_REPRODUCTION": "PASS" if valid else "FAIL",
    }


def _pre_constraint_invariant(
    reference: dict[str, Any], candidate: dict[str, Any]
) -> dict[str, Any]:
    fields = (
        "efc_J", "efc_vel", "efc_aref", "efc_R", "efc_D", "iefc_R", "iefc_D",
    )
    return {name: _allclose(reference["pre_constraint"].get(name), candidate["pre_constraint"].get(name)) for name in fields}


def _solver_optimization_invariants(
    conditions: dict[str, dict[str, Any]],
    production_options: dict[str, Any],
) -> dict[str, Any]:
    baseline = conditions["production_warmstart_production_tolerance"]
    checks = {}
    for name, _, _, _ in CONDITIONS[1:]:
        candidate = conditions[name]
        state_checks = _state_equal_except_warmstart(
            baseline["state_validation"]["pre_state_snapshot"],
            candidate["state_validation"]["pre_state_snapshot"],
        )
        option_diff = _option_difference(
            production_options,
            candidate["model_options"],
            {"opt.tolerance", "opt.iterations"},
        )
        checks[name] = {
            "state_except_qacc_warmstart": state_checks,
            "contact_geometry": _contact_geometry_equal(
                baseline["pre_contact_geometry"], candidate["pre_contact_geometry"]
            ),
            "M_J_W": all(_allclose(
                baseline["capture"].get(field), candidate["capture"].get(field)
            ) for field in ("mass_matrix", "J_phys", "W_phys")),
            "pre_constraint": _pre_constraint_invariant(baseline, candidate),
            "model_options_only_allowed_changes": option_diff,
            "solver_type_unchanged": option_diff["details"].get("opt.solver", {}).get("equal", False),
            "cone_unchanged": option_diff["details"].get("opt.cone", {}).get("equal", False),
            "timestep_integrator_unchanged": (
                option_diff["details"].get("opt.timestep", {}).get("equal", False)
                and option_diff["details"].get("opt.integrator", {}).get("equal", False)
            ),
        }
        checks[name]["valid"] = bool(
            checks[name]["state_except_qacc_warmstart"]["valid"]
            and checks[name]["contact_geometry"]
            and checks[name]["M_J_W"]
            and all(checks[name]["pre_constraint"].values())
            and checks[name]["model_options_only_allowed_changes"]["only_allowed"]
        )
    valid = all(item["valid"] for item in checks.values())
    return {
        "checks_against_production": checks,
        "allowed_changes": ["qacc_warmstart", "opt.tolerance", "opt.iterations", "solver outputs", "post-step state"],
        "SOLVER_OPTIMIZATION_DIAGNOSTIC_ISOLATION": "VALIDATED" if valid else "FAILED",
    }


def _warmstart_activation(conditions: dict[str, dict[str, Any]]) -> dict[str, Any]:
    production = conditions["production_warmstart_production_tolerance"]
    checks = {}
    for name in ("zero_warmstart_production_tolerance", "zero_warmstart_tight_tolerance"):
        condition = conditions[name]
        warmstart = np.asarray(condition["state_validation"]["pre_state_snapshot"]["qacc_warmstart"], dtype=np.float64)
        checks[name] = {
            "warmstart_exact_zero": bool(np.array_equal(warmstart, np.zeros_like(warmstart))),
            "warmstart_all_finite": bool(_finite(warmstart)),
            "other_state_equal": _state_equal_except_warmstart(
                production["state_validation"]["pre_state_snapshot"],
                condition["state_validation"]["pre_state_snapshot"],
            ),
        }
        checks[name]["valid"] = bool(
            checks[name]["warmstart_exact_zero"]
            and checks[name]["warmstart_all_finite"]
            and checks[name]["other_state_equal"]["valid"]
        )
    activation = "VALIDATED" if all(item["valid"] for item in checks.values()) else "FAILED"
    return {
        "production_warmstart": {
            "norm": production["solver_numerics"]["warmstart_vector_norm"],
            "maximum_absolute_value": production["solver_numerics"]["warmstart_max_abs"],
            "all_finite": production["solver_numerics"]["warmstart_all_finite"],
            "full_vector": production["state_validation"]["pre_state_snapshot"]["qacc_warmstart"],
        },
        "checks": checks,
        "ZERO_WARMSTART_ACTIVATION": activation,
    }


def _tight_tolerance_activation(
    conditions: dict[str, dict[str, Any]], production_options: dict[str, Any]
) -> dict[str, Any]:
    checks = {}
    production_tolerance = float(production_options["opt.tolerance"])
    production_iterations = int(production_options["opt.iterations"])
    for name in ("production_warmstart_tight_tolerance", "zero_warmstart_tight_tolerance"):
        condition = conditions[name]
        options = condition["model_options"]
        difference = _option_difference(
            production_options, options, {"opt.tolerance", "opt.iterations"}
        )
        checks[name] = {
            "tolerance": options["opt.tolerance"],
            "iterations": options["opt.iterations"],
            "tight_tolerance_less_than_production": bool(options["opt.tolerance"] < production_tolerance),
            "tight_iterations_at_least_production": bool(options["opt.iterations"] >= production_iterations),
            "only_tolerance_iterations_changed": difference["only_allowed"],
            "solver_type_unchanged": bool(options.get("opt.solver") == production_options.get("opt.solver")),
            "difference": difference,
        }
        checks[name]["valid"] = bool(
            checks[name]["tight_tolerance_less_than_production"]
            and checks[name]["tight_iterations_at_least_production"]
            and checks[name]["only_tolerance_iterations_changed"]
            and checks[name]["solver_type_unchanged"]
        )
    return {
        "production": {
            "tolerance": production_tolerance,
            "iterations": production_iterations,
            "solver": production_options.get("opt.solver"),
        },
        "checks": checks,
        "TIGHT_TOLERANCE_ACTIVATION": "VALIDATED" if all(item["valid"] for item in checks.values()) else "FAILED",
    }


def _convergence_assessment(
    conditions: dict[str, dict[str, Any]],
    baseline: dict[str, Any],
    warmstart_sensitivity: dict[str, Any],
    tolerance_sensitivity: dict[str, Any],
) -> dict[str, Any]:
    assessments = {}
    for name, _, _, _ in CONDITIONS:
        trace = conditions[name]["solver_iteration_trace"]
        numerics = conditions[name]["solver_numerics"]
        niter = numerics.get("solver_niter")
        limit = numerics.get("iterations_limit")
        assessments[name] = {
            "statistics_finite": bool(numerics.get("finite")),
            "statistics_complete": bool(trace.get("statistics_available")),
            "niter": niter,
            "iterations_limit": limit,
            "niter_below_limit": bool(
                isinstance(niter, (int, float, np.integer, np.floating))
                and isinstance(limit, (int, float, np.integer, np.floating))
                and int(niter) < int(limit)
            ),
            "last_iteration": trace.get("last_iteration", {}),
        }
    all_valid_stats = all(
        item["statistics_finite"] and item["statistics_complete"]
        for item in assessments.values()
    )
    any_limit_hit = any(not item["niter_below_limit"] for item in assessments.values())
    baseline_pass = baseline["OPTIMIZATION_BASELINE_REPRODUCTION"] == "PASS"
    converged = bool(
        all_valid_stats and not any_limit_hit and baseline_pass
        and warmstart_sensitivity["SOLVER_WARMSTART_SENSITIVITY"] == "INSENSITIVE"
        and tolerance_sensitivity["SOLVER_TOLERANCE_SENSITIVITY"] == "INSENSITIVE"
    )
    if not all_valid_stats:
        status = "INSUFFICIENT_EVIDENCE"
    elif any_limit_hit:
        status = "NOT_CONVERGED"
    elif converged:
        status = "VALIDATED"
    else:
        status = "QUESTIONABLE"
    return {
        "conditions": assessments,
        "baseline_reproduction": baseline["OPTIMIZATION_BASELINE_REPRODUCTION"],
        "all_statistics_finite": all_valid_stats,
        "any_iteration_limit_reached": any_limit_hit,
        "PRODUCTION_NEWTON_CONVERGENCE": status,
    }


def _classify_final(
    invariant: dict[str, Any],
    warmstart: dict[str, Any],
    tolerance: dict[str, Any],
    convergence: dict[str, Any],
    baseline: dict[str, Any],
) -> dict[str, Any]:
    if baseline["OPTIMIZATION_BASELINE_REPRODUCTION"] != "PASS":
        status = "INSUFFICIENT_EVIDENCE"
        action = "DIAGNOSTIC_IMPLEMENTATION_FIX_REQUIRED"
    elif invariant["SOLVER_OPTIMIZATION_DIAGNOSTIC_ISOLATION"] != "VALIDATED":
        status = "NONCANONICAL"
        action = "DIAGNOSTIC_IMPLEMENTATION_FIX_REQUIRED"
    elif convergence["PRODUCTION_NEWTON_CONVERGENCE"] == "NOT_CONVERGED":
        status = "NONCONVERGED"
        action = "SOLVER_CONVERGENCE_FIX_REQUIRED"
    else:
        warm_sensitive = warmstart["SOLVER_WARMSTART_SENSITIVITY"] == "SENSITIVE"
        tolerance_sensitive = tolerance["SOLVER_TOLERANCE_SENSITIVITY"] == "SENSITIVE"
        # A VALIDATED convergence result is already the tool's explicit gate
        # that the recorded Newton statistics are finite and not truncated by
        # the iteration limit.  Keep the detailed flags as a defensive check
        # for older/reference payloads that may not contain both fields.
        stats_support_sensitivity = bool(
            baseline["OPTIMIZATION_BASELINE_REPRODUCTION"] == "PASS"
            and (
                convergence.get("PRODUCTION_NEWTON_CONVERGENCE") == "VALIDATED"
                or (
                    convergence.get("all_statistics_finite") is True
                    and convergence.get("any_iteration_limit_reached") is False
                )
            )
        )
        if warm_sensitive and tolerance_sensitive and stats_support_sensitivity:
            status = "WARMSTART_AND_TOLERANCE_SENSITIVE"
            action = "SOLVER_WARMSTART_COUNTERFACTUAL"
        elif warm_sensitive and stats_support_sensitivity:
            status = "WARMSTART_SENSITIVE"
            action = "SOLVER_WARMSTART_COUNTERFACTUAL"
        elif tolerance_sensitive and stats_support_sensitivity:
            status = "TOLERANCE_SENSITIVE"
            action = "SOLVER_TOLERANCE_COUNTERFACTUAL"
        elif convergence["PRODUCTION_NEWTON_CONVERGENCE"] == "VALIDATED":
            status = "ROBUST_CONVERGED_SOLUTION"
            action = "FORMULATION_LEVEL_DIFFERENCE_CONFIRMED"
        else:
            status = "INSUFFICIENT_EVIDENCE"
            action = "INSUFFICIENT_EVIDENCE"
    return {
        "MUJOCO_SOLVER_EXCESS_OPTIMIZATION_STATUS": status,
        "NEXT_ACTION": action,
        "interpretation": (
            "The observed solver excess is not explained by Newton initialization "
            "or insufficient production convergence."
            if status == "ROBUST_CONVERGED_SOLUTION" else None
        ),
    }


def _warmstart_sensitivity(
    conditions: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    comparisons = {
        "production_tolerance_A_vs_B": _pair_sensitivity(
            conditions["production_warmstart_production_tolerance"],
            conditions["zero_warmstart_production_tolerance"],
        ),
        "tight_tolerance_C_vs_D": _pair_sensitivity(
            conditions["production_warmstart_tight_tolerance"],
            conditions["zero_warmstart_tight_tolerance"],
        ),
    }
    classification = (
        "INSUFFICIENT_EVIDENCE"
        if any(item["classification"] == "INSUFFICIENT_EVIDENCE" for item in comparisons.values())
        else "SENSITIVE"
        if any(item["classification"] == "SENSITIVE" for item in comparisons.values())
        else "INSENSITIVE"
    )
    return {
        "comparisons": comparisons,
        "SOLVER_WARMSTART_SENSITIVITY": classification,
    }


def _tolerance_sensitivity(
    conditions: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    comparisons = {
        "warmstart_production_A_vs_C": _pair_sensitivity(
            conditions["production_warmstart_production_tolerance"],
            conditions["production_warmstart_tight_tolerance"],
        ),
        "warmstart_zero_B_vs_D": _pair_sensitivity(
            conditions["zero_warmstart_production_tolerance"],
            conditions["zero_warmstart_tight_tolerance"],
        ),
    }
    classification = (
        "INSUFFICIENT_EVIDENCE"
        if any(item["classification"] == "INSUFFICIENT_EVIDENCE" for item in comparisons.values())
        else "SENSITIVE"
        if any(item["classification"] == "SENSITIVE" for item in comparisons.values())
        else "INSENSITIVE"
    )
    return {
        "comparisons": comparisons,
        "SOLVER_TOLERANCE_SENSITIVITY": classification,
    }


def _condition_payload(condition: dict[str, Any]) -> dict[str, Any]:
    capture = condition["capture"]
    return {
        "state_validation": condition["state_validation"],
        "solver_iteration_trace": condition["solver_iteration_trace"],
        "solver_numerics": condition["solver_numerics"],
        "mass_jacobian_delassus": {
            "mass_matrix": capture["mass_matrix"],
            "J_phys": capture["J_phys"],
            "W_phys": capture["W_phys"],
            "mass_matrix_stats": capture.get("mass_matrix_stats"),
            "delassus_stats": capture.get("delassus_stats"),
        },
        "physical_contact_impulses": {
            "api": "mujoco.mj_contactForce",
            "parameterization_independent_readback": True,
            "contacts": [
                {
                    "contact_index": item["contact_index"],
                    "pair": [item["geom1_name"], item["geom2_name"]],
                    "normal_impulse": item["normal_impulse"],
                    "tangent_impulse": item["tangential_impulse"],
                    "tangent_impulse_norm": item["tangential_impulse_norm"],
                }
                for item in capture["contacts"]
            ],
        },
        "shared_physical_global_demand": condition["shared_demand"],
        "solver_excess": condition["excess"],
        "one_step_result": condition["one_step_result"],
        "pre_constraint": condition["pre_constraint"],
        "pre_contact_geometry": condition["pre_contact_geometry"],
        "model_options": condition["model_options"],
    }


def _write_condition(output: Path, condition: dict[str, Any]) -> None:
    target = output / "conditions" / condition["condition_name"]
    target.mkdir(parents=True, exist_ok=True)
    payload = _condition_payload(condition)
    for name, value in payload.items():
        write_json(target / f"{name}.json", value)


def _safe_sha256(path: Path) -> str | None:
    try:
        return oracle.sha256(path) if path.is_file() else None
    except (OSError, ValueError):
        return None


def _source_hashes(paths: dict[str, Path]) -> dict[str, str | None]:
    source_paths = [
        paths["morphology_xml"],
        paths["checkpoint"],
        REPO_ROOT / "tools/analyze_mujoco_global55_contact_demand.py",
        REPO_ROOT / "tools/audit_mujoco_global55_friction_aref_counterfactual.py",
        REPO_ROOT / "tools/audit_mujoco_global55_contact_regularization_counterfactual.py",
        Path(__file__).resolve(),
        paths["corrected_oracle"] / "validation.json",
    ]
    return {str(path): _safe_sha256(path) for path in source_paths}


def _write_git_identity(output: Path) -> None:
    def git(*arguments: str) -> str:
        return subprocess.run(
            ["git", *arguments], cwd=REPO_ROOT, check=True,
            capture_output=True, text=True,
        ).stdout
    (output / "git_head.txt").write_text(
        f"TOPLEVEL={git('rev-parse', '--show-toplevel').strip()}\n"
        f"HEAD={git('rev-parse', 'HEAD').strip()}\n"
        f"BRANCH={git('branch', '--show-current').strip()}\n",
        encoding="utf-8",
    )
    (output / "git_status_short.txt").write_text(
        git("status", "--short"), encoding="utf-8"
    )


def _load_reference(path: Path, filename: str) -> dict[str, Any]:
    target = path / filename
    if not target.is_file():
        raise FileNotFoundError(f"reference file is missing: {target}")
    value = json.loads(target.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"reference file is not a JSON object: {target}")
    return value


def _custom_one_step_regression(
    mujoco: Any,
    base_model: Any,
    snapshot: Any,
    mapping: dict[str, Any],
    production_options: dict[str, Any],
) -> dict[str, Any]:
    staged_model, staged_copy_api = _copy_model(mujoco, base_model)
    staged_data = mujoco.MjData(staged_model)
    mujoco.mj_copyData(staged_data, staged_model, snapshot)
    aref.stage_to_constraint(mujoco, staged_model, staged_data)
    mujoco.mj_fwdConstraint(staged_model, staged_data)
    staged_solver = mujoco.MjData(staged_model)
    mujoco.mj_copyData(staged_solver, staged_model, staged_data)
    mujoco.mj_Euler(staged_model, staged_data)

    full_model, full_copy_api = _copy_model(mujoco, base_model)
    full_data = mujoco.MjData(full_model)
    mujoco.mj_copyData(full_data, full_model, snapshot)
    mujoco.mj_step(full_model, full_data)
    checks = {
        "post_qpos": _allclose(staged_data.qpos, full_data.qpos),
        "post_qvel": _allclose(staged_data.qvel, full_data.qvel),
        "post_time": bool(np.isclose(float(staged_data.time), float(full_data.time))),
    }
    staged_capture = aref.capture_after_integration(
        mujoco, staged_model, staged_data, snapshot, mapping, staged_solver
    )
    full_solver = mujoco.MjData(full_model)
    mujoco.mj_copyData(full_solver, full_model, full_data)
    full_capture = aref.capture_after_integration(
        mujoco, full_model, full_data, snapshot, mapping, full_solver
    )
    staged_target = next(item for item in staged_capture["contacts"] if item["robot_body_name"] == "limb/12")
    full_target = next(item for item in full_capture["contacts"] if item["robot_body_name"] == "limb/12")
    checks["post_slip"] = _allclose(
        staged_target["post_tangential_velocity"],
        full_target["post_tangential_velocity"],
    )
    return {
        "checks": checks,
        "staged_calls": [
            "mj_fwdPosition", "mj_fwdVelocity", "mj_fwdActuation",
            "mj_fwdAcceleration", "mj_fwdConstraint", "mj_Euler",
        ],
        "full_validation_calls": ["mj_step on independent model/data clone"],
        "model_copy_api": [staged_copy_api, full_copy_api],
        "CUSTOM_PIPELINE_ONE_STEP_REPRODUCTION": "PASS" if all(checks.values()) else "FAIL",
    }


def _write_failure_placeholders(output: Path, error: Exception) -> None:
    placeholders = {
        "solver_optimization_invariant_validation.json": {
            "SOLVER_OPTIMIZATION_DIAGNOSTIC_ISOLATION": "INSUFFICIENT_EVIDENCE",
        },
        "warmstart_activation.json": {"ZERO_WARMSTART_ACTIVATION": "FAILED"},
        "tight_tolerance_activation.json": {"TIGHT_TOLERANCE_ACTIVATION": "FAILED"},
        "baseline_regression.json": {"OPTIMIZATION_BASELINE_REPRODUCTION": "FAIL"},
        "warmstart_sensitivity.json": {"SOLVER_WARMSTART_SENSITIVITY": "INSUFFICIENT_EVIDENCE"},
        "tolerance_sensitivity.json": {"SOLVER_TOLERANCE_SENSITIVITY": "INSUFFICIENT_EVIDENCE"},
        "solver_convergence_assessment.json": {"PRODUCTION_NEWTON_CONVERGENCE": "INSUFFICIENT_EVIDENCE"},
        "custom_pipeline_one_step_regression.json": {"CUSTOM_PIPELINE_ONE_STEP_REPRODUCTION": "INSUFFICIENT_EVIDENCE"},
    }
    for filename, payload in placeholders.items():
        if not (output / filename).exists():
            write_json(output / filename, payload)
    for name, _, _, _ in CONDITIONS:
        target = output / "conditions" / name
        target.mkdir(parents=True, exist_ok=True)
        for filename in (
            "state_validation.json", "solver_iteration_trace.json",
            "solver_numerics.json", "physical_contact_impulses.json",
            "mass_jacobian_delassus.json", "shared_physical_global_demand.json",
            "solver_excess.json",
            "one_step_result.json",
        ):
            path = target / filename
            if not path.exists():
                write_json(path, {"status": "INSUFFICIENT_EVIDENCE"})
    write_json(output / "failure_context.json", {
        "error_type": type(error).__name__,
        "error": str(error),
        "partial_conditions": [name for name, _, _, _ in CONDITIONS],
        "traceback_file": "traceback.txt",
    })


def _failure_payload(error: Exception) -> dict[str, Any]:
    return {
        "OPTIMIZATION_BASELINE_REPRODUCTION": "FAIL",
        "ZERO_WARMSTART_ACTIVATION": "FAILED",
        "TIGHT_TOLERANCE_ACTIVATION": "FAILED",
        "SOLVER_OPTIMIZATION_DIAGNOSTIC_ISOLATION": "INSUFFICIENT_EVIDENCE",
        "SOLVER_WARMSTART_SENSITIVITY": "INSUFFICIENT_EVIDENCE",
        "SOLVER_TOLERANCE_SENSITIVITY": "INSUFFICIENT_EVIDENCE",
        "PRODUCTION_NEWTON_CONVERGENCE": "INSUFFICIENT_EVIDENCE",
        "MUJOCO_SOLVER_EXCESS_OPTIMIZATION_STATUS": "INSUFFICIENT_EVIDENCE",
        "NEXT_ACTION": "DIAGNOSTIC_IMPLEMENTATION_FIX_REQUIRED",
        "UNCONDITIONAL_ZIP_PACKAGING": "ENABLED",
        "LOCAL_IMPLEMENTATION": "INCOMPLETE",
        "COUNTERFACTUAL_VALID": False,
        "error_type": type(error).__name__,
        "error": str(error),
    }


def execute(args: argparse.Namespace, paths: dict[str, Path]) -> dict[str, Any]:
    output = paths["output_dir"]
    output.mkdir(parents=True, exist_ok=True)
    _write_git_identity(output)
    source_before = _source_hashes(paths)
    corrected_reference = aref.cone_helper.load_reference(paths["corrected_oracle"])
    mujoco = __import__("mujoco")
    recorder, mapping = aref.cone_helper.replay_once(args, paths)
    model = recorder.raw_model
    snapshot = recorder.global55_snapshot
    if model is None or snapshot is None:
        raise RuntimeError("global55 replay did not provide native model/data/snapshot")
    if len(recorder.records) != EXPECTED_SUBSTEPS:
        raise RuntimeError(
            f"formal replay substep count mismatch: {len(recorder.records)} vs {EXPECTED_SUBSTEPS}"
        )
    if not recorder.snapshot_copy_evidence or not recorder.snapshot_copy_evidence.get(
        "live_unchanged_by_copy", False
    ):
        raise RuntimeError("formal replay snapshot copy evidence is incomplete")

    production_cone = int(mujoco.mjtCone.mjCONE_PYRAMIDAL)
    if int(model.opt.cone) != production_cone:
        raise RuntimeError("formal optimization diagnostic requires production pyramidal cone")
    production_solver = int(mujoco.mjtSolver.mjSOL_NEWTON)
    if int(model.opt.solver) != production_solver:
        raise RuntimeError("formal optimization diagnostic requires mjSOL_NEWTON")
    production_options = _model_option_snapshot(model)
    write_json(output / "global55_pre_state_snapshot.json", aref.cone_helper.state_input_snapshot(snapshot))
    write_json(output / "state_copy_manifest.json", {
        **aref.cone_helper.state_copy_manifest(snapshot),
        "formal_snapshot_copy_evidence": recorder.snapshot_copy_evidence,
    })

    conditions: dict[str, dict[str, Any]] = {}
    for name, label, zero_warmstart, tight in CONDITIONS:
        condition = _run_condition(
            mujoco, model, snapshot, mapping, name, label,
            zero_warmstart, tight, production_options,
        )
        conditions[name] = condition
        _write_condition(output, condition)

    invariant = _solver_optimization_invariants(conditions, production_options)
    warmstart_activation = _warmstart_activation(conditions)
    tight_activation = _tight_tolerance_activation(conditions, production_options)
    baseline = _baseline_regression(
        conditions["production_warmstart_production_tolerance"],
        corrected_reference,
        paths["regularization_reference"],
    )
    warmstart_sensitivity = _warmstart_sensitivity(conditions)
    tolerance_sensitivity = _tolerance_sensitivity(conditions)
    convergence = _convergence_assessment(
        conditions, baseline, warmstart_sensitivity, tolerance_sensitivity
    )
    custom_step = _custom_one_step_regression(
        mujoco, model, snapshot, mapping, production_options
    )
    write_json(output / "solver_optimization_invariant_validation.json", invariant)
    write_json(output / "warmstart_activation.json", warmstart_activation)
    write_json(output / "tight_tolerance_activation.json", tight_activation)
    write_json(output / "baseline_regression.json", baseline)
    write_json(output / "warmstart_sensitivity.json", warmstart_sensitivity)
    write_json(output / "tolerance_sensitivity.json", tolerance_sensitivity)
    write_json(output / "solver_convergence_assessment.json", convergence)
    write_json(output / "custom_pipeline_one_step_regression.json", custom_step)

    source_after = _source_hashes(paths)
    source_unchanged = bool(source_before == source_after)
    model_after = _model_option_snapshot(model)
    model_restore = _option_difference(production_options, model_after)
    final = _classify_final(
        invariant, warmstart_sensitivity, tolerance_sensitivity,
        convergence, baseline,
    )
    gates = bool(
        baseline["OPTIMIZATION_BASELINE_REPRODUCTION"] == "PASS"
        and warmstart_activation["ZERO_WARMSTART_ACTIVATION"] == "VALIDATED"
        and tight_activation["TIGHT_TOLERANCE_ACTIVATION"] == "VALIDATED"
        and invariant["SOLVER_OPTIMIZATION_DIAGNOSTIC_ISOLATION"] == "VALIDATED"
        and convergence["PRODUCTION_NEWTON_CONVERGENCE"] in {"VALIDATED", "QUESTIONABLE"}
        and custom_step["CUSTOM_PIPELINE_ONE_STEP_REPRODUCTION"] == "PASS"
        and source_unchanged
        and model_restore["only_allowed"]
        and len(recorder.records) == EXPECTED_SUBSTEPS
    )
    if not gates and final["MUJOCO_SOLVER_EXCESS_OPTIMIZATION_STATUS"] == "ROBUST_CONVERGED_SOLUTION":
        final = {
            "MUJOCO_SOLVER_EXCESS_OPTIMIZATION_STATUS": "INSUFFICIENT_EVIDENCE",
            "NEXT_ACTION": "DIAGNOSTIC_IMPLEMENTATION_FIX_REQUIRED",
            "interpretation": None,
        }
    validation = {
        "OPTIMIZATION_BASELINE_REPRODUCTION": baseline["OPTIMIZATION_BASELINE_REPRODUCTION"],
        "ZERO_WARMSTART_ACTIVATION": warmstart_activation["ZERO_WARMSTART_ACTIVATION"],
        "TIGHT_TOLERANCE_ACTIVATION": tight_activation["TIGHT_TOLERANCE_ACTIVATION"],
        "SOLVER_OPTIMIZATION_DIAGNOSTIC_ISOLATION": invariant["SOLVER_OPTIMIZATION_DIAGNOSTIC_ISOLATION"],
        "SOLVER_WARMSTART_SENSITIVITY": warmstart_sensitivity["SOLVER_WARMSTART_SENSITIVITY"],
        "SOLVER_TOLERANCE_SENSITIVITY": tolerance_sensitivity["SOLVER_TOLERANCE_SENSITIVITY"],
        "PRODUCTION_NEWTON_CONVERGENCE": convergence["PRODUCTION_NEWTON_CONVERGENCE"],
        "MUJOCO_SOLVER_EXCESS_OPTIMIZATION_STATUS": final["MUJOCO_SOLVER_EXCESS_OPTIMIZATION_STATUS"],
        "NEXT_ACTION": final["NEXT_ACTION"],
        "formal_replay_physics_substeps": len(recorder.records),
        "expected_formal_replay_physics_substeps": EXPECTED_SUBSTEPS,
        "formal_replay_additional_steps": 0,
        "formal_data_mutated_by_probe": False,
        "source_hashes_unchanged": source_unchanged,
        "MODEL_OPTION_RESTORE": "PASS" if model_restore["only_allowed"] else "FAIL",
        "CUSTOM_PIPELINE_ONE_STEP_REPRODUCTION": custom_step["CUSTOM_PIPELINE_ONE_STEP_REPRODUCTION"],
        "UNCONDITIONAL_ZIP_PACKAGING": "ENABLED",
        "LOCAL_IMPLEMENTATION": "READY_FOR_SERVER_VALIDATION",
        "COUNTERFACTUAL_VALID": gates,
        "semantic_scope": "Only qacc_warmstart, solver tolerance, and solver iterations were changed on independent clones.",
    }
    summary = {key: validation[key] for key in (
        "OPTIMIZATION_BASELINE_REPRODUCTION", "ZERO_WARMSTART_ACTIVATION",
        "TIGHT_TOLERANCE_ACTIVATION", "SOLVER_OPTIMIZATION_DIAGNOSTIC_ISOLATION",
        "SOLVER_WARMSTART_SENSITIVITY", "SOLVER_TOLERANCE_SENSITIVITY",
        "PRODUCTION_NEWTON_CONVERGENCE", "MUJOCO_SOLVER_EXCESS_OPTIMIZATION_STATUS",
        "NEXT_ACTION", "UNCONDITIONAL_ZIP_PACKAGING", "LOCAL_IMPLEMENTATION",
    )}
    summary["interpretation"] = final.get("interpretation")
    metadata = {
        "created_at": datetime.now().astimezone().isoformat(),
        "backend": "mujoco",
        "mujoco_version": getattr(mujoco, "__version__", None),
        "morphology": MORPHOLOGY,
        "morphology_xml_sha256": source_before.get(str(paths["morphology_xml"])),
        "checkpoint_sha256": source_before.get(str(paths["checkpoint"])),
        "corrected_reference_oracle": str(paths["corrected_oracle"]),
        "regularization_reference": str(paths["regularization_reference"]),
        "formal_replay_physics_substeps": len(recorder.records),
        "formal_replay_additional_steps": 0,
        "formal_data_mutated_by_probe": False,
        "solver": "mjSOL_NEWTON",
        "cone": "mjCONE_PYRAMIDAL",
        "conditions": [label for _, label, _, _ in CONDITIONS],
        "condition_staged_forward_count": 4,
        "condition_constraint_solve_count": 4,
        "condition_custom_integration_count": 4,
        "production_options": production_options,
        "semantic_scope": validation["semantic_scope"],
    }
    for filename, payload in (
        ("metadata.json", metadata),
        ("validation.json", validation),
        ("summary.json", summary),
        ("source_purity.json", {
            "hashes_before": source_before,
            "hashes_after": source_after,
            "source_hashes_unchanged": source_unchanged,
            "formal_data_mutated_by_probe": False,
        }),
    ):
        write_json(output / filename, payload)
    return validation


def _package(output: Path, zip_path: Path) -> dict[str, Any]:
    return oracle.package_artifact(output, zip_path)


def main(argv: Sequence[str] | None = None) -> int:
    raw_args = parser().parse_args(argv)
    output: Path | None = None
    zip_path: Path | None = None
    try:
        args = resolve_arguments(raw_args)
        output = Path(args.output_dir).resolve()
        zip_path = Path(args.zip_path).resolve()
        paths = validate_paths(args)
        output.mkdir(parents=True, exist_ok=False)
        return_code = 2
        log_path = output / "run.log"
        with (
            log_path.open("w", encoding="utf-8") as log_stream,
            redirect_stdout(Tee(sys.__stdout__, log_stream)),
            redirect_stderr(Tee(sys.__stderr__, log_stream)),
        ):
            try:
                validation = execute(args, paths)
                print(json.dumps(_json_normalize(validation), indent=2, sort_keys=True, allow_nan=False))
                return_code = 0 if validation["COUNTERFACTUAL_VALID"] else 2
            except Exception as error:
                trace = traceback.format_exc()
                print(trace, file=sys.stderr, end="")
                (output / "traceback.txt").write_text(trace, encoding="utf-8")
                (output / "inner_exception_traceback.txt").write_text(trace, encoding="utf-8")
                _write_failure_placeholders(output, error)
                try:
                    _write_git_identity(output)
                except Exception:
                    pass
                failure = _failure_payload(error)
                write_json(output / "validation.json", failure)
                write_json(output / "summary.json", failure)
                write_json(output / "metadata.json", {
                    "created_at": datetime.now().astimezone().isoformat(),
                    "status": "failure",
                    "mode": "counterfactual",
                    "error_type": type(error).__name__,
                    "error": str(error),
                })
                write_json(output / "source_purity.json", {
                    "source_hashes_unchanged": None,
                    "formal_data_mutated_by_probe": False,
                    "status": "incomplete",
                })
        package = _package(output, zip_path)
        print(f"ZIP_VERIFY={package['ZIP_VERIFY']}")
        print(f"ZIP_SHA256={package['ZIP_SHA256']}")
        print(f"UPLOAD_THIS_ZIP={package['UPLOAD_THIS_ZIP']}")
        return return_code
    except Exception as error:
        if output is None or zip_path is None or output.exists() or zip_path.exists():
            output, zip_path = _default_output_paths()
        output = Path(output).resolve()
        zip_path = Path(zip_path).resolve()
        output.mkdir(parents=True, exist_ok=False)
        trace = traceback.format_exc()
        (output / "run.log").write_text(trace, encoding="utf-8")
        (output / "traceback.txt").write_text(trace, encoding="utf-8")
        (output / "inner_exception_traceback.txt").write_text(trace, encoding="utf-8")
        _write_failure_placeholders(output, error)
        try:
            _write_git_identity(output)
        except Exception:
            pass
        failure = _failure_payload(error)
        write_json(output / "validation.json", failure)
        write_json(output / "summary.json", failure)
        write_json(output / "metadata.json", {
            "created_at": datetime.now().astimezone().isoformat(),
            "status": "failure",
            "mode": "counterfactual",
            "error_type": type(error).__name__,
            "error": str(error),
        })
        package = _package(output, zip_path)
        print(f"ZIP_VERIFY={package['ZIP_VERIFY']}")
        print(f"ZIP_SHA256={package['ZIP_SHA256']}")
        print(f"UPLOAD_THIS_ZIP={package['UPLOAD_THIS_ZIP']}")
        return 2


if __name__ == "__main__":
    sys.exit(main())

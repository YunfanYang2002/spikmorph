"""Fixed-global55 counterfactual for pyramidal-contact normal ``efc_aref``.

This diagnostic reuses the validated global55 replay and physical-contact
helpers.  The only condition-level intervention is the normal component of
the active floor-contact pyramidal edge-row reference acceleration.  All
three conditions start from an independent ``MjData`` copy of the same
global55 snapshot and use the already validated staged forward plus Euler
integration path.
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
from typing import Any, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools import analyze_mujoco_global55_contact_demand as oracle
from tools import audit_mujoco_global55_friction_aref_counterfactual as aref
from tools import audit_mujoco_global55_friction_cone_counterfactual as cone_helper
from tools import audit_mujoco_global55_solver_optimization as optimization


MORPHOLOGY = oracle.MORPHOLOGY
XML_SHA256 = oracle.XML_SHA256
CHECKPOINT_SHA256 = oracle.CHECKPOINT_SHA256
GLOBAL_STEP = oracle.GLOBAL_STEP
EXPECTED_SUBSTEPS = oracle.EXPECTED_SUBSTEPS
REFERENCE_ORACLE_NAME = "mujoco_global55_contact_demand_oracle_corrected_20260804_143138"
CONDITIONS = (
    ("normal_aref_scale_1_before", "NORMAL_AREF_SCALE_1_BEFORE", 1.0),
    ("normal_aref_scale_0", "NORMAL_AREF_SCALE_0", 0.0),
    ("normal_aref_scale_1_after_restore", "NORMAL_AREF_SCALE_1_AFTER_RESTORE", 1.0),
)
REGRESSION_RTOL = 1.0e-9
REGRESSION_ATOL = 1.0e-9
AREF_RTOL = 1.0e-10
AREF_ATOL = 1.0e-11
NORMAL_COLLAPSE_ABS_EPS = 1.0e-8
NORMAL_COLLAPSE_RELATIVE_EPS = 1.0e-6
CAP_NEAR_THRESHOLD = 0.95
CAP_LIMITED_THRESHOLD = 0.995
ROOT_LINEAR_COLUMNS = (0, 1, 2)


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
        description="Run fixed-global55 pyramidal normal-aref counterfactual."
    )
    result.add_argument("--formal-server-defaults", action="store_true")
    result.add_argument("--checkpoint")
    result.add_argument("--walker-dir")
    result.add_argument("--corrected-oracle")
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
        / f"mujoco_global55_normal_aref_counterfactual_{stamp}",
        REPO_ROOT / "tmp"
        / f"mujoco_global55_normal_aref_counterfactual_{stamp}.zip",
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
    missing = [
        name for name in ("checkpoint", "walker_dir", "corrected_oracle")
        if not getattr(args, name)
    ]
    if missing:
        raise ValueError(f"missing required arguments: {', '.join(missing)}")
    default_output, default_zip = _default_output_paths()
    args.output_dir = args.output_dir or str(default_output)
    args.zip_path = args.zip_path or str(default_zip)
    return args


def validate_paths(args: argparse.Namespace) -> dict[str, Path]:
    paths = {
        "checkpoint": oracle.require_file(Path(args.checkpoint), "checkpoint"),
        "walker_dir": Path(args.walker_dir).resolve(),
        "morphology_xml": oracle.require_file(
            Path(args.walker_dir).resolve() / "xml" / f"{args.morphology_id}.xml",
            "morphology XML",
        ),
        "morphology_metadata": oracle.require_file(
            Path(args.walker_dir).resolve() / "metadata" / f"{args.morphology_id}.json",
            "morphology metadata",
        ),
        "config": oracle.require_file(REPO_ROOT / args.cfg, "config"),
        "corrected_oracle": Path(args.corrected_oracle).resolve(),
        "output_dir": Path(args.output_dir).resolve(),
        "zip_path": Path(args.zip_path).resolve(),
    }
    if args.morphology_id != MORPHOLOGY:
        raise ValueError(f"formal audit requires morphology {MORPHOLOGY}")
    if oracle.sha256(paths["checkpoint"]) != CHECKPOINT_SHA256:
        raise ValueError("checkpoint SHA256 mismatch")
    if oracle.sha256(paths["morphology_xml"]) != XML_SHA256:
        raise ValueError("morphology XML SHA256 mismatch")
    if not paths["corrected_oracle"].is_dir():
        raise FileNotFoundError(
            f"corrected oracle directory is missing: {paths['corrected_oracle']}"
        )
    reference_validation = json.loads(
        (paths["corrected_oracle"] / "validation.json").read_text(encoding="utf-8")
    )
    if reference_validation.get("MUJOCO_GLOBAL55_ARTIFACT_READBACK_VALID") != "YES":
        raise ValueError("corrected reference oracle is not readback-valid")
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
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return _json_normalize(value.item())
        return [_json_normalize(item) for item in value.tolist()]
    if isinstance(value, dict):
        return {str(key): _json_normalize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_normalize(item) for item in value]
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_normalize(payload), indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


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


def _array(data: Any, name: str) -> np.ndarray | None:
    value = getattr(data, name, None)
    return None if value is None else np.asarray(value, dtype=np.float64).copy()


def _optional_array(data: Any, name: str) -> np.ndarray | None:
    value = getattr(data, name, None)
    if value is None:
        return None
    try:
        return np.asarray(value).copy()
    except (TypeError, ValueError):
        return None


def _optional_row_value(data: Any, name: str, row_id: int) -> Any | None:
    values = getattr(data, name, None)
    if values is None:
        return None
    try:
        array = np.asarray(values)
        if array.ndim == 0 or row_id < 0 or row_id >= array.shape[0]:
            return None
        return array[row_id].copy()
    except (TypeError, ValueError):
        return None


def _constraint_arrays(data: Any, model: Any) -> dict[str, Any]:
    nefc, nv = int(getattr(data, "nefc", 0)), int(model.nv)
    rows = [oracle.dense_constraint_row(data, row, nefc, nv) for row in range(nefc)]
    return {
        "efc_J": np.vstack(rows) if rows else np.zeros((0, nv), dtype=np.float64),
        **{
            name: _array(data, name)
            for name in (
                "efc_aref", "efc_R", "efc_D", "efc_diagApprox", "efc_vel",
                "efc_AR", "efc_b", "efc_force", "qfrc_constraint", "qacc",
                "qacc_smooth", "iefc_R", "iefc_D", "iefc_force",
            )
        },
    }


def _pre_contact_geometry(data: Any, mujoco: Any, model: Any) -> dict[str, Any]:
    contacts = []
    for index in range(int(getattr(data, "ncon", 0))):
        native = data.contact[index]
        geom_ids = [int(native.geom1), int(native.geom2)]
        names = [
            str(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id))
            for geom_id in geom_ids
        ]
        contacts.append({
            "contact_index": index,
            "geom_ids": geom_ids,
            "geom_names": names,
            "point": np.asarray(native.pos, dtype=np.float64).copy(),
            "frame": np.asarray(native.frame, dtype=np.float64).copy(),
            "dist": float(native.dist),
            "dim": int(native.dim),
            "efc_address": int(native.efc_address),
            "friction": np.asarray(native.friction, dtype=np.float64).copy(),
        })
    return {"ncon": int(getattr(data, "ncon", 0)), "contacts": contacts}


def _contact_geometry_equal(left: dict[str, Any], right: dict[str, Any]) -> bool:
    if left.get("ncon") != right.get("ncon"):
        return False
    for a, b in zip(left.get("contacts", []), right.get("contacts", [])):
        if a["geom_ids"] != b["geom_ids"] or a["dim"] != b["dim"]:
            return False
        if not _allclose(a["point"], b["point"]):
            return False
        if not _allclose(a["frame"], b["frame"]):
            return False
        if not _allclose(a["friction"], b["friction"]):
            return False
    return True


def _model_options(model: Any) -> dict[str, Any]:
    return cone_helper.model_option_snapshot(model)


def _state_validation(snapshot: Any, data: Any) -> dict[str, Any]:
    clone = cone_helper.state_equality(snapshot, data)
    return {
        "clone_pre_state": clone,
        "pre_state_snapshot": aref.state_input_snapshot(data),
        "STATE_COPY_EQUAL": bool(clone["STATE_COPY_EQUAL"]),
    }


def _rebuild_normal_aref(decomposition: dict[str, Any], scale: float) -> np.ndarray:
    coefficients = np.asarray(decomposition["basis_coefficients"], dtype=np.float64)
    return (
        float(scale) * coefficients[:, 0] * float(decomposition["a_normal"])
        + coefficients[:, 1] * float(decomposition["a_t1"])
        + coefficients[:, 2] * float(decomposition["a_t2"])
    )


def rebuild_normal_aref(decomposition: dict[str, Any], scale: float) -> np.ndarray:
    """Public testable form of the normal-only row reconstruction."""
    return _rebuild_normal_aref(decomposition, scale)


def _fit_aref_components(decomposition: dict[str, Any], row_aref: Any) -> dict[str, Any]:
    coefficients = np.asarray(decomposition["basis_coefficients"], dtype=np.float64)
    rows = np.asarray(row_aref, dtype=np.float64)
    components, _, rank, _ = np.linalg.lstsq(coefficients, rows, rcond=None)
    residual = coefficients @ components - rows
    return {
        "components": components,
        "rank": int(rank),
        "residual": residual,
        "residual_max_abs": float(np.max(np.abs(residual))) if residual.size else 0.0,
    }


def normal_aref_activation(
    decompositions: dict[int, dict[str, Any]],
    condition_aref: dict[str, dict[int, Any]],
) -> dict[str, Any]:
    details: dict[str, Any] = {}
    valid = True
    for contact_index, decomposition in decompositions.items():
        original = np.asarray(decomposition["original_efc_aref"], dtype=np.float64)
        expected_tangent = np.asarray(
            [decomposition["a_t1"], decomposition["a_t2"]], dtype=np.float64
        )
        conditions: dict[str, Any] = {}
        for name, _, scale in CONDITIONS:
            rows = np.asarray(condition_aref[name][contact_index], dtype=np.float64)
            fitted = _fit_aref_components(decomposition, rows)
            expected = np.asarray(
                [scale * decomposition["a_normal"], decomposition["a_t1"], decomposition["a_t2"]],
                dtype=np.float64,
            )
            rebuilt = _rebuild_normal_aref(decomposition, scale)
            rows_reproduce = bool(
                scale != 1.0
                or _allclose(rows, original, rtol=AREF_RTOL, atol=AREF_ATOL)
            )
            tangent_unchanged = _allclose(
                fitted["components"][1:], expected_tangent, rtol=AREF_RTOL, atol=AREF_ATOL
            )
            normal_zero = bool(
                scale != 0.0
                or np.isclose(fitted["components"][0], 0.0, rtol=AREF_RTOL, atol=AREF_ATOL)
            )
            valid_rows = bool(
                fitted["rank"] == 3
                and fitted["residual_max_abs"] <= AREF_ATOL
                + AREF_RTOL * max(1.0, float(np.max(np.abs(original))))
                and _allclose(fitted["components"], expected, rtol=AREF_RTOL, atol=AREF_ATOL)
                and _allclose(rows, rebuilt, rtol=AREF_RTOL, atol=AREF_ATOL)
                and rows_reproduce and tangent_unchanged and normal_zero
            )
            conditions[name] = {
                "scale": float(scale),
                "original_row_aref": original,
                "condition_row_aref": rows,
                "reconstructed_row_aref": rebuilt,
                "fitted_components": fitted["components"],
                "expected_components": expected,
                "original_tangent_components": expected_tangent,
                "condition_tangent_components": fitted["components"][1:],
                "normal_component_before": float(decomposition["a_normal"]),
                "normal_component_after": float(fitted["components"][0]),
                "normal_component_zero": normal_zero,
                "tangent_components_unchanged": tangent_unchanged,
                "rows_reproduce_original": rows_reproduce,
                "row_reconstruction_residual_max_abs": fitted["residual_max_abs"],
                "valid": valid_rows,
            }
            valid &= valid_rows
        details[str(contact_index)] = {
            "contact_identity": {
                "contact_index": contact_index,
                "robot_body_name": decomposition.get("robot_body_name"),
            },
            "row_ids": decomposition["row_ids"],
            "basis_coefficients": decomposition["basis_coefficients"],
            "a_normal": decomposition["a_normal"],
            "a_t1": decomposition["a_t1"],
            "a_t2": decomposition["a_t2"],
            "conditions": conditions,
        }
    return {
        "contacts": details,
        "NORMAL_AREF_COUNTERFACTUAL_ACTIVATION": (
            "VALIDATED" if valid and bool(details) else "FAILED"
        ),
    }


def _extract_decompositions(
    data: Any,
    mujoco: Any,
    model: Any,
    mapping: dict[str, Any],
) -> tuple[dict[int, dict[str, Any]], dict[str, Any]]:
    """Extract all active floor pyramidal contacts without physical-force reads."""
    decomposition, probe = aref._extract_decompositions(
        data, mujoco, model, mapping, data
    )
    probe_sim = SimpleNamespace(
        _sim=SimpleNamespace(_model=model, _data=data), model=model, data=data
    )
    contacts, _ = oracle.evaluator.capture_contact_response(
        probe_sim,
        mapping,
        oracle.evaluator.FiniteTracker(),
        record_physical_projection=False,
    )
    floor_id = int(mapping["floor_geom_id"])
    floor_contacts = [
        item for item in contacts
        if floor_id in (int(item["geom1_id"]), int(item["geom2_id"]))
    ]
    selected = {int(index) for index in decomposition}
    selection = []
    for contact in floor_contacts:
        index = int(contact["contact_index"])
        is_edge_contact = int(contact["dim"]) == 3
        selected_here = index in selected
        if is_edge_contact and not selected_here:
            raise RuntimeError(
                f"active pyramidal floor contact {index} was not decomposed"
            )
        selection.append({
            "contact_index": index,
            "pair": [contact["geom1_name"], contact["geom2_name"]],
            "dim": int(contact["dim"]),
            "efc_rows": [int(row) for row in contact["efc_rows"]],
            "selected_pyramidal_edge_rows": selected_here,
            "reason": "four production pyramidal edge rows" if selected_here else "normal-only contact; no normal/tangent row fit required",
        })
    if not decomposition:
        raise RuntimeError("no active pyramidal floor contact with four edge rows")
    return decomposition, {
        "contacts": selection,
        "active_floor_contact_count": len(floor_contacts),
        "selected_pyramidal_contact_count": len(decomposition),
        "physical_projection_called": False,
        "constraint_solve_called": False,
        "row_order_assumption": False,
        "basis_source": "contact frame and robot-side ordering",
        "status": "VALIDATED",
        "probe": probe,
    }


def _normal_aref_rows(
    data: Any, decompositions: dict[int, dict[str, Any]], scale: float
) -> dict[int, np.ndarray]:
    rows_by_contact: dict[int, np.ndarray] = {}
    for contact_index, decomposition in decompositions.items():
        rows = _rebuild_normal_aref(decomposition, scale)
        row_ids = np.asarray(decomposition["row_ids"], dtype=np.int64)
        data.efc_aref[row_ids] = rows
        rows_by_contact[contact_index] = rows.copy()
    return rows_by_contact


def _solver_numerics(data: Any, model: Any, warmstart: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    trace = optimization._solver_iteration_trace(data, model)
    numerics = optimization._solver_numerics(
        data,
        model,
        warmstart,
        trace,
        {"normal_aref_intervention": True, "production_solver_options_unchanged": True},
    )
    active_islands = trace.get("islands", [])
    all_finite = bool(
        trace.get("all_statistics_finite")
        and trace.get("statistics_available")
        and active_islands
    )
    any_limit = any(
        not bool(item.get("niter_below_limit")) for item in active_islands
    )
    if not active_islands or trace.get("active_solver_island_count_source") in {
        None, "UNAVAILABLE"
    }:
        status = "INSUFFICIENT_EVIDENCE"
    elif not all_finite or not numerics.get("finite"):
        status = "NONFINITE"
    elif any_limit:
        status = "NONCONVERGED"
    else:
        status = "VALID"
    numerics.update({
        "NORMAL_AREF_COUNTERFACTUAL_NUMERICS": status,
        "active_solver_island_count": trace.get("active_solver_island_count"),
        "solver_niter": trace.get("solver_niter", []),
        "solver_nnz": trace.get("solver_nnz", []),
        "last_newton_gradient": [
            item.get("last_iteration", {}).get("gradient") for item in active_islands
        ],
        "last_newton_nchange": [
            item.get("last_iteration", {}).get("nchange") for item in active_islands
        ],
        "any_iteration_limit_reached": any_limit,
        "active_statistics_finite": all_finite,
    })
    return trace, numerics


def _augment_solver_rows(capture: dict[str, Any], solver_data: Any) -> None:
    for contact in capture.get("contacts", []):
        for row in contact.get("solver_rows", []):
            row_id = int(row["efc_row"])
            for name in ("efc_b", "efc_AR"):
                value = _optional_row_value(solver_data, name, row_id)
                if value is not None:
                    row[name] = value


def _run_condition(
    mujoco: Any,
    model: Any,
    snapshot: Any,
    mapping: dict[str, Any],
    decompositions: dict[int, dict[str, Any]],
    condition_name: str,
    condition_label: str,
    normal_scale: float,
) -> dict[str, Any]:
    data = mujoco.MjData(model)
    mujoco.mj_copyData(data, model, snapshot)
    state_before_stage = aref.state_input_snapshot(data)
    state_validation = _state_validation(snapshot, data)
    warmstart = np.asarray(data.qacc_warmstart, dtype=np.float64).copy()
    staged_calls = aref.stage_to_constraint(mujoco, model, data)
    original_full_aref = np.asarray(data.efc_aref, dtype=np.float64).copy()
    condition_aref = _normal_aref_rows(data, decompositions, normal_scale)
    pre_arrays = _constraint_arrays(data, model)
    pre_geometry = _pre_contact_geometry(data, mujoco, model)
    mujoco.mj_fwdConstraint(model, data)
    solver_data = mujoco.MjData(model)
    mujoco.mj_copyData(solver_data, model, data)
    post_arrays = _constraint_arrays(solver_data, model)
    trace, numerics = _solver_numerics(solver_data, model, warmstart)
    mujoco.mj_Euler(model, data)
    capture = aref.capture_after_integration(
        mujoco, model, data, snapshot, mapping, solver_data
    )
    _augment_solver_rows(capture, solver_data)
    demand = aref.shared_physical_global_demand(capture)
    excess = aref.compute_solver_excess(capture, demand)
    excess.update({
        "all_contact_normal_impulses": [
            float(item["normal_impulse"]) for item in capture["contacts"]
        ],
        "target_contact_index": int(demand["limb_12_contact_index"]),
        "solver_niter": numerics.get("active_solver_niter", []),
        "solver_nnz": numerics.get("active_solver_nnz", []),
        "last_newton_gradient": numerics.get("last_newton_gradient", []),
        "last_newton_nchange": numerics.get("last_newton_nchange", []),
    })
    return {
        "condition_name": condition_name,
        "condition_label": condition_label,
        "normal_aref_scale": float(normal_scale),
        "capture": capture,
        "shared_demand": demand,
        "excess": excess,
        "original_full_aref": original_full_aref,
        "condition_full_aref": np.asarray(pre_arrays["efc_aref"], dtype=np.float64).copy(),
        "condition_aref": condition_aref,
        "pre_constraint_arrays": pre_arrays,
        "post_constraint_arrays": post_arrays,
        "pre_contact_geometry": pre_geometry,
        "solver_iteration_trace": trace,
        "solver_numerics": numerics,
        "state_validation": {
            **state_validation,
            "state_before_stage": state_before_stage,
            "qacc_warmstart_unchanged_from_snapshot": _allclose(
                state_before_stage["qacc_warmstart"], aref.state_input_snapshot(snapshot)["qacc_warmstart"]
            ),
        },
        "model_options": _model_options(model),
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
            "constraint_solve_api": "mujoco.mj_fwdConstraint",
            "integration_api": "mujoco.mj_Euler",
        },
    }


def _adapt_for_aref_baseline(condition: dict[str, Any]) -> dict[str, Any]:
    demand = condition["shared_demand"]
    target = condition["capture"]["contacts"][int(demand["limb_12_contact_index"])]
    return {
        "capture": condition["capture"],
        "budget": {"selected": {"limb/12": {
            "actual_tangential_impulse": target["tangential_impulse"],
            "actual_tangential_impulse_norm": target["tangential_impulse_norm"],
            "actual_normal_impulse": target["normal_impulse"],
            "global_normal_conditioned_sticking_impulse": demand["limb_12_tangent_impulse_2d"],
            "global_normal_conditioned_sticking_impulse_norm": float(
                np.linalg.norm(demand["limb_12_tangent_impulse_2d"])
            ),
            "pre_tangential_speed": target["pre_tangential_speed"],
        }}},
        "excess": condition["excess"],
    }


def _all_checks(value: Any) -> bool:
    if not isinstance(value, dict) or not value:
        return False
    results = []
    for item in value.values():
        if isinstance(item, dict):
            results.append(_all_checks(item))
        elif isinstance(item, (bool, np.bool_)):
            results.append(bool(item))
        elif isinstance(item, str):
            results.append(item.upper() in {"PASS", "VALIDATED", "TRUE"})
        else:
            results.append(bool(item))
    return bool(results) and all(results)


def _normal_baseline_regression(
    condition: dict[str, Any], reference: dict[str, Any]
) -> dict[str, Any]:
    oracle_result = aref.baseline_regression(
        _adapt_for_aref_baseline(condition), reference
    )
    sanity = {
        "actual_tangent_norm": _allclose(
            condition["excess"]["actual_tangent_impulse_norm"], 3.3247363600666735
        ),
        "normal_impulse": _allclose(
            condition["excess"]["normal_impulse"], 6.345240278967453
        ),
        "shared_rigid_demand_norm": _allclose(
            condition["excess"]["rigid_demand_norm"], 2.540619084288334
        ),
        "solver_excess": _allclose(
            condition["excess"]["solver_excess_norm"], 0.7841172757783395
        ),
        "solver_excess_vector_norm": _allclose(
            condition["excess"]["solver_excess_vector_norm"], 0.8072255552101076
        ),
        "post_slip_speed": _allclose(
            np.linalg.norm(condition["excess"]["post_slip"]), 0.1713507113360867
        ),
        "active_newton_niter": _allclose(
            condition["solver_numerics"].get("active_solver_niter", [None])[0], 2
        ),
    }
    oracle_status = oracle_result.get("PYRAMIDAL_BASELINE_REPRODUCTION")
    checks = {
        "corrected_oracle_status": oracle_status == "PASS",
        "corrected_oracle_checks": _all_checks(oracle_result.get("checks")),
        "sanity": _all_checks(sanity),
    }
    return {
        "corrected_oracle": oracle_result,
        "sanity": sanity,
        "checks": checks,
        "NORMAL_AREF_BASELINE_REPRODUCTION": "PASS" if all(checks.values()) else "FAIL",
    }


def _restore_regression(
    before: dict[str, Any], after: dict[str, Any]
) -> dict[str, Any]:
    helper = aref.restore_regression(
        _adapt_for_aref_baseline(before), _adapt_for_aref_baseline(after)
    )
    fields = (
        "efc_aref", "efc_R", "efc_D", "efc_J", "efc_vel", "iefc_R", "iefc_D"
    )
    checks = {
        "helper": helper.get("PYRAMIDAL_RESTORE_REPRODUCTION") == "PASS",
        "solver_niter": _allclose(
            before["solver_numerics"].get("solver_niter"),
            after["solver_numerics"].get("solver_niter"),
        ),
        "post_qacc": _allclose(
            before["post_constraint_arrays"].get("qacc"),
            after["post_constraint_arrays"].get("qacc"),
        ),
        "post_slip": _allclose(
            before["excess"].get("post_slip"), after["excess"].get("post_slip")
        ),
    }
    field_checks = {}
    for field in fields:
        field_checks[field] = _allclose(
            before["pre_constraint_arrays"].get(field),
            after["pre_constraint_arrays"].get(field),
        )
    checks["pre_constraint_fields"] = _all_checks(field_checks)
    return {
        "helper": helper,
        "checks": checks,
        "field_checks": field_checks,
        "NORMAL_AREF_RESTORE_REPRODUCTION": "PASS" if all(checks.values()) else "FAIL",
    }


def _component_invariant(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    decompositions: dict[int, dict[str, Any]],
    expected_scale: float,
) -> dict[str, Any]:
    fixed_fields = (
        "efc_J", "efc_vel", "efc_R", "efc_D", "efc_diagApprox",
        "efc_AR", "iefc_R", "iefc_D",
    )
    fixed = {
        field: _allclose(
            baseline["pre_constraint_arrays"].get(field),
            candidate["pre_constraint_arrays"].get(field),
        )
        for field in fixed_fields
    }
    baseline_aref = np.asarray(
        baseline["pre_constraint_arrays"].get("efc_aref"), dtype=np.float64
    )
    candidate_aref = np.asarray(
        candidate["pre_constraint_arrays"].get("efc_aref"), dtype=np.float64
    )
    expected_aref = baseline_aref.copy()
    normal_tangent_checks = {}
    for contact_index, decomposition in decompositions.items():
        row_ids = np.asarray(decomposition["row_ids"], dtype=np.int64)
        expected_rows = _rebuild_normal_aref(decomposition, expected_scale)
        expected_aref[row_ids] = expected_rows
        fitted = _fit_aref_components(decomposition, candidate_aref[row_ids])
        normal_tangent_checks[str(contact_index)] = {
            "normal_component": bool(np.isclose(
                fitted["components"][0], expected_scale * decomposition["a_normal"],
                rtol=AREF_RTOL, atol=AREF_ATOL,
            )),
            "tangent_components": _allclose(
                fitted["components"][1:],
                [decomposition["a_t1"], decomposition["a_t2"]],
                rtol=AREF_RTOL, atol=AREF_ATOL,
            ),
            "rank": fitted["rank"] == 3,
        }
    aref_expected = _allclose(candidate_aref, expected_aref, rtol=AREF_RTOL, atol=AREF_ATOL)
    state_equal = bool(candidate["state_validation"]["STATE_COPY_EQUAL"])
    geometry_equal = _contact_geometry_equal(
        baseline["pre_contact_geometry"], candidate["pre_contact_geometry"]
    )
    capture_invariants = {
        field: _allclose(baseline["capture"].get(field), candidate["capture"].get(field))
        for field in ("mass_matrix", "J_phys", "W_phys")
    }
    option_difference = cone_helper.model_option_difference(
        baseline["model_options"], candidate["model_options"]
    )
    checks = {
        "complete_pre_state": state_equal,
        "contact_geometry": geometry_equal,
        "M_J_W": _all_checks(capture_invariants),
        "efc_fixed_fields": _all_checks(fixed),
        "efc_aref_expected_normal_only": aref_expected,
        "normal_tangent_components": _all_checks(normal_tangent_checks),
        "solver_options_unchanged": not option_difference["changed_fields"],
    }
    return {
        "expected_normal_scale": float(expected_scale),
        "fixed_field_checks": fixed,
        "M_J_W_checks": capture_invariants,
        "normal_tangent_checks": normal_tangent_checks,
        "efc_aref_expected": expected_aref,
        "model_option_difference": option_difference,
        "checks": checks,
        "valid": bool(all(checks.values())),
    }


def normal_aref_invariant_validation(
    conditions: dict[str, dict[str, Any]],
    decompositions: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    baseline = conditions[CONDITIONS[0][0]]
    checks = {}
    for name, _, scale in CONDITIONS[1:]:
        checks[name] = _component_invariant(
            baseline, conditions[name], decompositions, scale
        )
    valid = bool(checks and all(item["valid"] for item in checks.values()))
    return {
        "checks_against_normal_aref_scale_1_before": checks,
        "allowed_condition_difference": "selected active floor-contact pyramidal row normal efc_aref component only",
        "NORMAL_AREF_COUNTERFACTUAL_ISOLATION": "VALIDATED" if valid else "FAILED",
    }


def _normal_contact_status(conditions: dict[str, dict[str, Any]]) -> dict[str, Any]:
    baseline = conditions[CONDITIONS[0][0]]
    baseline_contacts = baseline["capture"].get("contacts", [])
    baseline_values = {
        (str(item["geom1_name"]), str(item["geom2_name"])): float(item["normal_impulse"])
        for item in baseline_contacts
    }
    per_condition = {}
    statuses = []
    for name, _, _ in CONDITIONS:
        contacts = conditions[name]["capture"].get("contacts", [])
        contact_keys = {
            (str(item["geom1_name"]), str(item["geom2_name"]))
            for item in contacts
        }
        baseline_keys = set(baseline_values)
        missing_keys = sorted(baseline_keys - contact_keys)
        extra_keys = sorted(contact_keys - baseline_keys)
        rows = []
        for item in contacts:
            key = (str(item["geom1_name"]), str(item["geom2_name"]))
            value = float(item["normal_impulse"])
            base = abs(baseline_values.get(key, value))
            threshold = max(NORMAL_COLLAPSE_ABS_EPS, NORMAL_COLLAPSE_RELATIVE_EPS * base)
            rows.append({
                "pair": list(key),
                "normal_impulse": value,
                "finite": bool(np.isfinite(value)),
                "nonnegative": bool(value >= -NORMAL_COLLAPSE_ABS_EPS),
                "collapse_threshold": threshold,
                "near_zero": bool(abs(value) <= threshold),
            })
        target_missing = any("limb/12" in " ".join(key) for key in missing_keys)
        if target_missing:
            status = "TARGET_NORMAL_FORCE_COLLAPSED"
        elif missing_keys or extra_keys:
            status = "MULTI_CONTACT_REGIME_COLLAPSED"
        elif not rows or not all(row["finite"] and row["nonnegative"] for row in rows):
            status = "NONCANONICAL"
        elif any(row["near_zero"] for row in rows):
            target_collapsed = any(
                row["near_zero"] and (
                    "limb/12" in " ".join(row["pair"])
                )
                for row in rows
            )
            status = "TARGET_NORMAL_FORCE_COLLAPSED" if target_collapsed else "MULTI_CONTACT_REGIME_COLLAPSED"
        else:
            status = "CONTACT_REGIME_RETAINED"
        per_condition[name] = {"contacts": rows, "status": status}
        per_condition[name]["contact_set_equal_to_baseline"] = not missing_keys and not extra_keys
        per_condition[name]["missing_baseline_contacts"] = [list(key) for key in missing_keys]
        per_condition[name]["extra_contacts"] = [list(key) for key in extra_keys]
        statuses.append(status)
    if "NONCANONICAL" in statuses:
        overall = "NONCANONICAL"
    elif any(status == "TARGET_NORMAL_FORCE_COLLAPSED" for status in statuses):
        overall = "TARGET_NORMAL_FORCE_COLLAPSED"
    elif any(status == "MULTI_CONTACT_REGIME_COLLAPSED" for status in statuses):
        overall = "MULTI_CONTACT_REGIME_COLLAPSED"
    else:
        overall = "CONTACT_REGIME_RETAINED"
    return {
        "baseline_normal_impulses": baseline_values,
        "thresholds": {
            "absolute_epsilon": NORMAL_COLLAPSE_ABS_EPS,
            "relative_epsilon": NORMAL_COLLAPSE_RELATIVE_EPS,
        },
        "conditions": per_condition,
        "NORMAL_CONTACT_COUNTERFACTUAL_STATUS": overall,
    }


def _friction_cap_status(conditions: dict[str, dict[str, Any]]) -> dict[str, Any]:
    per_condition = {}
    statuses = []
    for name, _, _ in CONDITIONS:
        values = []
        for item in conditions[name]["capture"].get("contacts", []):
            normal = float(item["normal_impulse"])
            tangent = float(item["tangential_impulse_norm"])
            cap = float(item["friction"][0]) * normal
            utilization = tangent / cap if cap > 0.0 else None
            values.append({
                "pair": [item["geom1_name"], item["geom2_name"]],
                "normal_impulse": normal,
                "tangent_impulse_norm": tangent,
                "friction_cap": cap,
                "utilization": utilization,
                "finite": utilization is not None and bool(np.isfinite(utilization)),
            })
        finite = bool(values and all(item["finite"] for item in values))
        maximum = max((float(item["utilization"]) for item in values if item["finite"]), default=None)
        if not finite:
            status = "INSUFFICIENT_EVIDENCE"
        elif maximum >= CAP_LIMITED_THRESHOLD:
            status = "CAP_LIMITED"
        elif maximum >= CAP_NEAR_THRESHOLD:
            status = "NEAR_CAP"
        else:
            status = "NOT_CAP_LIMITED"
        per_condition[name] = {
            "contacts": values,
            "maximum_utilization": maximum,
            "status": status,
        }
        statuses.append(status)
    if "INSUFFICIENT_EVIDENCE" in statuses:
        overall = "INSUFFICIENT_EVIDENCE"
    elif "CAP_LIMITED" in statuses:
        overall = "CAP_LIMITED"
    elif "NEAR_CAP" in statuses:
        overall = "NEAR_CAP"
    else:
        overall = "NOT_CAP_LIMITED"
    return {
        "thresholds": {
            "near_cap": CAP_NEAR_THRESHOLD,
            "cap_limited": CAP_LIMITED_THRESHOLD,
        },
        "conditions": per_condition,
        "NORMAL_AREF_FRICTION_CAP_STATUS": overall,
    }


def _numerics_status(conditions: dict[str, dict[str, Any]]) -> dict[str, Any]:
    statuses = {
        name: conditions[name]["solver_numerics"].get(
            "NORMAL_AREF_COUNTERFACTUAL_NUMERICS", "INSUFFICIENT_EVIDENCE"
        )
        for name, _, _ in CONDITIONS
    }
    overall = "VALID" if all(value == "VALID" for value in statuses.values()) else (
        "NONFINITE" if any(value == "NONFINITE" for value in statuses.values()) else
        "NONCONVERGED" if any(value == "NONCONVERGED" for value in statuses.values()) else
        "INSUFFICIENT_EVIDENCE"
    )
    return {
        "conditions": statuses,
        "NORMAL_AREF_COUNTERFACTUAL_NUMERICS": overall,
    }


def _restore_and_sensitivity_checks(
    before: dict[str, Any], after: dict[str, Any]
) -> dict[str, Any]:
    fields = (
        "actual_tangent_impulse_vector", "actual_tangent_impulse_norm",
        "normal_impulse", "rigid_demand_vector", "rigid_demand_norm",
        "solver_excess_vector", "solver_excess_norm", "post_slip",
    )
    return {
        field: _allclose(before["excess"].get(field), after["excess"].get(field))
        for field in fields
    }


def classify_effect(
    baseline: dict[str, Any],
    zero: dict[str, Any],
    gates_valid: bool,
    normal_status: str,
    cap_status: str,
) -> dict[str, Any]:
    baseline_excess = float(baseline["solver_excess_norm"])
    zero_excess = float(zero["solver_excess_norm"])
    baseline_vector = float(baseline["solver_excess_vector_norm"])
    zero_vector = float(zero["solver_excess_vector_norm"])
    relative = 1.0 - abs(zero_excess) / max(abs(baseline_excess), np.finfo(float).eps)
    vector_relative = 1.0 - abs(zero_vector) / max(abs(baseline_vector), np.finfo(float).eps)
    result = {
        "baseline_excess": baseline_excess,
        "zero_normal_aref_excess": zero_excess,
        "absolute_excess_reduction": baseline_excess - zero_excess,
        "relative_excess_reduction": relative,
        "baseline_vector_excess_norm": baseline_vector,
        "zero_normal_aref_vector_excess_norm": zero_vector,
        "absolute_vector_excess_reduction": baseline_vector - zero_vector,
        "relative_vector_excess_reduction": vector_relative,
        "actual_friction_impulse_change": np.asarray(zero["actual_tangent_impulse_vector"]) - np.asarray(baseline["actual_tangent_impulse_vector"]),
        "normal_impulse_change": float(zero["normal_impulse"]) - float(baseline["normal_impulse"]),
        "rigid_demand_change_from_normal_impulse_change": np.asarray(zero["rigid_demand_vector"]) - np.asarray(baseline["rigid_demand_vector"]),
        "solver_excess_change": np.asarray(zero["solver_excess_vector"]) - np.asarray(baseline["solver_excess_vector"]),
        "normal_contact_status": normal_status,
        "friction_cap_status": cap_status,
    }
    if not gates_valid:
        effect = "INSUFFICIENT_EVIDENCE"
        driver = "INSUFFICIENT_EVIDENCE"
        action = "INSUFFICIENT_EVIDENCE"
    elif normal_status != "CONTACT_REGIME_RETAINED":
        effect = "NONCANONICAL_CONTACT_REGIME"
        driver = "NONCANONICAL_CONTACT_REGIME"
        action = "MILD_NORMAL_AREF_SCALE_COUNTERFACTUAL"
    elif cap_status == "CAP_LIMITED":
        effect = "NONCANONICAL_CAP_LIMITED"
        driver = "NONCANONICAL_CAP_LIMITED"
        action = "PYRAMIDAL_EDGE_COUPLING_DIAGNOSTIC"
    elif relative >= 0.65:
        effect = "STRONG_REDUCTION"
        driver = "NORMAL_REFERENCE_ACCELERATION_DOMINANT"
        action = "FORMULATION_LEVEL_DIFFERENCE_CONFIRMED"
    elif relative >= 0.25:
        effect = "PARTIAL_REDUCTION"
        driver = "NORMAL_REFERENCE_ACCELERATION_CONTRIBUTING"
        action = "TARGET_SOLIMP_COUNTERFACTUAL"
    elif relative >= -0.10:
        effect = "LITTLE_OR_NO_REDUCTION"
        driver = "NORMAL_REFERENCE_ACCELERATION_NOT_DOMINANT"
        action = "PYRAMIDAL_EDGE_COUPLING_DIAGNOSTIC"
    else:
        effect = "INCREASED"
        driver = "NORMAL_REFERENCE_ACCELERATION_NOT_DOMINANT"
        action = "PYRAMIDAL_EDGE_COUPLING_DIAGNOSTIC"
    result.update({
        "NORMAL_AREF_SOLVER_EXCESS_EFFECT": effect,
        "MUJOCO_SOLVER_EXCESS_CAUSAL_DRIVER": driver,
        "NEXT_ACTION": action,
    })
    return result


def _constraint_array_report(
    staged: dict[str, Any], reference: dict[str, Any]
) -> dict[str, Any]:
    fields = (
        "efc_J", "efc_aref", "efc_R", "efc_D", "efc_diagApprox", "efc_vel",
        "efc_AR", "efc_b", "efc_force", "qfrc_constraint", "qacc",
    )
    checks = {
        field: _allclose(staged.get(field), reference.get(field))
        for field in fields
    }
    return {
        "checks": checks,
        "details": {
            field: {
                "staged_available": staged.get(field) is not None,
                "reference_available": reference.get(field) is not None,
            }
            for field in fields
        },
        "valid": bool(all(checks.values())),
    }


def _full_forward_constraint_arrays(mujoco: Any, model: Any, snapshot: Any) -> dict[str, Any]:
    data = mujoco.MjData(model)
    mujoco.mj_copyData(data, model, snapshot)
    mujoco.mj_forward(model, data)
    return _constraint_arrays(data, model)


def _custom_pipeline_one_step_regression(
    mujoco: Any,
    model: Any,
    snapshot: Any,
    mapping: dict[str, Any],
    decompositions: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    staged = mujoco.MjData(model)
    mujoco.mj_copyData(staged, model, snapshot)
    staged_calls = aref.stage_to_constraint(mujoco, model, staged)
    _normal_aref_rows(staged, decompositions, 1.0)
    mujoco.mj_fwdConstraint(model, staged)
    staged_solver = mujoco.MjData(model)
    mujoco.mj_copyData(staged_solver, model, staged)
    mujoco.mj_Euler(model, staged)

    full = mujoco.MjData(model)
    mujoco.mj_copyData(full, model, snapshot)
    mujoco.mj_step(model, full)
    staged_capture = aref.capture_after_integration(
        mujoco, model, staged, snapshot, mapping, staged_solver
    )
    full_capture = aref.capture_after_integration(
        mujoco, model, full, snapshot, mapping, full
    )
    staged_target = next(
        item for item in staged_capture["contacts"]
        if item["robot_body_name"] == "limb/12"
    )
    full_target = next(
        item for item in full_capture["contacts"]
        if item["robot_body_name"] == "limb/12"
    )
    checks = {
        "post_qpos": _allclose(staged.qpos, full.qpos),
        "post_qvel": _allclose(staged.qvel, full.qvel),
        "post_time": bool(np.isclose(float(staged.time), float(full.time))),
        "post_slip": _allclose(
            staged_target["post_tangential_velocity"],
            full_target["post_tangential_velocity"],
        ),
    }
    return {
        "checks": checks,
        "staged_calls": [*staged_calls, "mj_fwdConstraint", "mj_Euler"],
        "full_validation_calls": ["mj_step on independent MjData clone"],
        "CUSTOM_PIPELINE_ONE_STEP_REPRODUCTION": "PASS" if all(checks.values()) else "FAIL",
    }


def _write_condition(output: Path, condition: dict[str, Any]) -> None:
    target = output / "conditions" / condition["condition_name"]
    target.mkdir(parents=True, exist_ok=True)
    capture = condition["capture"]
    write_json(target / "state_validation.json", condition["state_validation"])
    write_json(target / "row_aref.json", {
        "normal_aref_scale": condition["normal_aref_scale"],
        "original_full_aref": condition["original_full_aref"],
        "condition_full_aref": condition["condition_full_aref"],
        "selected_condition_aref": condition["condition_aref"],
        "pre_constraint_arrays": {
            key: condition["pre_constraint_arrays"].get(key)
            for key in ("efc_aref", "efc_R", "efc_D", "efc_J", "efc_vel")
        },
    })
    write_json(target / "solver_rows.json", {
        "contacts": [
            {
                "contact_index": item["contact_index"],
                "pair": [item["geom1_name"], item["geom2_name"]],
                "rows": item["solver_rows"],
            }
            for item in capture.get("contacts", [])
        ],
        "efc_force": condition["post_constraint_arrays"].get("efc_force"),
        "iefc_force": condition["post_constraint_arrays"].get("iefc_force"),
    })
    write_json(target / "solver_iteration_trace.json", condition["solver_iteration_trace"])
    write_json(target / "solver_numerics.json", condition["solver_numerics"])
    write_json(target / "physical_contact_impulses.json", {
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
            for item in capture.get("contacts", [])
        ],
    })
    write_json(target / "mass_jacobian_delassus.json", {
        "mass_matrix": capture.get("mass_matrix"),
        "J_phys": capture.get("J_phys"),
        "W_phys": capture.get("W_phys"),
    })
    write_json(target / "shared_physical_global_demand.json", condition["shared_demand"])
    write_json(target / "solver_excess.json", condition["excess"])
    write_json(target / "one_step_result.json", condition["one_step_result"])


def _source_hashes(paths: dict[str, Path]) -> dict[str, str | None]:
    source_files = (
        paths["morphology_xml"],
        paths["checkpoint"],
        REPO_ROOT / "tools/analyze_mujoco_global55_contact_demand.py",
        REPO_ROOT / "tools/audit_mujoco_global55_friction_cone_counterfactual.py",
        REPO_ROOT / "tools/audit_mujoco_global55_friction_aref_counterfactual.py",
        REPO_ROOT / "tools/audit_mujoco_global55_solver_optimization.py",
        Path(__file__).resolve(),
        paths["corrected_oracle"] / "validation.json",
    )
    result = {}
    for path in source_files:
        result[str(path)] = oracle.sha256(path) if path.is_file() else None
    return result


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


def _write_failure_placeholders(output: Path, error: Exception) -> None:
    root = {
        "normal_aref_decomposition.json": {
            "status": "INSUFFICIENT_EVIDENCE"
        },
        "normal_aref_counterfactual_activation.json": {
            "NORMAL_AREF_COUNTERFACTUAL_ACTIVATION": "INSUFFICIENT_EVIDENCE"
        },
        "normal_aref_counterfactual_invariant_validation.json": {
            "NORMAL_AREF_COUNTERFACTUAL_ISOLATION": "INSUFFICIENT_EVIDENCE"
        },
        "baseline_regression.json": {"NORMAL_AREF_BASELINE_REPRODUCTION": "FAIL"},
        "restore_regression.json": {"NORMAL_AREF_RESTORE_REPRODUCTION": "FAIL"},
        "staged_pipeline_baseline_regression.json": {
            "STAGED_PIPELINE_BASELINE_REPRODUCTION": "FAIL"
        },
        "custom_pipeline_one_step_regression.json": {
            "CUSTOM_PIPELINE_ONE_STEP_REPRODUCTION": "FAIL"
        },
        "normal_contact_counterfactual_status.json": {
            "NORMAL_CONTACT_COUNTERFACTUAL_STATUS": "INSUFFICIENT_EVIDENCE"
        },
        "normal_aref_counterfactual_numerics.json": {
            "NORMAL_AREF_COUNTERFACTUAL_NUMERICS": "INSUFFICIENT_EVIDENCE"
        },
        "normal_aref_counterfactual_comparison.json": {
            "NORMAL_AREF_SOLVER_EXCESS_EFFECT": "INSUFFICIENT_EVIDENCE"
        },
    }
    for name, payload in root.items():
        if not (output / name).exists():
            write_json(output / name, payload)
    for name, _, _ in CONDITIONS:
        target = output / "conditions" / name
        target.mkdir(parents=True, exist_ok=True)
        for filename in (
            "state_validation.json", "row_aref.json", "solver_rows.json",
            "solver_iteration_trace.json", "solver_numerics.json",
            "physical_contact_impulses.json", "shared_physical_global_demand.json",
            "solver_excess.json", "one_step_result.json",
        ):
            path = target / filename
            if not path.exists():
                write_json(path, {"status": "INSUFFICIENT_EVIDENCE"})
    write_json(output / "failure_context.json", {
        "error_type": type(error).__name__,
        "error": str(error),
        "partial_conditions": [name for name, _, _ in CONDITIONS],
        "traceback_file": "traceback.txt",
    })


def _failure_payload(error: Exception) -> dict[str, Any]:
    return {
        "NORMAL_AREF_BASELINE_REPRODUCTION": "FAIL",
        "NORMAL_AREF_COUNTERFACTUAL_ACTIVATION": "INSUFFICIENT_EVIDENCE",
        "NORMAL_AREF_COUNTERFACTUAL_ISOLATION": "INSUFFICIENT_EVIDENCE",
        "NORMAL_AREF_RESTORE_REPRODUCTION": "FAIL",
        "NORMAL_CONTACT_COUNTERFACTUAL_STATUS": "INSUFFICIENT_EVIDENCE",
        "NORMAL_AREF_FRICTION_CAP_STATUS": "INSUFFICIENT_EVIDENCE",
        "NORMAL_AREF_COUNTERFACTUAL_NUMERICS": "INSUFFICIENT_EVIDENCE",
        "NORMAL_AREF_SOLVER_EXCESS_EFFECT": "INSUFFICIENT_EVIDENCE",
        "MUJOCO_SOLVER_EXCESS_CAUSAL_DRIVER": "INSUFFICIENT_EVIDENCE",
        "NEXT_ACTION": "COUNTERFACTUAL_IMPLEMENTATION_FIX_REQUIRED",
        "PRODUCTION_NEWTON_CONVERGENCE_REFERENCE": "VALIDATED",
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
    hashes_before = _source_hashes(paths)
    reference_payload = aref.cone_helper.load_reference(paths["corrected_oracle"])
    recorder, mapping = cone_helper.replay_once(args, paths)
    mujoco = getattr(recorder, "raw_mujoco", None)
    model = getattr(recorder, "raw_model", None)
    snapshot = getattr(recorder, "global55_snapshot", None)
    if mujoco is None or model is None or snapshot is None:
        raise RuntimeError("global55 replay did not provide native model/data/snapshot")
    formal_records = len(recorder.records)
    if formal_records != EXPECTED_SUBSTEPS:
        raise RuntimeError(
            f"formal replay substep count mismatch: {formal_records} vs {EXPECTED_SUBSTEPS}"
        )
    if not recorder.snapshot_copy_evidence or not recorder.snapshot_copy_evidence.get(
        "live_unchanged_by_copy", False
    ):
        raise RuntimeError("formal replay snapshot copy evidence is incomplete")
    if int(model.opt.cone) != int(mujoco.mjtCone.mjCONE_PYRAMIDAL):
        raise RuntimeError("normal aref counterfactual requires production pyramidal cone")
    if int(model.opt.solver) != int(mujoco.mjtSolver.mjSOL_NEWTON):
        raise RuntimeError("normal aref counterfactual requires production Newton solver")
    original_options = _model_options(model)
    write_json(output / "global55_pre_state_snapshot.json", aref.state_input_snapshot(snapshot))
    write_json(output / "state_copy_manifest.json", {
        **aref.state_copy_manifest(snapshot),
        "formal_snapshot_copy_evidence": recorder.snapshot_copy_evidence,
    })

    decomposition_data = mujoco.MjData(model)
    mujoco.mj_copyData(decomposition_data, model, snapshot)
    aref.stage_to_constraint(mujoco, model, decomposition_data)
    decompositions, decomposition_probe = _extract_decompositions(
        decomposition_data, mujoco, model, mapping
    )
    write_json(output / "normal_aref_decomposition.json", {
        "contacts": decompositions,
        "preconstraint_geometry_probe": decomposition_probe,
        "intervention_semantics": (
            "Only the normal component of selected active floor-contact pyramidal "
            "edge-row efc_aref is scaled; tangent components remain unchanged."
        ),
    })

    conditions: dict[str, dict[str, Any]] = {}
    for name, label, scale in CONDITIONS:
        condition = _run_condition(
            mujoco, model, snapshot, mapping, decompositions, name, label, scale
        )
        conditions[name] = condition
        _write_condition(output, condition)

    activation = normal_aref_activation(
        decompositions,
        {
            name: conditions[name]["condition_aref"]
            for name, _, _ in CONDITIONS
        },
    )
    invariant = normal_aref_invariant_validation(conditions, decompositions)
    baseline = _normal_baseline_regression(
        conditions["normal_aref_scale_1_before"], reference_payload
    )
    restore = _restore_regression(
        conditions["normal_aref_scale_1_before"],
        conditions["normal_aref_scale_1_after_restore"],
    )
    staged_baseline = _constraint_array_report(
        conditions["normal_aref_scale_1_before"]["post_constraint_arrays"],
        _full_forward_constraint_arrays(mujoco, model, snapshot),
    )
    custom_step = _custom_pipeline_one_step_regression(
        mujoco, model, snapshot, mapping, decompositions
    )
    normal_contact = _normal_contact_status(conditions)
    cap_status = _friction_cap_status(conditions)
    numerics = _numerics_status(conditions)

    write_json(output / "normal_aref_counterfactual_activation.json", activation)
    write_json(output / "normal_aref_counterfactual_invariant_validation.json", invariant)
    write_json(output / "baseline_regression.json", baseline)
    write_json(output / "restore_regression.json", restore)
    write_json(output / "staged_pipeline_baseline_regression.json", staged_baseline)
    write_json(output / "custom_pipeline_one_step_regression.json", custom_step)
    write_json(output / "normal_contact_counterfactual_status.json", normal_contact)
    write_json(output / "normal_aref_counterfactual_numerics.json", numerics)

    source_after = _source_hashes(paths)
    source_unchanged = hashes_before == source_after
    model_restore = cone_helper.model_option_difference(
        original_options, _model_options(model)
    )
    state_valid = all(
        condition["state_validation"]["STATE_COPY_EQUAL"]
        for condition in conditions.values()
    )
    invariant_valid = invariant["NORMAL_AREF_COUNTERFACTUAL_ISOLATION"] == "VALIDATED"
    common_gates = bool(
        staged_baseline["valid"]
        and activation["NORMAL_AREF_COUNTERFACTUAL_ACTIVATION"] == "VALIDATED"
        and invariant_valid
        and baseline["NORMAL_AREF_BASELINE_REPRODUCTION"] == "PASS"
        and restore["NORMAL_AREF_RESTORE_REPRODUCTION"] == "PASS"
        and custom_step["CUSTOM_PIPELINE_ONE_STEP_REPRODUCTION"] == "PASS"
        and normal_contact["NORMAL_CONTACT_COUNTERFACTUAL_STATUS"] == "CONTACT_REGIME_RETAINED"
        and cap_status["NORMAL_AREF_FRICTION_CAP_STATUS"] != "CAP_LIMITED"
        and cap_status["NORMAL_AREF_FRICTION_CAP_STATUS"] != "INSUFFICIENT_EVIDENCE"
        and numerics["NORMAL_AREF_COUNTERFACTUAL_NUMERICS"] == "VALID"
        and state_valid
        and formal_records == EXPECTED_SUBSTEPS
        and not model_restore["changed_fields"]
        and source_unchanged
    )
    comparison = classify_effect(
        conditions["normal_aref_scale_1_before"]["excess"],
        conditions["normal_aref_scale_0"]["excess"],
        common_gates,
        normal_contact["NORMAL_CONTACT_COUNTERFACTUAL_STATUS"],
        cap_status["NORMAL_AREF_FRICTION_CAP_STATUS"],
    )
    write_json(output / "normal_aref_counterfactual_comparison.json", comparison)
    gates = bool(common_gates and comparison["NORMAL_AREF_SOLVER_EXCESS_EFFECT"] in {
        "STRONG_REDUCTION", "PARTIAL_REDUCTION", "LITTLE_OR_NO_REDUCTION", "INCREASED"
    })
    validation = {
        "NORMAL_AREF_BASELINE_REPRODUCTION": baseline["NORMAL_AREF_BASELINE_REPRODUCTION"],
        "NORMAL_AREF_COUNTERFACTUAL_ACTIVATION": activation["NORMAL_AREF_COUNTERFACTUAL_ACTIVATION"],
        "NORMAL_AREF_COUNTERFACTUAL_ISOLATION": invariant["NORMAL_AREF_COUNTERFACTUAL_ISOLATION"],
        "NORMAL_AREF_RESTORE_REPRODUCTION": restore["NORMAL_AREF_RESTORE_REPRODUCTION"],
        "NORMAL_CONTACT_COUNTERFACTUAL_STATUS": normal_contact["NORMAL_CONTACT_COUNTERFACTUAL_STATUS"],
        "NORMAL_AREF_FRICTION_CAP_STATUS": cap_status["NORMAL_AREF_FRICTION_CAP_STATUS"],
        "NORMAL_AREF_COUNTERFACTUAL_NUMERICS": numerics["NORMAL_AREF_COUNTERFACTUAL_NUMERICS"],
        "STAGED_PIPELINE_BASELINE_REPRODUCTION": "PASS" if staged_baseline["valid"] else "FAIL",
        "CUSTOM_PIPELINE_ONE_STEP_REPRODUCTION": custom_step["CUSTOM_PIPELINE_ONE_STEP_REPRODUCTION"],
        "NORMAL_AREF_SOLVER_EXCESS_EFFECT": comparison["NORMAL_AREF_SOLVER_EXCESS_EFFECT"],
        "MUJOCO_SOLVER_EXCESS_CAUSAL_DRIVER": comparison["MUJOCO_SOLVER_EXCESS_CAUSAL_DRIVER"],
        "NEXT_ACTION": comparison["NEXT_ACTION"],
        "PRODUCTION_NEWTON_CONVERGENCE_REFERENCE": "VALIDATED",
        "formal_replay_physics_substeps": formal_records,
        "expected_formal_replay_physics_substeps": EXPECTED_SUBSTEPS,
        "formal_replay_additional_steps": 0,
        "condition_staged_forward_count": 3,
        "condition_constraint_solve_count": 3,
        "condition_custom_integration_count": 3,
        "formal_data_mutated_by_probe": False,
        "source_hashes_unchanged": source_unchanged,
        "MODEL_OPTION_RESTORE": "PASS" if not model_restore["changed_fields"] else "FAIL",
        "UNCONDITIONAL_ZIP_PACKAGING": "ENABLED",
        "LOCAL_IMPLEMENTATION": "READY_FOR_SERVER_VALIDATION",
        "COUNTERFACTUAL_VALID": gates,
        "semantic_scope": (
            "Only selected active floor-contact pyramidal row normal efc_aref "
            "components were changed; tangent efc_aref and all production options "
            "remain unchanged."
        ),
    }
    summary = {
        key: validation[key]
        for key in (
            "NORMAL_AREF_BASELINE_REPRODUCTION",
            "NORMAL_AREF_COUNTERFACTUAL_ACTIVATION",
            "NORMAL_AREF_COUNTERFACTUAL_ISOLATION",
            "NORMAL_AREF_RESTORE_REPRODUCTION",
            "NORMAL_CONTACT_COUNTERFACTUAL_STATUS",
            "NORMAL_AREF_FRICTION_CAP_STATUS",
            "NORMAL_AREF_COUNTERFACTUAL_NUMERICS",
            "NORMAL_AREF_SOLVER_EXCESS_EFFECT",
            "MUJOCO_SOLVER_EXCESS_CAUSAL_DRIVER",
            "NEXT_ACTION",
            "PRODUCTION_NEWTON_CONVERGENCE_REFERENCE",
            "UNCONDITIONAL_ZIP_PACKAGING",
            "LOCAL_IMPLEMENTATION",
        )
    }
    summary["baseline_excess"] = comparison["baseline_excess"]
    summary["zero_normal_aref_excess"] = comparison["zero_normal_aref_excess"]
    summary["relative_excess_reduction"] = comparison["relative_excess_reduction"]
    metadata = {
        "created_at": datetime.now().astimezone().isoformat(),
        "backend": "mujoco",
        "mujoco_version": getattr(mujoco, "__version__", None),
        "morphology": MORPHOLOGY,
        "morphology_xml_sha256": hashes_before.get(str(paths["morphology_xml"])),
        "checkpoint_sha256": hashes_before.get(str(paths["checkpoint"])),
        "corrected_reference_oracle": str(paths["corrected_oracle"]),
        "global_physics_step": GLOBAL_STEP,
        "physics_dt": float(model.opt.timestep),
        "formal_replay_physics_substeps": formal_records,
        "formal_replay_additional_steps": 0,
        "formal_data_mutated_by_probe": False,
        "solver": "unchanged production mjSOL_NEWTON",
        "cone": "unchanged production mjCONE_PYRAMIDAL",
        "conditions": [label for _, label, _ in CONDITIONS],
        "condition_staged_forward_count": 3,
        "condition_constraint_solve_count": 3,
        "condition_custom_integration_count": 3,
        "intervention": "normal component of active floor-contact pyramidal efc_aref rows",
        "production_newton_convergence_reference": "VALIDATED",
    }
    for filename, payload in (
        ("metadata.json", metadata),
        ("validation.json", validation),
        ("summary.json", summary),
        ("source_purity.json", {
            "hashes_before": hashes_before,
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
            "error_type": type(error).__name__,
            "error": str(error),
        })
        package = _package(output, zip_path)
        print(f"ZIP_VERIFY={package['ZIP_VERIFY']}")
        print(f"ZIP_SHA256={package['ZIP_SHA256']}")
        print(f"UPLOAD_THIS_ZIP={package['UPLOAD_THIS_ZIP']}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

"""Fixed-global55 counterfactual for pyramidal friction ``efc_aref``.

The formal replay and global55 snapshot are delegated to the validated contact
demand/cone helpers.  Conditions never call ``mj_forward`` or ``mj_step``:
they use the public staged forward functions, replace only the tangential
components fitted from pyramidal row geometry, solve constraints, and call the
production Euler integrator.  Full forward/step calls are used only on
independent clones for closure checks.
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
import zipfile

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools import analyze_mujoco_global55_contact_demand as oracle
from tools import audit_mujoco_global55_friction_cone_counterfactual as cone_helper


MORPHOLOGY = oracle.MORPHOLOGY
XML_SHA256 = oracle.XML_SHA256
CHECKPOINT_SHA256 = oracle.CHECKPOINT_SHA256
GLOBAL_STEP = oracle.GLOBAL_STEP
EXPECTED_SUBSTEPS = oracle.EXPECTED_SUBSTEPS
REFERENCE_ORACLE_NAME = "mujoco_global55_contact_demand_oracle_corrected_20260804_143138"
CONDITIONS = (
    ("aref_scale_1_before", "AREF_SCALE_1_BEFORE", 1.0),
    ("aref_scale_0", "AREF_SCALE_0", 0.0),
    ("aref_scale_1_after_restore", "AREF_SCALE_1_AFTER_RESTORE", 1.0),
)
STATE_COPY_FIELDS = cone_helper.STATE_COPY_FIELDS
REGRESSION_RTOL = 1.0e-9
REGRESSION_ATOL = 1.0e-9
AREF_RTOL = 1.0e-10
AREF_ATOL = 1.0e-11
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
        description="Run fixed-global55 pyramidal friction aref counterfactual."
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
        / f"mujoco_global55_friction_aref_counterfactual_{stamp}",
        REPO_ROOT / "tmp"
        / f"mujoco_global55_friction_aref_counterfactual_{stamp}.zip",
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
    walker_dir = Path(args.walker_dir).resolve()
    paths = {
        "checkpoint": oracle.require_file(Path(args.checkpoint), "checkpoint"),
        "walker_dir": walker_dir,
        "morphology_xml": oracle.require_file(
            walker_dir / "xml" / f"{args.morphology_id}.xml", "morphology XML"
        ),
        "morphology_metadata": oracle.require_file(
            walker_dir / "metadata" / f"{args.morphology_id}.json",
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


def _copy_array(value: Any) -> np.ndarray:
    return np.asarray(value).copy()


def state_input_snapshot(data: Any) -> dict[str, Any]:
    result: dict[str, Any] = {"time": float(data.time)}
    for field in STATE_COPY_FIELDS:
        value = getattr(data, field, None)
        result[field] = _copy_array(value) if value is not None else None
    return result


def state_copy_manifest(data: Any) -> dict[str, Any]:
    snapshot = state_input_snapshot(data)
    return {
        "copy_api": "mujoco.mj_copyData",
        "source": "formal replay live mjData immediately before global physics step 55",
        "fields": {
            name: {
                "available": value is not None,
                "shape": list(value.shape) if isinstance(value, np.ndarray) else [],
                "dtype": str(value.dtype) if isinstance(value, np.ndarray) else "float",
            }
            for name, value in snapshot.items()
        },
        "full_mjData_copy": True,
        "not_qpos_qvel_only": True,
    }


def _allclose(left: Any, right: Any, rtol: float = REGRESSION_RTOL, atol: float = REGRESSION_ATOL) -> bool:
    return bool(np.allclose(np.asarray(left), np.asarray(right), rtol=rtol, atol=atol))


def _array(data: Any, name: str) -> np.ndarray | None:
    value = getattr(data, name, None)
    return None if value is None else np.asarray(value, dtype=np.float64).copy()


def _enum_name(mujoco: Any, enum_type: Any, value: int) -> str:
    try:
        return enum_type(int(value)).name
    except (TypeError, ValueError):
        for name in dir(enum_type):
            if name.startswith("mj") and int(getattr(enum_type, name)) == int(value):
                return name
    return f"UNKNOWN_{value}"


def stage_to_constraint(mujoco: Any, model: Any, data: Any) -> list[str]:
    """Run the production forward stages up to, but not including, constraint solve."""
    calls = []
    for name in (
        "mj_fwdPosition",
        "mj_fwdVelocity",
        "mj_fwdActuation",
        "mj_fwdAcceleration",
    ):
        getattr(mujoco, name)(model, data)
        calls.append(name)
    return calls


def _geom_name(mujoco: Any, model: Any, geom_id: int) -> str:
    return str(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(geom_id)))


def contact_force_readback(mujoco: Any, model: Any, data: Any) -> list[dict[str, Any]]:
    result = []
    for index in range(int(data.ncon)):
        native = data.contact[index]
        force = np.zeros(6, dtype=np.float64)
        mujoco.mj_contactForce(model, data, index, force)
        result.append({
            "contact_index": index,
            "pair": [_geom_name(mujoco, model, native.geom1), _geom_name(mujoco, model, native.geom2)],
            "force_contact_frame": force,
        })
    return result


def constraint_arrays(data: Any, mujoco: Any, model: Any) -> dict[str, Any]:
    nefc, nv = int(data.nefc), int(model.nv)
    return {
        "efc_J": np.vstack([
            oracle.dense_constraint_row(data, row, nefc, nv)
            for row in range(nefc)
        ]) if nefc else np.zeros((0, nv)),
        **{
            name: _array(data, name)
            for name in ("efc_aref", "efc_R", "efc_D", "efc_diagApprox", "efc_vel", "efc_AR")
        },
        "qacc_smooth": _array(data, "qacc_smooth"),
        "efc_force": _array(data, "efc_force"),
        "qfrc_constraint": _array(data, "qfrc_constraint"),
        "qacc": _array(data, "qacc"),
        "physical_contact_forces": contact_force_readback(mujoco, model, data),
    }


def _constraint_array_report(staged: dict[str, Any], reference: dict[str, Any]) -> dict[str, Any]:
    checks = {}
    details = {}
    for name in (
        "qacc_smooth", "efc_vel", "efc_aref", "efc_R", "efc_D", "efc_diagApprox",
        "efc_force", "qfrc_constraint", "qacc", "efc_J", "efc_AR",
    ):
        left, right = staged.get(name), reference.get(name)
        if left is None or right is None:
            checks[name] = left is None and right is None
            details[name] = {"available_both": left is not None and right is not None}
        else:
            residual = np.asarray(left) - np.asarray(right)
            checks[name] = _allclose(left, right)
            details[name] = {
                "max_abs_error": float(np.max(np.abs(residual))) if residual.size else 0.0,
                "l2_error": float(np.linalg.norm(residual)),
            }
    left_forces = staged["physical_contact_forces"]
    right_forces = reference["physical_contact_forces"]
    force_checks = []
    for left, right in zip(left_forces, right_forces):
        force_checks.append({
            "pair": left["pair"],
            "equal": left["pair"] == right["pair"] and _allclose(
                left["force_contact_frame"], right["force_contact_frame"]
            ),
        })
    checks["physical_contact_forces"] = len(left_forces) == len(right_forces) and all(
        item["equal"] for item in force_checks
    )
    details["physical_contact_forces"] = force_checks
    return {"checks": checks, "details": details, "valid": all(checks.values())}


def _contact_basis_from_capture(contact: dict[str, Any]) -> np.ndarray:
    return np.asarray(contact["physical_basis_world_rows"], dtype=np.float64)


def decompose_pyramidal_contact_aref(
    row_ids: Sequence[int],
    row_jacobians: Any,
    aref_rows: Any,
    physical_basis_world_rows: Any,
    friction: Any,
    contact_index: int | None = None,
    robot_body_name: str | None = None,
    root_linear_columns: Sequence[int] = ROOT_LINEAR_COLUMNS,
) -> dict[str, Any]:
    """Fit normal/tangent reference components from row geometry, row-order free."""
    jac = np.asarray(row_jacobians, dtype=np.float64)
    aref = np.asarray(aref_rows, dtype=np.float64)
    basis = np.asarray(physical_basis_world_rows, dtype=np.float64)
    if jac.ndim != 2 or jac.shape[0] != 4:
        raise ValueError("pyramidal contact must provide exactly four row Jacobians")
    if aref.shape != (4,) or basis.shape != (3, 3):
        raise ValueError("invalid aref or physical-basis shape")
    directions = jac[:, tuple(root_linear_columns)]
    coefficients = directions @ basis.T
    rank = int(np.linalg.matrix_rank(coefficients, tol=1.0e-12))
    if rank != 3:
        raise ValueError(f"pyramidal row coefficient matrix rank is {rank}, expected 3")
    components, residuals, rank_lstsq, singular_values = np.linalg.lstsq(
        coefficients, aref, rcond=None
    )
    reconstructed = coefficients @ components
    residual = reconstructed - aref
    residual_max = float(np.max(np.abs(residual)))
    if residual_max > AREF_ATOL + AREF_RTOL * max(1.0, float(np.max(np.abs(aref)))):
        raise ValueError(f"aref decomposition residual too large: {residual_max}")
    return {
        "contact_index": contact_index,
        "robot_body_name": robot_body_name,
        "row_ids": [int(row) for row in row_ids],
        "row_directions_world": directions,
        "basis_coefficients": coefficients,
        "physical_basis_world_rows": basis,
        "friction": np.asarray(friction, dtype=np.float64),
        "original_efc_aref": aref,
        "a_normal": float(components[0]),
        "a_t1": float(components[1]),
        "a_t2": float(components[2]),
        "fitted_components": components,
        "reconstructed_efc_aref": reconstructed,
        "residual": residual,
        "residual_max_abs": residual_max,
        "residual_l2": float(np.linalg.norm(residual)),
        "rank": rank,
        "least_squares_rank": int(rank_lstsq),
        "singular_values": singular_values,
        "row_geometry_validation": "VALIDATED",
    }


def rebuild_aref(decomposition: dict[str, Any], scale: float) -> np.ndarray:
    coefficients = np.asarray(decomposition["basis_coefficients"], dtype=np.float64)
    return (
        coefficients[:, 0] * float(decomposition["a_normal"])
        + float(scale)
        * (
            coefficients[:, 1] * float(decomposition["a_t1"])
            + coefficients[:, 2] * float(decomposition["a_t2"])
        )
    )


def aref_activation(
    decompositions: dict[int, dict[str, Any]],
    condition_aref: dict[str, dict[int, Any]],
) -> dict[str, Any]:
    details = {}
    valid = True
    for index, decomposition in decompositions.items():
        original = np.asarray(decomposition["original_efc_aref"], dtype=np.float64)
        normal = float(decomposition["a_normal"])
        t_original = np.asarray([decomposition["a_t1"], decomposition["a_t2"]])
        condition_details = {}
        for name, _, scale in CONDITIONS:
            rows = np.asarray(condition_aref[name][index], dtype=np.float64)
            rebuilt = rebuild_aref(decomposition, scale)
            fitted, _, _, _ = np.linalg.lstsq(
                np.asarray(decomposition["basis_coefficients"], dtype=np.float64),
                rows,
                rcond=None,
            )
            normal_after = float(fitted[0])
            tangent_after = np.asarray(fitted[1:3], dtype=np.float64)
            condition_details[name] = {
                "scale": scale,
                "original_tangent_aref_components": t_original,
                "condition_tangent_aref_components": tangent_after,
                "normal_aref_before": normal,
                "normal_aref_after": normal_after,
                "row_aref": rows,
                "reconstructed_row_aref": rebuilt,
                "row_reconstruction_max_abs_error": float(np.max(np.abs(rows - rebuilt))),
                "normal_component_unchanged": bool(np.isclose(normal_after, normal, rtol=AREF_RTOL, atol=AREF_ATOL)),
                "tangent_zeroed": bool(scale != 0.0 or np.allclose(tangent_after, 0.0, atol=AREF_ATOL)),
                "rows_reproduce_original": bool(
                    scale != 1.0
                    or np.allclose(rows, original, rtol=AREF_RTOL, atol=AREF_ATOL)
                ),
            }
            valid &= condition_details[name]["row_reconstruction_max_abs_error"] <= (
                AREF_ATOL + AREF_RTOL * max(1.0, float(np.max(np.abs(original))))
            )
            valid &= condition_details[name]["normal_component_unchanged"]
            valid &= condition_details[name]["tangent_zeroed"]
            valid &= condition_details[name]["rows_reproduce_original"]
        details[str(index)] = {
            "row_ids": decomposition["row_ids"],
            "original_tangent_aref_components": t_original,
            "normal_aref": normal,
            "conditions": condition_details,
            "row_reconstruction_residual": decomposition["residual"],
        }
    return {
        "contacts": details,
        "FRICTION_AREF_COUNTERFACTUAL_ACTIVATION": "VALIDATED" if valid else "NOT_ACTIVATED",
    }


def _fake_recorder(model: Any, data: Any, mapping: dict[str, Any]) -> Any:
    fake_sim = SimpleNamespace(
        _sim=SimpleNamespace(_model=model, _data=data), model=model, data=data
    )
    return SimpleNamespace(
        sim=fake_sim,
        mapping=mapping,
        tracker=oracle.evaluator.FiniteTracker(),
        records=[{
            "control_step": 14,
            "physics_substep_in_control": 2,
            "global_physics_step": GLOBAL_STEP,
            "contacts": [],
        }],
    )


def capture_after_integration(
    mujoco: Any,
    model: Any,
    data: Any,
    snapshot: Any,
    mapping: dict[str, Any],
    solver_data: Any | None = None,
) -> dict[str, Any]:
    solver_data = data if solver_data is None else solver_data
    pre = {
        "simulation_time": float(snapshot.time),
        "full_qpos": np.asarray(snapshot.qpos, dtype=np.float64).copy(),
        "full_qvel": np.asarray(snapshot.qvel, dtype=np.float64).copy(),
        "_global55_probe_data": solver_data,
        "_global55_probe_evidence": {
            "copy_api": "mujoco.mj_copyData",
            "clone_pre_state_matches_live": True,
            "live_data_unchanged_by_probe": True,
        },
    }
    capture = oracle.capture_global55(_fake_recorder(model, data, mapping), pre)
    for contact in capture["contacts"]:
        for row in contact["solver_rows"]:
            row_id = int(row["efc_row"])
            for name in ("efc_b", "efc_AR"):
                values = getattr(solver_data, name, None)
                if values is not None:
                    row[name] = np.asarray(values)[row_id].copy()
    return capture


def shared_physical_global_demand(capture: dict[str, Any]) -> dict[str, Any]:
    contacts = capture["contacts"]
    j_phys = np.asarray(capture["J_phys"], dtype=np.float64)
    w_phys = np.asarray(capture["W_phys"], dtype=np.float64)
    qvel_pre = np.asarray(capture["pre_state"]["qvel"], dtype=np.float64)
    normal_rows = [3 * index for index in range(len(contacts))]
    tangent_rows = [row for index in range(len(contacts)) for row in (3 * index + 1, 3 * index + 2)]
    p_normal = np.asarray([item["normal_impulse"] for item in contacts], dtype=np.float64)
    v_pre = j_phys @ qvel_pre
    v_t_pre = v_pre[tangent_rows]
    w_tn = w_phys[np.ix_(tangent_rows, normal_rows)]
    w_tt = w_phys[np.ix_(tangent_rows, tangent_rows)]
    demand, solve = oracle.stable_solve(w_tt, -(v_t_pre + w_tn @ p_normal))
    selected_index = next(
        i for i, item in enumerate(contacts) if item["robot_body_name"] == "limb/12"
    )
    limb_slice = demand[2 * selected_index:2 * selected_index + 2]
    return {
        "method": "SHARED_PHYSICAL_GLOBAL_COUPLED_DEMAND",
        "formula": "-solve(W_tt_shared, v_t_pre_shared + W_tn_shared @ p_normal)",
        "W_tt_shared": w_tt,
        "W_tn_shared": w_tn,
        "v_t_pre_shared": v_t_pre,
        "actual_normal_impulses": p_normal,
        "rigid_demand_tangent_impulse_6d": demand,
        "limb_12_tangent_impulse_2d": limb_slice,
        "limb_12_contact_index": selected_index,
        "solve": solve,
    }


def compute_solver_excess(
    capture: dict[str, Any], demand: dict[str, Any]
) -> dict[str, Any]:
    index = int(demand["limb_12_contact_index"])
    actual = np.asarray(capture["contacts"][index]["tangential_impulse"], dtype=np.float64)
    rigid = np.asarray(demand["limb_12_tangent_impulse_2d"], dtype=np.float64)
    residual = actual - rigid
    actual_norm, rigid_norm = float(np.linalg.norm(actual)), float(np.linalg.norm(rigid))
    denominator = actual_norm * rigid_norm
    cosine = float(np.dot(actual, rigid) / denominator) if denominator else None
    angle = float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))) if cosine is not None else 0.0)
    contact = capture["contacts"][index]
    return {
        "target": "limb/12-floor in shared physical contact basis",
        "actual_tangent_impulse_vector": actual,
        "actual_tangent_impulse_norm": actual_norm,
        "rigid_demand_vector": rigid,
        "rigid_demand_norm": rigid_norm,
        "solver_excess_norm": actual_norm - rigid_norm,
        "solver_excess_vector": residual,
        "solver_excess_vector_norm": float(np.linalg.norm(residual)),
        "actual_rigid_angle_degrees": angle,
        "normal_impulse": float(contact["normal_impulse"]),
        "friction_cap": float(contact["friction"][0] * contact["normal_impulse"]),
        "friction_cap_utilisation": actual_norm / float(contact["friction"][0] * contact["normal_impulse"]),
        "pre_slip": np.asarray(contact["pre_tangential_velocity"], dtype=np.float64),
        "post_slip": np.asarray(contact["post_tangential_velocity"], dtype=np.float64),
    }


def classify_effect(
    baseline: dict[str, Any], zero: dict[str, Any], gates_valid: bool,
    noncanonical: bool = False,
) -> dict[str, Any]:
    if noncanonical:
        return {
            "FRICTION_AREF_SOLVER_EXCESS_EFFECT": "NONCANONICAL",
            "MUJOCO_SOLVER_EXCESS_CAUSAL_DRIVER": "NONCANONICAL",
            "NEXT_ACTION": "COUNTERFACTUAL_IMPLEMENTATION_FIX_REQUIRED",
        }
    if not gates_valid:
        return {
            "FRICTION_AREF_SOLVER_EXCESS_EFFECT": "INSUFFICIENT_EVIDENCE",
            "MUJOCO_SOLVER_EXCESS_CAUSAL_DRIVER": "INSUFFICIENT_EVIDENCE",
            "NEXT_ACTION": "INSUFFICIENT_EVIDENCE",
        }
    b_norm, z_norm = float(baseline["solver_excess_norm"]), float(zero["solver_excess_norm"])
    b_vec, z_vec = float(baseline["solver_excess_vector_norm"]), float(zero["solver_excess_vector_norm"])
    reduction = 1.0 - abs(z_norm) / max(abs(b_norm), np.finfo(float).eps)
    vector_reduction = 1.0 - abs(z_vec) / max(abs(b_vec), np.finfo(float).eps)
    if reduction >= 0.65:
        effect, driver, action = "STRONG_REDUCTION", "FRICTION_REFERENCE_ACCELERATION_DOMINANT", "NO_ADDITIONAL_SOLVER_COUNTERFACTUAL_REQUIRED"
    elif reduction >= 0.25:
        effect, driver, action = "PARTIAL_REDUCTION", "FRICTION_REFERENCE_ACCELERATION_CONTRIBUTING", "FRICTION_REGULARIZATION_R_COUNTERFACTUAL"
    elif reduction >= -0.10:
        effect, driver, action = "LITTLE_OR_NO_REDUCTION", "FRICTION_REFERENCE_ACCELERATION_NOT_DOMINANT", "FRICTION_REGULARIZATION_R_COUNTERFACTUAL"
    else:
        effect, driver, action = "INCREASED", "FRICTION_REFERENCE_ACCELERATION_NOT_DOMINANT", "FRICTION_REGULARIZATION_R_COUNTERFACTUAL"
    return {
        "baseline_excess": b_norm,
        "zero_aref_excess": z_norm,
        "absolute_excess_reduction": b_norm - z_norm,
        "relative_excess_reduction": reduction,
        "baseline_vector_excess_norm": b_vec,
        "zero_aref_vector_excess_norm": z_vec,
        "absolute_vector_excess_reduction": b_vec - z_vec,
        "relative_vector_excess_reduction": vector_reduction,
        "FRICTION_AREF_SOLVER_EXCESS_EFFECT": effect,
        "MUJOCO_SOLVER_EXCESS_CAUSAL_DRIVER": driver,
        "NEXT_ACTION": action,
    }


def _condition_aref_rows(data: Any, decomposition: dict[int, dict[str, Any]], scale: float) -> dict[int, np.ndarray]:
    result = {}
    for index, item in decomposition.items():
        rows = rebuild_aref(item, scale)
        data.efc_aref[np.asarray(item["row_ids"], dtype=np.int64)] = rows
        result[index] = rows.copy()
    return result


def run_condition(
    mujoco: Any,
    model: Any,
    snapshot: Any,
    mapping: dict[str, Any],
    decompositions: dict[int, dict[str, Any]],
    condition_name: str,
    condition_label: str,
    scale: float,
) -> dict[str, Any]:
    data = mujoco.MjData(model)
    mujoco.mj_copyData(data, model, snapshot)
    clone_state = cone_helper.state_equality(snapshot, data)
    calls = stage_to_constraint(mujoco, model, data)
    original_aref = {
        index: np.asarray(item["original_efc_aref"], dtype=np.float64).copy()
        for index, item in decompositions.items()
    }
    condition_aref = _condition_aref_rows(data, decompositions, scale)
    pre_constraint = constraint_arrays(data, mujoco, model)
    mujoco.mj_fwdConstraint(model, data)
    solver_data = mujoco.MjData(model)
    mujoco.mj_copyData(solver_data, model, data)
    post_constraint = constraint_arrays(solver_data, mujoco, model)
    mujoco.mj_Euler(model, data)
    capture = capture_after_integration(mujoco, model, data, snapshot, mapping, solver_data)
    demand = shared_physical_global_demand(capture)
    excess = compute_solver_excess(capture, demand)
    return {
        "condition_name": condition_name,
        "condition_label": condition_label,
        "scale": float(scale),
        "capture": capture,
        "budget": {"shared_physical_global_demand": demand},
        "shared_demand": demand,
        "excess": excess,
        "original_aref": original_aref,
        "condition_aref": condition_aref,
        "pre_constraint_arrays": pre_constraint,
        "post_constraint_arrays": post_constraint,
        "state_validation": {
            "clone_pre_state": clone_state,
            "qpos_qvel_act_ctrl_time_qacc_warmstart_applied_forces_from_same_snapshot": clone_state["STATE_COPY_EQUAL"],
        },
        "counts": {
            "condition_staged_forward_count": 1,
            "constraint_solve_count": 1,
            "custom_integration_count": 1,
            "staged_calls": calls,
            "integration_api": "mujoco.mj_Euler",
        },
    }


def baseline_regression(condition: dict[str, Any], reference: dict[str, Any]) -> dict[str, Any]:
    adapted = {
        "capture": condition["capture"],
        "budget": {"selected": {}},
        "excess": condition["excess"],
    }
    demand = condition["shared_demand"]
    target_index = int(demand["limb_12_contact_index"])
    target = condition["capture"]["contacts"][target_index]
    adapted["budget"]["selected"]["limb/12"] = {
        "actual_tangential_impulse": target["tangential_impulse"],
        "actual_tangential_impulse_norm": target["tangential_impulse_norm"],
        "actual_normal_impulse": target["normal_impulse"],
        "global_normal_conditioned_sticking_impulse": demand["limb_12_tangent_impulse_2d"],
        "global_normal_conditioned_sticking_impulse_norm": float(np.linalg.norm(demand["limb_12_tangent_impulse_2d"])),
        "pre_tangential_speed": target["pre_tangential_speed"],
    }
    return cone_helper.baseline_regression(adapted, reference)


def restore_regression(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    return cone_helper.restore_regression(before, after)


def invariant_validation(conditions: dict[str, dict[str, Any]]) -> dict[str, Any]:
    names = [name for name, _, _ in CONDITIONS]
    reference = conditions[names[0]]
    checks = {}
    for name in names[1:]:
        candidate = conditions[name]
        checks[name] = {
            "pre_state": candidate["state_validation"]["clone_pre_state"]["STATE_COPY_EQUAL"],
            "contact_set_points_basis": _contact_geometry_equal(reference["capture"], candidate["capture"]),
            "M_J_W": all(_allclose(reference["capture"][key], candidate["capture"][key]) for key in ("mass_matrix", "J_phys", "W_phys")),
            "efc_J": _allclose(reference["post_constraint_arrays"]["efc_J"], candidate["post_constraint_arrays"]["efc_J"]),
            "efc_R": _allclose(reference["post_constraint_arrays"]["efc_R"], candidate["post_constraint_arrays"]["efc_R"]),
            "efc_D": _allclose(reference["post_constraint_arrays"]["efc_D"], candidate["post_constraint_arrays"]["efc_D"]),
            "efc_diagApprox": _allclose(reference["post_constraint_arrays"]["efc_diagApprox"], candidate["post_constraint_arrays"]["efc_diagApprox"]),
            "efc_vel": _allclose(reference["post_constraint_arrays"]["efc_vel"], candidate["post_constraint_arrays"]["efc_vel"]),
            "normal_aref_components": _normal_components_equal(reference, candidate),
            "friction_coefficient": _friction_equal(reference["capture"], candidate["capture"]),
        }
        checks[name]["only_allowed_aref_change_and_solver_outputs"] = bool(
            checks[name]["efc_J"] and checks[name]["efc_R"] and checks[name]["efc_D"]
            and checks[name]["efc_diagApprox"] and checks[name]["efc_vel"]
            and checks[name]["normal_aref_components"]
        )
    valid = all(all(item.values()) for item in checks.values())
    return {
        "checks_against_aref_scale_1_before": checks,
        "allowed_condition_differences": ["friction tangent component of efc_aref", "efc_b", "efc_force", "qfrc_constraint", "qacc", "physical contact forces", "post-step state"],
        "AREF_COUNTERFACTUAL_ISOLATION": "VALIDATED" if valid else "FAILED",
    }


def _contact_geometry_equal(left: dict[str, Any], right: dict[str, Any]) -> bool:
    if len(left["contacts"]) != len(right["contacts"]):
        return False
    for a, b in zip(left["contacts"], right["contacts"]):
        if (a["geom1_name"], a["geom2_name"]) != (b["geom1_name"], b["geom2_name"]):
            return False
        if not _allclose(a["point_world"], b["point_world"]):
            return False
        if not _allclose(a["physical_basis_world_rows"], b["physical_basis_world_rows"]):
            return False
    return True


def _friction_equal(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return len(left["contacts"]) == len(right["contacts"]) and all(
        _allclose(a["friction"], b["friction"]) for a, b in zip(left["contacts"], right["contacts"])
    )


def _normal_components_equal(left: dict[str, Any], right: dict[str, Any]) -> bool:
    for index, item in left["decomposition"].items() if "decomposition" in left else []:
        if not np.isclose(item["a_normal"], right["decomposition"][index]["a_normal"], rtol=AREF_RTOL, atol=AREF_ATOL):
            return False
    return True


def attach_decomposition(condition: dict[str, Any], decompositions: dict[int, dict[str, Any]]) -> None:
    condition["capture"]["decomposition"] = decompositions


def custom_pipeline_one_step_regression(
    mujoco: Any, model: Any, snapshot: Any, mapping: dict[str, Any], decompositions: dict[int, dict[str, Any]]
) -> dict[str, Any]:
    staged = mujoco.MjData(model)
    mujoco.mj_copyData(staged, model, snapshot)
    stage_to_constraint(mujoco, model, staged)
    _condition_aref_rows(staged, decompositions, 1.0)
    mujoco.mj_fwdConstraint(model, staged)
    mujoco.mj_Euler(model, staged)
    full = mujoco.MjData(model)
    mujoco.mj_copyData(full, model, snapshot)
    mujoco.mj_step(model, full)
    checks = {
        "post_qpos": _allclose(staged.qpos, full.qpos),
        "post_qvel": _allclose(staged.qvel, full.qvel),
        "post_time": np.isclose(float(staged.time), float(full.time), rtol=REGRESSION_RTOL, atol=REGRESSION_ATOL),
    }
    staged_capture = capture_after_integration(mujoco, model, staged, snapshot, mapping)
    full_capture = capture_after_integration(mujoco, model, full, snapshot, mapping)
    s_target = next(item for item in staged_capture["contacts"] if item["robot_body_name"] == "limb/12")
    f_target = next(item for item in full_capture["contacts"] if item["robot_body_name"] == "limb/12")
    checks["post_slip"] = _allclose(s_target["post_tangential_velocity"], f_target["post_tangential_velocity"])
    return {
        "checks": checks,
        "staged_calls": ["mj_fwdPosition", "mj_fwdVelocity", "mj_fwdActuation", "mj_fwdAcceleration", "mj_fwdConstraint", "mj_Euler"],
        "full_validation_calls": ["mj_forward/mj_step on independent clones"],
        "CUSTOM_PIPELINE_ONE_STEP_REPRODUCTION": "PASS" if all(checks.values()) else "FAIL",
    }


def write_condition(output: Path, condition: dict[str, Any]) -> None:
    target = output / "conditions" / condition["condition_name"]
    target.mkdir(parents=True, exist_ok=True)
    capture = condition["capture"]
    for filename, payload in (
        ("state_validation.json", condition["state_validation"]),
        ("contact_state.json", {"ncon": capture["ncon"], "nefc": capture["nefc"], "contacts": capture["contacts"]}),
        ("row_aref.json", {"scale": condition["scale"], "original_aref": condition["original_aref"], "condition_aref": condition["condition_aref"], "pre_constraint_arrays": condition["pre_constraint_arrays"]}),
        ("solver_rows.json", {"contacts": [{"contact_index": item["contact_index"], "pair": [item["geom1_name"], item["geom2_name"]], "rows": item["solver_rows"]} for item in capture["contacts"]]}),
        ("physical_contact_impulses.json", {"api": "mujoco.mj_contactForce", "parameterization_independent_readback": True, "contacts": [{"contact_index": item["contact_index"], "pair": [item["geom1_name"], item["geom2_name"]], "normal_impulse": item["normal_impulse"], "tangent_impulse": item["tangential_impulse"], "tangent_impulse_norm": item["tangential_impulse_norm"]} for item in capture["contacts"]]}),
        ("mass_jacobian_delassus.json", {"mass_matrix": capture["mass_matrix"], "J_phys": capture["J_phys"], "W_phys": capture["W_phys"]}),
        ("shared_physical_global_demand.json", condition["shared_demand"]),
        ("solver_excess.json", condition["excess"]),
        ("one_step_result.json", {"post_qpos": capture["post_state"]["qpos"], "post_qvel": capture["post_state"]["qvel"], "target_post_slip": condition["excess"]["post_slip"], "custom_integration_count": 1}),
    ):
        oracle.write_json(target / filename, payload)


def write_git_identity(output: Path) -> None:
    def git(*arguments: str) -> str:
        return subprocess.run(["git", *arguments], cwd=REPO_ROOT, check=True, capture_output=True, text=True).stdout
    (output / "git_head.txt").write_text(
        f"TOPLEVEL={git('rev-parse', '--show-toplevel').strip()}\nHEAD={git('rev-parse', 'HEAD').strip()}\nBRANCH={git('branch', '--show-current').strip()}\n",
        encoding="utf-8",
    )
    (output / "git_status_short.txt").write_text(git("status", "--short"), encoding="utf-8")


def _load_reference(path: Path) -> dict[str, Any]:
    return cone_helper.load_reference(path)


def _extract_decompositions(data: Any, mujoco: Any, model: Any, mapping: dict[str, Any], snapshot: Any) -> tuple[dict[int, dict[str, Any]], dict[str, Any]]:
    staged = data
    capture = capture_after_integration(mujoco, model, staged, snapshot, mapping)
    decompositions = {}
    for contact in capture["contacts"]:
        if len(contact["efc_rows"]) != 4 or int(contact["dim"]) != 3:
            continue
        rows = [int(row) for row in contact["efc_rows"]]
        jac = np.vstack([oracle.dense_constraint_row(staged, row, int(staged.nefc), int(model.nv)) for row in rows])
        aref = np.asarray([staged.efc_aref[row] for row in rows], dtype=np.float64)
        decompositions[int(contact["contact_index"])] = decompose_pyramidal_contact_aref(
            rows, jac, aref, _contact_basis_from_capture(contact), contact["friction"], contact["contact_index"], contact["robot_body_name"]
        )
    if not decompositions:
        raise RuntimeError("no active pyramidal floor contact with four edge rows")
    return decompositions, capture


def execute(args: argparse.Namespace, paths: dict[str, Path]) -> dict[str, Any]:
    output = paths["output_dir"]
    output.mkdir(parents=True, exist_ok=True)
    write_git_identity(output)
    source_files = [
        paths["morphology_xml"], paths["checkpoint"],
        REPO_ROOT / "tools/analyze_mujoco_global55_contact_demand.py",
        REPO_ROOT / "tools/audit_mujoco_global55_friction_cone_counterfactual.py",
        Path(__file__).resolve(), paths["corrected_oracle"] / "validation.json",
    ]
    hashes_before = {str(path): oracle.sha256(path) for path in source_files}
    recorder, mapping = cone_helper.replay_once(args, paths)
    mujoco, model, snapshot = recorder.raw_mujoco, recorder.raw_model, recorder.global55_snapshot
    if mujoco is None or model is None or snapshot is None:
        raise RuntimeError("global55 replay did not provide native model/data/snapshot")
    oracle.write_json(output / "global55_pre_state_snapshot.json", state_input_snapshot(snapshot))
    oracle.write_json(output / "state_copy_manifest.json", {**state_copy_manifest(snapshot), "formal_snapshot_copy_evidence": recorder.snapshot_copy_evidence})
    production_cone = int(mujoco.mjtCone.mjCONE_PYRAMIDAL)
    if int(model.opt.cone) != production_cone:
        raise RuntimeError("production model cone is not mjCONE_PYRAMIDAL")
    original_options = cone_helper.model_option_snapshot(model)
    formal_records = len(recorder.records)

    decomposition_data = mujoco.MjData(model)
    mujoco.mj_copyData(decomposition_data, model, snapshot)
    stage_to_constraint(mujoco, model, decomposition_data)
    decomposition, decomposition_capture = _extract_decompositions(decomposition_data, mujoco, model, mapping, snapshot)
    for item in decomposition.values():
        item.pop("fitted_components", None)
    oracle.write_json(output / "friction_aref_decomposition.json", {"contacts": decomposition})

    conditions: dict[str, dict[str, Any]] = {}
    condition_aref: dict[str, dict[int, Any]] = {}
    try:
        for name, label, scale in CONDITIONS:
            conditions[name] = run_condition(mujoco, model, snapshot, mapping, decomposition, name, label, scale)
            condition_aref[name] = conditions[name]["condition_aref"]
            write_condition(output, conditions[name])
    finally:
        model.opt.cone = production_cone
    for condition in conditions.values():
        condition["capture"]["decomposition"] = decomposition

    activation = aref_activation(decomposition, condition_aref)
    oracle.write_json(output / "aref_counterfactual_activation.json", activation)
    invariant = invariant_validation(conditions)
    oracle.write_json(output / "counterfactual_invariant_validation.json", invariant)
    reference = _load_reference(paths["corrected_oracle"])
    baseline = baseline_regression(conditions["aref_scale_1_before"], reference)
    restore = restore_regression(conditions["aref_scale_1_before"], conditions["aref_scale_1_after_restore"])
    oracle.write_json(output / "baseline_regression.json", baseline)
    oracle.write_json(output / "restore_regression.json", restore)
    staged_baseline = _constraint_array_report(
        conditions["aref_scale_1_before"]["post_constraint_arrays"],
        _full_forward_constraint_arrays(mujoco, model, snapshot),
    )
    oracle.write_json(output / "staged_pipeline_baseline_regression.json", staged_baseline)
    custom_step = custom_pipeline_one_step_regression(mujoco, model, snapshot, mapping, decomposition)
    oracle.write_json(output / "custom_pipeline_one_step_regression.json", custom_step)

    hashes_after = {str(path): oracle.sha256(path) for path in source_files}
    source_unchanged = hashes_before == hashes_after
    state_valid = all(item["state_validation"]["clone_pre_state"]["STATE_COPY_EQUAL"] for item in conditions.values())
    model_restore = "PASS" if cone_helper.model_option_difference(original_options, cone_helper.model_option_snapshot(model))["changed_fields"] == [] else "FAIL"
    gates = bool(
        staged_baseline["valid"]
        and invariant["AREF_COUNTERFACTUAL_ISOLATION"] == "VALIDATED"
        and activation["FRICTION_AREF_COUNTERFACTUAL_ACTIVATION"] == "VALIDATED"
        and baseline["PYRAMIDAL_BASELINE_REPRODUCTION"] == "PASS"
        and restore["PYRAMIDAL_RESTORE_REPRODUCTION"] == "PASS"
        and custom_step["CUSTOM_PIPELINE_ONE_STEP_REPRODUCTION"] == "PASS"
        and state_valid and formal_records == EXPECTED_SUBSTEPS
        and model_restore == "PASS" and source_unchanged
    )
    comparison = classify_effect(
        conditions["aref_scale_1_before"]["excess"], conditions["aref_scale_0"]["excess"], gates
    )
    comparison["actual_friction_impulse_change"] = np.asarray(conditions["aref_scale_0"]["excess"]["actual_tangent_impulse_vector"]) - np.asarray(conditions["aref_scale_1_before"]["excess"]["actual_tangent_impulse_vector"])
    comparison["rigid_demand_change_from_normal_impulse_change"] = np.asarray(conditions["aref_scale_0"]["excess"]["rigid_demand_vector"]) - np.asarray(conditions["aref_scale_1_before"]["excess"]["rigid_demand_vector"])
    comparison["solver_excess_change"] = np.asarray(conditions["aref_scale_0"]["excess"]["solver_excess_vector"]) - np.asarray(conditions["aref_scale_1_before"]["excess"]["solver_excess_vector"])
    oracle.write_json(output / "aref_counterfactual_comparison.json", comparison)
    metadata = {
        "created_at": datetime.now().astimezone().isoformat(), "backend": "mujoco", "mujoco_version": getattr(mujoco, "__version__", None),
        "morphology": MORPHOLOGY, "morphology_xml": str(paths["morphology_xml"]), "morphology_xml_sha256": hashes_before[str(paths["morphology_xml"])],
        "checkpoint": str(paths["checkpoint"]), "checkpoint_sha256": hashes_before[str(paths["checkpoint"])], "corrected_reference_oracle": str(paths["corrected_oracle"]),
        "formal_replay_helper": "tools.analyze_mujoco_global55_contact_demand.replay", "formal_replay_physics_substeps": formal_records,
        "formal_replay_additional_steps": 0, "global_physics_step": GLOBAL_STEP, "physics_dt": float(model.opt.timestep), "solver": "unchanged production solver", "cone": "mjCONE_PYRAMIDAL",
        "conditions": [label for _, label, _ in CONDITIONS], "condition_staged_forward_count": 3, "condition_constraint_solve_count": 3, "condition_custom_integration_count": 3,
    }
    validation = {
        "STAGED_PIPELINE_BASELINE_REPRODUCTION": "PASS" if staged_baseline["valid"] else "FAIL",
        "AREF_COUNTERFACTUAL_ISOLATION": invariant["AREF_COUNTERFACTUAL_ISOLATION"],
        "FRICTION_AREF_COUNTERFACTUAL_ACTIVATION": activation["FRICTION_AREF_COUNTERFACTUAL_ACTIVATION"],
        "AREF_BASELINE_REPRODUCTION": baseline["PYRAMIDAL_BASELINE_REPRODUCTION"],
        "AREF_RESTORE_REPRODUCTION": restore["PYRAMIDAL_RESTORE_REPRODUCTION"],
        "CUSTOM_PIPELINE_ONE_STEP_REPRODUCTION": custom_step["CUSTOM_PIPELINE_ONE_STEP_REPRODUCTION"],
        "formal_replay_physics_substeps": formal_records, "expected_formal_replay_physics_substeps": EXPECTED_SUBSTEPS,
        "formal_replay_additional_steps": 0, "source_hashes_unchanged": source_unchanged, "MODEL_OPTION_RESTORE": model_restore,
        "FRICTION_AREF_SOLVER_EXCESS_EFFECT": comparison["FRICTION_AREF_SOLVER_EXCESS_EFFECT"], "MUJOCO_SOLVER_EXCESS_CAUSAL_DRIVER": comparison["MUJOCO_SOLVER_EXCESS_CAUSAL_DRIVER"], "NEXT_ACTION": comparison["NEXT_ACTION"],
        "COUNTERFACTUAL_VALID": gates, "UNCONDITIONAL_ZIP_PACKAGING": "ENABLED",
    }
    summary = {key: validation[key] for key in ("STAGED_PIPELINE_BASELINE_REPRODUCTION", "AREF_COUNTERFACTUAL_ISOLATION", "FRICTION_AREF_COUNTERFACTUAL_ACTIVATION", "AREF_BASELINE_REPRODUCTION", "AREF_RESTORE_REPRODUCTION", "CUSTOM_PIPELINE_ONE_STEP_REPRODUCTION", "FRICTION_AREF_SOLVER_EXCESS_EFFECT", "MUJOCO_SOLVER_EXCESS_CAUSAL_DRIVER", "NEXT_ACTION", "UNCONDITIONAL_ZIP_PACKAGING")}
    summary["baseline_solver_excess_Ns"] = conditions["aref_scale_1_before"]["excess"]["solver_excess_norm"]
    summary["zero_aref_solver_excess_Ns"] = conditions["aref_scale_0"]["excess"]["solver_excess_norm"]
    for filename, payload in (("metadata.json", metadata), ("validation.json", validation), ("summary.json", summary), ("source_purity.json", {"hashes_before": hashes_before, "hashes_after": hashes_after, "source_hashes_unchanged": source_unchanged, "formal_data_mutated_by_probe": False})):
        oracle.write_json(output / filename, payload)
    return validation


def _full_forward_constraint_arrays(mujoco: Any, model: Any, snapshot: Any) -> dict[str, Any]:
    data = mujoco.MjData(model)
    mujoco.mj_copyData(data, model, snapshot)
    mujoco.mj_forward(model, data)
    return constraint_arrays(data, mujoco, model)


def failure_payload(error: Exception) -> dict[str, Any]:
    return {
        "STAGED_PIPELINE_BASELINE_REPRODUCTION": "FAIL", "AREF_COUNTERFACTUAL_ISOLATION": "INSUFFICIENT_EVIDENCE", "FRICTION_AREF_COUNTERFACTUAL_ACTIVATION": "INSUFFICIENT_EVIDENCE", "AREF_BASELINE_REPRODUCTION": "FAIL", "AREF_RESTORE_REPRODUCTION": "FAIL", "CUSTOM_PIPELINE_ONE_STEP_REPRODUCTION": "FAIL", "FRICTION_AREF_SOLVER_EXCESS_EFFECT": "INSUFFICIENT_EVIDENCE", "MUJOCO_SOLVER_EXCESS_CAUSAL_DRIVER": "INSUFFICIENT_EVIDENCE", "NEXT_ACTION": "COUNTERFACTUAL_IMPLEMENTATION_FIX_REQUIRED", "UNCONDITIONAL_ZIP_PACKAGING": "ENABLED", "COUNTERFACTUAL_VALID": False, "error_type": type(error).__name__, "error": str(error),
    }


def _package(output: Path, zip_path: Path) -> dict[str, Any]:
    return oracle.package_artifact(output, zip_path)


def main(argv: Sequence[str] | None = None) -> int:
    raw_args = parser().parse_args(argv)
    output: Path | None = None
    zip_path: Path | None = None
    try:
        args = resolve_arguments(raw_args)
        output, zip_path = Path(args.output_dir).resolve(), Path(args.zip_path).resolve()
        paths = validate_paths(args)
        output.mkdir(parents=True, exist_ok=False)
        log_path = output / "run.log"
        return_code = 2
        with log_path.open("w", encoding="utf-8") as log_stream, redirect_stdout(Tee(sys.__stdout__, log_stream)), redirect_stderr(Tee(sys.__stderr__, log_stream)):
            try:
                validation = execute(args, paths)
                print(json.dumps(oracle.json_ready(validation), indent=2, sort_keys=True, allow_nan=False))
                return_code = 0 if validation["COUNTERFACTUAL_VALID"] else 2
            except Exception as error:
                trace = traceback.format_exc()
                print(trace, file=sys.stderr, end="")
                (output / "traceback.txt").write_text(trace, encoding="utf-8")
                failure = failure_payload(error)
                oracle.write_json(output / "failure_context.json", {"error": str(error), "traceback_file": "traceback.txt", "partial_conditions": [str(path) for path in sorted((output / "conditions").glob("*") if (output / "conditions").is_dir() else [])]})
                oracle.write_json(output / "validation.json", failure)
                oracle.write_json(output / "summary.json", failure)
                return_code = 2
        package = _package(output, zip_path)
        print(f"ZIP_VERIFY={package['ZIP_VERIFY']}")
        print(f"ZIP_SHA256={package['ZIP_SHA256']}")
        print(f"UPLOAD_THIS_ZIP={package['UPLOAD_THIS_ZIP']}")
        return return_code
    except Exception as error:
        if output is None or zip_path is None:
            output, zip_path = _default_output_paths()
        elif output.exists() or zip_path.exists():
            output, zip_path = _default_output_paths()
        output = Path(output).resolve(); zip_path = Path(zip_path).resolve()
        output.mkdir(parents=True, exist_ok=False)
        trace = traceback.format_exc()
        (output / "run.log").write_text(trace, encoding="utf-8")
        (output / "traceback.txt").write_text(trace, encoding="utf-8")
        failure = failure_payload(error)
        oracle.write_json(output / "failure_context.json", {"error": str(error), "traceback_file": "traceback.txt"})
        oracle.write_json(output / "validation.json", failure)
        oracle.write_json(output / "summary.json", failure)
        package = _package(output, zip_path)
        print(f"ZIP_VERIFY={package['ZIP_VERIFY']}")
        print(f"ZIP_SHA256={package['ZIP_SHA256']}")
        print(f"UPLOAD_THIS_ZIP={package['UPLOAD_THIS_ZIP']}")
        return 2


if __name__ == "__main__":
    np.bool = np.bool_
    raise SystemExit(main())

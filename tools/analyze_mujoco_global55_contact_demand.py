"""Extract a read-only MuJoCo global-55 contact-demand oracle.

The replay uses the formal evaluator environment/checkpoint restoration and its
existing one-callback-per-live-mj_step instrumentation.  At global step 55 a
complete pre-integration mjData clone receives one diagnostic mj_forward; the
formal live mjData still receives exactly one mj_step and no extra forward.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import sys
import traceback
from types import SimpleNamespace
from typing import Any, Sequence
import zipfile

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools import evaluate_mujoco_checkpoint as evaluator


MORPHOLOGY = "floor-1409-0-3-01-15-56-55"
XML_SHA256 = "da2156e9be4b706a34599a32bc8f3f1a2037fa8ecfb44042bc22e1740e9382a0"
CHECKPOINT_SHA256 = "cdf53f3e427a5a1c081ddd4d78230574fd694a5cefd97b30184a3762d9943d03"
GLOBAL_STEP = 55
CONTROL_STEPS = 30
EXPECTED_SUBSTEPS = 120
SELECTED = (("limb/12", "limby/12"), ("limb/11", "limby/11"))
REGRESSION = {
    "Fn": 1269.0480557934907,
    "Ft_norm": 664.9472720133347,
    "normal_generalized": 332.635590617929,
    "friction_generalized": -240.195861797270,
    "total_generalized": 92.439728820659,
}
ISAAC_REFERENCE = {
    "backend": "isaac",
    "semantics": "fixed comparison values; apparent metrics are not exact J M^-1 J^T metrics",
    "global_physics_step": 55,
    "body": "limb/12",
    "pre_tangential_speed_m_per_s": 0.5088208518817131,
    "post_tangential_speed_m_per_s": 0.023172343566941596,
    "actual_friction_impulse_N_s": 1.8164226398028622,
    "actual_normal_impulse_N_s": 5.117247619628906,
    "apparent_directional_mass_kg": 3.9189994935896055,
    "apparent_sticking_demand_N_s": 1.9037494498107959,
}
GENERALIZED_COLUMN_ORDER_FILE = "generalized_dof_order.json"
GENERALIZED_COLUMN_ORDER = "MUJOCO_QVEL_DOF_ORDER"
CLOSURE_RELATIVE_L2_TOLERANCE = 1.0e-8
CLOSURE_MAX_ABS_TOLERANCE = 1.0e-8
PHYSICAL_JACOBIAN_CONTRACT = {
    "PHYSICAL_JACOBIAN_CONVENTION": "WORLD_POINT_AT_CONTACT",
    "PHYSICAL_JACOBIAN_API": "mujoco.mj_jac",
    "PHYSICAL_JACOBIAN_POINT": "exact mjContact.pos",
    "PHYSICAL_JACOBIAN_POINT_FRAME": "world",
    "PHYSICAL_JACOBIAN_BODY": "robot contact body",
    "PHYSICAL_JACOBIAN_OUTPUT": "world-frame translational velocity Jacobian of the specified contact point",
    "PHYSICAL_JACOBIAN_REFERENCE": "WORLD_POINT_AT_CONTACT",
    "LINK_COM_BODY_ORIGIN_SHIFT": "NOT_APPLICABLE_ALREADY_EVALUATED_AT_CONTACT_POINT",
}


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="Replay exactly 120 formal MuJoCo substeps and inspect global55."
    )
    result.add_argument("--checkpoint", required=True)
    result.add_argument("--walker-dir", required=True)
    result.add_argument("--morphology-id", default=MORPHOLOGY)
    result.add_argument("--existing-oracle", required=True)
    result.add_argument("--output-dir", required=True)
    result.add_argument("--cfg", default="configs/ft.yaml")
    result.add_argument("--seed", type=int, default=1409)
    result.add_argument("--device", default="cpu")
    return result


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(json_ready(payload), indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def json_ready(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return json_ready(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        scalar = float(value)
        return scalar if math.isfinite(scalar) else None
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    return value


def all_numeric_values_finite(value: Any) -> bool:
    if isinstance(value, np.ndarray):
        return bool(np.isfinite(value).all())
    if isinstance(value, (np.floating, float)):
        return math.isfinite(float(value))
    if isinstance(value, dict):
        return all(all_numeric_values_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(all_numeric_values_finite(item) for item in value)
    return True


def require_file(path: Path, label: str) -> Path:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} is missing: {resolved}")
    return resolved


def pair_matches(contact: dict[str, Any], robot_geom: str) -> bool:
    return {contact["geom1_name"], contact["geom2_name"]} == {
        robot_geom,
        "floor/0",
    }


def dense_constraint_row(data: Any, row: int, nefc: int, nv: int) -> np.ndarray:
    indices, values = evaluator.constraint_jacobian_row(data, row, nefc, nv)
    dense = np.zeros(nv, dtype=np.float64)
    dense[np.asarray(indices, dtype=np.int64)] = np.asarray(values, dtype=np.float64)
    return dense


def stable_solve(matrix: np.ndarray, rhs: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    matrix = np.asarray(matrix, dtype=np.float64)
    rhs = np.asarray(rhs, dtype=np.float64)
    rank = int(np.linalg.matrix_rank(matrix, tol=1.0e-12))
    condition = float(np.linalg.cond(matrix)) if matrix.size else 0.0
    use_pinv = rank < matrix.shape[0] or not math.isfinite(condition) or condition > 1.0e12
    if use_pinv:
        solution = np.linalg.pinv(matrix, rcond=1.0e-12) @ rhs
        method = "numpy.linalg.pinv_rcond_1e-12"
    else:
        solution = np.linalg.solve(matrix, rhs)
        method = "numpy.linalg.solve"
    residual = matrix @ solution - rhs
    return solution, {
        "method": method,
        "rank": rank,
        "shape": list(matrix.shape),
        "condition_number": condition,
        "residual_max_abs": float(np.max(np.abs(residual))) if residual.size else 0.0,
        "residual_l2": float(np.linalg.norm(residual)),
    }


def point_jacobian_and_velocity(
    mujoco: Any,
    model: Any,
    data: Any,
    body_id: int,
    point_world: np.ndarray,
    qvel: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    jacp = np.zeros((3, int(model.nv)), dtype=np.float64)
    jacr = np.zeros_like(jacp)
    body_jacp = np.zeros_like(jacp)
    body_jacr = np.zeros_like(jacp)
    mujoco.mj_jac(model, data, jacp, jacr, point_world, int(body_id))
    mujoco.mj_jacBody(model, data, body_jacp, body_jacr, int(body_id))
    from_jacobian = jacp @ qvel
    body_origin = np.asarray(data.xpos[body_id], dtype=np.float64)
    from_rigid = (
        body_jacp @ qvel
        + np.cross(body_jacr @ qvel, point_world - body_origin)
    )
    return jacp, {
        "point_velocity_world_from_Jqvel": from_jacobian,
        "point_velocity_world_from_rigid_kinematics": from_rigid,
        "mapping_max_abs_error": float(np.max(np.abs(from_jacobian - from_rigid))),
    }


def expanded_mass_matrix(mujoco: Any, model: Any, data: Any) -> tuple[np.ndarray, dict[str, Any]]:
    nv = int(model.nv)
    packed_before = np.asarray(data.qM, dtype=np.float64).copy()
    mass = np.zeros((nv, nv), dtype=np.float64)
    mujoco.mj_fullM(model, mass, data.qM)
    packed_after = np.asarray(data.qM, dtype=np.float64).copy()
    symmetry = float(np.max(np.abs(mass - mass.T)))
    eigenvalues = np.linalg.eigvalsh((mass + mass.T) * 0.5)
    probe_rhs = np.asarray(data.qfrc_constraint, dtype=np.float64)
    _, solve_check = stable_solve(mass, probe_rhs)
    return mass, {
        "mass_matrix_shape": [nv, nv],
        "symmetry_max_abs_error": symmetry,
        "minimum_eigenvalue": float(np.min(eigenvalues)),
        "maximum_eigenvalue": float(np.max(eigenvalues)),
        "condition_number": float(np.linalg.cond(mass)),
        "linear_solve_check": solve_check,
        "api": "mujoco.mj_fullM(model, dense_scratch, data.qM)",
        "live_qM_mutated": bool(not np.array_equal(packed_before, packed_after)),
    }


def rich_solver_rows(
    data: Any, rows: Sequence[int], nefc: int, nv: int
) -> list[dict[str, Any]]:
    records = []
    for row in rows:
        jacobian = dense_constraint_row(data, int(row), nefc, nv)
        kbip = np.asarray(data.efc_KBIP[row], dtype=np.float64)
        records.append(
            {
                "efc_row": int(row),
                "efc_type": int(data.efc_type[row]),
                "efc_id": int(data.efc_id[row]),
                "efc_force": float(data.efc_force[row]),
                "efc_vel": float(data.efc_vel[row]),
                "efc_aref": float(data.efc_aref[row]),
                "efc_R": float(data.efc_R[row]),
                "efc_D": float(data.efc_D[row]),
                "efc_diagApprox": float(data.efc_diagApprox[row]),
                "efc_KBIP": kbip,
                "efc_KBIP_components": {
                    "K": float(kbip[0]),
                    "B": float(kbip[1]),
                    "impedance": float(kbip[2]),
                    "impedance_derivative": float(kbip[3]),
                },
                "efc_state": int(data.efc_state[row]),
                "J_row": jacobian,
            }
        )
    return records


def physical_basis(frame: np.ndarray, robot_side_factor: int) -> np.ndarray:
    """Return right-handed [normal, tangent1, tangent2] world-axis rows."""
    factor = int(robot_side_factor)
    basis = np.vstack((factor * frame[0], frame[1], factor * frame[2]))
    return basis


def force_to_impulse(force: Any, physics_dt: float) -> np.ndarray:
    """Convert one post-solve force readback to its one-substep impulse."""
    return np.asarray(force, dtype=np.float64) * float(physics_dt)


def package_artifact(output_dir: Path, zip_path: Path) -> dict[str, Any]:
    """Package one success or failure artifact and verify it immediately."""
    output_dir = Path(output_dir).resolve()
    zip_path = Path(zip_path).resolve()
    if not output_dir.is_dir():
        raise FileNotFoundError(f"artifact directory is missing: {output_dir}")
    if zip_path.exists():
        raise FileExistsError(f"refusing to overwrite ZIP: {zip_path}")
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "x", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(output_dir.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(output_dir.parent))
    with zipfile.ZipFile(zip_path) as archive:
        bad_member = archive.testzip()
        file_count = len(archive.namelist())
    digest = sha256(zip_path)
    sidecar = Path(str(zip_path) + ".sha256")
    sidecar.write_text(f"{digest}  {zip_path.name}\n", encoding="utf-8")
    return {
        "ZIP_VERIFY": "PASS" if bad_member is None else f"FAIL:{bad_member}",
        "ZIP_SHA256": digest,
        "ZIP_FILE_COUNT": file_count,
        "UPLOAD_THIS_ZIP": str(zip_path),
        "SHA256_SIDECAR": str(sidecar),
    }


def _array_snapshot(data: Any, name: str) -> np.ndarray | None:
    value = getattr(data, name, None)
    if value is None:
        return None
    return np.asarray(value).copy()


def live_data_fingerprint(data: Any) -> dict[str, Any]:
    return {
        "time": float(data.time),
        "ncon": int(getattr(data, "ncon", 0)),
        "nefc": int(getattr(data, "nefc", 0)),
        **{
            name: _array_snapshot(data, name)
            for name in (
                "qpos", "qvel", "act", "ctrl", "qfrc_applied",
                "xfrc_applied", "mocap_pos", "mocap_quat", "qacc_warmstart",
                "qM", "qacc", "qacc_smooth", "qfrc_smooth",
                "qfrc_constraint", "efc_force", "efc_J", "xpos", "xquat",
            )
        },
    }


def fingerprints_equal(left: dict[str, Any], right: dict[str, Any]) -> bool:
    if left.keys() != right.keys():
        return False
    for name in left:
        if name in ("time", "ncon", "nefc"):
            if left[name] != right[name]:
                return False
        elif left[name] is None or right[name] is None:
            if left[name] is not right[name]:
                return False
        elif not np.array_equal(left[name], right[name]):
            return False
    return True


def clone_and_forward_preintegration_data(
    mujoco: Any, model: Any, live_data: Any
) -> tuple[Any, dict[str, Any]]:
    """Run one forward on a full clone without changing formal live mjData."""
    live_before = live_data_fingerprint(live_data)
    probe_data = mujoco.MjData(model)
    mujoco.mj_copyData(probe_data, model, live_data)
    clone_state_matches = fingerprints_equal(
        live_before, live_data_fingerprint(probe_data)
    )
    mujoco.mj_forward(model, probe_data)
    live_after = live_data_fingerprint(live_data)
    return probe_data, {
        "copy_api": "mujoco.mj_copyData",
        "probe_forward_api": "mujoco.mj_forward",
        "probe_mj_forward_count": 1,
        "extra_live_mj_forward_count": 0,
        "extra_live_mj_step_count": 0,
        "clone_pre_state_matches_live": clone_state_matches,
        "live_data_unchanged_by_probe": fingerprints_equal(
            live_before, live_after
        ),
    }


def build_generalized_dof_order(
    mujoco: Any, model: Any
) -> dict[str, Any]:
    joint_names = evaluator._runtime_object_names(model, "joint", int(model.njnt))
    body_names = evaluator._runtime_object_names(model, "body", int(model.nbody))
    type_names = {
        int(mujoco.mjtJoint.mjJNT_FREE): "free",
        int(mujoco.mjtJoint.mjJNT_BALL): "ball",
        int(mujoco.mjtJoint.mjJNT_SLIDE): "slide",
        int(mujoco.mjtJoint.mjJNT_HINGE): "hinge",
    }
    free_semantics = (
        ("root_translation_x", "global-frame linear velocity x"),
        ("root_translation_y", "global-frame linear velocity y"),
        ("root_translation_z", "global-frame linear velocity z"),
        ("root_rotation_x", "free-joint local body-frame angular velocity x"),
        ("root_rotation_y", "free-joint local body-frame angular velocity y"),
        ("root_rotation_z", "free-joint local body-frame angular velocity z"),
    )
    entries = []
    for dof_index in range(int(model.nv)):
        joint_id = int(model.dof_jntid[dof_index])
        joint_type_value = int(model.jnt_type[joint_id])
        joint_type = type_names.get(joint_type_value, f"unknown_{joint_type_value}")
        dof_address = int(model.jnt_dofadr[joint_id])
        local_index = dof_index - dof_address
        body_id = int(model.jnt_bodyid[joint_id])
        if joint_type == "free":
            coordinate_label, coordinate_semantics = free_semantics[local_index]
            joint_axis = None
        elif joint_type == "ball":
            coordinate_label = f"ball_rotation_{'xyz'[local_index]}"
            coordinate_semantics = (
                "ball-joint local tangent-space angular velocity component"
            )
            joint_axis = None
        elif joint_type == "slide":
            coordinate_label = "slide_velocity"
            coordinate_semantics = (
                "scalar translational velocity along compiled joint axis"
            )
            joint_axis = np.asarray(model.jnt_axis[joint_id], dtype=np.float64)
        elif joint_type == "hinge":
            coordinate_label = "hinge_angular_velocity"
            coordinate_semantics = (
                "scalar angular velocity about compiled joint axis"
            )
            joint_axis = np.asarray(model.jnt_axis[joint_id], dtype=np.float64)
        else:
            coordinate_label = "unknown"
            coordinate_semantics = "unknown joint-coordinate semantics"
            joint_axis = None
        entries.append(
            {
                "dof_index": dof_index,
                "joint_id": joint_id,
                "joint_name": joint_names[joint_id],
                "joint_type": joint_type,
                "joint_type_numeric": joint_type_value,
                "joint_dof_address": dof_address,
                "joint_qpos_address": int(model.jnt_qposadr[joint_id]),
                "local_dof_index_within_joint": local_index,
                "body_id": body_id,
                "body_name": body_names[body_id],
                "joint_axis": joint_axis,
                "coordinate_label": coordinate_label,
                "coordinate_semantics": coordinate_semantics,
            }
        )
    free_count = sum(item["joint_type"] == "free" for item in entries)
    scalar_count = sum(
        item["joint_type"] in ("hinge", "slide") for item in entries
    )
    explicit = (
        len(entries) == int(model.nv)
        and [item["dof_index"] for item in entries] == list(range(int(model.nv)))
        and all(not item["joint_type"].startswith("unknown_") for item in entries)
    )
    return {
        "GENERALIZED_DOF_ORDER": "EXPLICIT" if explicit else "INCOMPLETE",
        "GENERALIZED_DOF_COUNT": len(entries),
        "generalized_column_order": GENERALIZED_COLUMN_ORDER,
        "source": "runtime MjModel dof_jntid, jnt_dofadr, jnt_qposadr, jnt_type and jnt_bodyid",
        "free_joint_velocity_frame_note": (
            "free-joint linear velocity is global-frame; angular velocity is "
            "in the local body frame"
        ),
        "free_joint_dof_count": free_count,
        "scalar_joint_dof_count": scalar_count,
        "dofs": entries,
    }


def _enum_name(enum_type: Any, numeric_value: int) -> str:
    try:
        return enum_type(int(numeric_value)).name
    except (TypeError, ValueError):
        for name in dir(enum_type):
            if name.startswith("mj") and int(getattr(enum_type, name)) == int(numeric_value):
                return name
    return f"UNKNOWN_{numeric_value}"


def inspect_integrator_semantics(
    mujoco: Any, model: Any, probe_data: Any
) -> dict[str, Any]:
    integrator_value = int(model.opt.integrator)
    integrator_name = _enum_name(mujoco.mjtIntegrator, integrator_value)
    disableflags = int(model.opt.disableflags)
    eulerdamp_bit = int(mujoco.mjtDisableBit.mjDSBL_EULERDAMP)
    damper_bit = int(mujoco.mjtDisableBit.mjDSBL_DAMPER)
    eulerdamp_disabled = bool(disableflags & eulerdamp_bit)
    damper_disabled = bool(disableflags & damper_bit)
    damping = np.asarray(model.dof_damping, dtype=np.float64).copy()
    qderiv = getattr(probe_data, "qDeriv", None)
    qderiv_array = np.asarray(qderiv, dtype=np.float64).copy() if qderiv is not None else None
    if integrator_name == "mjINT_EULER":
        implicit_damping = not eulerdamp_disabled and not damper_disabled
        classification = (
            "EULER_WITH_IMPLICIT_JOINT_DAMPING"
            if implicit_damping
            else "EULER_WITHOUT_IMPLICIT_JOINT_DAMPING"
        )
    elif integrator_name == "mjINT_IMPLICITFAST":
        implicit_damping = None
        classification = "IMPLICITFAST"
    elif integrator_name == "mjINT_IMPLICIT":
        implicit_damping = None
        classification = "IMPLICIT"
    elif integrator_name == "mjINT_RK4":
        implicit_damping = None
        classification = "RK4"
    else:
        implicit_damping = None
        classification = "INSUFFICIENT_EVIDENCE"
    return {
        "model_opt_integrator_numeric": integrator_value,
        "model_opt_integrator_enum_name": integrator_name,
        "model_opt_timestep": float(model.opt.timestep),
        "model_opt_disableflags": disableflags,
        "EULERDAMP_disable_bit": eulerdamp_bit,
        "EULERDAMP_disabled": eulerdamp_disabled,
        "DAMPER_disable_bit": damper_bit,
        "DAMPER_disabled": damper_disabled,
        "euler_implicit_joint_damping_effective": implicit_damping,
        "dof_damping": damping,
        "dof_damping_min": float(np.min(damping)),
        "dof_damping_max": float(np.max(damping)),
        "dof_damping_nonzero_count": int(np.count_nonzero(damping)),
        "qDeriv_available": qderiv_array is not None,
        "qDeriv_shape": list(qderiv_array.shape) if qderiv_array is not None else None,
        "qDeriv_layout": (
            "runtime binding raw layout; not assumed dense unless shape is nv by nv"
            if qderiv_array is not None else None
        ),
        "INTEGRATOR_SEMANTICS": classification,
    }


def build_integration_matrix(
    raw_mass: np.ndarray,
    timestep: float,
    damping: np.ndarray,
    integrator: dict[str, Any],
    qderiv: np.ndarray | None = None,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    raw_mass = np.asarray(raw_mass, dtype=np.float64)
    damping = np.asarray(damping, dtype=np.float64)
    name = integrator["model_opt_integrator_enum_name"]
    if name == "mjINT_EULER":
        if integrator["euler_implicit_joint_damping_effective"]:
            delta = float(timestep) * np.diag(damping)
            matrix = raw_mass + delta
            formula = "M + dt * diag(model.dof_damping)"
        else:
            delta = np.zeros_like(raw_mass)
            matrix = raw_mass.copy()
            formula = "M because Euler implicit joint damping is disabled"
        status = "VALIDATED"
    elif name in ("mjINT_IMPLICIT", "mjINT_IMPLICITFAST"):
        if qderiv is not None and np.asarray(qderiv).shape == raw_mass.shape:
            delta = -float(timestep) * np.asarray(qderiv, dtype=np.float64)
            matrix = raw_mass + delta
            formula = "M - dt * qDeriv"
            status = "VALIDATED"
        else:
            return None, {
                "INTEGRATION_MATRIX_CONSTRUCTION": "UNSUPPORTED",
                "formula": "M - dt * qDeriv",
                "reason": "dense nv by nv qDeriv is unavailable",
            }
    else:
        return None, {
            "INTEGRATION_MATRIX_CONSTRUCTION": "UNSUPPORTED",
            "formula": None,
            "reason": f"integrator {name} is unsupported for current closure",
        }
    off_diagonal_delta = delta - np.diag(np.diag(delta))
    return matrix, {
        "INTEGRATION_MATRIX_CONSTRUCTION": status,
        "formula": formula,
        "shape": list(matrix.shape),
        "all_finite": bool(np.isfinite(matrix).all()),
        "symmetry_max_abs_error": float(np.max(np.abs(matrix - matrix.T))),
        "condition_number": float(np.linalg.cond(matrix)),
        "delta": delta,
        "delta_off_diagonal_max_abs": float(np.max(np.abs(off_diagonal_delta))),
    }


def vector_error_metrics(candidate: Any, reference: Any) -> dict[str, Any]:
    candidate = np.asarray(candidate, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    residual = candidate - reference
    denominator = max(float(np.linalg.norm(reference)), np.finfo(np.float64).eps)
    return {
        "residual": residual,
        "relative_l2_error": float(np.linalg.norm(residual) / denominator),
        "max_abs_error": float(np.max(np.abs(residual))) if residual.size else 0.0,
    }


def velocity_closures(
    capture: dict[str, Any], integration_matrix: np.ndarray | None
) -> dict[str, dict[str, Any]]:
    """Close MuJoCo's one-step velocity update in generalized coordinates."""
    unsupported = {
        "status": "INSUFFICIENT_EVIDENCE",
        "reason": "integration-effective matrix construction is unsupported",
    }
    if integration_matrix is None:
        return {
            "generalized": {"GENERALIZED_VELOCITY_CLOSURE": "INSUFFICIENT_EVIDENCE", **unsupported},
            "constraint": {"CONTACT_CONSTRAINT_GENERALIZED_CLOSURE": "INSUFFICIENT_EVIDENCE", **unsupported},
            "physical": {"PHYSICAL_CONTACT_IMPULSE_MAPPING": "INSUFFICIENT_EVIDENCE", **unsupported},
        }
    state = capture["solver_phase_state"]
    dt = float(capture["post_simulation_time"] - capture["pre_simulation_time"])
    pre_qvel = np.asarray(capture["pre_state"]["qvel"], dtype=np.float64)
    post_qvel = np.asarray(capture["post_state"]["qvel"], dtype=np.float64)
    observed_delta = post_qvel - pre_qvel
    qfrc_smooth = np.asarray(state["qfrc_smooth"], dtype=np.float64)
    qfrc_constraint = np.asarray(state["qfrc_constraint"], dtype=np.float64)
    smooth_accel, smooth_solve = stable_solve(integration_matrix, qfrc_smooth)
    constraint_accel, constraint_solve = stable_solve(
        integration_matrix, qfrc_constraint
    )
    total_accel, total_solve = stable_solve(
        integration_matrix, qfrc_smooth + qfrc_constraint
    )
    smooth_delta = dt * smooth_accel
    constraint_delta = dt * constraint_accel
    predicted_delta = dt * total_accel
    errors = vector_error_metrics(predicted_delta, observed_delta)
    generalized_pass = bool(
        errors["relative_l2_error"] <= CLOSURE_RELATIVE_L2_TOLERANCE
        and errors["max_abs_error"] <= CLOSURE_MAX_ABS_TOLERANCE
        and total_solve["residual_max_abs"] <= CLOSURE_MAX_ABS_TOLERANCE
    )
    decomposition = vector_error_metrics(
        smooth_delta + constraint_delta, predicted_delta
    )
    j_phys = np.asarray(capture["J_phys"], dtype=np.float64)
    projected_constraint_delta = j_phys @ constraint_delta
    constraint_pass = bool(
        decomposition["max_abs_error"] <= CLOSURE_MAX_ABS_TOLERANCE
        and constraint_solve["residual_max_abs"] <= CLOSURE_MAX_ABS_TOLERANCE
    )

    physical_contact_impulse = np.concatenate(
        [
            np.concatenate(
                ([float(contact["normal_impulse"])], np.asarray(contact["tangential_impulse"], dtype=np.float64))
            )
            for contact in capture["contacts"]
        ]
    )
    physical_impulse = np.asarray(capture["J_phys"], dtype=np.float64).T @ physical_contact_impulse
    applyft_physical_impulse = np.zeros(int(capture["nv"]), dtype=np.float64)
    floor_efc_impulse = np.zeros_like(physical_impulse)
    for contact in capture["contacts"]:
        projection = contact["formal_physical_projection"]
        applyft_physical_impulse += dt * np.asarray(
            projection["qfrc_total"], dtype=np.float64
        )
        floor_efc_impulse += dt * np.asarray(
            projection["qfrc_constraint_rows_contact"], dtype=np.float64
        )
    physical_vs_floor = vector_error_metrics(physical_impulse, floor_efc_impulse)
    jacobian_vs_applyft = vector_error_metrics(
        physical_impulse, applyft_physical_impulse
    )
    all_constraint_impulse = dt * qfrc_constraint
    physical_vs_all_constraints = vector_error_metrics(
        physical_impulse, all_constraint_impulse
    )
    nonfloor_constraint_impulse = all_constraint_impulse - floor_efc_impulse
    total_scale = max(float(np.linalg.norm(all_constraint_impulse)), np.finfo(float).eps)
    nonfloor_ratio = float(np.linalg.norm(nonfloor_constraint_impulse) / total_scale)
    physical_delta, physical_solve = stable_solve(
        integration_matrix, physical_impulse
    )
    physical_mapping_closes = bool(
        physical_vs_floor["relative_l2_error"] <= CLOSURE_RELATIVE_L2_TOLERANCE
        and physical_vs_floor["max_abs_error"] <= CLOSURE_MAX_ABS_TOLERANCE
        and jacobian_vs_applyft["max_abs_error"] <= CLOSURE_MAX_ABS_TOLERANCE
        and physical_solve["residual_max_abs"] <= CLOSURE_MAX_ABS_TOLERANCE
    )
    if not physical_mapping_closes:
        physical_status = "FAIL"
    elif nonfloor_ratio <= CLOSURE_RELATIVE_L2_TOLERANCE:
        physical_status = "PASS"
    else:
        physical_status = "PARTIAL"
    return {
        "generalized": {
            "GENERALIZED_VELOCITY_CLOSURE": "PASS" if generalized_pass else "FAIL",
            "integration_equation": "qvel_post - qvel_pre = dt * solve(M_integration, qfrc_smooth + qfrc_constraint)",
            "qvel_pre": pre_qvel,
            "qvel_post": post_qvel,
            "observed_delta_qvel": observed_delta,
            "predicted_delta_qvel": predicted_delta,
            **errors,
            "total_linear_solve": total_solve,
            "qfrc_smooth_source": state["qfrc_smooth_source"],
        },
        "constraint": {
            "CONTACT_CONSTRAINT_GENERALIZED_CLOSURE": "PASS" if constraint_pass else "FAIL",
            "terminology_note": "constraint includes all active MuJoCo constraints, not only contacts",
            "predicted_smooth_delta_qvel": smooth_delta,
            "predicted_constraint_delta_qvel": constraint_delta,
            "predicted_total_from_component_sum": smooth_delta + constraint_delta,
            "predicted_contact_velocity_delta_from_all_constraints": projected_constraint_delta,
            "component_additivity": decomposition,
            "smooth_linear_solve": smooth_solve,
            "constraint_linear_solve": constraint_solve,
        },
        "physical": {
            "PHYSICAL_CONTACT_IMPULSE_MAPPING": physical_status,
            "physical_floor_contact_generalized_impulse": physical_impulse,
            "physical_contact_impulse_vector_ordered_by_J_rows": physical_contact_impulse,
            "applyFT_floor_contact_generalized_impulse": applyft_physical_impulse,
            "J_transpose_vs_applyFT": jacobian_vs_applyft,
            "floor_contact_efc_generalized_impulse": floor_efc_impulse,
            "physical_vs_floor_efc": physical_vs_floor,
            "physical_vs_all_constraints": physical_vs_all_constraints,
            "physical_to_generalized_impulse_residual": physical_vs_all_constraints["residual"],
            "physical_to_generalized_impulse_relative_l2": physical_vs_all_constraints["relative_l2_error"],
            "physical_to_generalized_impulse_max_abs_error": physical_vs_all_constraints["max_abs_error"],
            "all_constraint_generalized_impulse": all_constraint_impulse,
            "nonfloor_constraint_generalized_impulse": nonfloor_constraint_impulse,
            "nonfloor_constraint_impulse_relative_l2": nonfloor_ratio,
            "predicted_delta_qvel_from_physical_floor_contacts": physical_delta,
            "predicted_delta_v_contact_from_physical_floor_contacts": np.asarray(
                capture["J_phys"], dtype=np.float64
            ) @ physical_delta,
            "physical_impulse_linear_solve": physical_solve,
            "interpretation": "PARTIAL means physical floor-contact wrench closes its own EFC rows but non-floor constraints also contribute to total qfrc_constraint",
        },
    }


def load_old_artifact_regression(
    existing_oracle: Path, current_budget: dict[str, Any], capture: dict[str, Any]
) -> dict[str, Any]:
    old_budget = json.loads(
        (existing_oracle / "global55_effective_mass_budget.json").read_text(encoding="utf-8")
    )
    old_contacts = json.loads(
        (existing_oracle / "global55_contacts.json").read_text(encoding="utf-8")
    )["all_active_robot_floor_contacts"]
    result: dict[str, Any] = {"rtol": 1.0e-9, "atol": 1.0e-9, "joints": {}}
    overall = True
    for geom, _ in SELECTED:
        old_contact = next(item for item in old_contacts if pair_matches(item, geom))
        current_contact = selected_contact(capture, geom)
        old_selected = old_budget["selected"][geom]
        current_selected = current_budget["selected"][geom]
        old_values = {
            "directional_effective_mass_kg": old_selected["directional_effective_mass_kg"],
            "uncoupled_sticking_impulse_norm": old_selected["uncoupled_sticking_impulse_norm"],
            "global_normal_conditioned_sticking_impulse_norm": old_selected["global_normal_conditioned_sticking_impulse_norm"],
            "actual_normal_impulse": old_selected["actual_normal_impulse"],
            "actual_tangential_impulse_norm": old_selected["actual_tangential_impulse_norm"],
            "pre_tangential_speed": old_selected["pre_tangential_speed"],
            "post_tangential_speed": old_contact["post_tangential_speed"],
        }
        current_values = {
            "directional_effective_mass_kg": current_selected["directional_effective_mass_kg"],
            "uncoupled_sticking_impulse_norm": current_selected["uncoupled_sticking_impulse_norm"],
            "global_normal_conditioned_sticking_impulse_norm": current_selected["global_normal_conditioned_sticking_impulse_norm"],
            "actual_normal_impulse": current_selected["actual_normal_impulse"],
            "actual_tangential_impulse_norm": current_selected["actual_tangential_impulse_norm"],
            "pre_tangential_speed": current_selected["pre_tangential_speed"],
            "post_tangential_speed": current_contact["post_tangential_speed"],
        }
        checks = {
            name: bool(np.isclose(current_values[name], old_values[name], rtol=1e-9, atol=1e-9))
            for name in old_values
        }
        valid = all(checks.values())
        result["joints"][geom] = {
            "old": old_values,
            "current": current_values,
            "checks": checks,
            "valid": valid,
        }
        overall &= valid
    result["OLD_ARTIFACT_REGRESSION"] = "PASS" if overall else "FAIL"
    return result


def capture_global55(
    recorder: evaluator.JointLimitSubstepRecorder,
    pre: dict[str, Any],
) -> dict[str, Any]:
    mujoco, model, live_data = evaluator._native_model_data(recorder.sim)
    data = pre.pop("_global55_probe_data")
    probe_evidence = pre.pop("_global55_probe_evidence")
    nv, nq, nefc = int(model.nv), int(model.nq), int(data.nefc)
    qfrc_applied_before = np.asarray(data.qfrc_applied, dtype=np.float64).copy()
    mass, mass_stats = expanded_mass_matrix(mujoco, model, data)
    formal_record = recorder.records[-1]
    floor_id = int(recorder.mapping["floor_geom_id"])
    probe_sim = SimpleNamespace(
        _sim=SimpleNamespace(_model=model, _data=data), model=model, data=data
    )
    probe_contacts, _ = evaluator.capture_contact_response(
        probe_sim,
        recorder.mapping,
        recorder.tracker,
        record_physical_projection=True,
        pre_qvel=pre["full_qvel"],
    )
    floor_contacts = [
        contact
        for contact in probe_contacts
        if floor_id in (int(contact["geom1_id"]), int(contact["geom2_id"]))
    ]
    if not floor_contacts:
        raise RuntimeError("global55 contains no active robot-floor contacts")

    jacobian_blocks = []
    contact_records = []
    point_mapping_valid = True
    pyramidal = int(model.opt.cone) == int(mujoco.mjtCone.mjCONE_PYRAMIDAL)
    pre_qvel = np.asarray(pre["full_qvel"], dtype=np.float64)
    post_qvel = np.asarray(live_data.qvel, dtype=np.float64).copy()
    dt = float(model.opt.timestep)
    for contact in floor_contacts:
        index = int(contact["contact_index"])
        native = data.contact[index]
        geom1, geom2 = int(native.geom1), int(native.geom2)
        body1, body2 = int(model.geom_bodyid[geom1]), int(model.geom_bodyid[geom2])
        robot_body = body2 if body1 == 0 else body1
        if robot_body == 0 or (body1 != 0 and body2 != 0):
            raise RuntimeError(f"floor contact {index} does not have one robot side")
        projection = contact["physical_projection"]
        if not projection["physical_wrench_sign_valid"]:
            raise RuntimeError(f"physical wrench sign is invalid for contact {index}")
        sign = int(projection["robot_side_sign"])
        robot_side_factor = sign if robot_body == body2 else -sign
        frame = np.asarray(native.frame, dtype=np.float64).reshape(3, 3)
        basis = physical_basis(frame, robot_side_factor)
        point = np.asarray(native.pos, dtype=np.float64)
        jac_world, pre_velocity = point_jacobian_and_velocity(
            mujoco, model, data, robot_body, point, pre_qvel
        )
        _, post_velocity = point_jacobian_and_velocity(
            mujoco, model, data, robot_body, point, post_qvel
        )
        jac_phys = basis @ jac_world
        jacobian_blocks.append(jac_phys)
        force_world = np.asarray(
            projection[
                "total_force_world_on_body2"
                if robot_body == body2
                else "total_force_world_on_body1"
            ],
            dtype=np.float64,
        )
        force_phys = basis @ force_world
        rows = evaluator.contact_efc_rows(
            int(native.efc_address), int(native.dim), pyramidal, nefc
        )
        pre_phys = jac_phys @ pre_qvel
        post_phys = jac_phys @ post_qvel
        basis_check = basis @ basis.T
        point_mapping_valid &= (
            pre_velocity["mapping_max_abs_error"] <= 1.0e-9
            and post_velocity["mapping_max_abs_error"] <= 1.0e-9
        )
        contact_records.append(
            {
                "contact_index": index,
                "geom1_id": geom1,
                "geom1_name": contact["geom1_name"],
                "body1_id": body1,
                "body1_name": contact["body1_name"],
                "geom2_id": geom2,
                "geom2_name": contact["geom2_name"],
                "body2_id": body2,
                "body2_name": contact["body2_name"],
                "robot_body_id": robot_body,
                "robot_body_name": recorder.mapping["body_names"][robot_body],
                "ground_body_id": 0,
                "ground_velocity_world": [0.0, 0.0, 0.0],
                "point_world": point,
                "contact_frame_world_rows": frame,
                "physical_basis_world_rows": basis,
                "physical_basis_semantics": [
                    "normal direction of physical force on robot",
                    "right-handed tangent1",
                    "right-handed tangent2",
                ],
                "basis_orthonormality_max_abs_error": float(
                    np.max(np.abs(basis_check - np.eye(3)))
                ),
                "basis_determinant": float(np.linalg.det(basis)),
                "friction": np.asarray(native.friction, dtype=np.float64),
                "dist": float(native.dist),
                "dim": int(native.dim),
                "efc_address": int(native.efc_address),
                "efc_rows": rows,
                "J_point_world": jac_world,
                "J_physical": jac_phys,
                "pre_point_velocity": pre_velocity,
                "post_point_velocity": post_velocity,
                "pre_velocity_physical": pre_phys,
                "post_velocity_physical": post_phys,
                "pre_normal_velocity": float(pre_phys[0]),
                "pre_tangential_velocity": pre_phys[1:3],
                "pre_tangential_speed": float(np.linalg.norm(pre_phys[1:3])),
                "pre_slip_direction": (
                    pre_phys[1:3] / np.linalg.norm(pre_phys[1:3])
                    if np.linalg.norm(pre_phys[1:3]) > 1.0e-12
                    else None
                ),
                "post_normal_velocity": float(post_phys[0]),
                "post_tangential_velocity": post_phys[1:3],
                "post_tangential_speed": float(np.linalg.norm(post_phys[1:3])),
                "force_world_on_robot": force_world,
                "force_physical": force_phys,
                "normal_force": float(force_phys[0]),
                "tangential_force": force_phys[1:3],
                "tangential_force_norm": float(np.linalg.norm(force_phys[1:3])),
                "normal_impulse": float(force_to_impulse(force_phys[0], dt)),
                "tangential_impulse": force_to_impulse(force_phys[1:3], dt),
                "tangential_impulse_norm": float(
                    np.linalg.norm(force_to_impulse(force_phys[1:3], dt))
                ),
                "constraint_generalized_force": np.asarray(
                    projection["qfrc_constraint_rows_contact"], dtype=np.float64
                ),
                "solver_rows": rich_solver_rows(data, rows, nefc, nv),
                "formal_physical_projection": projection,
            }
        )

    j_phys = np.vstack(jacobian_blocks)
    solved, mass_solve = stable_solve(mass, j_phys.T)
    w_phys = j_phys @ solved
    w_symmetry = float(np.max(np.abs(w_phys - w_phys.T)))
    w_eigenvalues = np.linalg.eigvalsh((w_phys + w_phys.T) * 0.5)
    qfrc_applied_after = np.asarray(data.qfrc_applied, dtype=np.float64).copy()
    physical_projection_scratch_unchanged = all(
        bool(contact["physical_projection"]["qfrc_applied_unchanged"])
        for contact in floor_contacts
    )
    formal_data_mutated = bool(
        not np.array_equal(qfrc_applied_before, qfrc_applied_after)
        or not physical_projection_scratch_unchanged
        or mass_stats["live_qM_mutated"]
    )
    generalized_dof_order = build_generalized_dof_order(mujoco, model)
    integrator_semantics = inspect_integrator_semantics(mujoco, model, data)
    qderiv = getattr(data, "qDeriv", None)
    integration_matrix, integration_matrix_report = build_integration_matrix(
        mass,
        float(model.opt.timestep),
        np.asarray(model.dof_damping, dtype=np.float64),
        integrator_semantics,
        np.asarray(qderiv, dtype=np.float64) if qderiv is not None else None,
    )
    return {
        "capture_phase": "PRE_INTEGRATION_CLONE_FORWARD_PLUS_FORMAL_POST_INTEGRATION_STATE",
        "solver_linearization_configuration": "PRE_INTEGRATION",
        "control_step": int(formal_record["control_step"]),
        "physics_substep_in_control": int(formal_record["physics_substep_in_control"]),
        "global_physics_step": int(formal_record["global_physics_step"]),
        "pre_simulation_time": float(pre["simulation_time"]),
        "post_simulation_time": float(live_data.time),
        "nq": nq,
        "nv": nv,
        "ncon": int(data.ncon),
        "nefc": nefc,
        "pre_state": {
            "qpos": np.asarray(pre["full_qpos"], dtype=np.float64),
            "qvel": pre_qvel,
        },
        "solver_phase_state": {
            "qacc_smooth": np.asarray(data.qacc_smooth, dtype=np.float64).copy(),
            "qacc": np.asarray(data.qacc, dtype=np.float64).copy(),
            "qfrc_applied": qfrc_applied_after,
            "qfrc_actuator": np.asarray(data.qfrc_actuator, dtype=np.float64).copy(),
            "qfrc_passive": np.asarray(data.qfrc_passive, dtype=np.float64).copy(),
            "qfrc_bias": np.asarray(data.qfrc_bias, dtype=np.float64).copy(),
            "qfrc_constraint": np.asarray(data.qfrc_constraint, dtype=np.float64).copy(),
            "qfrc_smooth": (
                np.asarray(data.qfrc_smooth, dtype=np.float64).copy()
                if hasattr(data, "qfrc_smooth")
                else np.asarray(data.qfrc_applied, dtype=np.float64)
                + np.asarray(data.qfrc_actuator, dtype=np.float64)
                + np.asarray(data.qfrc_passive, dtype=np.float64)
                - np.asarray(data.qfrc_bias, dtype=np.float64)
            ),
            "qfrc_smooth_source": (
                "runtime data.qfrc_smooth"
                if hasattr(data, "qfrc_smooth")
                else "qfrc_applied + qfrc_actuator + qfrc_passive - qfrc_bias"
            ),
        },
        "post_state": {
            "qpos": np.asarray(live_data.qpos, dtype=np.float64).copy(),
            "qvel": post_qvel,
        },
        "mass_matrix": mass,
        "mass_matrix_stats": mass_stats,
        "generalized_dof_order": generalized_dof_order,
        "integrator_semantics": integrator_semantics,
        "integration_matrix": integration_matrix,
        "integration_matrix_report": integration_matrix_report,
        "contacts": contact_records,
        "J_phys": j_phys,
        "W_phys": w_phys,
        "delassus_stats": {
            "shape": list(w_phys.shape),
            "symmetry_max_abs_error": w_symmetry,
            "eigenvalues": w_eigenvalues,
            "minimum_eigenvalue": float(np.min(w_eigenvalues)),
            "condition_number": float(np.linalg.cond(w_phys)),
            "mass_linear_solve": mass_solve,
        },
        "point_velocity_mapping_valid": bool(point_mapping_valid),
        "qfrc_applied_before_probe": qfrc_applied_before,
        "qfrc_applied_after_probe": qfrc_applied_after,
        "physical_projection_scratch_targets_unchanged": physical_projection_scratch_unchanged,
        "formal_data_mutated_by_probe": bool(
            formal_data_mutated or not probe_evidence["live_data_unchanged_by_probe"]
        ),
        "probe_evidence": probe_evidence,
        "formal_record": formal_record,
    }


class DemandRecorder(evaluator.JointLimitSubstepRecorder):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.global55: dict[str, Any] | None = None

    def capture_pre_step(self) -> dict[str, Any]:
        pre = super().capture_pre_step()
        if self.global_physics_step + 1 == GLOBAL_STEP:
            mujoco, model, live_data = evaluator._native_model_data(self.sim)
            probe_data, evidence = clone_and_forward_preintegration_data(
                mujoco, model, live_data
            )
            pre["_global55_probe_data"] = probe_data
            pre["_global55_probe_evidence"] = evidence
        return pre

    def capture_post_step(self, pre: dict[str, Any]) -> None:
        super().capture_post_step(pre)
        if self.global_physics_step == GLOBAL_STEP:
            self.global55 = capture_global55(self, pre)


def selected_contact(capture: dict[str, Any], geom_name: str) -> dict[str, Any]:
    matches = [
        contact for contact in capture["contacts"] if pair_matches(contact, geom_name)
    ]
    if len(matches) != 1:
        raise RuntimeError(f"expected one {geom_name}-floor contact, found {len(matches)}")
    return matches[0]


def regression_check(capture: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {"tolerance_rtol": 1.0e-9, "tolerance_atol": 1.0e-9}
    all_valid = True
    reference = None
    for geom, joint in SELECTED:
        contact = selected_contact(capture, geom)
        projection = contact["formal_physical_projection"]
        observed = {
            "Fn": float(projection["Fn"]),
            "Ft_norm": float(projection["friction_force_norm"]),
            "normal_generalized": float(projection["selected_joints"][joint]["normal"]),
            "friction_generalized": float(projection["selected_joints"][joint]["friction"]),
            "total_generalized": float(projection["selected_joints"][joint]["total"]),
        }
        expected = REGRESSION if geom == "limb/12" else (reference or REGRESSION)
        checks = {
            name: bool(np.isclose(observed[name], expected[name], rtol=1.0e-9, atol=1.0e-9))
            for name in expected
        }
        valid = all(checks.values())
        result[geom] = {
            "joint": joint,
            "observed": observed,
            "expected": expected,
            "absolute_error": {name: observed[name] - expected[name] for name in expected},
            "checks": checks,
            "valid": valid,
        }
        if geom == "limb/12":
            reference = observed
        all_valid &= valid
    result["MUJOCO_GLOBAL55_REGRESSION"] = "PASS" if all_valid else "FAIL"
    return result


def demand_budget(capture: dict[str, Any]) -> dict[str, Any]:
    j_phys = np.asarray(capture["J_phys"], dtype=np.float64)
    w_phys = np.asarray(capture["W_phys"], dtype=np.float64)
    qvel_pre = np.asarray(capture["pre_state"]["qvel"], dtype=np.float64)
    contacts = capture["contacts"]
    count = len(contacts)
    normal_rows = [3 * index for index in range(count)]
    tangent_rows = [row for index in range(count) for row in (3 * index + 1, 3 * index + 2)]
    p_normal = np.asarray([contact["normal_impulse"] for contact in contacts])
    p_tangent = np.concatenate(
        [np.asarray(contact["tangential_impulse"], dtype=np.float64) for contact in contacts]
    )
    v_pre = j_phys @ qvel_pre
    v_t_pre_all = v_pre[tangent_rows]
    delta_v_t_normals_all = w_phys[np.ix_(tangent_rows, normal_rows)] @ p_normal
    v_t_after_normals_all = v_t_pre_all + delta_v_t_normals_all
    w_tt_global = w_phys[np.ix_(tangent_rows, tangent_rows)]
    p_t_stick_global, global_stick_solve = stable_solve(
        w_tt_global, -v_t_after_normals_all
    )
    selected = {}
    for geom, joint in SELECTED:
        contact = selected_contact(capture, geom)
        index = next(
            candidate
            for candidate, item in enumerate(contacts)
            if item is contact
        )
        tangent = [3 * index + 1, 3 * index + 2]
        own_normal = [3 * index]
        other_normals = [row for row in normal_rows if row not in own_normal]
        other_tangents = [row for row in tangent_rows if row not in tangent]
        vt = v_pre[tangent]
        speed = float(np.linalg.norm(vt))
        slip = vt / speed if speed > 1.0e-12 else np.zeros(2)
        wtt = w_phys[np.ix_(tangent, tangent)]
        directional_inverse_mass = float(slip @ wtt @ slip)
        uncoupled, uncoupled_solve = stable_solve(wtt, -vt)
        own_normal_delta = w_phys[np.ix_(tangent, own_normal)] @ p_normal[[index]]
        other_normal_delta = (
            w_phys[np.ix_(tangent, other_normals)]
            @ p_normal[[normal_rows.index(row) for row in other_normals]]
            if other_normals else np.zeros(2)
        )
        selected_global = p_t_stick_global[2 * index : 2 * index + 2]
        actual_tangent = np.asarray(contact["tangential_impulse"])
        other_tangent_actual = (
            w_phys[np.ix_(tangent, other_tangents)]
            @ p_tangent[[tangent_rows.index(row) for row in other_tangents]]
            if other_tangents else np.zeros(2)
        )
        selected[geom] = {
            "joint": joint,
            "contact_index_in_floor_stack": index,
            "pre_tangential_velocity": vt,
            "pre_tangential_speed": speed,
            "slip_direction": slip,
            "W_tt_local": wtt,
            "W_tn_own": w_phys[np.ix_(tangent, own_normal)],
            "W_tn_other_floor_contacts": w_phys[np.ix_(tangent, other_normals)],
            "directional_inverse_effective_mass_1_per_kg": directional_inverse_mass,
            "directional_effective_mass_kg": (
                1.0 / directional_inverse_mass if directional_inverse_mass > 0.0 else None
            ),
            "uncoupled_sticking_impulse": uncoupled,
            "uncoupled_sticking_impulse_norm": float(np.linalg.norm(uncoupled)),
            "uncoupled_sticking_solve": uncoupled_solve,
            "delta_v_t_from_own_normal": own_normal_delta,
            "delta_v_t_from_other_floor_normals": other_normal_delta,
            "delta_v_t_from_all_floor_normals": own_normal_delta + other_normal_delta,
            "v_t_after_all_floor_normals": vt + own_normal_delta + other_normal_delta,
            "global_normal_conditioned_sticking_impulse": selected_global,
            "global_normal_conditioned_sticking_impulse_norm": float(np.linalg.norm(selected_global)),
            "actual_tangential_impulse": actual_tangent,
            "actual_tangential_impulse_norm": float(np.linalg.norm(actual_tangent)),
            "actual_normal_impulse": float(contact["normal_impulse"]),
            "friction_cap_mu_pn": float(contact["friction"][0] * contact["normal_impulse"]),
            "delta_v_t_from_other_actual_tangential_impulses": other_tangent_actual,
            "cross_contact_tangent_block_frobenius_norm": float(
                np.linalg.norm(w_phys[np.ix_(tangent, other_tangents)])
            ),
            "solver_rows": contact["solver_rows"],
        }

    return {
        "contact_row_mapping": [
            {
                "stack_index": index,
                "contact_index": contact["contact_index"],
                "robot_geom": (
                    contact["geom2_name"] if contact["geom1_name"] == "floor/0" else contact["geom1_name"]
                ),
                "normal_row": normal_rows[index],
                "tangent_rows": tangent_rows[2 * index : 2 * index + 2],
            }
            for index, contact in enumerate(contacts)
        ],
        "normal_rows": normal_rows,
        "tangent_rows": tangent_rows,
        "W_tt_global": w_tt_global,
        "W_tn_global": w_phys[np.ix_(tangent_rows, normal_rows)],
        "actual_normal_impulses": p_normal,
        "actual_tangential_impulses": p_tangent,
        "delta_v_t_from_all_normals": delta_v_t_normals_all,
        "v_t_after_all_normals": v_t_after_normals_all,
        "global_normal_conditioned_sticking_impulse": p_t_stick_global,
        "global_sticking_solve": global_stick_solve,
        "selected": selected,
        "metric_semantics": {
            "effective_mass": "rigid unregularized directional metric from physical J M^-1 J^T",
            "sticking_demands": "rigid unregularized diagnostics; not asserted equal to MuJoCo solver solution",
            "normal_conditioning": "all active robot-floor normal impulses and global tangential cross-contact block",
        },
    }


def classifications(
    regression: dict[str, Any],
    capture: dict[str, Any],
    budget: dict[str, Any],
) -> dict[str, Any]:
    if regression["MUJOCO_GLOBAL55_REGRESSION"] != "PASS":
        return {
            "MUJOCO_GLOBAL55_FRICTION_REGIME": "INSUFFICIENT_EVIDENCE",
            "MUJOCO_GLOBAL55_DEMAND_DECOMPOSITION": "INSUFFICIENT_EVIDENCE",
            "DOMINANT_FRICTION_IMPULSE_GAP_DRIVER": "INSUFFICIENT_EVIDENCE",
            "reason": "global55 regression failed",
        }
    limb12 = budget["selected"]["limb/12"]
    actual = limb12["actual_tangential_impulse_norm"]
    cap = limb12["friction_cap_mu_pn"]
    rigid = limb12["global_normal_conditioned_sticking_impulse_norm"]
    cap_ratio = actual / cap if cap else None
    demand_ratio = actual / rigid if rigid else None
    coupling_ratio = float(
        np.linalg.norm(limb12["delta_v_t_from_all_floor_normals"])
        / max(limb12["pre_tangential_speed"], 1.0e-12)
    )
    cross_ratio = float(
        np.linalg.norm(limb12["delta_v_t_from_other_actual_tangential_impulses"])
        / max(limb12["pre_tangential_speed"], 1.0e-12)
    )
    positive_regularization = any(
        float(row["efc_R"]) > 0.0 for row in limb12["solver_rows"]
    )
    if cap_ratio is not None and cap_ratio >= 0.95:
        regime = "CAP_LIMITED"
    elif demand_ratio is not None and abs(demand_ratio - 1.0) <= 0.1:
        regime = "DEMAND_LIMITED"
    elif cross_ratio >= 0.1:
        regime = "COUPLED_MULTI_CONTACT"
    elif positive_regularization and demand_ratio is not None and demand_ratio < 0.9:
        regime = "REGULARIZATION_LIMITED"
    else:
        regime = "INSUFFICIENT_EVIDENCE"

    signals = []
    mujoco_slip = limb12["pre_tangential_speed"]
    mujoco_mass = limb12["directional_effective_mass_kg"]
    if abs(mujoco_slip - ISAAC_REFERENCE["pre_tangential_speed_m_per_s"]) > 0.1 * ISAAC_REFERENCE["pre_tangential_speed_m_per_s"]:
        signals.append("PRE_CONTACT_SLIP")
    if mujoco_mass is not None and abs(mujoco_mass - ISAAC_REFERENCE["apparent_directional_mass_kg"]) > 0.1 * ISAAC_REFERENCE["apparent_directional_mass_kg"]:
        signals.append("TANGENTIAL_EFFECTIVE_MASS")
    if coupling_ratio >= 0.1:
        signals.append("NORMAL_TANGENTIAL_COUPLING")
    if cross_ratio >= 0.1:
        signals.append("MULTI_CONTACT_COUPLING")
    if regime == "REGULARIZATION_LIMITED":
        signals.append("SOLVER_REGULARIZATION")
    driver = signals[0] if len(signals) == 1 else ("MIXED" if signals else "INSUFFICIENT_EVIDENCE")
    decomposition = "VALIDATED" if regime != "INSUFFICIENT_EVIDENCE" else "PARTIAL"
    return {
        "actual_over_friction_cap": cap_ratio,
        "actual_over_normal_conditioned_rigid_demand": demand_ratio,
        "normal_coupling_velocity_ratio": coupling_ratio,
        "other_contact_tangent_velocity_ratio": cross_ratio,
        "positive_solver_regularization_rows": positive_regularization,
        "driver_signals": signals,
        "MUJOCO_GLOBAL55_FRICTION_REGIME": regime,
        "MUJOCO_GLOBAL55_DEMAND_DECOMPOSITION": decomposition,
        "DOMINANT_FRICTION_IMPULSE_GAP_DRIVER": driver,
    }


def replay(args: argparse.Namespace, paths: dict[str, Path]) -> tuple[Any, Any]:
    import torch

    runtime_args = SimpleNamespace(
        checkpoint=str(paths["checkpoint"]),
        walker_dir=str(paths["walker_dir"]),
        morphology_id=args.morphology_id,
        cfg=args.cfg,
        seed=args.seed,
        device=args.device,
        reset_noise_scale=0.0,
    )
    evaluator.configure_runtime(runtime_args, paths)
    envs, model, _ = evaluator.build_runtime(runtime_args, paths)
    tracker = evaluator.FiniteTracker()
    base_env = None
    original_sim = None
    recorder = None
    try:
        observation = envs.reset()
        base_env = evaluator.unwrap_single_mujoco_env(envs)
        trajectory = evaluator.build_state_trajectory_metadata(base_env)
        mapping = evaluator.build_joint_limit_probe_mapping(
            base_env,
            [joint for _, joint in SELECTED],
            trajectory,
            contact_probe_body_names=(),
            enable_contact_mapping=True,
        )
        original_sim = base_env.sim
        recorder = DemandRecorder(
            original_sim,
            int(base_env.frame_skip),
            mapping,
            tracker,
            record_contact_response=True,
            record_physical_projection=True,
        )
        proxy = evaluator.JointLimitRecordingSimProxy(original_sim, recorder)
        base_env.sim = proxy
        for control_step in range(1, CONTROL_STEPS + 1):
            with torch.no_grad():
                _, distribution, _, _ = model(observation)
                action, _ = evaluator.choose_raw_action(distribution, observation, "zero")
            recorder.begin_control_step(0, control_step)
            observation, _, done, _ = envs.step(action)
            recorder.end_control_step()
            if proxy.callback_error is not None:
                raise RuntimeError("substep demand instrumentation failed") from proxy.callback_error
            if bool(done[0]):
                raise RuntimeError(f"formal environment terminated before control step {control_step}")
        if recorder.global55 is None:
            raise RuntimeError("global55 capture was not produced")
        return recorder, mapping
    finally:
        if base_env is not None and original_sim is not None:
            base_env.sim = original_sim
        envs.close()


def validate_paths(args: argparse.Namespace) -> dict[str, Path]:
    walker_dir = Path(args.walker_dir).resolve()
    paths = {
        "checkpoint": require_file(Path(args.checkpoint), "checkpoint"),
        "walker_dir": walker_dir,
        "morphology_xml": require_file(walker_dir / "xml" / f"{args.morphology_id}.xml", "morphology XML"),
        "morphology_metadata": require_file(walker_dir / "metadata" / f"{args.morphology_id}.json", "morphology metadata"),
        "config": require_file(REPO_ROOT / args.cfg, "config"),
        "existing_oracle": Path(args.existing_oracle).resolve(),
        "output_dir": Path(args.output_dir).resolve(),
    }
    if not paths["existing_oracle"].is_dir():
        raise FileNotFoundError(f"existing oracle is missing: {paths['existing_oracle']}")
    if paths["output_dir"].exists():
        raise FileExistsError(f"refusing to overwrite output: {paths['output_dir']}")
    if args.morphology_id != MORPHOLOGY:
        raise ValueError(f"formal oracle requires morphology {MORPHOLOGY}")
    if sha256(paths["checkpoint"]) != CHECKPOINT_SHA256:
        raise ValueError("checkpoint SHA256 mismatch")
    if sha256(paths["morphology_xml"]) != XML_SHA256:
        raise ValueError("morphology XML SHA256 mismatch")
    return paths


def run(args: argparse.Namespace, paths: dict[str, Path]) -> dict[str, Any]:
    output = paths["output_dir"]
    output.mkdir(parents=True)
    hashes_before = {
        "morphology_xml": sha256(paths["morphology_xml"]),
        "checkpoint": sha256(paths["checkpoint"]),
        "existing_oracle_validation": (
            sha256(paths["existing_oracle"] / "validation.json")
            if (paths["existing_oracle"] / "validation.json").is_file()
            else None
        ),
    }
    recorder, mapping = replay(args, paths)
    capture = recorder.global55
    assert capture is not None
    regression = regression_check(capture)
    budget = demand_budget(capture) if regression["MUJOCO_GLOBAL55_REGRESSION"] == "PASS" else None
    closures = velocity_closures(capture, capture["integration_matrix"])
    old_regression = (
        load_old_artifact_regression(paths["existing_oracle"], budget, capture)
        if budget is not None else {"OLD_ARTIFACT_REGRESSION": "FAIL"}
    )
    interpretation = (
        classifications(regression, capture, budget)
        if budget is not None
        else classifications(regression, capture, {})
    )
    hashes_after = {
        "morphology_xml": sha256(paths["morphology_xml"]),
        "checkpoint": sha256(paths["checkpoint"]),
        "existing_oracle_validation": (
            sha256(paths["existing_oracle"] / "validation.json")
            if (paths["existing_oracle"] / "validation.json").is_file()
            else None
        ),
    }
    source_unchanged = hashes_before == hashes_after
    record_count_valid = len(recorder.records) == EXPECTED_SUBSTEPS
    mass_valid = (
        capture["mass_matrix_stats"]["symmetry_max_abs_error"] <= 1.0e-10
        and capture["mass_matrix_stats"]["minimum_eigenvalue"] > 0.0
        and capture["mass_matrix_stats"]["linear_solve_check"]["residual_max_abs"] <= 1.0e-8
    )
    delassus_valid = capture["delassus_stats"]["symmetry_max_abs_error"] <= 1.0e-9
    dof_order_valid = bool(
        capture["generalized_dof_order"]["GENERALIZED_DOF_ORDER"] == "EXPLICIT"
        and capture["generalized_dof_order"]["GENERALIZED_DOF_COUNT"] == int(capture["nv"]) == 19
        and capture["generalized_dof_order"]["free_joint_dof_count"] == 6
        and capture["generalized_dof_order"]["scalar_joint_dof_count"] == 13
    )
    efc_rows_have_nv_columns = all(
        np.asarray(row["J_row"]).shape == (int(capture["nv"]),)
        for contact in capture["contacts"]
        for row in contact["solver_rows"]
    )
    jacobian_order_valid = bool(
        np.asarray(capture["J_phys"]).shape[1] == int(capture["nv"])
        and np.asarray(capture["mass_matrix"]).shape
        == (int(capture["nv"]), int(capture["nv"]))
        and efc_rows_have_nv_columns
    )
    integration_valid = (
        capture["integration_matrix_report"]["INTEGRATION_MATRIX_CONSTRUCTION"]
        == "VALIDATED"
    )
    generalized_closure_valid = (
        closures["generalized"]["GENERALIZED_VELOCITY_CLOSURE"] == "PASS"
    )
    constraint_closure_valid = (
        closures["constraint"]["CONTACT_CONSTRAINT_GENERALIZED_CLOSURE"] == "PASS"
    )
    physical_mapping_valid = (
        closures["physical"]["PHYSICAL_CONTACT_IMPULSE_MAPPING"] in ("PASS", "PARTIAL")
    )
    mujoco_version = __import__("mujoco").__version__
    identity_valid = bool(
        mujoco_version == "3.8.1"
        and hashes_before["morphology_xml"] == XML_SHA256
        and hashes_before["checkpoint"] == CHECKPOINT_SHA256
    )
    oracle_valid = bool(
        identity_valid
        and regression["MUJOCO_GLOBAL55_REGRESSION"] == "PASS"
        and old_regression["OLD_ARTIFACT_REGRESSION"] == "PASS"
        and record_count_valid
        and capture["point_velocity_mapping_valid"]
        and mass_valid
        and delassus_valid
        and dof_order_valid
        and jacobian_order_valid
        and integration_valid
        and generalized_closure_valid
        and constraint_closure_valid
        and physical_mapping_valid
        and source_unchanged
        and not capture["formal_data_mutated_by_probe"]
    )
    metadata = {
        "created_at": datetime.now().astimezone().isoformat(),
        "backend": "mujoco",
        "mujoco_version": mujoco_version,
        "morphology": MORPHOLOGY,
        "morphology_xml": str(paths["morphology_xml"]),
        "morphology_xml_sha256": hashes_before["morphology_xml"],
        "checkpoint": str(paths["checkpoint"]),
        "checkpoint_sha256": hashes_before["checkpoint"],
        "existing_reference_oracle": str(paths["existing_oracle"]),
        "action_mode": "zero",
        "reset_noise_scale": 0.0,
        "control_steps": CONTROL_STEPS,
        "physics_substeps": len(recorder.records),
        "physics_dt": float(
            capture["post_simulation_time"] - capture["pre_simulation_time"]
        ),
        "capture_phase": capture["capture_phase"],
        "capture_phase_detail": "pre generalized state cloned immediately before the sole live mj_step; one mj_forward runs only on the clone to produce internally consistent pre-integration qM/J/contact/efc/force readback; qvel_post comes from the sole formal live mj_step",
        "EXTRA_PHYSICS_STEPS": 0,
        "EXTRA_MJ_FORWARD_CALLS_ON_FORMAL_DATA": 0,
        "DIAGNOSTIC_CLONE_MJ_FORWARD_CALLS": capture["probe_evidence"]["probe_mj_forward_count"],
        "FORMAL_DATA_MUTATED_BY_PROBE": capture["formal_data_mutated_by_probe"],
        "physical_basis": "right-handed rows [normal force direction on robot, tangent1, tangent2]",
        "jacobian": "mujoco.mj_jac at exact mjContact.pos on robot body relative to static world",
        "generalized_column_order": GENERALIZED_COLUMN_ORDER,
        "generalized_column_order_file": GENERALIZED_COLUMN_ORDER_FILE,
        "JACOBIAN_COLUMN_ORDER": "EXPLICIT_MUJOCO_QVEL_ORDER",
        **PHYSICAL_JACOBIAN_CONTRACT,
        "CLOSURE_LINEARIZATION_CONFIGURATION": "PRE_INTEGRATION",
    }
    pre_state = {
        key: capture[key]
        for key in (
            "capture_phase", "control_step", "physics_substep_in_control",
            "global_physics_step", "pre_simulation_time", "nq", "nv", "ncon", "nefc",
        )
    }
    pre_state["integrator_input_state"] = capture["pre_state"]
    pre_state["same_solve_force_and_acceleration_readback"] = capture["solver_phase_state"]
    post_state = {
        "post_simulation_time": capture["post_simulation_time"],
        **capture["post_state"],
    }
    contacts = {
        "all_active_robot_floor_contacts": capture["contacts"],
        "count": len(capture["contacts"]),
    }
    order_reference = {
        "generalized_column_order": GENERALIZED_COLUMN_ORDER,
        "generalized_column_order_file": GENERALIZED_COLUMN_ORDER_FILE,
        "nv": int(capture["nv"]),
    }
    mass_report = {
        "mass_matrix": capture["mass_matrix"],
        "row_order": GENERALIZED_COLUMN_ORDER,
        "column_order": GENERALIZED_COLUMN_ORDER,
        **order_reference,
        **capture["mass_matrix_stats"],
    }
    jacobian_report = {
        "J_phys": capture["J_phys"],
        "contact_row_mapping": budget["contact_row_mapping"] if budget else None,
        "point_velocity_mapping_valid": capture["point_velocity_mapping_valid"],
        "per_contact": [
            {
                "contact_index": item["contact_index"],
                "robot_body_name": item["robot_body_name"],
                "point_world": item["point_world"],
                "physical_basis_world_rows": item["physical_basis_world_rows"],
                "J_point_world": item["J_point_world"],
                "J_physical": item["J_physical"],
                "pre_point_velocity": item["pre_point_velocity"],
                "post_point_velocity": item["post_point_velocity"],
            }
            for item in capture["contacts"]
        ],
        **order_reference,
        "JACOBIAN_COLUMN_ORDER": "EXPLICIT_MUJOCO_QVEL_ORDER",
        **PHYSICAL_JACOBIAN_CONTRACT,
        "physical_basis_rows": ["normal force direction on robot", "tangent1", "tangent2"],
        "physical_basis_frame": "world",
        "physical_basis_handedness": "right-handed",
    }
    delassus = {
        "W_phys": capture["W_phys"],
        **capture["delassus_stats"],
        "W_tt_global": budget["W_tt_global"] if budget else None,
        "W_tn_global": budget["W_tn_global"] if budget else None,
        "physical_row_order": "same stacked contact order documented in global55_physical_jacobians.json",
        **order_reference,
    }
    solver_rows = {
        "representation": "MuJoCo pyramidal EFC rows; distinct from physical 3D basis",
        **order_reference,
        "contacts": [
            {
                "contact_index": item["contact_index"],
                "robot_body_name": item["robot_body_name"],
                "efc_address": item["efc_address"],
                "efc_rows": item["efc_rows"],
                "solver_rows": item["solver_rows"],
            }
            for item in capture["contacts"]
        ],
    }
    comparison = {
        "mujoco": budget["selected"]["limb/12"] if budget else None,
        "isaac_fixed_reference": ISAAC_REFERENCE,
        "interpretation": interpretation,
        "warning": "Isaac apparent mass/demand values are not exact Delassus metrics",
    }
    if budget is not None:
        comparison.update(
            {
                "MUJOCO_GLOBAL55_LIMB12_PRE_TANGENTIAL_SPEED": budget["selected"]["limb/12"]["pre_tangential_speed"],
                "MUJOCO_GLOBAL55_LIMB11_PRE_TANGENTIAL_SPEED": budget["selected"]["limb/11"]["pre_tangential_speed"],
            }
        )
    validation = {
        "MUJOCO_GLOBAL55_ORACLE_VALID": oracle_valid,
        "MUJOCO_GLOBAL55_ARTIFACT_READBACK_VALID": "YES" if oracle_valid else "NO",
        "MUJOCO_GLOBAL55_REGRESSION": regression["MUJOCO_GLOBAL55_REGRESSION"],
        "IDENTITY": "PASS" if identity_valid else "FAIL",
        "OLD_ARTIFACT_REGRESSION": old_regression["OLD_ARTIFACT_REGRESSION"],
        "MUJOCO_GLOBAL55_FRICTION_REGIME": interpretation["MUJOCO_GLOBAL55_FRICTION_REGIME"],
        "MUJOCO_GLOBAL55_DEMAND_DECOMPOSITION": interpretation["MUJOCO_GLOBAL55_DEMAND_DECOMPOSITION"],
        "DOMINANT_FRICTION_IMPULSE_GAP_DRIVER": interpretation["DOMINANT_FRICTION_IMPULSE_GAP_DRIVER"],
        "GLOBAL55_SOLVER_CAPTURE_PHASE": capture["capture_phase"],
        "GENERALIZED_DOF_ORDER": capture["generalized_dof_order"]["GENERALIZED_DOF_ORDER"],
        "JACOBIAN_COLUMN_ORDER": "EXPLICIT_MUJOCO_QVEL_ORDER" if jacobian_order_valid else "INCOMPLETE",
        "PHYSICAL_JACOBIAN_CONVENTION": "WORLD_POINT_AT_CONTACT" if jacobian_order_valid else "INCOMPLETE",
        "CLOSURE_LINEARIZATION_CONFIGURATION": "PRE_INTEGRATION",
        "INTEGRATION_MATRIX_CONSTRUCTION": capture["integration_matrix_report"]["INTEGRATION_MATRIX_CONSTRUCTION"],
        "GENERALIZED_VELOCITY_CLOSURE": closures["generalized"]["GENERALIZED_VELOCITY_CLOSURE"],
        "CONTACT_CONSTRAINT_GENERALIZED_CLOSURE": closures["constraint"]["CONTACT_CONSTRAINT_GENERALIZED_CLOSURE"],
        "PHYSICAL_CONTACT_IMPULSE_MAPPING": closures["physical"]["PHYSICAL_CONTACT_IMPULSE_MAPPING"],
        "POINT_VELOCITY_MAPPING_VALID": capture["point_velocity_mapping_valid"],
        "M_J_EFC_SHARE_GENERALIZED_COLUMN_ORDER": jacobian_order_valid,
        "mass_matrix_valid": mass_valid,
        "delassus_matrix_valid": delassus_valid,
        "record_count": len(recorder.records),
        "expected_record_count": EXPECTED_SUBSTEPS,
        "record_count_valid": record_count_valid,
        "exactly_120_formal_substeps": record_count_valid,
        "extra_physics_steps": 0,
        "extra_live_mj_forward_calls": 0,
        "diagnostic_clone_mj_forward_calls": capture["probe_evidence"]["probe_mj_forward_count"],
        "diagnostic_clone_pre_state_matches_live": capture["probe_evidence"]["clone_pre_state_matches_live"],
        "live_data_unchanged_by_probe": capture["probe_evidence"]["live_data_unchanged_by_probe"],
        "formal_data_mutated_by_probe": capture["formal_data_mutated_by_probe"],
        "source_hashes_unchanged": source_unchanged,
        "hashes_before": hashes_before,
        "hashes_after": hashes_after,
        "all_numerical_outputs_finite": all_numeric_values_finite(
            {
                "capture": capture,
                "budget": budget,
                "interpretation": interpretation,
                "closures": closures,
            }
        ),
        "nonfinite_json_policy": "serialize non-finite diagnostics as null",
    }
    if budget is not None:
        validation.update(
            {
                "MUJOCO_GLOBAL55_LIMB12_PRE_TANGENTIAL_SPEED": budget["selected"]["limb/12"]["pre_tangential_speed"],
                "MUJOCO_GLOBAL55_LIMB11_PRE_TANGENTIAL_SPEED": budget["selected"]["limb/11"]["pre_tangential_speed"],
            }
        )
    for filename, payload in (
        ("metadata.json", metadata),
        ("global55_pre_state.json", pre_state),
        ("global55_post_state.json", post_state),
        ("global55_contacts.json", contacts),
        ("raw_mass_matrix.json", mass_report),
        ("global55_mass_matrix.json", mass_report),
        (GENERALIZED_COLUMN_ORDER_FILE, capture["generalized_dof_order"]),
        ("integrator_semantics.json", capture["integrator_semantics"]),
        ("integration_effective_mass_matrix.json", {
            "matrix": capture["integration_matrix"], **order_reference,
            **{key: value for key, value in capture["integration_matrix_report"].items() if key != "delta"},
        }),
        ("integration_matrix_delta.json", {
            "delta": capture["integration_matrix_report"].get("delta"), **order_reference,
            "formula": capture["integration_matrix_report"].get("formula"),
        }),
        ("global55_physical_jacobians.json", jacobian_report),
        ("global55_delassus_matrix.json", delassus),
        ("global55_solver_rows.json", solver_rows),
        ("global55_effective_mass_budget.json", budget),
        ("isaac_fixed_reference.json", ISAAC_REFERENCE),
        ("comparison.json", comparison),
        ("global55_regression.json", regression),
        ("old_artifact_regression.json", old_regression),
        ("generalized_velocity_closure.json", closures["generalized"]),
        ("contact_constraint_velocity_closure.json", closures["constraint"]),
        ("physical_contact_impulse_closure.json", closures["physical"]),
        ("source_purity.json", {
            "hashes_before": hashes_before,
            "hashes_after": hashes_after,
            "source_hashes_unchanged": source_unchanged,
            "formal_data_mutated_by_probe": capture["formal_data_mutated_by_probe"],
            "probe_evidence": capture["probe_evidence"],
        }),
        ("summary.json", {
            "MUJOCO_GLOBAL55_ARTIFACT_READBACK_VALID": "YES" if oracle_valid else "NO",
            "GLOBAL55_SOLVER_CAPTURE_PHASE": capture["capture_phase"],
            "GENERALIZED_VELOCITY_CLOSURE": closures["generalized"]["GENERALIZED_VELOCITY_CLOSURE"],
            "CONTACT_CONSTRAINT_GENERALIZED_CLOSURE": closures["constraint"]["CONTACT_CONSTRAINT_GENERALIZED_CLOSURE"],
            "PHYSICAL_CONTACT_IMPULSE_MAPPING": closures["physical"]["PHYSICAL_CONTACT_IMPULSE_MAPPING"],
        }),
        ("validation.json", validation),
    ):
        write_json(output / filename, payload)
    return validation


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    output = Path(args.output_dir).resolve()
    try:
        paths = validate_paths(args)
        validation = run(args, paths)
        print(json.dumps(json_ready({
            "output_dir": str(output),
            "MUJOCO_GLOBAL55_ORACLE_VALID": validation["MUJOCO_GLOBAL55_ORACLE_VALID"],
            "MUJOCO_GLOBAL55_ARTIFACT_READBACK_VALID": validation["MUJOCO_GLOBAL55_ARTIFACT_READBACK_VALID"],
            "MUJOCO_GLOBAL55_REGRESSION": validation["MUJOCO_GLOBAL55_REGRESSION"],
            "MUJOCO_GLOBAL55_FRICTION_REGIME": validation["MUJOCO_GLOBAL55_FRICTION_REGIME"],
            "MUJOCO_GLOBAL55_LIMB12_PRE_TANGENTIAL_SPEED": validation.get("MUJOCO_GLOBAL55_LIMB12_PRE_TANGENTIAL_SPEED"),
            "MUJOCO_GLOBAL55_LIMB11_PRE_TANGENTIAL_SPEED": validation.get("MUJOCO_GLOBAL55_LIMB11_PRE_TANGENTIAL_SPEED"),
            "MUJOCO_GLOBAL55_DEMAND_DECOMPOSITION": validation["MUJOCO_GLOBAL55_DEMAND_DECOMPOSITION"],
            "DOMINANT_FRICTION_IMPULSE_GAP_DRIVER": validation["DOMINANT_FRICTION_IMPULSE_GAP_DRIVER"],
        }), indent=2, allow_nan=False))
        return 0 if validation["MUJOCO_GLOBAL55_ORACLE_VALID"] else 2
    except Exception as error:
        output.mkdir(parents=True, exist_ok=True)
        trace = traceback.format_exc()
        print(trace, file=sys.stderr, end="")
        (output / "traceback.txt").write_text(trace, encoding="utf-8")
        failure = {
            "MUJOCO_GLOBAL55_ORACLE_VALID": False,
            "MUJOCO_GLOBAL55_ARTIFACT_READBACK_VALID": "NO",
            "MUJOCO_GLOBAL55_REGRESSION": "FAIL",
            "GENERALIZED_DOF_ORDER": "INSUFFICIENT_EVIDENCE",
            "JACOBIAN_COLUMN_ORDER": "INSUFFICIENT_EVIDENCE",
            "PHYSICAL_JACOBIAN_CONVENTION": "INSUFFICIENT_EVIDENCE",
            "INTEGRATION_MATRIX_CONSTRUCTION": "INSUFFICIENT_EVIDENCE",
            "GENERALIZED_VELOCITY_CLOSURE": "INSUFFICIENT_EVIDENCE",
            "CONTACT_CONSTRAINT_GENERALIZED_CLOSURE": "INSUFFICIENT_EVIDENCE",
            "PHYSICAL_CONTACT_IMPULSE_MAPPING": "INSUFFICIENT_EVIDENCE",
            "MUJOCO_GLOBAL55_FRICTION_REGIME": "INSUFFICIENT_EVIDENCE",
            "MUJOCO_GLOBAL55_DEMAND_DECOMPOSITION": "INSUFFICIENT_EVIDENCE",
            "DOMINANT_FRICTION_IMPULSE_GAP_DRIVER": "INSUFFICIENT_EVIDENCE",
            "error_type": type(error).__name__,
            "error": str(error),
        }
        write_json(output / "validation.json", failure)
        write_json(output / "summary.json", failure)
        write_json(output / "source_purity.json", {
            "status": "INCOMPLETE_DUE_TO_FAILURE",
            "formal_dynamics_mutation_requested": False,
            "traceback_file": "traceback.txt",
        })
        return 2


if __name__ == "__main__":
    np.bool = np.bool_
    raise SystemExit(main())

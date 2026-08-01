"""Extract a read-only MuJoCo global-55 contact-demand oracle.

The replay uses the formal evaluator environment/checkpoint restoration and its
existing one-callback-per-live-mj_step instrumentation.  The additional probe
only reads the solver data left by the formal step and performs NumPy linear
algebra; it never calls mj_step, mj_forward, or writes live force arrays.
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


def capture_global55(
    recorder: evaluator.JointLimitSubstepRecorder,
    pre: dict[str, Any],
) -> dict[str, Any]:
    mujoco, model, data = evaluator._native_model_data(recorder.sim)
    nv, nq, nefc = int(model.nv), int(model.nq), int(data.nefc)
    qfrc_applied_before = np.asarray(data.qfrc_applied, dtype=np.float64).copy()
    mass, mass_stats = expanded_mass_matrix(mujoco, model, data)
    formal_record = recorder.records[-1]
    floor_id = int(recorder.mapping["floor_geom_id"])
    floor_contacts = [
        contact
        for contact in formal_record["contacts"]
        if floor_id in (int(contact["geom1_id"]), int(contact["geom2_id"]))
    ]
    if not floor_contacts:
        raise RuntimeError("global55 contains no active robot-floor contacts")

    jacobian_blocks = []
    contact_records = []
    point_mapping_valid = True
    pyramidal = int(model.opt.cone) == int(mujoco.mjtCone.mjCONE_PYRAMIDAL)
    pre_qvel = np.asarray(pre["full_qvel"], dtype=np.float64)
    post_qvel = np.asarray(data.qvel, dtype=np.float64)
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
    return {
        "capture_phase": "AFTER_MJ_STEP_WITH_SOLVER_DATA_STILL_VALID",
        "control_step": int(formal_record["control_step"]),
        "physics_substep_in_control": int(formal_record["physics_substep_in_control"]),
        "global_physics_step": int(formal_record["global_physics_step"]),
        "pre_simulation_time": float(pre["simulation_time"]),
        "post_simulation_time": float(data.time),
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
            "qfrc_applied": qfrc_applied_after,
            "qfrc_actuator": np.asarray(data.qfrc_actuator, dtype=np.float64).copy(),
            "qfrc_passive": np.asarray(data.qfrc_passive, dtype=np.float64).copy(),
            "qfrc_bias": np.asarray(data.qfrc_bias, dtype=np.float64).copy(),
            "qfrc_constraint": np.asarray(data.qfrc_constraint, dtype=np.float64).copy(),
        },
        "post_state": {
            "qpos": np.asarray(data.qpos, dtype=np.float64).copy(),
            "qvel": post_qvel,
        },
        "mass_matrix": mass,
        "mass_matrix_stats": mass_stats,
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
        "formal_data_mutated_by_probe": formal_data_mutated,
        "formal_record": formal_record,
    }


class DemandRecorder(evaluator.JointLimitSubstepRecorder):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.global55: dict[str, Any] | None = None

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
    mass = np.asarray(capture["mass_matrix"], dtype=np.float64)
    j_phys = np.asarray(capture["J_phys"], dtype=np.float64)
    w_phys = np.asarray(capture["W_phys"], dtype=np.float64)
    qvel_pre = np.asarray(capture["pre_state"]["qvel"], dtype=np.float64)
    qvel_post = np.asarray(capture["post_state"]["qvel"], dtype=np.float64)
    contacts = capture["contacts"]
    count = len(contacts)
    normal_rows = [3 * index for index in range(count)]
    tangent_rows = [row for index in range(count) for row in (3 * index + 1, 3 * index + 2)]
    p_normal = np.asarray([contact["normal_impulse"] for contact in contacts])
    p_tangent = np.concatenate(
        [np.asarray(contact["tangential_impulse"], dtype=np.float64) for contact in contacts]
    )
    v_pre = j_phys @ qvel_pre
    v_post = j_phys @ qvel_post
    v_t_pre_all = v_pre[tangent_rows]
    delta_v_t_normals_all = w_phys[np.ix_(tangent_rows, normal_rows)] @ p_normal
    v_t_after_normals_all = v_t_pre_all + delta_v_t_normals_all
    w_tt_global = w_phys[np.ix_(tangent_rows, tangent_rows)]
    p_t_stick_global, global_stick_solve = stable_solve(
        w_tt_global, -v_t_after_normals_all
    )
    actual_impulse = np.zeros(3 * count, dtype=np.float64)
    actual_impulse[normal_rows] = p_normal
    actual_impulse[tangent_rows] = p_tangent
    predicted_contact_delta = w_phys @ actual_impulse
    observed_delta = v_post - v_pre
    contact_only_residual = observed_delta - predicted_contact_delta

    dt = float(capture["post_simulation_time"] - capture["pre_simulation_time"])
    qacc_smooth = np.asarray(capture["solver_phase_state"]["qacc_smooth"])
    qfrc_constraint = np.asarray(capture["solver_phase_state"]["qfrc_constraint"])
    constraint_acceleration, constraint_solve = stable_solve(mass, qfrc_constraint)
    full_predicted_delta = j_phys @ (dt * (qacc_smooth + constraint_acceleration))
    full_residual = observed_delta - full_predicted_delta
    contact_scale = max(1.0, float(np.linalg.norm(observed_delta)))
    contact_only_relative = float(np.linalg.norm(contact_only_residual) / contact_scale)
    full_relative = float(np.linalg.norm(full_residual) / contact_scale)
    if contact_only_relative <= 1.0e-6:
        closure = "PASS"
    elif full_relative <= 1.0e-6:
        closure = "RESIDUAL_EXPLAINED_BY_NONCONTACT_AND_REGULARIZATION"
    else:
        closure = "FAIL"

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
        "post_solve_closure": {
            "observed_delta_v_contact": observed_delta,
            "predicted_delta_v_from_floor_contact_impulses": predicted_contact_delta,
            "contact_only_residual": contact_only_residual,
            "contact_only_relative_l2": contact_only_relative,
            "predicted_delta_v_from_qacc_smooth_and_all_constraints": full_predicted_delta,
            "all_dynamics_residual": full_residual,
            "all_dynamics_relative_l2": full_relative,
            "constraint_mass_solve": constraint_solve,
            "CONTACT_IMPULSE_VELOCITY_CLOSURE": closure,
        },
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
    closure = budget["post_solve_closure"]["CONTACT_IMPULSE_VELOCITY_CLOSURE"]
    decomposition = "VALIDATED" if closure != "FAIL" and regime != "INSUFFICIENT_EVIDENCE" else "PARTIAL"
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
    oracle_valid = bool(
        regression["MUJOCO_GLOBAL55_REGRESSION"] == "PASS"
        and record_count_valid
        and capture["point_velocity_mapping_valid"]
        and mass_valid
        and delassus_valid
        and source_unchanged
        and not capture["formal_data_mutated_by_probe"]
    )
    metadata = {
        "created_at": datetime.now().astimezone().isoformat(),
        "backend": "mujoco",
        "mujoco_version": __import__("mujoco").__version__,
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
        "capture_phase_detail": "pre generalized state saved immediately before the sole live mj_step; contact/efc/qM and solver force arrays read immediately after that same mj_step before any next simulation call",
        "EXTRA_PHYSICS_STEPS": 0,
        "EXTRA_MJ_FORWARD_CALLS_ON_FORMAL_DATA": 0,
        "FORMAL_DATA_MUTATED_BY_PROBE": capture["formal_data_mutated_by_probe"],
        "physical_basis": "right-handed rows [normal force direction on robot, tangent1, tangent2]",
        "jacobian": "mujoco.mj_jac at exact mjContact.pos on robot body relative to static world",
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
    mass_report = {"mass_matrix": capture["mass_matrix"], **capture["mass_matrix_stats"]}
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
    }
    delassus = {
        "W_phys": capture["W_phys"],
        **capture["delassus_stats"],
        "W_tt_global": budget["W_tt_global"] if budget else None,
        "W_tn_global": budget["W_tn_global"] if budget else None,
    }
    solver_rows = {
        "representation": "MuJoCo pyramidal EFC rows; distinct from physical 3D basis",
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
        "MUJOCO_GLOBAL55_REGRESSION": regression["MUJOCO_GLOBAL55_REGRESSION"],
        "MUJOCO_GLOBAL55_FRICTION_REGIME": interpretation["MUJOCO_GLOBAL55_FRICTION_REGIME"],
        "MUJOCO_GLOBAL55_DEMAND_DECOMPOSITION": interpretation["MUJOCO_GLOBAL55_DEMAND_DECOMPOSITION"],
        "DOMINANT_FRICTION_IMPULSE_GAP_DRIVER": interpretation["DOMINANT_FRICTION_IMPULSE_GAP_DRIVER"],
        "GLOBAL55_SOLVER_CAPTURE_PHASE": capture["capture_phase"],
        "POINT_VELOCITY_MAPPING_VALID": capture["point_velocity_mapping_valid"],
        "mass_matrix_valid": mass_valid,
        "delassus_matrix_valid": delassus_valid,
        "record_count": len(recorder.records),
        "expected_record_count": EXPECTED_SUBSTEPS,
        "record_count_valid": record_count_valid,
        "exactly_120_formal_substeps": record_count_valid,
        "extra_physics_steps": 0,
        "extra_live_mj_forward_calls": 0,
        "formal_data_mutated_by_probe": capture["formal_data_mutated_by_probe"],
        "source_hashes_unchanged": source_unchanged,
        "hashes_before": hashes_before,
        "hashes_after": hashes_after,
        "all_numerical_outputs_finite": all_numeric_values_finite(
            {
                "capture": capture,
                "budget": budget,
                "interpretation": interpretation,
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
        ("global55_mass_matrix.json", mass_report),
        ("global55_physical_jacobians.json", jacobian_report),
        ("global55_delassus_matrix.json", delassus),
        ("global55_solver_rows.json", solver_rows),
        ("global55_effective_mass_budget.json", budget),
        ("isaac_fixed_reference.json", ISAAC_REFERENCE),
        ("comparison.json", comparison),
        ("global55_regression.json", regression),
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
        traceback.print_exc()
        write_json(output / "validation.json", {
            "MUJOCO_GLOBAL55_ORACLE_VALID": False,
            "MUJOCO_GLOBAL55_REGRESSION": "FAIL",
            "MUJOCO_GLOBAL55_FRICTION_REGIME": "INSUFFICIENT_EVIDENCE",
            "MUJOCO_GLOBAL55_DEMAND_DECOMPOSITION": "INSUFFICIENT_EVIDENCE",
            "DOMINANT_FRICTION_IMPULSE_GAP_DRIVER": "INSUFFICIENT_EVIDENCE",
            "error_type": type(error).__name__,
            "error": str(error),
        })
        return 2


if __name__ == "__main__":
    np.bool = bool
    raise SystemExit(main())

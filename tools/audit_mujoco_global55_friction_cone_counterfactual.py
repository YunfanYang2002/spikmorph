"""Fixed-global55 MuJoCo pyramidal-versus-elliptic cone counterfactual.

The formal environment is replayed once through the already validated global55
oracle helper.  Immediately before formal physics step 55, a complete mjData
snapshot is copied.  Three isolated solver/step clone pairs are then evaluated
from that identical snapshot; the live replay data is never stepped by this
tool outside the original 120-substep replay.
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
from unittest import mock

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools import analyze_mujoco_global55_contact_demand as oracle


MORPHOLOGY = oracle.MORPHOLOGY
XML_SHA256 = oracle.XML_SHA256
CHECKPOINT_SHA256 = oracle.CHECKPOINT_SHA256
GLOBAL_STEP = oracle.GLOBAL_STEP
CONTROL_STEPS = oracle.CONTROL_STEPS
EXPECTED_SUBSTEPS = oracle.EXPECTED_SUBSTEPS
REFERENCE_ORACLE_NAME = "mujoco_global55_contact_demand_oracle_corrected_20260804_143138"
CONDITIONS = (
    ("pyramidal_before", "PYRAMIDAL_BEFORE", "pyramidal"),
    ("elliptic", "ELLIPTIC", "elliptic"),
    ("pyramidal_after_restore", "PYRAMIDAL_AFTER_RESTORE", "pyramidal"),
)
STATE_COPY_FIELDS = (
    "qpos", "qvel", "act", "ctrl", "mocap_pos", "mocap_quat",
    "userdata", "qacc_warmstart", "qfrc_applied", "xfrc_applied",
)
REGRESSION_RTOL = 1.0e-9
REGRESSION_ATOL = 1.0e-9


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
        description="Run one fixed-global55 MuJoCo friction-cone counterfactual."
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


def resolve_arguments(args: argparse.Namespace) -> argparse.Namespace:
    stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    if args.formal_server_defaults:
        batch = REPO_ROOT / "output/diagnostics/mujoco_control_51k_20260727_091638"
        manifest = json.loads((batch / "manifest.json").read_text(encoding="utf-8"))
        args.checkpoint = str(
            batch / "jobs/job_000_seed1409_lr0p00015/Unimal-v0.pt"
        )
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
    if not args.output_dir:
        args.output_dir = str(
            REPO_ROOT / "output/diagnostics"
            / f"mujoco_global55_friction_cone_counterfactual_{stamp}"
        )
    if not args.zip_path:
        args.zip_path = str(
            REPO_ROOT / "tmp"
            / f"mujoco_global55_friction_cone_counterfactual_{stamp}.zip"
        )
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


def state_snapshots_equal(left: dict[str, Any], right: dict[str, Any]) -> bool:
    if left.keys() != right.keys():
        return False
    return all(
        np.array_equal(left[name], right[name])
        if isinstance(left[name], np.ndarray)
        else left[name] == right[name]
        for name in left
    )


def state_equality(reference: Any, candidate: Any) -> dict[str, Any]:
    left = state_input_snapshot(reference)
    right = state_input_snapshot(candidate)
    fields: dict[str, Any] = {}
    valid = True
    for name in left:
        if left[name] is None or right[name] is None:
            equal = left[name] is right[name]
            max_abs = None
        elif isinstance(left[name], np.ndarray):
            equal = bool(np.array_equal(left[name], right[name]))
            difference = np.asarray(left[name], dtype=np.float64) - np.asarray(
                right[name], dtype=np.float64
            )
            max_abs = float(np.max(np.abs(difference))) if difference.size else 0.0
        else:
            equal = left[name] == right[name]
            max_abs = abs(float(left[name]) - float(right[name]))
        fields[name] = {"equal": equal, "max_abs_mismatch": max_abs}
        valid &= equal
    return {"STATE_COPY_EQUAL": valid, "fields": fields}


class Global55SnapshotRecorder(oracle.evaluator.JointLimitSubstepRecorder):
    """Capture one complete pre-step snapshot without forwarding or stepping it."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.global55: dict[str, Any] | None = None
        self.global55_snapshot: Any | None = None
        self.snapshot_copy_evidence: dict[str, Any] | None = None
        self.raw_mujoco: Any | None = None
        self.raw_model: Any | None = None

    def capture_pre_step(self) -> dict[str, Any]:
        pre = super().capture_pre_step()
        if self.global_physics_step + 1 == GLOBAL_STEP:
            mujoco, model, live_data = oracle.evaluator._native_model_data(self.sim)
            before = state_input_snapshot(live_data)
            snapshot = mujoco.MjData(model)
            mujoco.mj_copyData(snapshot, model, live_data)
            after = state_input_snapshot(live_data)
            self.raw_mujoco = mujoco
            self.raw_model = model
            self.global55_snapshot = snapshot
            self.snapshot_copy_evidence = {
                "snapshot_equals_live": state_equality(live_data, snapshot),
                "live_unchanged_by_copy": state_snapshots_equal(before, after),
                "extra_formal_steps": 0,
                "extra_formal_forwards": 0,
            }
        return pre

    def capture_post_step(self, pre: dict[str, Any]) -> None:
        super().capture_post_step(pre)
        if self.global_physics_step == GLOBAL_STEP:
            self.global55 = {
                "snapshot_captured": self.global55_snapshot is not None,
                "control_step": int(self.records[-1]["control_step"]),
                "physics_substep_in_control": int(
                    self.records[-1]["physics_substep_in_control"]
                ),
                "global_physics_step": int(self.records[-1]["global_physics_step"]),
            }


def replay_once(
    args: argparse.Namespace, paths: dict[str, Path]
) -> tuple[Global55SnapshotRecorder, dict[str, Any]]:
    replay_paths = {
        **paths,
        "existing_oracle": paths["corrected_oracle"],
    }
    with mock.patch.object(oracle, "DemandRecorder", Global55SnapshotRecorder):
        recorder, mapping = oracle.replay(args, replay_paths)
    if recorder.global55_snapshot is None:
        raise RuntimeError("global55 pre-state snapshot was not captured")
    return recorder, mapping


def _optional_array(model: Any, name: str) -> np.ndarray | None:
    value = getattr(model, name, None)
    return np.asarray(value).copy() if value is not None else None


def model_option_snapshot(model: Any) -> dict[str, Any]:
    option_fields = (
        "cone", "integrator", "solver", "iterations", "ls_iterations",
        "tolerance", "timestep", "disableflags",
    )
    return {
        **{
            f"opt.{name}": (
                int(getattr(model.opt, name))
                if name in ("cone", "integrator", "solver", "iterations", "ls_iterations", "disableflags")
                else float(getattr(model.opt, name))
            )
            for name in option_fields
        },
        **{
            name: _optional_array(model, name)
            for name in (
                "geom_friction", "pair_friction", "geom_solref", "geom_solimp",
                "pair_solref", "pair_solimp", "jnt_solref", "jnt_solimp",
                "dof_damping",
            )
        },
    }


def model_option_difference(
    reference: dict[str, Any], candidate: dict[str, Any]
) -> dict[str, Any]:
    changed = []
    details = {}
    for name in reference:
        left, right = reference[name], candidate[name]
        if isinstance(left, np.ndarray):
            equal = isinstance(right, np.ndarray) and np.array_equal(left, right)
        else:
            equal = left == right
        details[name] = {"equal": bool(equal)}
        if not equal:
            changed.append(name)
    return {"changed_fields": changed, "only_changed_field": changed[0] if len(changed) == 1 else None, "details": details}


def _fake_recorder(
    model: Any,
    step_data: Any,
    mapping: dict[str, Any],
) -> Any:
    fake_sim = SimpleNamespace(
        _sim=SimpleNamespace(_model=model, _data=step_data),
        model=model,
        data=step_data,
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


def run_condition(
    mujoco: Any,
    model: Any,
    snapshot: Any,
    mapping: dict[str, Any],
    condition_name: str,
    condition_label: str,
    cone_value: int,
) -> dict[str, Any]:
    model.opt.cone = int(cone_value)
    options_before = model_option_snapshot(model)

    solver_data = mujoco.MjData(model)
    mujoco.mj_copyData(solver_data, model, snapshot)
    solver_copy = state_equality(snapshot, solver_data)
    snapshot_before_forward = state_input_snapshot(snapshot)
    mujoco.mj_forward(model, solver_data)
    snapshot_after_forward = state_input_snapshot(snapshot)

    step_data = mujoco.MjData(model)
    mujoco.mj_copyData(step_data, model, snapshot)
    step_copy = state_equality(snapshot, step_data)
    snapshot_before_step = state_input_snapshot(snapshot)
    mujoco.mj_step(model, step_data)
    snapshot_after_step = state_input_snapshot(snapshot)

    probe_evidence = {
        "copy_api": "mujoco.mj_copyData",
        "probe_forward_api": "mujoco.mj_forward",
        "probe_mj_forward_count": 1,
        "probe_mj_step_count": 1,
        "extra_live_mj_forward_count": 0,
        "extra_live_mj_step_count": 0,
        "clone_pre_state_matches_live": solver_copy["STATE_COPY_EQUAL"],
        "live_data_unchanged_by_probe": all(
            np.array_equal(snapshot_before_forward[name], snapshot_after_forward[name])
            if isinstance(snapshot_before_forward[name], np.ndarray)
            else snapshot_before_forward[name] == snapshot_after_forward[name]
            for name in snapshot_before_forward
        ) and all(
            np.array_equal(snapshot_before_step[name], snapshot_after_step[name])
            if isinstance(snapshot_before_step[name], np.ndarray)
            else snapshot_before_step[name] == snapshot_after_step[name]
            for name in snapshot_before_step
        ),
    }
    pre = {
        "simulation_time": float(snapshot.time),
        "full_qpos": np.asarray(snapshot.qpos, dtype=np.float64).copy(),
        "full_qvel": np.asarray(snapshot.qvel, dtype=np.float64).copy(),
        "_global55_probe_data": solver_data,
        "_global55_probe_evidence": probe_evidence,
    }
    capture = oracle.capture_global55(
        _fake_recorder(model, step_data, mapping), pre
    )
    options_after = model_option_snapshot(model)
    budget = oracle.demand_budget(capture)
    target_contact = oracle.selected_contact(capture, "limb/12")
    target_budget = budget["selected"]["limb/12"]
    excess = compute_solver_excess(target_contact, target_budget)

    contact_parameterization = [
        {
            "contact_index": item["contact_index"],
            "pair": [item["geom1_name"], item["geom2_name"]],
            "robot_body_name": item["robot_body_name"],
            "dim": item["dim"],
            "efc_address": item["efc_address"],
            "efc_rows": item["efc_rows"],
            "row_count": len(item["efc_rows"]),
            "efc_types": [row["efc_type"] for row in item["solver_rows"]],
            "efc_ids": [row["efc_id"] for row in item["solver_rows"]],
        }
        for item in capture["contacts"]
    ]
    physical_impulses = {
        "api": "mujoco.mj_contactForce",
        "parameterization_independent_readback": True,
        "contacts": [
            {
                "contact_index": item["contact_index"],
                "pair": [item["geom1_name"], item["geom2_name"]],
                "robot_body_name": item["robot_body_name"],
                "physical_basis_world_rows": item["physical_basis_world_rows"],
                "normal_impulse": item["normal_impulse"],
                "tangent_impulse": item["tangential_impulse"],
                "tangent_impulse_norm": item["tangential_impulse_norm"],
                "contact_force_contact_frame": item["formal_physical_projection"]["force_contact_frame"],
            }
            for item in capture["contacts"]
        ],
    }
    return {
        "condition_name": condition_name,
        "condition_label": condition_label,
        "cone_numeric": int(model.opt.cone),
        "options_before": options_before,
        "options_after": options_after,
        "options_changed_during_condition": model_option_difference(
            options_before, options_after
        ),
        "state_validation": {
            "solver_clone": solver_copy,
            "step_clone": step_copy,
            "snapshot_unchanged_by_solver_and_step_clones": probe_evidence["live_data_unchanged_by_probe"],
        },
        "capture": capture,
        "budget": budget,
        "excess": excess,
        "contact_parameterization": contact_parameterization,
        "physical_impulses": physical_impulses,
        "counts": {"probe_mj_forward": 1, "probe_mj_step": 1},
    }


def compute_solver_excess(
    target_contact: dict[str, Any], target_budget: dict[str, Any]
) -> dict[str, Any]:
    actual = np.asarray(target_budget["actual_tangential_impulse"], dtype=np.float64)
    rigid = np.asarray(
        target_budget["global_normal_conditioned_sticking_impulse"], dtype=np.float64
    )
    residual = actual - rigid
    denominator = float(np.linalg.norm(actual) * np.linalg.norm(rigid))
    cosine = float(np.dot(actual, rigid) / denominator) if denominator else None
    angle = (
        float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))
        if cosine is not None else None
    )
    return {
        "target": "limb/12-floor in limby/12-aligned shared physical basis",
        "actual_tangent_impulse": actual,
        "actual_tangent_impulse_norm": float(np.linalg.norm(actual)),
        "rigid_demand_impulse": rigid,
        "rigid_demand_impulse_norm": float(np.linalg.norm(rigid)),
        "solver_excess_norm": float(np.linalg.norm(actual) - np.linalg.norm(rigid)),
        "solver_excess_vector": residual,
        "solver_excess_vector_norm": float(np.linalg.norm(residual)),
        "actual_rigid_cosine_similarity": cosine,
        "actual_rigid_angle_degrees": angle,
        "normal_impulse": float(target_budget["actual_normal_impulse"]),
        "friction_cap": float(target_budget["friction_cap_mu_pn"]),
        "friction_cap_utilisation": float(
            np.linalg.norm(actual) / target_budget["friction_cap_mu_pn"]
        ),
        "pre_slip": np.asarray(target_contact["pre_tangential_velocity"]),
        "post_slip": np.asarray(target_contact["post_tangential_velocity"]),
    }


def _pair_key(contact: dict[str, Any]) -> tuple[str, str]:
    return tuple(sorted((contact["geom1_name"], contact["geom2_name"])))


def _allclose(left: Any, right: Any, rtol: float = REGRESSION_RTOL, atol: float = REGRESSION_ATOL) -> bool:
    return bool(np.allclose(np.asarray(left), np.asarray(right), rtol=rtol, atol=atol))


def load_reference(path: Path) -> dict[str, Any]:
    return {
        name: json.loads((path / filename).read_text(encoding="utf-8"))
        for name, filename in (
            ("contacts", "global55_contacts.json"),
            ("budget", "global55_effective_mass_budget.json"),
            ("mass", "raw_mass_matrix.json"),
            ("jacobian", "global55_physical_jacobians.json"),
            ("delassus", "global55_delassus_matrix.json"),
            ("solver_rows", "global55_solver_rows.json"),
            ("validation", "validation.json"),
        )
    }


def baseline_regression(
    condition: dict[str, Any], reference: dict[str, Any]
) -> dict[str, Any]:
    capture, budget = condition["capture"], condition["budget"]
    current_contacts = {_pair_key(item): item for item in capture["contacts"]}
    old_contacts = {
        _pair_key(item): item
        for item in reference["contacts"]["all_active_robot_floor_contacts"]
    }
    checks: dict[str, bool] = {
        "contact_set": set(current_contacts) == set(old_contacts),
        "mass_matrix": _allclose(capture["mass_matrix"], reference["mass"]["mass_matrix"]),
        "physical_jacobian": _allclose(capture["J_phys"], reference["jacobian"]["J_phys"]),
        "physical_delassus": _allclose(capture["W_phys"], reference["delassus"]["W_phys"]),
    }
    for key in sorted(set(current_contacts) & set(old_contacts)):
        current, old = current_contacts[key], old_contacts[key]
        label = "__".join(key)
        checks[f"{label}.point"] = _allclose(current["point_world"], old["point_world"])
        checks[f"{label}.basis"] = _allclose(current["physical_basis_world_rows"], old["physical_basis_world_rows"])
        checks[f"{label}.pre_slip"] = _allclose(current["pre_tangential_velocity"], old["pre_tangential_velocity"])
        checks[f"{label}.normal_impulse"] = _allclose(current["normal_impulse"], old["normal_impulse"])
        checks[f"{label}.tangent_impulse"] = _allclose(current["tangential_impulse"], old["tangential_impulse"])
        checks[f"{label}.post_slip"] = _allclose(current["post_tangential_velocity"], old["post_tangential_velocity"])
        checks[f"{label}.row_forces"] = _allclose(
            [row["efc_force"] for row in current["solver_rows"]],
            [row["efc_force"] for row in old["solver_rows"]],
        )
    current_target = budget["selected"]["limb/12"]
    old_target = reference["budget"]["selected"]["limb/12"]
    for name in (
        "actual_tangential_impulse", "actual_tangential_impulse_norm",
        "actual_normal_impulse", "global_normal_conditioned_sticking_impulse",
        "global_normal_conditioned_sticking_impulse_norm", "pre_tangential_speed",
    ):
        checks[f"limb12.{name}"] = _allclose(current_target[name], old_target[name])
    checks["sanity.actual_tangent_vector"] = _allclose(
        current_target["actual_tangential_impulse"],
        [0.2817184097958504, -3.312779286570642],
    )
    checks["sanity.actual_tangent_norm"] = _allclose(
        current_target["actual_tangential_impulse_norm"], 3.3247363600666735
    )
    checks["sanity.actual_normal_impulse"] = _allclose(
        current_target["actual_normal_impulse"], 6.345240278967453
    )
    checks["sanity.shared_rigid_demand"] = _allclose(
        current_target["global_normal_conditioned_sticking_impulse_norm"],
        2.540619084288334,
    )
    checks["sanity.solver_excess"] = _allclose(
        condition["excess"]["solver_excess_norm"], 0.7841172757783395
    )
    return {
        "tolerance_rtol": REGRESSION_RTOL,
        "tolerance_atol": REGRESSION_ATOL,
        "checks": checks,
        "PYRAMIDAL_BASELINE_REPRODUCTION": "PASS" if all(checks.values()) else "FAIL",
    }


def restore_regression(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    left_capture, right_capture = before["capture"], after["capture"]
    left_contacts = {_pair_key(item): item for item in left_capture["contacts"]}
    right_contacts = {_pair_key(item): item for item in right_capture["contacts"]}
    checks = {
        "mass_matrix": _allclose(left_capture["mass_matrix"], right_capture["mass_matrix"]),
        "physical_jacobian": _allclose(left_capture["J_phys"], right_capture["J_phys"]),
        "physical_delassus": _allclose(left_capture["W_phys"], right_capture["W_phys"]),
        "contact_set": set(left_contacts) == set(right_contacts),
        "contact_points": set(left_contacts) == set(right_contacts) and all(
            _allclose(left_contacts[key]["point_world"], right_contacts[key]["point_world"])
            for key in left_contacts
        ),
        "contact_bases": set(left_contacts) == set(right_contacts) and all(
            _allclose(
                left_contacts[key]["physical_basis_world_rows"],
                right_contacts[key]["physical_basis_world_rows"],
            )
            for key in left_contacts
        ),
        "normal_impulse_vector": _allclose(
            [item["normal_impulse"] for item in left_capture["contacts"]],
            [item["normal_impulse"] for item in right_capture["contacts"]],
        ),
        "target_tangent_impulse": _allclose(
            before["excess"]["actual_tangent_impulse"],
            after["excess"]["actual_tangent_impulse"],
        ),
        "target_rigid_demand": _allclose(
            before["excess"]["rigid_demand_impulse"],
            after["excess"]["rigid_demand_impulse"],
        ),
        "solver_excess": _allclose(
            before["excess"]["solver_excess_norm"],
            after["excess"]["solver_excess_norm"],
        ),
        "post_slip": _allclose(before["excess"]["post_slip"], after["excess"]["post_slip"]),
        "constraint_rows": set(left_contacts) == set(right_contacts) and all(
            _allclose(
                [
                    (row["efc_type"], row["efc_id"], row["efc_force"])
                    for row in left_contacts[key]["solver_rows"]
                ],
                [
                    (row["efc_type"], row["efc_id"], row["efc_force"])
                    for row in right_contacts[key]["solver_rows"]
                ],
            )
            for key in left_contacts
        ),
    }
    return {
        "checks": checks,
        "PYRAMIDAL_RESTORE_REPRODUCTION": "PASS" if all(checks.values()) else "FAIL",
    }


def constraint_activation(
    conditions: dict[str, dict[str, Any]], pyramidal_value: int, elliptic_value: int
) -> dict[str, Any]:
    target_names = {"limb/11", "limb/12"}
    details = {}
    valid = True
    for name, _, kind in CONDITIONS:
        condition = conditions[name]
        targets = [
            item for item in condition["contact_parameterization"]
            if item["robot_body_name"] in target_names
        ]
        expected_rows = 4 if kind == "pyramidal" else 3
        expected_cone = pyramidal_value if kind == "pyramidal" else elliptic_value
        condition_valid = bool(
            len(targets) == 2
            and all(item["row_count"] == expected_rows for item in targets)
            and condition["cone_numeric"] == expected_cone
        )
        details[name] = {
            "cone_numeric": condition["cone_numeric"],
            "expected_cone_numeric": expected_cone,
            "contact_count": int(condition["capture"]["ncon"]),
            "nefc": int(condition["capture"]["nefc"]),
            "expected_target_row_count": expected_rows,
            "target_contacts": targets,
            "target_constraint_types": sorted({
                value for item in targets for value in item["efc_types"]
            }),
            "target_contact_dimensions": [item["dim"] for item in targets],
            "valid": condition_valid,
        }
        valid &= condition_valid
    actual_pyramidal_counts = [
        item["row_count"]
        for item in details["pyramidal_before"]["target_contacts"]
    ]
    actual_elliptic_counts = [
        item["row_count"] for item in details["elliptic"]["target_contacts"]
    ]
    return {
        "conditions": details,
        "representation_changed": bool(
            actual_pyramidal_counts
            and actual_elliptic_counts
            and actual_pyramidal_counts != actual_elliptic_counts
        ),
        "CONE_COUNTERFACTUAL_ACTIVATION": "VALIDATED" if valid else "NOT_ACTIVATED",
    }


def classify_effect(
    baseline: dict[str, Any],
    elliptic: dict[str, Any],
    gates_valid: bool,
    noncanonical: bool = False,
) -> dict[str, Any]:
    if noncanonical:
        return {
            "FRICTION_CONE_SOLVER_EXCESS_EFFECT": "NONCANONICAL",
            "MUJOCO_SOLVER_EXCESS_CAUSAL_DRIVER": "NONCANONICAL",
            "NEXT_ACTION": "COUNTERFACTUAL_IMPLEMENTATION_FIX_REQUIRED",
        }
    if not gates_valid:
        return {
            "FRICTION_CONE_SOLVER_EXCESS_EFFECT": "INSUFFICIENT_EVIDENCE",
            "MUJOCO_SOLVER_EXCESS_CAUSAL_DRIVER": "INSUFFICIENT_EVIDENCE",
            "NEXT_ACTION": "COUNTERFACTUAL_IMPLEMENTATION_FIX_REQUIRED",
        }
    baseline_norm = float(baseline["solver_excess_norm"])
    elliptic_norm = float(elliptic["solver_excess_norm"])
    baseline_vector = float(baseline["solver_excess_vector_norm"])
    elliptic_vector = float(elliptic["solver_excess_vector_norm"])
    norm_reduction = 1.0 - abs(elliptic_norm) / abs(baseline_norm)
    vector_reduction = 1.0 - abs(elliptic_vector) / abs(baseline_vector)
    if norm_reduction >= 0.65:
        effect = "STRONG_REDUCTION"
        driver = "PYRAMIDAL_CONE_PARAMETERIZATION_DOMINANT"
        next_action = "NO_ADDITIONAL_SOLVER_COUNTERFACTUAL_REQUIRED"
    elif norm_reduction >= 0.25:
        effect = "PARTIAL_REDUCTION"
        driver = "PYRAMIDAL_CONE_PARAMETERIZATION_CONTRIBUTING"
        next_action = "FRICTION_AREF_REGULARIZATION_COUNTERFACTUAL"
    elif norm_reduction >= -0.10:
        effect = "LITTLE_OR_NO_REDUCTION"
        driver = "PYRAMIDAL_CONE_PARAMETERIZATION_NOT_DOMINANT"
        next_action = "FRICTION_AREF_REGULARIZATION_COUNTERFACTUAL"
    else:
        effect = "INCREASED"
        driver = "PYRAMIDAL_CONE_PARAMETERIZATION_NOT_DOMINANT"
        next_action = "FRICTION_AREF_REGULARIZATION_COUNTERFACTUAL"
    return {
        "baseline_excess": baseline_norm,
        "elliptic_excess": elliptic_norm,
        "absolute_excess_reduction": baseline_norm - elliptic_norm,
        "relative_excess_reduction": norm_reduction,
        "baseline_vector_residual_norm": baseline_vector,
        "elliptic_vector_residual_norm": elliptic_vector,
        "absolute_vector_residual_reduction": baseline_vector - elliptic_vector,
        "relative_vector_residual_reduction": vector_reduction,
        "FRICTION_CONE_SOLVER_EXCESS_EFFECT": effect,
        "MUJOCO_SOLVER_EXCESS_CAUSAL_DRIVER": driver,
        "NEXT_ACTION": next_action,
    }


def write_condition(output: Path, condition: dict[str, Any]) -> None:
    target = output / "conditions" / condition["condition_name"]
    target.mkdir(parents=True)
    capture = condition["capture"]
    for filename, payload in (
        ("state_validation.json", condition["state_validation"]),
        ("contact_state.json", {
            "ncon": capture["ncon"], "nefc": capture["nefc"],
            "contacts": capture["contacts"],
        }),
        ("solver_rows.json", {
            "cone_numeric": condition["cone_numeric"],
            "contacts": [
                {"contact_index": item["contact_index"], "pair": [item["geom1_name"], item["geom2_name"]], "rows": item["solver_rows"]}
                for item in capture["contacts"]
            ],
        }),
        ("physical_contact_impulses.json", condition["physical_impulses"]),
        ("mass_jacobian_delassus.json", {
            "mass_matrix": capture["mass_matrix"],
            "J_phys": capture["J_phys"],
            "W_phys": capture["W_phys"],
            "generalized_column_order": oracle.GENERALIZED_COLUMN_ORDER,
        }),
        ("shared_physical_global_demand.json", {
            "method": "SHARED_PHYSICAL_GLOBAL_COUPLED_DEMAND",
            "formula": "-solve(W_tt_shared_6x6, v_t_pre_shared_6 + W_tn_shared_6x3 @ p_normal_3)",
            "budget": condition["budget"],
        }),
        ("solver_excess.json", condition["excess"]),
        ("one_step_result.json", {
            "post_qpos": capture["post_state"]["qpos"],
            "post_qvel": capture["post_state"]["qvel"],
            "target_post_slip": condition["excess"]["post_slip"],
            "mj_step_count": 1,
        }),
    ):
        oracle.write_json(target / filename, payload)


def write_git_identity(output: Path) -> None:
    def git(*arguments: str) -> str:
        return subprocess.run(
            ["git", *arguments],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout

    identity = (
        f"TOPLEVEL={git('rev-parse', '--show-toplevel').strip()}\n"
        f"HEAD={git('rev-parse', 'HEAD').strip()}\n"
        f"BRANCH={git('branch', '--show-current').strip()}\n"
    )
    (output / "git_head.txt").write_text(identity, encoding="utf-8")
    try:
        artifact_relative = output.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        artifact_relative = ""
    status_lines = [
        line for line in git("status", "--short").splitlines()
        if not artifact_relative or artifact_relative not in line.replace("\\", "/")
    ]
    (output / "git_status_short.txt").write_text(
        "\n".join(status_lines) + ("\n" if status_lines else ""),
        encoding="utf-8",
    )


def execute(args: argparse.Namespace, paths: dict[str, Path]) -> dict[str, Any]:
    output = paths["output_dir"]
    output.mkdir(parents=True, exist_ok=True)
    write_git_identity(output)
    source_files = [
        paths["morphology_xml"], paths["checkpoint"],
        REPO_ROOT / "tools/analyze_mujoco_global55_contact_demand.py",
        Path(__file__).resolve(),
        paths["corrected_oracle"] / "validation.json",
    ]
    hashes_before = {str(path): oracle.sha256(path) for path in source_files}
    recorder, mapping = replay_once(args, paths)
    mujoco, model, snapshot = (
        recorder.raw_mujoco, recorder.raw_model, recorder.global55_snapshot
    )
    assert mujoco is not None and model is not None and snapshot is not None
    oracle.write_json(output / "global55_pre_state_snapshot.json", state_input_snapshot(snapshot))
    oracle.write_json(output / "state_copy_manifest.json", {
        **state_copy_manifest(snapshot),
        "formal_snapshot_copy_evidence": recorder.snapshot_copy_evidence,
    })
    original_options = model_option_snapshot(model)
    original_cone = int(model.opt.cone)
    pyramidal_value = int(mujoco.mjtCone.mjCONE_PYRAMIDAL)
    elliptic_value = int(mujoco.mjtCone.mjCONE_ELLIPTIC)
    if original_cone != pyramidal_value:
        raise RuntimeError("production model cone is not mjCONE_PYRAMIDAL")

    conditions: dict[str, dict[str, Any]] = {}
    try:
        for name, label, kind in CONDITIONS:
            cone = pyramidal_value if kind == "pyramidal" else elliptic_value
            conditions[name] = run_condition(
                mujoco, model, snapshot, mapping, name, label, cone
            )
            write_condition(output, conditions[name])
    finally:
        model.opt.cone = original_cone
    restored_options = model_option_snapshot(model)
    model_restore_diff = model_option_difference(original_options, restored_options)
    model_restore = "PASS" if not model_restore_diff["changed_fields"] else "FAIL"

    option_comparison = {
        "original": original_options,
        "conditions": {
            name: {
                "before": condition["options_before"],
                "after": condition["options_after"],
                "changed_during_condition": condition["options_changed_during_condition"],
            }
            for name, condition in conditions.items()
        },
        "original_to_elliptic": model_option_difference(
            original_options, conditions["elliptic"]["options_before"]
        ),
        "original_to_pyramidal_before": model_option_difference(
            original_options, conditions["pyramidal_before"]["options_before"]
        ),
        "original_to_pyramidal_after_restore": model_option_difference(
            original_options,
            conditions["pyramidal_after_restore"]["options_before"],
        ),
        "restored": restored_options,
        "restore_difference": model_restore_diff,
        "MODEL_OPTION_RESTORE": model_restore,
    }
    only_cone_changed = (
        option_comparison["original_to_elliptic"]["changed_fields"] == ["opt.cone"]
    )
    model_option_purity = bool(
        only_cone_changed
        and not option_comparison["original_to_pyramidal_before"]["changed_fields"]
        and not option_comparison["original_to_pyramidal_after_restore"]["changed_fields"]
    )
    activation = constraint_activation(conditions, pyramidal_value, elliptic_value)
    reference = load_reference(paths["corrected_oracle"])
    baseline = baseline_regression(conditions["pyramidal_before"], reference)
    restore = restore_regression(
        conditions["pyramidal_before"], conditions["pyramidal_after_restore"]
    )
    all_state_equal = all(
        condition["state_validation"]["solver_clone"]["STATE_COPY_EQUAL"]
        and condition["state_validation"]["step_clone"]["STATE_COPY_EQUAL"]
        and condition["state_validation"]["snapshot_unchanged_by_solver_and_step_clones"]
        for condition in conditions.values()
    )
    probe_forward_count = sum(
        condition["counts"]["probe_mj_forward"] for condition in conditions.values()
    )
    probe_step_count = sum(
        condition["counts"]["probe_mj_step"] for condition in conditions.values()
    )
    formal_copy_valid = bool(
        recorder.snapshot_copy_evidence
        and recorder.snapshot_copy_evidence["snapshot_equals_live"]["STATE_COPY_EQUAL"]
        and recorder.snapshot_copy_evidence["live_unchanged_by_copy"]
    )
    no_condition_option_drift = all(
        not condition["options_changed_during_condition"]["changed_fields"]
        for condition in conditions.values()
    )
    noncanonical = bool(
        restore["PYRAMIDAL_RESTORE_REPRODUCTION"] != "PASS"
        or model_restore != "PASS"
        or not all_state_equal
        or not formal_copy_valid
        or not no_condition_option_drift
    )
    gates_valid = bool(
        baseline["PYRAMIDAL_BASELINE_REPRODUCTION"] == "PASS"
        and restore["PYRAMIDAL_RESTORE_REPRODUCTION"] == "PASS"
        and activation["CONE_COUNTERFACTUAL_ACTIVATION"] == "VALIDATED"
        and model_restore == "PASS"
        and only_cone_changed
        and model_option_purity
        and all_state_equal
        and formal_copy_valid
        and no_condition_option_drift
        and len(recorder.records) == EXPECTED_SUBSTEPS
        and probe_forward_count == 3
        and probe_step_count == 3
    )
    comparison = classify_effect(
        conditions["pyramidal_before"]["excess"],
        conditions["elliptic"]["excess"],
        gates_valid,
        noncanonical=noncanonical,
    )
    hashes_after = {str(path): oracle.sha256(path) for path in source_files}
    source_unchanged = hashes_before == hashes_after
    if not source_unchanged:
        gates_valid = False
        comparison = {
            "FRICTION_CONE_SOLVER_EXCESS_EFFECT": "NONCANONICAL",
            "MUJOCO_SOLVER_EXCESS_CAUSAL_DRIVER": "NONCANONICAL",
            "NEXT_ACTION": "COUNTERFACTUAL_IMPLEMENTATION_FIX_REQUIRED",
        }
    snapshot_payload = state_input_snapshot(snapshot)
    state_equality_report = {
        "conditions": {
            name: condition["state_validation"] for name, condition in conditions.items()
        },
        "all_conditions_equal_same_snapshot": all_state_equal,
    }
    metadata = {
        "created_at": datetime.now().astimezone().isoformat(),
        "backend": "mujoco",
        "mujoco_version": getattr(mujoco, "__version__", None),
        "morphology": MORPHOLOGY,
        "morphology_xml": str(paths["morphology_xml"]),
        "morphology_xml_sha256": hashes_before[str(paths["morphology_xml"])],
        "checkpoint": str(paths["checkpoint"]),
        "checkpoint_sha256": hashes_before[str(paths["checkpoint"])],
        "corrected_reference_oracle": str(paths["corrected_oracle"]),
        "formal_replay_helper": "tools.analyze_mujoco_global55_contact_demand.replay",
        "formal_replay_physics_substeps": len(recorder.records),
        "global_physics_step": GLOBAL_STEP,
        "physics_dt": float(model.opt.timestep),
        "action_mode": "zero",
        "reset_noise_scale": 0.0,
        "conditions": [label for _, label, _ in CONDITIONS],
        "isaac_external_reference_solver_excess_Ns": 0.012,
        "isaac_reference_is_success_gate": False,
    }
    runtime_identity_valid = bool(
        metadata["mujoco_version"] == "3.8.1"
        and metadata["morphology_xml_sha256"] == XML_SHA256
        and metadata["checkpoint_sha256"] == CHECKPOINT_SHA256
    )
    if not runtime_identity_valid:
        gates_valid = False
        comparison = {
            "FRICTION_CONE_SOLVER_EXCESS_EFFECT": "NONCANONICAL",
            "MUJOCO_SOLVER_EXCESS_CAUSAL_DRIVER": "NONCANONICAL",
            "NEXT_ACTION": "COUNTERFACTUAL_IMPLEMENTATION_FIX_REQUIRED",
        }
    validation = {
        "PYRAMIDAL_BASELINE_REPRODUCTION": baseline["PYRAMIDAL_BASELINE_REPRODUCTION"],
        "CONE_COUNTERFACTUAL_ACTIVATION": activation["CONE_COUNTERFACTUAL_ACTIVATION"],
        "PYRAMIDAL_RESTORE_REPRODUCTION": restore["PYRAMIDAL_RESTORE_REPRODUCTION"],
        "MODEL_OPTION_RESTORE": model_restore,
        "only_model_option_changed_is_cone": only_cone_changed,
        "model_option_purity": model_option_purity,
        "runtime_identity": "PASS" if runtime_identity_valid else "FAIL",
        "condition_state_equality": all_state_equal,
        "formal_replay_physics_substeps": len(recorder.records),
        "expected_formal_replay_physics_substeps": EXPECTED_SUBSTEPS,
        "probe_mj_forward_count": probe_forward_count,
        "probe_mj_step_count": probe_step_count,
        "extra_formal_replay_steps": 0,
        "formal_snapshot_copy_valid": formal_copy_valid,
        "formal_data_mutated_by_probe": not formal_copy_valid,
        "condition_model_option_drift": not no_condition_option_drift,
        "source_hashes_unchanged": source_unchanged,
        "all_numerical_outputs_finite": oracle.all_numeric_values_finite(
            {"conditions": conditions, "comparison": comparison}
        ),
        **{key: comparison[key] for key in (
            "FRICTION_CONE_SOLVER_EXCESS_EFFECT",
            "MUJOCO_SOLVER_EXCESS_CAUSAL_DRIVER", "NEXT_ACTION",
        )},
        "COUNTERFACTUAL_VALID": bool(gates_valid and source_unchanged),
        "UNCONDITIONAL_ZIP_PACKAGING": "ENABLED",
    }
    summary = {
        **{key: validation[key] for key in (
            "PYRAMIDAL_BASELINE_REPRODUCTION", "CONE_COUNTERFACTUAL_ACTIVATION",
            "PYRAMIDAL_RESTORE_REPRODUCTION", "MODEL_OPTION_RESTORE",
            "FRICTION_CONE_SOLVER_EXCESS_EFFECT",
            "MUJOCO_SOLVER_EXCESS_CAUSAL_DRIVER", "NEXT_ACTION",
        )},
        "baseline_solver_excess_Ns": conditions["pyramidal_before"]["excess"]["solver_excess_norm"],
        "elliptic_solver_excess_Ns": conditions["elliptic"]["excess"]["solver_excess_norm"],
        "comparison": comparison,
    }
    for filename, payload in (
        ("metadata.json", metadata),
        ("global55_pre_state_snapshot.json", snapshot_payload),
        ("state_copy_manifest.json", {
            **state_copy_manifest(snapshot),
            "formal_snapshot_copy_evidence": recorder.snapshot_copy_evidence,
        }),
        ("condition_state_equality.json", state_equality_report),
        ("model_option_comparison.json", option_comparison),
        ("constraint_parameterization.json", activation),
        ("baseline_regression.json", baseline),
        ("restore_regression.json", restore),
        ("cone_counterfactual_comparison.json", comparison),
        ("validation.json", validation),
        ("summary.json", summary),
        ("source_purity.json", {
            "hashes_before": hashes_before,
            "hashes_after": hashes_after,
            "source_hashes_unchanged": source_unchanged,
            "model_option_restore": model_restore,
            "formal_data_mutated_by_probe": not formal_copy_valid,
            "condition_model_option_drift": not no_condition_option_drift,
        }),
    ):
        oracle.write_json(output / filename, payload)
    return validation


def failure_payload(error: Exception) -> dict[str, Any]:
    return {
        "PYRAMIDAL_BASELINE_REPRODUCTION": "FAIL",
        "CONE_COUNTERFACTUAL_ACTIVATION": "INSUFFICIENT_EVIDENCE",
        "PYRAMIDAL_RESTORE_REPRODUCTION": "FAIL",
        "MODEL_OPTION_RESTORE": "FAIL",
        "FRICTION_CONE_SOLVER_EXCESS_EFFECT": "INSUFFICIENT_EVIDENCE",
        "MUJOCO_SOLVER_EXCESS_CAUSAL_DRIVER": "INSUFFICIENT_EVIDENCE",
        "NEXT_ACTION": "COUNTERFACTUAL_IMPLEMENTATION_FIX_REQUIRED",
        "COUNTERFACTUAL_VALID": False,
        "UNCONDITIONAL_ZIP_PACKAGING": "ENABLED",
        "error_type": type(error).__name__,
        "error": str(error),
    }


def main(argv: Sequence[str] | None = None) -> int:
    raw_args = parser().parse_args(argv)
    try:
        args = resolve_arguments(raw_args)
    except Exception as error:
        stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
        output = Path(
            raw_args.output_dir
            or REPO_ROOT / "output/diagnostics"
            / f"mujoco_global55_friction_cone_counterfactual_{stamp}"
        ).resolve()
        zip_path = Path(
            raw_args.zip_path
            or REPO_ROOT / "tmp"
            / f"mujoco_global55_friction_cone_counterfactual_{stamp}.zip"
        ).resolve()
        output.mkdir(parents=True, exist_ok=False)
        trace = traceback.format_exc()
        (output / "run.log").write_text(trace, encoding="utf-8")
        (output / "traceback.txt").write_text(trace, encoding="utf-8")
        failure = failure_payload(error)
        oracle.write_json(output / "validation.json", failure)
        oracle.write_json(output / "summary.json", failure)
        oracle.write_json(output / "source_purity.json", {
            "status": "INCOMPLETE_DUE_TO_ARGUMENT_RESOLUTION_FAILURE",
            "formal_dynamics_mutation_requested": False,
            "traceback_file": "traceback.txt",
        })
        package = oracle.package_artifact(output, zip_path)
        print(f"ZIP_VERIFY={package['ZIP_VERIFY']}")
        print(f"ZIP_SHA256={package['ZIP_SHA256']}")
        print(f"UPLOAD_THIS_ZIP={package['UPLOAD_THIS_ZIP']}")
        return 2
    output = Path(args.output_dir).resolve()
    zip_path = Path(args.zip_path).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite output: {output}")
    if zip_path.exists():
        raise FileExistsError(f"refusing to overwrite ZIP: {zip_path}")
    output.mkdir(parents=True, exist_ok=False)
    log_path = output / "run.log"
    return_code = 2
    with log_path.open("w", encoding="utf-8") as log_stream:
        with redirect_stdout(Tee(sys.__stdout__, log_stream)), redirect_stderr(
            Tee(sys.__stderr__, log_stream)
        ):
            try:
                paths = validate_paths(args)
                paths["output_dir"] = output
                validation = execute(args, paths)
                return_code = 0 if validation["COUNTERFACTUAL_VALID"] else 2
                print(json.dumps(oracle.json_ready(validation), indent=2, allow_nan=False))
            except Exception as error:
                trace = traceback.format_exc()
                print(trace, file=sys.stderr, end="")
                (output / "traceback.txt").write_text(trace, encoding="utf-8")
                failure = failure_payload(error)
                oracle.write_json(output / "validation.json", failure)
                oracle.write_json(output / "summary.json", failure)
                oracle.write_json(output / "source_purity.json", {
                    "status": "INCOMPLETE_DUE_TO_FAILURE",
                    "formal_dynamics_mutation_requested": False,
                    "traceback_file": "traceback.txt",
                })
    package = oracle.package_artifact(output, zip_path)
    print(f"ZIP_VERIFY={package['ZIP_VERIFY']}")
    print(f"ZIP_SHA256={package['ZIP_SHA256']}")
    print(f"UPLOAD_THIS_ZIP={package['UPLOAD_THIS_ZIP']}")
    return return_code


if __name__ == "__main__":
    np.bool = np.bool_
    raise SystemExit(main())

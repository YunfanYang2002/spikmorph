"""Numerically evaluate a formal spikmorph MuJoCo checkpoint.

The evaluator reuses the repository environment factory, VecNormalize,
ActorCritic model, and native checkpoint restoration.  It does not change
task, reset, reward, termination, actuator, physics, or PPO semantics.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
from types import MethodType
from typing import Any, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

ACTION_MODES = ("zero", "mean", "sample")
OUTPUT_FILENAMES = ("metadata.json", "summary.json", "transitions.jsonl")
JOINT_LIMIT_OUTPUT_FILENAMES = (
    "substeps.jsonl",
    "joint_mapping.json",
    "first_contact_and_limit_summary.json",
    "validation.json",
    "contact_generalized_response_summary.json",
    "physical_contact_substeps.jsonl",
    "contact_frame_validation.json",
    "physical_vs_constraint_generalized.json",
    "selected_joint_physical_decomposition.json",
    "unit_force_projection.json",
    "joint_contact_geometry.json",
    "run.log",
)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="Evaluate one formal MuJoCo checkpoint without rendering."
    )
    result.add_argument("--checkpoint", required=True)
    result.add_argument("--walker-dir", required=True)
    result.add_argument("--morphology-id", required=True)
    result.add_argument("--action-mode", choices=ACTION_MODES, required=True)
    result.add_argument("--episodes", type=int, default=5)
    result.add_argument("--seed", type=int, default=1409)
    result.add_argument("--output-dir", required=True)
    result.add_argument("--device", default="cpu")
    result.add_argument("--cfg", default="configs/ft.yaml")
    result.add_argument(
        "--record-state-trajectory",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    result.add_argument("--max-eval-steps", type=int)
    result.add_argument("--reset-noise-scale", type=float)
    result.add_argument(
        "--record-joint-limit-substeps",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    result.add_argument(
        "--joint-limit-probe-names",
        nargs="+",
        default=[],
    )
    result.add_argument(
        "--record-contact-generalized-response",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    result.add_argument(
        "--contact-probe-body-names",
        nargs="+",
        default=[],
    )
    result.add_argument(
        "--record-physical-contact-projection",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    return result


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_output(*args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def require_file(path: Path, label: str) -> Path:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    return path


def validate_args(args: argparse.Namespace) -> dict[str, Path]:
    if args.episodes <= 0:
        raise ValueError("--episodes must be positive")
    max_eval_steps = getattr(args, "max_eval_steps", None)
    reset_noise_scale = getattr(args, "reset_noise_scale", None)
    if max_eval_steps is not None and max_eval_steps <= 0:
        raise ValueError("--max-eval-steps must be positive")
    if reset_noise_scale is not None and reset_noise_scale < 0.0:
        raise ValueError("--reset-noise-scale must be non-negative")
    record_limit_substeps = bool(
        getattr(args, "record_joint_limit_substeps", False)
    )
    record_contact_response = bool(
        getattr(args, "record_contact_generalized_response", False)
    )
    record_physical_projection = bool(
        getattr(args, "record_physical_contact_projection", False)
    )
    probe_names = list(getattr(args, "joint_limit_probe_names", []))
    if record_limit_substeps:
        if args.action_mode != "zero":
            raise ValueError("joint-limit substep recording requires --action-mode zero")
        if args.episodes != 1:
            raise ValueError("joint-limit substep recording requires --episodes 1")
        if max_eval_steps is None:
            raise ValueError(
                "joint-limit substep recording requires --max-eval-steps"
            )
        if reset_noise_scale != 0.0:
            raise ValueError(
                "joint-limit substep recording requires --reset-noise-scale 0.0"
            )
        if not probe_names:
            raise ValueError(
                "joint-limit substep recording requires --joint-limit-probe-names"
            )
        if len(set(probe_names)) != len(probe_names):
            raise ValueError("--joint-limit-probe-names must be unique")
    if record_contact_response:
        body_names = list(getattr(args, "contact_probe_body_names", []))
        if not record_limit_substeps:
            raise ValueError(
                "contact generalized response requires --record-joint-limit-substeps"
            )
        if not body_names:
            raise ValueError(
                "contact generalized response requires --contact-probe-body-names"
            )
        if len(set(body_names)) != len(body_names):
            raise ValueError("--contact-probe-body-names must be unique")
    if record_physical_projection and not record_limit_substeps:
        raise ValueError(
            "physical contact projection requires --record-joint-limit-substeps"
        )
    checkpoint = require_file(Path(args.checkpoint), "checkpoint")
    walker_dir = Path(args.walker_dir).resolve()
    morphology_xml = require_file(
        walker_dir / "xml" / f"{args.morphology_id}.xml", "morphology XML"
    )
    morphology_metadata = require_file(
        walker_dir / "metadata" / f"{args.morphology_id}.json",
        "morphology metadata",
    )
    config = require_file(REPO_ROOT / args.cfg, "config")
    output_dir = Path(args.output_dir).resolve()
    output_names = list(OUTPUT_FILENAMES)
    if record_limit_substeps:
        output_names.extend(JOINT_LIMIT_OUTPUT_FILENAMES)
    existing = [output_dir / name for name in output_names]
    collisions = [path for path in existing if path.exists()]
    if collisions:
        raise FileExistsError(
            "refusing to overwrite evaluator outputs: "
            + ", ".join(str(path) for path in collisions)
        )
    return {
        "checkpoint": checkpoint,
        "walker_dir": walker_dir,
        "morphology_xml": morphology_xml,
        "morphology_metadata": morphology_metadata,
        "config": config,
        "output_dir": output_dir,
    }


class FiniteTracker:
    def __init__(self) -> None:
        self.all_values_finite = True

    def scalar(self, value: Any) -> float | int | None:
        if value is None:
            return None
        try:
            result = float(value)
        except (TypeError, ValueError):
            self.all_values_finite = False
            return None
        if not math.isfinite(result):
            self.all_values_finite = False
            return None
        return result

    def vector(self, value: Any) -> list[float] | None:
        import numpy as np

        result = np.asarray(value, dtype=np.float64).reshape(-1)
        if not np.isfinite(result).all():
            self.all_values_finite = False
            return None
        return result.tolist()


def info_scalar(
    info: dict[str, Any], tracker: FiniteTracker, *keys: str
) -> float | None:
    for key in keys:
        if key not in info:
            continue
        value = info[key]
        try:
            if hasattr(value, "reshape"):
                value = value.reshape(-1)[0]
            elif isinstance(value, (list, tuple)):
                value = value[0]
        except (IndexError, TypeError, ValueError):
            tracker.all_values_finite = False
            return None
        return tracker.scalar(value)
    return None


def official_fall_measurement(
    info: dict[str, Any], tracker: FiniteTracker
) -> dict[str, Any]:
    torso_height = info_scalar(info, tracker, "formal_torso_height")
    threshold = info_scalar(info, tracker, "formal_fallen_threshold")
    return {
        "formal_torso_height": torso_height,
        "formal_fallen_threshold": threshold,
        "formal_torso_height_source": "official_termination_info",
        "formal_torso_height_status": (
            "available"
            if torso_height is not None
            else "unavailable_from_official_termination_info"
        ),
    }


def choose_raw_action(distribution, observation, action_mode: str):
    """Return the unmodified policy-space action and valid-dimension mask."""
    import torch

    if action_mode == "zero":
        action = torch.zeros_like(distribution.mean)
    elif action_mode == "mean":
        action = distribution.mean
    elif action_mode == "sample":
        action = distribution.sample()
    else:
        raise ValueError(f"unsupported action mode: {action_mode}")
    valid_mask = ~observation["act_padding_mask"].bool()
    return action, valid_mask


def raw_action_diagnostics(action, valid_mask, tracker: FiniteTracker):
    import torch

    valid = action[valid_mask]
    if valid.numel() == 0:
        return {
            "action_l2": 0.0,
            "raw_action_min": None,
            "raw_action_max": None,
            "raw_action_out_of_bounds_fraction": 0.0,
            "_valid_action_count": 0,
            "_out_of_bounds_count": 0,
            "_action_values_finite": True,
        }
    finite = bool(torch.isfinite(valid).all().item())
    tracker.all_values_finite &= finite
    if not finite:
        return {
            "action_l2": None,
            "raw_action_min": None,
            "raw_action_max": None,
            "raw_action_out_of_bounds_fraction": None,
            "_valid_action_count": int(valid.numel()),
            "_out_of_bounds_count": 0,
            "_action_values_finite": False,
        }
    out_of_bounds = int((valid.abs() > 1.0).sum().item())
    count = int(valid.numel())
    return {
        "action_l2": tracker.scalar(torch.linalg.vector_norm(valid).item()),
        "raw_action_min": tracker.scalar(valid.min().item()),
        "raw_action_max": tracker.scalar(valid.max().item()),
        "raw_action_out_of_bounds_fraction": out_of_bounds / count,
        "_valid_action_count": count,
        "_out_of_bounds_count": out_of_bounds,
        "_action_values_finite": True,
    }


def mean_or_none(values: Sequence[float | int | None]) -> float | None:
    valid = [float(value) for value in values if value is not None]
    return sum(valid) / len(valid) if valid else None


def evaluator_cutoff_reached(step_count: int, max_eval_steps: int | None) -> bool:
    return max_eval_steps is not None and step_count >= max_eval_steps


def native_reset_metadata(
    effective_noise_scale: float, requested_noise_scale: float | None
) -> dict[str, Any]:
    noise_active = bool(float(effective_noise_scale) != 0.0)
    return {
        "reset_noise_scale": float(effective_noise_scale),
        "requested_reset_noise_scale": requested_noise_scale,
        "reset_state_noise_active": noise_active,
        "qpos_qvel_noise_preserved": noise_active,
        "deterministic_reset_effective": not noise_active,
        "deterministic_reset_forced": bool(requested_noise_scale == 0.0),
    }


def _indices(address: int | tuple[int, int]) -> list[int]:
    if isinstance(address, tuple):
        return list(range(int(address[0]), int(address[1])))
    return [int(address)]


def unwrap_single_mujoco_env(envs: Any) -> Any:
    current = envs
    seen: set[int] = set()
    while id(current) not in seen:
        seen.add(id(current))
        if "venv" in vars(current):
            current = vars(current)["venv"]
            continue
        if "envs" in vars(current):
            candidates = vars(current)["envs"]
            if len(candidates) != 1:
                raise RuntimeError("state trajectory requires exactly one raw env")
            current = candidates[0]
            continue
        if "env" in vars(current):
            current = vars(current)["env"]
            continue
        break
    base = getattr(current, "unwrapped", current)
    if getattr(base, "sim", None) is None:
        raise RuntimeError("MuJoCo simulation is unavailable for state trajectory")
    return base


def find_env_wrapper(envs: Any, class_name: str) -> Any:
    current = envs
    seen: set[int] = set()
    while id(current) not in seen:
        seen.add(id(current))
        if current.__class__.__name__ == class_name:
            return current
        if "venv" in vars(current):
            current = vars(current)["venv"]
            continue
        if "envs" in vars(current):
            candidates = vars(current)["envs"]
            if len(candidates) != 1:
                raise RuntimeError("state trajectory requires exactly one raw env")
            current = candidates[0]
            continue
        if "env" in vars(current):
            current = vars(current)["env"]
            continue
        break
    raise RuntimeError(f"required wrapper is unavailable: {class_name}")


def install_pre_autoreset_state_capture(
    termination_wrapper: Any,
    base_env: Any,
    metadata: dict[str, Any],
    tracker: FiniteTracker,
    snapshots: dict[str, dict[str, Any]],
) -> None:
    """Capture post-termination MuJoCo state before DummyVecEnv auto-reset."""
    original_step = termination_wrapper.step

    def step(self: Any, action: Any):
        result = original_step(action)
        snapshots["latest"] = capture_state_trajectory(
            base_env, metadata, tracker
        )
        return result

    termination_wrapper.step = MethodType(step, termination_wrapper)


def build_state_trajectory_metadata(base_env: Any) -> dict[str, Any]:
    model = base_env.sim.model
    type_names = {0: "free", 1: "ball", 2: "slide", 3: "hinge"}
    joints = []
    for joint_id, raw_name in enumerate(model.joint_names):
        name = str(raw_name)
        qpos_indices = _indices(model.get_joint_qpos_addr(name))
        qvel_indices = _indices(model.get_joint_qvel_addr(name))
        joints.append(
            {
                "joint_id": joint_id,
                "joint_name": name,
                "source_mjcf_name": name,
                "joint_type": type_names.get(int(model.jnt_type[joint_id]), "unknown"),
                "qpos_indices": qpos_indices,
                "qvel_indices": qvel_indices,
                "joint_range": [float(value) for value in model.jnt_range[joint_id]],
            }
        )
    free_joints = [joint for joint in joints if joint["joint_type"] == "free"]
    if len(free_joints) != 1:
        raise RuntimeError(f"expected one free root joint, found {len(free_joints)}")
    root = free_joints[0]
    ordinary = [joint for joint in joints if joint is not root]
    if any(len(joint["qpos_indices"]) != 1 or len(joint["qvel_indices"]) != 1 for joint in ordinary):
        raise RuntimeError("ordinary agent joints must have scalar qpos/qvel")
    if any(joint["joint_type"] != "hinge" for joint in ordinary):
        raise RuntimeError("ordinary policy joints must all be one-DOF hinge joints")
    actuator_names = tuple(getattr(model, "actuator_names", ()))
    actuators = []
    for actuator_id in range(int(model.nu)):
        joint_id = int(model.actuator_trnid[actuator_id, 0])
        actuators.append(
            {
                "actuator_id": actuator_id,
                "actuator_name": str(actuator_names[actuator_id]) if actuator_id < len(actuator_names) else f"actuator/{actuator_id}",
                "joint_id": joint_id,
                "joint_name": str(model.joint_names[joint_id]),
                "ctrl_index": actuator_id,
                "actuator_force_index": actuator_id,
            }
        )
    for policy_index, joint in enumerate(ordinary):
        actuator_indices = [
            item["actuator_id"]
            for item in actuators
            if item["joint_id"] == joint["joint_id"]
        ]
        if len(actuator_indices) != 1:
            raise RuntimeError(
                "each ordinary policy joint must map to exactly one actuator: "
                f"joint={joint['joint_name']!r}, actuators={actuator_indices}"
            )
        joint.update(
            {
                "policy_index": policy_index,
                "source_joint_name": joint["joint_name"],
                "actuator_indices": actuator_indices,
                "qfrc_indices": list(joint["qvel_indices"]),
            }
        )
    actuator_joint_names = [item["joint_name"] for item in actuators]
    ordinary_joint_names = [joint["joint_name"] for joint in ordinary]
    if actuator_joint_names != ordinary_joint_names:
        raise RuntimeError(
            "formal policy actuator order does not match source ordinary-joint "
            f"order: actuators={actuator_joint_names}, joints={ordinary_joint_names}"
        )
    dof_mapping = []
    for joint in joints:
        components = (
            ("linear_x", "linear_y", "linear_z", "angular_x", "angular_y", "angular_z")
            if joint["joint_type"] == "free"
            else ("velocity",)
        )
        dof_mapping.extend(
            {
                "qvel_index": index,
                "qfrc_index": index,
                "joint_name": joint["joint_name"],
                "component": components[offset],
            }
            for offset, index in enumerate(joint["qvel_indices"])
        )
    if len(dof_mapping) != int(model.nv):
        raise RuntimeError(
            f"generalized DOF mapping mismatch: {len(dof_mapping)} != {model.nv}"
        )
    torso_id = int(model.body_name2id("torso/0"))
    root_body_id = int(model.jnt_bodyid[root["joint_id"]])
    root_body_name = str(model.body_id2name(root_body_id))
    root["body_id"] = root_body_id
    root["body_name"] = root_body_name
    root["qfrc_indices"] = list(root["qvel_indices"])
    joint_names = [joint["joint_name"] for joint in ordinary]
    return {
        "physics_timestep": float(model.opt.timestep),
        "frame_skip": int(base_env.frame_skip),
        "control_dt": float(model.opt.timestep * base_env.frame_skip),
        "root_free_joint": root,
        "root_qpos_convention": "[world_x, world_y, world_z, quat_w, quat_x, quat_y, quat_z]",
        "root_qvel_convention": (
            "native MuJoCo generalized free-joint [linear_xyz, angular_xyz]; "
            "retained for diagnostics and not asserted equivalent to world-frame body velocity"
        ),
        "body_velocity_convention": (
            "sim.data.body_xvelp/body_xvelr compatibility API; modern MuJoCo "
            "uses mj_objectVelocity with flg_local=0, world-aligned Cartesian"
        ),
        "root_body_id": root_body_id,
        "root_body_name": root_body_name,
        "root_body_is_torso_body": root_body_id == torso_id,
        "ordinary_joint_mapping": ordinary,
        "all_ordinary_joints_one_dof_hinge": True,
        "joint_order_semantics": (
            "source MJCF ordinary-joint order, validated equal to formal "
            "MuJoCo actuator/control order"
        ),
        "ordered_joint_names": joint_names,
        "joint_names": joint_names,
        "joint_indices": list(range(len(joint_names))),
        "joint_index_map": {
            str(index): name for index, name in enumerate(joint_names)
        },
        "generalized_dof_mapping": dof_mapping,
        "actuator_mapping": actuators,
        "body_mapping": [
            {"body_id": index, "body_name": str(name)}
            for index, name in enumerate(model.body_names)
        ],
        "torso_body_id": torso_id,
        "torso_body_name": "torso/0",
        "cross_backend_alias_status": (
            "not_asserted_without_isaac_articulation_root_frame_mapping"
        ),
        "direct_field_availability": {
            name: hasattr(base_env.sim.data, name)
            for name in (
                "ctrl",
                "actuator_force",
                "qfrc_actuator",
                "qfrc_passive",
                "contact",
            )
        },
        "force_semantics": {
            "native_action": "policy-space action after padding removal",
            "actuator_ctrl": "MuJoCo actuator control input in model actuator order",
            "actuator_force": "MuJoCo actuator scalar force in model actuator order",
            "qfrc_actuator": "actuator contribution to generalized force in nv/DOF order",
            "qfrc_passive": "passive generalized force in nv/DOF order",
            "joint_qfrc_mapping": "source joint qvel/DOF address; not policy array index",
        },
        "coordinate_conventions": {
            "positions": "MuJoCo world frame, metres",
            "env_origin_world_xyz": [0.0, 0.0, 0.0],
            "root_local_to_env_origin": (
                "identical to MuJoCo world xyz for this single-environment evaluator"
            ),
            "body_orientation": "quaternion wxyz",
            "body_linear_velocity": "world-aligned Cartesian, metres/second",
            "body_angular_velocity": "world-aligned Cartesian, radians/second",
            "ordinary_joint_qpos": "native scalar joint coordinates, radians for hinge joints",
            "ordinary_joint_qvel": "native scalar joint velocities, radians/second for hinge joints",
        },
    }


def capture_state_trajectory(base_env: Any, metadata: dict[str, Any], tracker: FiniteTracker) -> dict[str, Any]:
    import numpy as np

    sim = base_env.sim
    data, model = sim.data, sim.model
    root = metadata["root_free_joint"]
    qpos = np.asarray(data.qpos)
    qvel = np.asarray(data.qvel)
    root_qpos = qpos[root["qpos_indices"]]
    root_qvel = qvel[root["qvel_indices"]]
    root_body_id = metadata["root_body_id"]
    torso_id = metadata["torso_body_id"]
    body_xpos = np.asarray(data.body_xpos)
    body_xquat = np.asarray(data.body_xquat)
    body_xvelp = np.asarray(data.body_xvelp)
    body_xvelr = np.asarray(data.body_xvelr)
    contacts = []
    for contact in data.contact[: int(data.ncon)]:
        geom1, geom2 = int(contact.geom1), int(contact.geom2)
        body1, body2 = int(model.geom_bodyid[geom1]), int(model.geom_bodyid[geom2])
        contacts.append(
            {
                "geom1_id": geom1,
                "geom1_name": model.geom_id2name(geom1),
                "body1_id": body1,
                "body1_name": model.body_id2name(body1),
                "geom2_id": geom2,
                "geom2_name": model.geom_id2name(geom2),
                "body2_id": body2,
                "body2_name": model.body_id2name(body2),
            }
        )
    ordinary = metadata["ordinary_joint_mapping"]
    actuator_force = getattr(data, "actuator_force", None)
    ordinary_actuator_indices = [
        joint["actuator_indices"][0] for joint in ordinary
    ]
    ordinary_qfrc_indices = [joint["qfrc_indices"][0] for joint in ordinary]
    joint_qpos = [
        tracker.scalar(qpos[joint["qpos_indices"][0]]) for joint in ordinary
    ]
    joint_qvel = [
        tracker.scalar(qvel[joint["qvel_indices"][0]]) for joint in ordinary
    ]
    return {
        "simulation_time": tracker.scalar(data.time),
        "backend_episode_step": int(base_env.step_count),
        "full_qpos": tracker.vector(qpos),
        "full_qvel": tracker.vector(qvel),
        "root_world_position_xyz": tracker.vector(body_xpos[root_body_id]),
        "root_local_to_env_position_xyz": tracker.vector(
            body_xpos[root_body_id]
        ),
        "root_world_orientation_wxyz": tracker.vector(body_xquat[root_body_id]),
        "root_world_linear_velocity_xyz": tracker.vector(body_xvelp[root_body_id]),
        "root_world_angular_velocity_xyz": tracker.vector(body_xvelr[root_body_id]),
        "torso_world_position_xyz": tracker.vector(body_xpos[torso_id]),
        "torso_world_orientation_wxyz": tracker.vector(body_xquat[torso_id]),
        "torso_world_linear_velocity_xyz": tracker.vector(body_xvelp[torso_id]),
        "torso_world_angular_velocity_xyz": tracker.vector(body_xvelr[torso_id]),
        "root_free_joint_position": tracker.vector(root_qpos[:3]),
        "root_free_joint_orientation_wxyz": tracker.vector(root_qpos[3:7]),
        "root_generalized_qvel": tracker.vector(root_qvel),
        "root_qfrc_actuator": tracker.vector(data.qfrc_actuator[root["qfrc_indices"]]),
        "root_qfrc_passive": tracker.vector(data.qfrc_passive[root["qfrc_indices"]]),
        "joint_qpos": joint_qpos,
        "joint_qvel": joint_qvel,
        "ordered_joint_qpos": joint_qpos,
        "ordered_joint_qvel": joint_qvel,
        "actuator_ctrl": tracker.vector(data.ctrl),
        "actuator_force": (
            tracker.vector(actuator_force) if actuator_force is not None else None
        ),
        "joint_actuator_ctrl": tracker.vector(data.ctrl[ordinary_actuator_indices]),
        "joint_actuator_force": (
            tracker.vector(actuator_force[ordinary_actuator_indices])
            if actuator_force is not None
            else None
        ),
        "joint_qfrc_actuator": tracker.vector(data.qfrc_actuator[ordinary_qfrc_indices]),
        "joint_qfrc_passive": tracker.vector(data.qfrc_passive[ordinary_qfrc_indices]),
        "qfrc_actuator": tracker.vector(data.qfrc_actuator),
        "qfrc_passive": tracker.vector(data.qfrc_passive),
        "contact_count": int(data.ncon),
        "contacts": contacts,
    }


def _runtime_object_names(model: Any, kind: str, count: int) -> list[str]:
    """Read compiled names without relying on native convenience attributes."""
    raw_model = getattr(model, "_model", model)
    try:
        from metamorph.utils import mujoco_compat as mjc

        if mjc.BACKEND == "mujoco":
            object_type = getattr(mjc.mujoco.mjtObj, f"mjOBJ_{kind.upper()}")
            return [
                mjc.mujoco.mj_id2name(raw_model, object_type, index) or ""
                for index in range(int(count))
            ]
    except (AttributeError, ImportError):
        pass
    return [str(name) for name in getattr(model, f"{kind}_names")]


def build_joint_limit_probe_mapping(
    base_env: Any,
    probe_names: Sequence[str],
    trajectory_metadata: dict[str, Any],
    contact_probe_body_names: Sequence[str] = (),
    enable_contact_mapping: bool = False,
) -> dict[str, Any]:
    model = base_env.sim.model
    joint_names = _runtime_object_names(model, "joint", int(model.njnt))
    body_names = _runtime_object_names(model, "body", int(model.nbody))
    geom_names = _runtime_object_names(model, "geom", int(model.ngeom))
    mappings = []
    for name in probe_names:
        matches = [index for index, candidate in enumerate(joint_names) if candidate == name]
        if len(matches) != 1:
            raise ValueError(f"expected one compiled joint named {name!r}, found {len(matches)}")
        joint_id = matches[0]
        if int(model.jnt_type[joint_id]) != 3:
            raise ValueError(f"joint-limit probe {name!r} is not a hinge joint")
        if not bool(model.jnt_limited[joint_id]):
            raise ValueError(f"joint-limit probe {name!r} is not limited")
        qpos_address = int(model.jnt_qposadr[joint_id])
        dof_address = int(model.jnt_dofadr[joint_id])
        mappings.append(
            {
                "joint_name": name,
                "joint_id": joint_id,
                "qpos_address": qpos_address,
                "dof_address": dof_address,
                "joint_body_id": int(model.jnt_bodyid[joint_id]),
                "joint_body_name": body_names[int(model.jnt_bodyid[joint_id])],
                "jnt_range": [float(value) for value in model.jnt_range[joint_id]],
                "jnt_margin": float(model.jnt_margin[joint_id]),
                "jnt_solref": [float(value) for value in model.jnt_solref[joint_id]],
                "jnt_solimp": [float(value) for value in model.jnt_solimp[joint_id]],
                "jnt_stiffness": float(model.jnt_stiffness[joint_id]),
                "dof_damping": float(model.dof_damping[dof_address]),
                "dof_armature": float(model.dof_armature[dof_address]),
                "dof_frictionloss": float(model.dof_frictionloss[dof_address]),
            }
        )
    body_probes = []
    for name in contact_probe_body_names:
        matches = [index for index, candidate in enumerate(body_names) if candidate == name]
        if len(matches) != 1:
            raise ValueError(
                f"expected one compiled body named {name!r}, found {len(matches)}"
            )
        body_probes.append({"body_name": name, "body_id": matches[0]})
    floor_geom_ids = [
        index for index, name in enumerate(geom_names) if name == "floor/0"
    ]
    if (contact_probe_body_names or enable_contact_mapping) and len(floor_geom_ids) != 1:
        raise ValueError(
            f"expected one compiled floor geom named 'floor/0', found {len(floor_geom_ids)}"
        )
    return {
        "schema_version": "spikmorph-mujoco-joint-limit-mapping-v1",
        "constraint_row_identity": "efc_type == mjCNSTR_LIMIT_JOINT and efc_id == compiled joint_id; resolved independently after every mj_step",
        "joint_limit_constraint_type": "mjCNSTR_LIMIT_JOINT",
        "joints": mappings,
        "joint_names": joint_names,
        "body_names": body_names,
        "geom_names": geom_names,
        "contact_probe_bodies": body_probes,
        "floor_geom_id": floor_geom_ids[0] if floor_geom_ids else None,
        "floor_geom_name": "floor/0" if floor_geom_ids else None,
        "root_free_joint": trajectory_metadata["root_free_joint"],
        "root_body_id": trajectory_metadata["root_body_id"],
        "root_body_name": trajectory_metadata["root_body_name"],
        "torso_body_id": trajectory_metadata["torso_body_id"],
        "torso_body_name": trajectory_metadata["torso_body_name"],
    }


def constraint_jacobian_row(data: Any, row: int, nefc: int, nv: int) -> tuple[list[int], list[float]]:
    """Return one efc_J row for either MuJoCo dense or sparse storage."""
    import numpy as np

    jacobian = np.asarray(data.efc_J, dtype=np.float64).reshape(-1)
    if jacobian.size == nefc * nv:
        dense = jacobian.reshape(nefc, nv)[row]
        indices = np.flatnonzero(dense).astype(int).tolist()
        return indices, [float(dense[index]) for index in indices]
    row_nnz = np.asarray(data.efc_J_rownnz, dtype=np.int64)
    row_address = np.asarray(data.efc_J_rowadr, dtype=np.int64)
    column_indices = np.asarray(data.efc_J_colind, dtype=np.int64)
    start = int(row_address[row])
    count = int(row_nnz[row])
    stop = start + count
    return column_indices[start:stop].astype(int).tolist(), jacobian[start:stop].astype(float).tolist()


def _selected_constraint_reconstruction(data: Any, dof_address: int, nefc: int, nv: int) -> float:
    total = 0.0
    for row in range(nefc):
        indices, values = constraint_jacobian_row(data, row, nefc, nv)
        if dof_address in indices:
            total += values[indices.index(dof_address)] * float(data.efc_force[row])
    return float(total)


def contact_efc_rows(
    efc_address: int, dim: int, pyramidal: bool, nefc: int
) -> list[int]:
    """Map one mjContact to all of its dynamic constraint rows."""
    if efc_address < 0:
        return []
    row_count = dim if not pyramidal else (1 if dim == 1 else 2 * (dim - 1))
    stop = efc_address + row_count
    if stop > nefc:
        raise ValueError(
            f"contact efc rows [{efc_address}, {stop}) exceed nefc={nefc}"
        )
    return list(range(efc_address, stop))


def aggregate_constraint_rows(
    data: Any, rows: Sequence[int], nefc: int, nv: int
) -> tuple[list[dict[str, Any]], list[float]]:
    """Compute the exact sum of J_row^T * efc_force over selected rows."""
    generalized = [0.0] * int(nv)
    row_records = []
    for row in rows:
        indices, values = constraint_jacobian_row(data, int(row), nefc, nv)
        force = float(data.efc_force[row])
        for index, value in zip(indices, values):
            generalized[index] += float(value) * force
        row_records.append(
            {
                "efc_row": int(row),
                "efc_type": int(data.efc_type[row]),
                "efc_id": int(data.efc_id[row]),
                "efc_force": force,
                "J_row": {"dof_indices": indices, "values": values},
            }
        )
    return row_records, generalized


def _native_model_data(sim: Any) -> tuple[Any, Any, Any]:
    from metamorph.utils import mujoco_compat as mjc

    raw_sim = getattr(sim, "_sim", sim)
    raw_model = getattr(raw_sim, "_model", None)
    raw_data = getattr(raw_sim, "_data", None)
    if mjc.BACKEND != "mujoco" or raw_model is None or raw_data is None:
        raise RuntimeError(
            "contact response oracle requires the modern native MuJoCo backend"
        )
    return mjc.mujoco, raw_model, raw_data


def make_body_kinematics_snapshot_data(sim: Any) -> Any:
    mujoco, model, _ = _native_model_data(sim)
    return mujoco.MjData(model)


def capture_native_body_states(
    sim: Any,
    bodies: Sequence[dict[str, Any]],
    tracker: FiniteTracker,
    snapshot_data: Any,
) -> dict[str, dict[str, Any]]:
    """Derive synchronized Cartesian state from live generalized state on isolated mjData."""
    import numpy as np

    mujoco, model, live_data = _native_model_data(sim)
    snapshot_data.qpos[:] = live_data.qpos
    snapshot_data.qvel[:] = live_data.qvel
    if snapshot_data.act.size:
        snapshot_data.act[:] = live_data.act
    if snapshot_data.mocap_pos.size:
        snapshot_data.mocap_pos[:] = live_data.mocap_pos
        snapshot_data.mocap_quat[:] = live_data.mocap_quat
    snapshot_data.time = live_data.time
    mujoco.mj_forward(model, snapshot_data)
    result = {}
    for item in bodies:
        body_id = int(item["body_id"])
        spatial_velocity = np.zeros(6, dtype=np.float64)
        mujoco.mj_objectVelocity(
            model,
            snapshot_data,
            mujoco.mjtObj.mjOBJ_BODY,
            body_id,
            spatial_velocity,
            0,
        )
        result[item["body_name"]] = {
            "xpos": tracker.vector(snapshot_data.xpos[body_id]),
            "xquat_wxyz": tracker.vector(snapshot_data.xquat[body_id]),
            "linear_velocity_world_at_body_origin": tracker.vector(
                spatial_velocity[3:6]
            ),
            "angular_velocity_world": tracker.vector(spatial_velocity[0:3]),
        }
    return result


def contact_frame_to_world(frame_raw: Any, vector_contact: Any) -> Any:
    """Transform a vector using mjContact.frame's world-axis rows."""
    import numpy as np

    frame = np.asarray(frame_raw, dtype=np.float64).reshape(3, 3)
    return frame.T @ np.asarray(vector_contact, dtype=np.float64)


def contact_frame_validation(frame_raw: Any) -> dict[str, Any]:
    import numpy as np

    frame = np.asarray(frame_raw, dtype=np.float64).reshape(3, 3)
    identity_error = frame @ frame.T - np.eye(3, dtype=np.float64)
    return {
        "raw_rows": frame.reshape(-1).tolist(),
        "normal_world": frame[0].tolist(),
        "tangent1_world": frame[1].tolist(),
        "tangent2_world": frame[2].tolist(),
        "orthonormality_max_abs_error": float(np.max(np.abs(identity_error))),
        "determinant": float(np.linalg.det(frame)),
        "right_handed_determinant_error": float(abs(np.linalg.det(frame) - 1.0)),
    }


def vector_comparison(candidate: Any, reference: Any) -> dict[str, Any]:
    import numpy as np

    candidate = np.asarray(candidate, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    delta = candidate - reference
    candidate_norm = float(np.linalg.norm(candidate))
    reference_norm = float(np.linalg.norm(reference))
    denominator = candidate_norm * reference_norm
    return {
        "max_abs_error": float(np.max(np.abs(delta))) if delta.size else 0.0,
        "rms_error": float(np.sqrt(np.mean(delta * delta))) if delta.size else 0.0,
        "candidate_norm": candidate_norm,
        "reference_norm": reference_norm,
        "norm_ratio": candidate_norm / reference_norm if reference_norm else None,
        "cosine_similarity": (
            float(np.dot(candidate, reference) / denominator)
            if denominator
            else None
        ),
    }


def _apply_ft_scratch(
    mujoco: Any,
    model: Any,
    data: Any,
    force_world: Any,
    torque_world: Any,
    point_world: Any,
    body_id: int,
) -> Any:
    import numpy as np

    target = np.zeros(int(model.nv), dtype=np.float64)
    if int(body_id) != 0:
        mujoco.mj_applyFT(
            model,
            data,
            np.asarray(force_world, dtype=np.float64),
            np.asarray(torque_world, dtype=np.float64),
            np.asarray(point_world, dtype=np.float64),
            int(body_id),
            target,
        )
    return target


def apply_contact_pair_scratch(
    mujoco: Any,
    model: Any,
    data: Any,
    force_world_on_geom2: Any,
    torque_world_on_geom2: Any,
    point_world: Any,
    body1: int,
    body2: int,
) -> Any:
    """Apply equal/opposite contact wrench to scratch qfrc, never live qfrc_applied."""
    result = _apply_ft_scratch(
        mujoco, model, data, force_world_on_geom2, torque_world_on_geom2,
        point_world, body2,
    )
    result += _apply_ft_scratch(
        mujoco, model, data, -force_world_on_geom2, -torque_world_on_geom2,
        point_world, body1,
    )
    return result


def physical_contact_projection(
    mujoco: Any,
    model: Any,
    data: Any,
    contact: Any,
    wrench_contact: Any,
    constraint_generalized: Any,
    body1: int,
    body2: int,
    selected_dofs: dict[str, int],
) -> dict[str, Any]:
    """Independent physical-wrench projection and full-nv sign validation."""
    import numpy as np

    frame = np.asarray(contact.frame, dtype=np.float64).reshape(3, 3)
    wrench = np.asarray(wrench_contact, dtype=np.float64)
    point = np.asarray(contact.pos, dtype=np.float64)
    force_cf_normal = np.asarray([wrench[0], 0.0, 0.0])
    force_cf_friction = np.asarray([0.0, wrench[1], wrench[2]])
    torque_cf_zero = np.zeros(3, dtype=np.float64)
    force_world_normal = contact_frame_to_world(frame, force_cf_normal)
    force_world_friction = contact_frame_to_world(frame, force_cf_friction)
    force_world_total = contact_frame_to_world(frame, wrench[:3])
    torque_world_total = contact_frame_to_world(frame, wrench[3:])
    qfrc_before = np.asarray(data.qfrc_applied, dtype=np.float64).copy()
    sign_candidates = {}
    candidate_vectors = {}
    for sign in (1, -1):
        projected = apply_contact_pair_scratch(
            mujoco, model, data,
            sign * force_world_total, sign * torque_world_total,
            point, body1, body2,
        )
        candidate_vectors[sign] = projected
        sign_candidates[str(sign)] = vector_comparison(
            projected, constraint_generalized
        )
    selected_sign = min(
        (1, -1), key=lambda sign: sign_candidates[str(sign)]["max_abs_error"]
    )
    selected_error = sign_candidates[str(selected_sign)]["max_abs_error"]
    other_error = sign_candidates[str(-selected_sign)]["max_abs_error"]
    reference_scale = max(1.0, float(np.max(np.abs(constraint_generalized))))
    strict_tolerance = 1.0e-8 * reference_scale + 1.0e-10
    sign_valid = bool(
        selected_error <= strict_tolerance and other_error > strict_tolerance
    )
    sign = selected_sign
    qfrc_normal = apply_contact_pair_scratch(
        mujoco, model, data, sign * force_world_normal, sign * torque_cf_zero,
        point, body1, body2,
    )
    qfrc_friction = apply_contact_pair_scratch(
        mujoco, model, data, sign * force_world_friction, sign * torque_cf_zero,
        point, body1, body2,
    )
    qfrc_total = candidate_vectors[sign]
    component_delta = qfrc_normal + qfrc_friction - qfrc_total
    torque_norm = float(np.linalg.norm(wrench[3:]))
    component_error = float(np.max(np.abs(component_delta)))
    qfrc_unchanged = bool(np.array_equal(qfrc_before, data.qfrc_applied))

    # For a robot-floor contact, report point-force geometry on the robot side.
    robot_body = body2 if body2 != 0 and body1 == 0 else body1
    robot_side_factor = sign if robot_body == body2 else -sign
    unit_normal_world = robot_side_factor * frame[0]
    qfrc_unit_normal = _apply_ft_scratch(
        mujoco, model, data, unit_normal_world, torque_cf_zero, point, robot_body
    )
    prescribed_unit_normal = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
    prescribed_qfrc_unit_normal = _apply_ft_scratch(
        mujoco, model, data, prescribed_unit_normal, torque_cf_zero, point,
        robot_body,
    )
    joint_contact_geometry = {}
    for name, dof in selected_dofs.items():
        joint_id = int(model.dof_jntid[dof])
        anchor = np.asarray(data.xanchor[joint_id], dtype=np.float64)
        axis = np.asarray(data.xaxis[joint_id], dtype=np.float64)
        lever_arm = point - anchor
        moment_arm = np.cross(lever_arm, prescribed_unit_normal)
        kappa_analytic = float(np.dot(axis, moment_arm))
        kappa_mj_apply_ft = float(prescribed_qfrc_unit_normal[dof])
        difference = kappa_analytic - kappa_mj_apply_ft
        scale = max(1.0, abs(kappa_analytic), abs(kappa_mj_apply_ft))
        tolerance = 1.0e-12 * scale + 1.0e-14
        joint_contact_geometry[name] = {
            "joint_id": joint_id,
            "qpos_address": int(model.jnt_qposadr[joint_id]),
            "qvel_dof_address": int(model.jnt_dofadr[joint_id]),
            "qpos": float(data.qpos[int(model.jnt_qposadr[joint_id])]),
            "qvel": float(data.qvel[int(model.jnt_dofadr[joint_id])]),
            "joint_anchor_world": anchor.tolist(),
            "joint_axis_world": axis.tolist(),
            "contact_point_world": point.tolist(),
            "lever_arm_world": lever_arm.tolist(),
            "unit_normal_world": prescribed_unit_normal.tolist(),
            "lever_arm_cross_unit_normal_world": moment_arm.tolist(),
            "axis_norm": float(np.linalg.norm(axis)),
            "axis_dot_lever_arm": float(np.dot(axis, lever_arm)),
            "axis_dot_unit_normal": float(np.dot(axis, prescribed_unit_normal)),
            "axis_dot_r_cross_unit_normal": kappa_analytic,
            "kappa_analytic": kappa_analytic,
            "kappa_mj_applyFT": kappa_mj_apply_ft,
            "difference": difference,
            "abs_difference": abs(difference),
            "numerical_tolerance": tolerance,
            "within_numerical_tolerance": bool(abs(difference) <= tolerance),
        }
    friction_robot_world = robot_side_factor * force_world_friction
    friction_norm = float(np.linalg.norm(friction_robot_world))
    qfrc_unit_friction = None
    unit_friction_world = None
    if friction_norm > 1.0e-12:
        unit_friction_world = friction_robot_world / friction_norm
        qfrc_unit_friction = _apply_ft_scratch(
            mujoco, model, data, unit_friction_world, torque_cf_zero,
            point, robot_body,
        )
    selected = {
        name: {
            "normal": float(qfrc_normal[dof]),
            "friction": float(qfrc_friction[dof]),
            "total": float(qfrc_total[dof]),
            "constraint_rows_total": float(constraint_generalized[dof]),
            "kappa_normal": float(qfrc_unit_normal[dof]),
            "kappa_friction_direction": (
                float(qfrc_unit_friction[dof])
                if qfrc_unit_friction is not None else None
            ),
        }
        for name, dof in selected_dofs.items()
    }
    return {
        "frame_validation": contact_frame_validation(frame),
        "force_contact_frame": wrench[:3].tolist(),
        "torque_contact_frame": wrench[3:].tolist(),
        "Fn": float(wrench[0]),
        "Ft1": float(wrench[1]),
        "Ft2": float(wrench[2]),
        "friction_force_norm": float(np.linalg.norm(wrench[1:3])),
        "contact_torque_norm": torque_norm,
        "contact_frame_force_component_closure_max_abs_error": float(
            np.max(np.abs(force_cf_normal + force_cf_friction - wrench[:3]))
        ),
        "world_frame_force_component_closure_max_abs_error": float(
            np.max(np.abs(force_world_normal + force_world_friction - force_world_total))
        ),
        "unexpected_contact_torque": bool(int(contact.dim) == 3 and torque_norm > 1.0e-10),
        "normal_force_world_unsigned": force_world_normal.tolist(),
        "friction_force_world_unsigned": force_world_friction.tolist(),
        "total_force_world_unsigned": force_world_total.tolist(),
        "normal_torque_world_unsigned": torque_cf_zero.tolist(),
        "friction_torque_world_unsigned": torque_cf_zero.tolist(),
        "total_torque_world_unsigned": torque_world_total.tolist(),
        "robot_side_sign": int(sign),
        "normal_force_world_on_body2": (sign * force_world_normal).tolist(),
        "friction_force_world_on_body2": (sign * force_world_friction).tolist(),
        "total_force_world_on_body2": (sign * force_world_total).tolist(),
        "normal_torque_world_on_body2": torque_cf_zero.tolist(),
        "friction_torque_world_on_body2": torque_cf_zero.tolist(),
        "total_torque_world_on_body2": (sign * torque_world_total).tolist(),
        "normal_force_world_on_body1": (-sign * force_world_normal).tolist(),
        "friction_force_world_on_body1": (-sign * force_world_friction).tolist(),
        "total_force_world_on_body1": (-sign * force_world_total).tolist(),
        "sign_candidates": sign_candidates,
        "sign_strict_tolerance": strict_tolerance,
        "physical_wrench_sign_valid": sign_valid,
        "qfrc_normal": qfrc_normal.tolist(),
        "qfrc_friction": qfrc_friction.tolist(),
        "qfrc_total": qfrc_total.tolist(),
        "qfrc_constraint_rows_contact": np.asarray(constraint_generalized).tolist(),
        "component_reconstruction_max_abs_error": component_error,
        "physical_component_reconstruction_valid": bool(
            component_error <= strict_tolerance and torque_norm <= 1.0e-10
        ),
        "total_vs_constraint": vector_comparison(
            qfrc_total, constraint_generalized
        ),
        "qfrc_applied_unchanged": qfrc_unchanged,
        "robot_body_id_for_unit_projection": int(robot_body),
        "unit_normal_world_robot_side": unit_normal_world.tolist(),
        "unit_friction_world_robot_side": (
            unit_friction_world.tolist() if unit_friction_world is not None else None
        ),
        "qfrc_per_unit_normal_force": qfrc_unit_normal.tolist(),
        "prescribed_unit_normal_world": prescribed_unit_normal.tolist(),
        "qfrc_per_prescribed_unit_normal_force": (
            prescribed_qfrc_unit_normal.tolist()
        ),
        "joint_contact_geometry": joint_contact_geometry,
        "qfrc_per_unit_friction_direction": (
            qfrc_unit_friction.tolist() if qfrc_unit_friction is not None else None
        ),
        "selected_joints": selected,
    }


def capture_contact_response(
    sim: Any,
    mapping: dict[str, Any],
    tracker: FiniteTracker,
    record_physical_projection: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    """Read mjContact, mj_contactForce and each contact's EFC contribution."""
    import numpy as np

    mujoco, raw_model, raw_data = _native_model_data(sim)
    model, data = sim.model, sim.data
    nefc, nv = int(data.nefc), int(model.nv)
    pyramidal = int(raw_model.opt.cone) == int(mujoco.mjtCone.mjCONE_PYRAMIDAL)
    selected = {
        item["joint_name"]: int(item["dof_address"])
        for item in mapping["joints"]
    }
    total_selected = {name: 0.0 for name in selected}
    contacts = []
    for contact_index in range(int(data.ncon)):
        contact = raw_data.contact[contact_index]
        geom1, geom2 = int(contact.geom1), int(contact.geom2)
        body1 = int(raw_model.geom_bodyid[geom1])
        body2 = int(raw_model.geom_bodyid[geom2])
        rows = contact_efc_rows(
            int(contact.efc_address), int(contact.dim), pyramidal, nefc
        )
        row_records, generalized = aggregate_constraint_rows(
            data, rows, nefc, nv
        )
        joint_limit_type = int(
            mujoco.mjtConstraint.mjCNSTR_LIMIT_JOINT
        )
        if any(
            row_record["efc_type"] == joint_limit_type
            for row_record in row_records
        ):
            raise RuntimeError(
                f"contact {contact_index} row range includes a joint-limit row"
            )
        selected_contribution = {
            name: tracker.scalar(generalized[dof])
            for name, dof in selected.items()
        }
        for name, value in selected_contribution.items():
            total_selected[name] += float(value)
        wrench = np.zeros(6, dtype=np.float64)
        mujoco.mj_contactForce(raw_model, raw_data, contact_index, wrench)
        frame = np.asarray(contact.frame, dtype=np.float64).reshape(3, 3)
        world_force_on_geom2 = frame.T @ wrench[0:3]
        world_torque_on_geom2 = frame.T @ wrench[3:6]
        contact_record = {
                "contact_index": contact_index,
                "geom1_id": geom1,
                "geom1_name": mapping["geom_names"][geom1],
                "body1_id": body1,
                "body1_name": mapping["body_names"][body1],
                "geom2_id": geom2,
                "geom2_name": mapping["geom_names"][geom2],
                "body2_id": body2,
                "body2_name": mapping["body_names"][body2],
                "position_world": tracker.vector(contact.pos),
                "contact_frame_world_rows": tracker.vector(frame.reshape(-1)),
                "normal_world_geom1_to_geom2": tracker.vector(frame[0]),
                "dist": tracker.scalar(contact.dist),
                "includemargin": tracker.scalar(contact.includemargin),
                "friction": tracker.vector(contact.friction),
                "solref": tracker.vector(contact.solref),
                "solimp": tracker.vector(contact.solimp),
                "dim": int(contact.dim),
                "efc_address": int(contact.efc_address),
                "efc_rows": rows,
                "constraint_rows": row_records,
                "constraint_rows_exclude_joint_limits": True,
                "contact_frame_wrench_on_geom2": tracker.vector(wrench),
                "normal_force_on_geom2": tracker.scalar(wrench[0]),
                "tangential_force_components_on_geom2": tracker.vector(wrench[1:3]),
                "contact_frame_torque_on_geom2": tracker.vector(wrench[3:6]),
                "world_force_on_geom2": tracker.vector(world_force_on_geom2),
                "world_force_on_geom1": tracker.vector(-world_force_on_geom2),
                "world_torque_on_geom2": tracker.vector(world_torque_on_geom2),
                "world_torque_on_geom1": tracker.vector(-world_torque_on_geom2),
                "selected_dof_generalized_contribution": selected_contribution,
                "is_floor_contact": (
                    geom1 == mapping["floor_geom_id"]
                    or geom2 == mapping["floor_geom_id"]
                ),
            }
        if record_physical_projection:
            contact_record["physical_projection"] = physical_contact_projection(
                mujoco, raw_model, raw_data, contact, wrench,
                np.asarray(generalized, dtype=np.float64), body1, body2, selected,
            )
        contacts.append(contact_record)
    return contacts, total_selected


class JointLimitSubstepRecorder:
    """Observe each formal mj_step result without extra simulation calls."""

    def __init__(
        self,
        sim: Any,
        frame_skip: int,
        mapping: dict[str, Any],
        tracker: FiniteTracker,
        record_contact_response: bool = False,
        record_physical_projection: bool = False,
    ) -> None:
        self.sim = sim
        self.frame_skip = int(frame_skip)
        self.mapping = mapping
        self.tracker = tracker
        self.records: list[dict[str, Any]] = []
        self.control_step: int | None = None
        self.episode = 0
        self.physics_substep = 0
        self.global_physics_step = 0
        self.record_contact_response = bool(record_contact_response)
        self.record_physical_projection = bool(record_physical_projection)
        self.body_snapshot_data = (
            make_body_kinematics_snapshot_data(sim)
            if mapping["contact_probe_bodies"]
            else None
        )

    def begin_control_step(self, episode: int, control_step: int) -> None:
        self.episode = int(episode)
        self.control_step = int(control_step)
        self.physics_substep = 0

    def end_control_step(self) -> None:
        if self.physics_substep != self.frame_skip:
            raise RuntimeError(f"expected {self.frame_skip} physics substeps, observed {self.physics_substep}")

    def _state(self) -> dict[str, Any]:
        data = self.sim.data
        root = self.mapping["root_free_joint"]
        root_body_id = int(self.mapping["root_body_id"])
        torso_body_id = int(self.mapping["torso_body_id"])
        return {
            "root_position": self.tracker.vector(data.body_xpos[root_body_id]),
            "root_quaternion_wxyz": self.tracker.vector(data.body_xquat[root_body_id]),
            "root_linear_velocity": self.tracker.vector(data.body_xvelp[root_body_id]),
            "root_angular_velocity": self.tracker.vector(data.body_xvelr[root_body_id]),
            "root_generalized_qpos": self.tracker.vector(data.qpos[root["qpos_indices"]]),
            "root_generalized_qvel": self.tracker.vector(data.qvel[root["qvel_indices"]]),
            "torso_height": self.tracker.scalar(data.body_xpos[torso_body_id][2]),
        }

    def capture_pre_step(self) -> dict[str, Any]:
        data = self.sim.data
        return {
            "simulation_time": self.tracker.scalar(data.time),
            "root": self._state(),
            "joints": {
                item["joint_name"]: {
                    "qpos": self.tracker.scalar(data.qpos[item["qpos_address"]]),
                    "qvel": self.tracker.scalar(data.qvel[item["dof_address"]]),
                }
                for item in self.mapping["joints"]
            },
            "contact_probe_bodies": (
                capture_native_body_states(
                    self.sim,
                    self.mapping["contact_probe_bodies"],
                    self.tracker,
                    self.body_snapshot_data,
                )
                if self.body_snapshot_data is not None
                else {}
            ),
        }

    def capture_post_step(self, pre: dict[str, Any]) -> None:
        if self.control_step is None:
            raise RuntimeError("physics step occurred outside a control step")
        from metamorph.utils import mujoco_compat as mjc

        data = self.sim.data
        model = self.sim.model
        nefc = int(data.nefc)
        nv = int(model.nv)
        limit_type = int(mjc.mujoco.mjtConstraint.mjCNSTR_LIMIT_JOINT)
        contacts = []
        contact_generalized_totals = {
            item["joint_name"]: 0.0 for item in self.mapping["joints"]
        }
        limb_floor_contact = False
        body_names = self.mapping["body_names"]
        geom_names = self.mapping["geom_names"]
        for contact in data.contact[: int(data.ncon)]:
            geom1, geom2 = int(contact.geom1), int(contact.geom2)
            body1, body2 = int(model.geom_bodyid[geom1]), int(model.geom_bodyid[geom2])
            pair = {
                "geom1_id": geom1, "geom1_name": geom_names[geom1],
                "body1_id": body1, "body1_name": body_names[body1],
                "geom2_id": geom2, "geom2_name": geom_names[geom2],
                "body2_id": body2, "body2_name": body_names[body2],
            }
            contacts.append(pair)
            floor_geom_id = self.mapping["floor_geom_id"]
            limb_floor_contact |= (
                floor_geom_id in (geom1, geom2)
                and "limb/0" in (pair["body1_name"], pair["body2_name"])
            )
        if self.record_contact_response or self.record_physical_projection:
            contacts, contact_generalized_totals = capture_contact_response(
                self.sim,
                self.mapping,
                self.tracker,
                record_physical_projection=self.record_physical_projection,
            )
            floor_geom_id = self.mapping["floor_geom_id"]
            limb_floor_contact = any(
                floor_geom_id in (contact["geom1_id"], contact["geom2_id"])
                and "limb/0" in (contact["body1_name"], contact["body2_name"])
                for contact in contacts
            )

        joint_records = []
        for item in self.mapping["joints"]:
            joint_id, dof = int(item["joint_id"]), int(item["dof_address"])
            rows = [row for row in range(nefc) if int(data.efc_type[row]) == limit_type and int(data.efc_id[row]) == joint_id]
            if len(rows) > 1:
                raise RuntimeError(f"multiple joint-limit rows found for {item['joint_name']}: {rows}")
            if rows:
                row = rows[0]
                indices, values = constraint_jacobian_row(data, row, nefc, nv)
                coefficient = float(values[indices.index(dof)]) if dof in indices else 0.0
                force = float(data.efc_force[row])
                kbip = [float(value) for value in data.efc_KBIP[row]]
                constraint = {
                    "limit_constraint_present": True,
                    "efc_row": row, "efc_type": int(data.efc_type[row]), "efc_id": int(data.efc_id[row]),
                    "efc_pos": self.tracker.scalar(data.efc_pos[row]),
                    "efc_margin": self.tracker.scalar(data.efc_margin[row]),
                    "efc_vel": self.tracker.scalar(data.efc_vel[row]),
                    "efc_aref": self.tracker.scalar(data.efc_aref[row]),
                    "efc_diagApprox": self.tracker.scalar(data.efc_diagApprox[row]),
                    "efc_KBIP": self.tracker.vector(kbip),
                    "efc_KBIP_components": {"K": kbip[0], "B": kbip[1], "impedance": kbip[2], "impedance_derivative": kbip[3]},
                    "efc_D": self.tracker.scalar(data.efc_D[row]),
                    "efc_R": self.tracker.scalar(data.efc_R[row]),
                    "efc_force": self.tracker.scalar(force),
                    "efc_state": int(data.efc_state[row]),
                    "J_row": {"dof_indices": indices, "values": values},
                    "selected_dof_J": self.tracker.scalar(coefficient),
                    "selected_dof_limit_generalized_force": self.tracker.scalar(coefficient * force),
                }
            else:
                constraint = {field: None for field in (
                    "efc_row", "efc_type", "efc_id", "efc_pos", "efc_margin", "efc_vel", "efc_aref",
                    "efc_diagApprox", "efc_KBIP", "efc_KBIP_components", "efc_D", "efc_R", "efc_force",
                    "efc_state", "J_row", "selected_dof_J", "selected_dof_limit_generalized_force",
                )}
                constraint["limit_constraint_present"] = False
            reconstructed = _selected_constraint_reconstruction(data, dof, nefc, nv)
            qfrc_constraint = float(data.qfrc_constraint[dof])
            joint_records.append({
                "joint_name": item["joint_name"], "joint_id": joint_id,
                "qpos_address": int(item["qpos_address"]), "dof_address": dof,
                "pre_step_qpos": pre["joints"][item["joint_name"]]["qpos"],
                "pre_step_qvel": pre["joints"][item["joint_name"]]["qvel"],
                "post_step_qpos": self.tracker.scalar(data.qpos[item["qpos_address"]]),
                "post_step_qvel": self.tracker.scalar(data.qvel[dof]),
                **constraint,
                "qfrc_constraint": self.tracker.scalar(qfrc_constraint),
                "qfrc_constraint_reconstructed_from_JT_efc_force": self.tracker.scalar(reconstructed),
                "qfrc_constraint_reconstruction_error": self.tracker.scalar(reconstructed - qfrc_constraint),
                "qfrc_passive": self.tracker.scalar(data.qfrc_passive[dof]),
                "qfrc_actuator": self.tracker.scalar(data.qfrc_actuator[dof]),
                "qfrc_applied": self.tracker.scalar(data.qfrc_applied[dof]),
                "qfrc_bias": self.tracker.scalar(data.qfrc_bias[dof]),
                "qacc_smooth": self.tracker.scalar(data.qacc_smooth[dof]),
            })
        self.global_physics_step += 1
        selected_constraint_force = {
            item["joint_name"]: self.tracker.scalar(
                data.qfrc_constraint[item["dof_address"]]
            )
            for item in self.mapping["joints"]
        }
        self.records.append({
            "episode": self.episode, "control_step": self.control_step,
            "physics_substep_in_control": self.physics_substep,
            "global_physics_step": self.global_physics_step,
            "pre_step_simulation_time": pre["simulation_time"],
            "simulation_time": self.tracker.scalar(data.time),
            "pre_step_root": pre["root"], "post_step_root": self._state(),
            "pre_step_contact_probe_bodies": pre["contact_probe_bodies"],
            "post_step_contact_probe_bodies": (
                capture_native_body_states(
                    self.sim,
                    self.mapping["contact_probe_bodies"],
                    self.tracker,
                    self.body_snapshot_data,
                )
                if self.body_snapshot_data is not None
                else {}
            ),
            "joints": joint_records, "nefc": nefc, "ncon": int(data.ncon),
            "contacts": contacts, "contains_limb_0_floor_0_contact": limb_floor_contact,
            "physical_contact_projection_enabled": self.record_physical_projection,
            "sum_contact_generalized_force_selected_dofs": {
                name: self.tracker.scalar(value)
                for name, value in contact_generalized_totals.items()
            },
            "qfrc_constraint_selected_dofs": selected_constraint_force,
            "contact_vs_qfrc_constraint_reconstruction_error": {
                name: self.tracker.scalar(
                    contact_generalized_totals[name]
                    - selected_constraint_force[name]
                )
                for name in selected_constraint_force
            },
        })
        self.physics_substep += 1


class JointLimitRecordingSimProxy:
    def __init__(self, sim: Any, recorder: JointLimitSubstepRecorder) -> None:
        self._sim, self._recorder = sim, recorder
        self.callback_error: Exception | None = None

    def __getattr__(self, name: str) -> Any:
        return getattr(self._sim, name)

    def step(self) -> Any:
        pre = None
        if self.callback_error is None:
            try:
                pre = self._recorder.capture_pre_step()
            except Exception as error:
                self.callback_error = error
        result = self._sim.step()
        if self.callback_error is None:
            try:
                self._recorder.capture_post_step(pre)
            except Exception as error:
                self.callback_error = error
        return result


def build_joint_limit_oracle_outputs(
    records: Sequence[dict[str, Any]],
    mapping: dict[str, Any],
    expected_control_steps: int | None,
    frame_skip: int,
) -> dict[str, Any]:
    joint_names = [item["joint_name"] for item in mapping["joints"]]
    first_contact = next(
        (record for record in records if record["contains_limb_0_floor_0_contact"]),
        None,
    )
    joint_summary = {}
    finite_limit_forces = True
    max_reconstruction_error = 0.0
    for joint_name in joint_names:
        samples = [
            (record, next(item for item in record["joints"] if item["joint_name"] == joint_name))
            for record in records
        ]
        mapping_item = next(item for item in mapping["joints"] if item["joint_name"] == joint_name)
        upper = float(mapping_item["jnt_range"][1])
        first_constraint = next((pair for pair in samples if pair[1]["limit_constraint_present"]), None)
        first_penetration = next((pair for pair in samples if pair[1]["post_step_qpos"] > upper), None)
        first_force = next(
            (
                pair for pair in samples
                if pair[1]["selected_dof_limit_generalized_force"] is not None
                and pair[1]["selected_dof_limit_generalized_force"] != 0.0
            ),
            None,
        )
        force_samples = [
            pair for pair in samples
            if pair[1]["selected_dof_limit_generalized_force"] is not None
        ]
        for _, item in force_samples:
            finite_limit_forces &= math.isfinite(item["selected_dof_limit_generalized_force"])
        max_force = max(
            force_samples,
            key=lambda pair: abs(pair[1]["selected_dof_limit_generalized_force"]),
            default=None,
        )
        max_penetration = max(
            samples,
            key=lambda pair: max(0.0, pair[1]["post_step_qpos"] - upper),
            default=None,
        )
        for _, item in samples:
            error = item["qfrc_constraint_reconstruction_error"]
            if error is not None and math.isfinite(error):
                max_reconstruction_error = max(max_reconstruction_error, abs(error))

        def location(pair: Any) -> dict[str, Any] | None:
            if pair is None:
                return None
            record, item = pair
            return {
                "control_step": record["control_step"],
                "physics_substep_in_control": record["physics_substep_in_control"],
                "global_physics_step": record["global_physics_step"],
                "post_step_qpos": item["post_step_qpos"],
                "efc_force": item.get("efc_force"),
                "selected_dof_limit_generalized_force": item["selected_dof_limit_generalized_force"],
            }

        joint_summary[joint_name] = {
            "first_limit_constraint_control_step": first_constraint[0]["control_step"] if first_constraint else None,
            "first_limit_constraint_substep": first_constraint[0]["physics_substep_in_control"] if first_constraint else None,
            "first_limit_constraint": location(first_constraint),
            "first_positive_source_limit_penetration": location(first_penetration),
            "first_nonzero_limit_force": location(first_force),
            "max_limit_force_steps_1_30": location(max_force),
            "max_penetration_steps_1_30": location(max_penetration),
            "max_positive_penetration_radians": (
                max(0.0, max_penetration[1]["post_step_qpos"] - upper)
                if max_penetration is not None else None
            ),
        }
    detailed_indices = [
        {
            "record_index": index,
            "control_step": record["control_step"],
            "physics_substep_in_control": record["physics_substep_in_control"],
            "global_physics_step": record["global_physics_step"],
        }
        for index, record in enumerate(records)
        if 8 <= int(record["control_step"]) <= 15
    ]
    expected_count = (
        int(expected_control_steps) * int(frame_skip)
        if expected_control_steps is not None else None
    )
    summary = {
        "schema_version": "spikmorph-mujoco-joint-limit-substep-summary-v1",
        "first_ground_contact_control_step": first_contact["control_step"] if first_contact else None,
        "first_ground_contact_substep": first_contact["physics_substep_in_control"] if first_contact else None,
        "first_ground_contact_global_physics_step": first_contact["global_physics_step"] if first_contact else None,
        "joints": joint_summary,
        "control_steps_8_through_15_record_indices": detailed_indices,
    }
    validation = {
        "schema_version": "spikmorph-mujoco-joint-limit-substep-validation-v1",
        "expected_record_count": expected_count,
        "actual_record_count": len(records),
        "record_count_matches": expected_count is None or len(records) == expected_count,
        "physics_substeps_per_control": int(frame_skip),
        "all_selected_dof_limit_generalized_forces_finite": finite_limit_forces,
        "max_abs_qfrc_constraint_reconstruction_error": max_reconstruction_error,
        "JT_efc_force_reconstruction_sign_and_value_match": max_reconstruction_error <= 1e-7,
        "limit_constraint_presence_steps_1_through_8": {
            joint_name: sum(
                1
                for record in records
                if int(record["control_step"]) <= 8
                for item in record["joints"]
                if item["joint_name"] == joint_name
                and item["limit_constraint_present"]
            )
            for joint_name in joint_names
        },
        "default_mode_note": "no proxy is installed unless --record-joint-limit-substeps is enabled",
    }
    contact_summary = (
        build_contact_generalized_response_summary(records, mapping)
        if mapping.get("contact_probe_bodies")
        else None
    )
    if contact_summary is not None:
        validation["global_55_contact_reconstruction"] = contact_summary.get(
            "global_55_contact_reconstruction"
        )
    physical_outputs = build_physical_contact_projection_outputs(records, mapping)
    if physical_outputs is not None:
        validation.update(physical_outputs["validation"])
    return {
        "records": list(records),
        "mapping": mapping,
        "summary": summary,
        "validation": validation,
        "contact_summary": contact_summary,
        "physical_outputs": physical_outputs,
    }


def build_contact_generalized_response_summary(
    records: Sequence[dict[str, Any]], mapping: dict[str, Any]
) -> dict[str, Any]:
    floor_geom_id = mapping["floor_geom_id"]
    selected_names = [item["joint_name"] for item in mapping["joints"]]
    window = {}
    for global_step in (54, 55, 56):
        record = next(
            (
                candidate for candidate in records
                if int(candidate["global_physics_step"]) == global_step
            ),
            None,
        )
        if record is None:
            window[str(global_step)] = None
            continue
        focus_contacts = []
        for contact in record["contacts"]:
            if (
                floor_geom_id in (contact["geom1_id"], contact["geom2_id"])
                and any(
                    body in (contact["body1_name"], contact["body2_name"])
                    for body in ("limb/12", "limb/11")
                )
            ):
                focus_contacts.append(contact)
        window[str(global_step)] = {
            "control_step": record["control_step"],
            "physics_substep_in_control": record["physics_substep_in_control"],
            "simulation_time": record["simulation_time"],
            "pre_step_contact_probe_bodies": record[
                "pre_step_contact_probe_bodies"
            ],
            "post_step_contact_probe_bodies": record[
                "post_step_contact_probe_bodies"
            ],
            "focus_limb_12_limb_11_floor_contacts": focus_contacts,
            "all_contacts": record["contacts"],
            "sum_contact_generalized_force_selected_dofs": record[
                "sum_contact_generalized_force_selected_dofs"
            ],
            "qfrc_constraint_selected_dofs": record[
                "qfrc_constraint_selected_dofs"
            ],
            "contact_vs_qfrc_constraint_reconstruction_error": record[
                "contact_vs_qfrc_constraint_reconstruction_error"
            ],
        }
    global_55 = window.get("55")
    reconstruction = None
    if global_55 is not None:
        reconstruction = {
            name: {
                "sum_contact_generalized_force": global_55[
                    "sum_contact_generalized_force_selected_dofs"
                ][name],
                "qfrc_constraint": global_55[
                    "qfrc_constraint_selected_dofs"
                ][name],
                "error": global_55[
                    "contact_vs_qfrc_constraint_reconstruction_error"
                ][name],
            }
            for name in selected_names
        }
    return {
        "schema_version": "spikmorph-mujoco-contact-generalized-response-v1",
        "focus_global_physics_steps": [54, 55, 56],
        "floor_identity": {
            "geom_id": floor_geom_id,
            "geom_name": mapping["floor_geom_name"],
            "body_name_is_not_used_for_floor_detection": True,
        },
        "velocity_source": "mujoco.mj_objectVelocity on isolated mjData synchronized from each pre/post live generalized qpos/qvel",
        "velocity_frame": "world-oriented; linear component evaluated at body origin",
        "body_kinematics_purity": "mj_forward is called only on isolated MjData; live mjData and formal stepping are not modified",
        "contact_frame_convention": "rows are world-frame contact axes; row 0 is normal from geom1 toward geom2",
        "contact_wrench_convention": "mj_contactForce raw 6-vector is force/torque on geom2 by geom1 in the contact frame; equal-and-opposite world vectors are also recorded for geom1",
        "constraint_reconstruction": "sum each contact's dynamic rows from contact.efc_address using current cone type and contact.dim; accumulate J_row^T * efc_force",
        "window": window,
        "global_55_contact_reconstruction": reconstruction,
    }


def build_physical_contact_projection_outputs(
    records: Sequence[dict[str, Any]], mapping: dict[str, Any]
) -> dict[str, Any] | None:
    physical_records = []
    frame_checks = []
    comparisons = []
    for record in records:
        if not record.get("physical_contact_projection_enabled", False):
            continue
        projected_contacts = [
            contact for contact in record.get("contacts", [])
            if "physical_projection" in contact
        ]
        physical_records.append({
            "control_step": record["control_step"],
            "physics_substep_in_control": record["physics_substep_in_control"],
            "global_physics_step": record["global_physics_step"],
            "simulation_time": record["simulation_time"],
            "contacts": projected_contacts,
        })
        for contact in projected_contacts:
            projection = contact["physical_projection"]
            identity = {
                "global_physics_step": record["global_physics_step"],
                "contact_index": contact["contact_index"],
                "geom1_name": contact["geom1_name"],
                "geom2_name": contact["geom2_name"],
            }
            frame_checks.append({**identity, **projection["frame_validation"]})
            comparisons.append({
                **identity,
                "robot_side_sign": projection["robot_side_sign"],
                "physical_wrench_sign_valid": projection[
                    "physical_wrench_sign_valid"
                ],
                "sign_candidates": projection["sign_candidates"],
                "total_vs_constraint": projection["total_vs_constraint"],
                "component_reconstruction_max_abs_error": projection[
                    "component_reconstruction_max_abs_error"
                ],
                "physical_component_reconstruction_valid": projection[
                    "physical_component_reconstruction_valid"
                ],
                "qfrc_applied_unchanged": projection["qfrc_applied_unchanged"],
            })
    if not physical_records:
        return None

    floor_geom_id = mapping["floor_geom_id"]
    global_55 = next(
        (record for record in physical_records if record["global_physics_step"] == 55),
        None,
    )
    selected = {}
    unit_projection = {}
    joint_contact_geometry = {}
    frozen_old = {
        "limby/12": {
            "normal": 332.635590617929,
            "friction": -240.195861797270,
            "total": 92.439728820659,
        },
        "limby/11": {
            "normal": 332.635590617929,
            "friction": -240.195861797270,
            "total": 92.439728820659,
        },
    }
    if global_55 is not None:
        for joint_name, body_name in (("limby/12", "limb/12"), ("limby/11", "limb/11")):
            contact = next(
                (
                    item for item in global_55["contacts"]
                    if floor_geom_id in (item["geom1_id"], item["geom2_id"])
                    and body_name in (item["body1_name"], item["body2_name"])
                ),
                None,
            )
            if contact is None:
                selected[joint_name] = None
                unit_projection[joint_name] = None
                continue
            projection = contact["physical_projection"]
            values = projection["selected_joints"][joint_name]
            old = frozen_old[joint_name]
            selected[joint_name] = {
                "contact_index": contact["contact_index"],
                "geom1_name": contact["geom1_name"],
                "geom2_name": contact["geom2_name"],
                "physical": {
                    key: values[key] for key in ("normal", "friction", "total")
                },
                "frozen_previous_constraint_row_decomposition": old,
                "difference_physical_minus_old": {
                    key: values[key] - old[key]
                    for key in ("normal", "friction", "total")
                },
                "oracle_gates": {
                    "physical_wrench_sign_valid": projection[
                        "physical_wrench_sign_valid"
                    ],
                    "physical_component_reconstruction_valid": projection[
                        "physical_component_reconstruction_valid"
                    ],
                    "total_vs_constraint": projection["total_vs_constraint"],
                },
            }
            unit_projection[joint_name] = {
                "contact_index": contact["contact_index"],
                "robot_body_id": projection["robot_body_id_for_unit_projection"],
                "contact_position_world": contact["position_world"],
                "unit_normal_world_robot_side": projection[
                    "unit_normal_world_robot_side"
                ],
                "unit_friction_world_robot_side": projection[
                    "unit_friction_world_robot_side"
                ],
                "qfrc_per_unit_normal_force": projection[
                    "qfrc_per_unit_normal_force"
                ],
                "qfrc_per_unit_friction_direction": projection[
                    "qfrc_per_unit_friction_direction"
                ],
                "kappa_normal": values["kappa_normal"],
                "kappa_friction_direction": values[
                    "kappa_friction_direction"
                ],
                "interpretation": "point-force geometry/Jacobian coefficient, not a solver coefficient",
            }
            geometry = projection["joint_contact_geometry"][joint_name]
            joint_contact_geometry[joint_name] = {
                "global_physics_step": 55,
                "contact_index": contact["contact_index"],
                "geom1_name": contact["geom1_name"],
                "geom2_name": contact["geom2_name"],
                "robot_body_id": projection[
                    "robot_body_id_for_unit_projection"
                ],
                "runtime_fields": {
                    "joint_anchor_world": "data.xanchor[joint_id]",
                    "joint_axis_world": "data.xaxis[joint_id]",
                    "contact_point_world": "mjContact.pos",
                },
                **geometry,
            }

    validation_comparisons = [
        item for item in comparisons
        if item["global_physics_step"] == 55
        and item["total_vs_constraint"]["reference_norm"] > 1.0e-12
    ]
    all_sign_valid = bool(validation_comparisons) and all(
        item["physical_wrench_sign_valid"] for item in validation_comparisons
    )
    all_total_valid = bool(validation_comparisons) and all(
        item["total_vs_constraint"]["max_abs_error"]
        <= 1.0e-8 * max(1.0, item["total_vs_constraint"]["reference_norm"])
        + 1.0e-10
        for item in validation_comparisons
    )
    all_components_valid = bool(validation_comparisons) and all(
        item["physical_component_reconstruction_valid"]
        for item in validation_comparisons
    )
    all_scratch_unchanged = bool(validation_comparisons) and all(
        item["qfrc_applied_unchanged"] for item in validation_comparisons
    )
    selected_available = all(selected.get(name) is not None for name in frozen_old)
    selected_match = selected_available and all(
        max(abs(value) for value in selected[name]["difference_physical_minus_old"].values())
        <= 1.0e-6
        for name in frozen_old
    )
    oracle_valid = all_sign_valid and all_total_valid and all_scratch_unchanged
    geometry_available = all(
        joint_contact_geometry.get(name) is not None for name in frozen_old
    )
    geometry_valid = geometry_available and all(
        item["within_numerical_tolerance"]
        and abs(item["axis_norm"] - 1.0) <= 1.0e-12
        and item["unit_normal_world"] == [0.0, 0.0, 1.0]
        for item in joint_contact_geometry.values()
    )
    joint_geometry_verdict = (
        "READY" if oracle_valid and geometry_valid else "INSUFFICIENT_EVIDENCE"
    )
    if oracle_valid and all_components_valid and selected_match:
        verdict = "CONFIRMED"
    elif oracle_valid and all_components_valid and selected_available:
        verdict = "REFUTED"
    else:
        verdict = "INSUFFICIENT_EVIDENCE"
    return {
        "records": physical_records,
        "contact_frame_validation": {
            "frame_storage": "three world-frame axes stored as rows: normal, tangent1, tangent2",
            "checks": frame_checks,
        },
        "physical_vs_constraint_generalized": {
            "comparison_scope": "full nv vector for each contact",
            "comparisons": comparisons,
        },
        "selected_joint_physical_decomposition": {
            "global_physics_step": 55,
            "old_reference_provenance": "task-frozen previous pyramidal constraint-row decomposition; limby/11 specified symmetric to limby/12",
            "selected": selected,
            "mujoco_component_decomposition": verdict,
        },
        "unit_force_projection": {
            "global_physics_step": 55,
            "selected": unit_projection,
        },
        "joint_contact_geometry": {
            "schema_version": "spikmorph-mujoco-joint-contact-geometry-v1",
            "global_physics_step": 55,
            "formula": "joint_axis_world dot (lever_arm_world cross unit_normal_world)",
            "generalized_coordinate_sign": "MuJoCo runtime data.xaxis direction; no fitted sign",
            "extra_mj_step_calls": 0,
            "extra_mj_forward_calls": 0,
            "selected": joint_contact_geometry,
            "MUJOCO_JOINT_CONTACT_GEOMETRY": joint_geometry_verdict,
        },
        "validation": {
            "physical_wrench_sign_valid": all_sign_valid,
            "physical_wrench_oracle_valid": oracle_valid,
            "physical_component_reconstruction_valid": all_components_valid,
            "qfrc_applied_unchanged": all_scratch_unchanged,
            "selected_joint_old_component_match": bool(selected_match),
            "mujoco_component_decomposition": verdict,
            "extra_mj_step_calls": 0,
            "extra_mj_forward_calls": 0,
            "joint_contact_geometry_available": geometry_available,
            "joint_contact_geometry_within_numerical_tolerance": geometry_valid,
            "MUJOCO_JOINT_CONTACT_GEOMETRY": joint_geometry_verdict,
        },
    }


def configure_runtime(args: argparse.Namespace, paths: dict[str, Path]) -> None:
    from metamorph.config import cfg
    from tools.train_ppo import calculate_max_limbs_joints

    cfg.merge_from_file(str(paths["config"]))
    cfg.ENV.WALKER_DIR = str(paths["walker_dir"])
    cfg.ENV.WALKERS = [args.morphology_id]
    cfg.RNG_SEED = int(args.seed)
    cfg.DEVICE = str(args.device)
    cfg.PPO.NUM_ENVS = 1
    cfg.VECENV.TYPE = "DummyVecEnv"
    cfg.PPO.CHECKPOINT_PATH = str(paths["checkpoint"])
    reset_noise_scale = getattr(args, "reset_noise_scale", None)
    if reset_noise_scale is not None:
        cfg.ENV.RESET_NOISE_SCALE = float(reset_noise_scale)
    calculate_max_limbs_joints()


def build_runtime(args: argparse.Namespace, paths: dict[str, Path]):
    import torch

    from metamorph.algos.ppo.envs import (
        get_vec_normalize,
        make_vec_envs,
        set_ob_rms,
    )
    from metamorph.algos.ppo.inherit_weight import restore_from_checkpoint
    from metamorph.algos.ppo.model import ActorCritic
    from metamorph.config import cfg
    from metamorph.utils import sample as su

    su.set_seed(args.seed)
    torch.set_num_threads(1)
    envs = make_vec_envs(
        training=False,
        norm_rew=False,
        num_env=1,
        seed=args.seed,
    )
    model = ActorCritic(envs.observation_space, envs.action_space)
    ob_rms = restore_from_checkpoint(
        model, map_location=torch.device(args.device)
    )
    if ob_rms is None:
        envs.close()
        raise ValueError("checkpoint does not contain observation RMS state")
    set_ob_rms(envs, ob_rms)
    vec_normalize = get_vec_normalize(envs)
    vec_normalize.eval()
    model.to(torch.device(args.device))
    model.eval()
    # Make the sample action stream explicit and independent of model
    # initialization RNG consumption.
    torch.manual_seed(args.seed)
    return envs, model, vec_normalize


def evaluate(args: argparse.Namespace, paths: dict[str, Path]):
    import torch

    from metamorph.config import cfg

    envs, model, vec_normalize = build_runtime(args, paths)
    tracker = FiniteTracker()
    transitions = []
    episode_records = []
    total_valid_actions = 0
    total_out_of_bounds_actions = 0
    all_action_values_finite = True
    base_env = None
    trajectory_metadata = None
    trajectory_snapshots: dict[str, dict[str, Any]] = {}
    joint_limit_mapping = None
    joint_limit_recorder = None
    joint_limit_proxy = None
    original_sim = None
    try:
        observation = envs.reset()
        if args.record_state_trajectory or args.record_joint_limit_substeps:
            base_env = unwrap_single_mujoco_env(envs)
            trajectory_metadata = build_state_trajectory_metadata(base_env)
        if args.record_state_trajectory:
            termination_wrapper = find_env_wrapper(envs, "TerminateOnFalling")
            install_pre_autoreset_state_capture(
                termination_wrapper,
                base_env,
                trajectory_metadata,
                tracker,
                trajectory_snapshots,
            )
        if args.record_joint_limit_substeps:
            joint_limit_mapping = build_joint_limit_probe_mapping(
                base_env,
                args.joint_limit_probe_names,
                trajectory_metadata,
                contact_probe_body_names=(
                    args.contact_probe_body_names
                    if args.record_contact_generalized_response
                    else ()
                ),
                enable_contact_mapping=(
                    args.record_contact_generalized_response
                    or args.record_physical_contact_projection
                ),
            )
            original_sim = base_env.sim
            joint_limit_recorder = JointLimitSubstepRecorder(
                original_sim,
                base_env.frame_skip,
                joint_limit_mapping,
                tracker,
                record_contact_response=args.record_contact_generalized_response,
                record_physical_projection=args.record_physical_contact_projection,
            )
            joint_limit_proxy = JointLimitRecordingSimProxy(
                original_sim, joint_limit_recorder
            )
            base_env.sim = joint_limit_proxy
        for episode_index in range(args.episodes):
            episode_reward = 0.0
            episode_rewards_finite = True
            episode_length = 0
            episode_velocities = []
            initial_x = None
            final_x = None
            episode_fallen = False
            episode_terminated = False
            episode_truncated = False
            evaluator_cutoff = False

            while True:
                with torch.no_grad():
                    _, distribution, _, _ = model(observation)
                    raw_action, valid_mask = choose_raw_action(
                        distribution, observation, args.action_mode
                    )
                action_stats = raw_action_diagnostics(
                    raw_action, valid_mask, tracker
                )
                native_action = (
                    tracker.vector(raw_action[valid_mask].detach().cpu().numpy())
                    if trajectory_metadata is not None
                    else None
                )
                total_valid_actions += action_stats.pop("_valid_action_count")
                total_out_of_bounds_actions += action_stats.pop(
                    "_out_of_bounds_count"
                )
                all_action_values_finite &= action_stats.pop(
                    "_action_values_finite"
                )

                if args.record_state_trajectory and trajectory_metadata is not None:
                    trajectory_snapshots.clear()
                if joint_limit_recorder is not None:
                    joint_limit_recorder.begin_control_step(
                        episode_index, episode_length + 1
                    )
                next_observation, reward_tensor, done_array, infos = envs.step(
                    raw_action
                )
                if joint_limit_recorder is not None:
                    joint_limit_recorder.end_control_step()
                    if joint_limit_proxy.callback_error is not None:
                        raise RuntimeError(
                            "joint-limit substep instrumentation failed"
                        ) from joint_limit_proxy.callback_error
                info = dict(infos[0])
                reward = tracker.scalar(reward_tensor.reshape(-1)[0].item())
                done = bool(done_array[0])
                truncated = bool(done and info.get("timeout", False))
                terminated = bool(done and not truncated)
                fallen = bool(info.get("fallen", False))
                x_velocity = info_scalar(
                    info, tracker, "x_vel", "x_velocity"
                )
                torso_x = info_scalar(info, tracker, "x_pos")
                fall_measurement = official_fall_measurement(info, tracker)

                if initial_x is None:
                    before = info.get("xy_pos_before")
                    if before is not None:
                        initial_x = info_scalar(
                            {"before": before}, tracker, "before"
                        )
                    elif torso_x is not None:
                        initial_x = torso_x
                if torso_x is not None:
                    final_x = torso_x
                if x_velocity is not None:
                    episode_velocities.append(x_velocity)
                if reward is not None:
                    episode_reward += reward
                else:
                    episode_rewards_finite = False
                episode_length += 1

                record = {
                    "episode": episode_index,
                    "step": episode_length,
                    "reward": reward,
                    "raw_reward": reward,
                    "reported_x_velocity": x_velocity,
                    "root_x": None,
                    "torso_x": torso_x,
                    **fall_measurement,
                    "fallen": fallen,
                    "terminated": terminated,
                    "truncated": truncated,
                    **action_stats,
                }
                if args.record_state_trajectory and trajectory_metadata is not None and base_env is not None:
                    record["native_action"] = native_action
                    record["torso_height_above_ground"] = record[
                        "formal_torso_height"
                    ]
                    record["control_step"] = episode_length
                    record["backend_episode_step"] = episode_length
                    record["elapsed_control_time"] = episode_length * float(
                        trajectory_metadata["control_dt"]
                    )
                    snapshot = trajectory_snapshots.get("latest")
                    if snapshot is None:
                        raise RuntimeError(
                            "pre-auto-reset state capture did not run for transition"
                        )
                    record.update(snapshot)
                    record["state_snapshot_status"] = (
                        "post_termination_pre_dummy_vec_env_auto_reset"
                    )
                transitions.append(record)

                observation = next_observation
                if done:
                    episode_fallen = fallen
                    episode_terminated = terminated
                    episode_truncated = truncated
                    break
                if evaluator_cutoff_reached(episode_length, args.max_eval_steps):
                    evaluator_cutoff = True
                    if not args.record_joint_limit_substeps:
                        observation = envs.reset()
                    break

            displacement = (
                final_x - initial_x
                if final_x is not None and initial_x is not None
                else None
            )
            episode_records.append(
                {
                    "episode": episode_index,
                    "length": episode_length,
                    "cumulative_reward": (
                        tracker.scalar(episode_reward)
                        if episode_rewards_finite
                        else None
                    ),
                    "mean_reported_x_velocity": mean_or_none(
                        episode_velocities
                    ),
                    "net_displacement": tracker.scalar(displacement),
                    "fallen": episode_fallen,
                    "terminated": episode_terminated,
                    "truncated": episode_truncated,
                    "evaluator_cutoff": evaluator_cutoff,
                }
            )
    finally:
        if base_env is not None and original_sim is not None:
            base_env.sim = original_sim
        envs.close()

    restored_policy_std_mean = tracker.scalar(
        torch.exp(model.log_std).mean().detach().cpu().item()
    )
    formal_height_count = sum(
        transition["formal_torso_height"] is not None
        for transition in transitions
    )
    summary = {
        "schema_version": "spikmorph-mujoco-checkpoint-evaluation-v1",
        "action_mode": args.action_mode,
        "checkpoint_path": str(paths["checkpoint"]),
        "checkpoint_sha256": sha256(paths["checkpoint"]),
        "episode_count": args.episodes,
        "completed_episode_count": len(episode_records),
        "mean_episode_length": mean_or_none(
            [record["length"] for record in episode_records]
        ),
        "mean_reported_x_velocity": mean_or_none(
            [
                transition["reported_x_velocity"]
                for transition in transitions
            ]
        ),
        "mean_cumulative_reward": mean_or_none(
            [record["cumulative_reward"] for record in episode_records]
        ),
        "mean_net_displacement": mean_or_none(
            [record["net_displacement"] for record in episode_records]
        ),
        "fallen_fraction": (
            sum(record["fallen"] for record in episode_records)
            / len(episode_records)
            if episode_records
            else None
        ),
        "terminated_count": sum(
            record["terminated"] for record in episode_records
        ),
        "truncated_count": sum(
            record["truncated"] for record in episode_records
        ),
        "raw_sampled_action_out_of_bounds_fraction": (
            total_out_of_bounds_actions / total_valid_actions
            if total_valid_actions and all_action_values_finite
            else (0.0 if all_action_values_finite else None)
        ),
        "formal_torso_height_source": "official_termination_info",
        "formal_torso_height_available_count": formal_height_count,
        "formal_torso_height_missing_count": (
            len(transitions) - formal_height_count
        ),
        "all_values_finite": tracker.all_values_finite,
        "episodes": episode_records,
        "transition_count": len(transitions),
        "evaluator_max_steps": args.max_eval_steps,
    }
    if args.record_state_trajectory:
        summary.update(
            {
                "environment_completed_episode_count": sum(
                    not record["evaluator_cutoff"] for record in episode_records
                ),
                "evaluator_cutoff_count": sum(
                    record["evaluator_cutoff"] for record in episode_records
                ),
            }
        )
    metadata = {
        "schema_version": "spikmorph-mujoco-checkpoint-evaluation-metadata-v1",
        "created_at": datetime.now().astimezone().isoformat(),
        "git_head": git_output("rev-parse", "HEAD"),
        "git_status_short": git_output("status", "--short").splitlines(),
        "evaluator_source_path": str(Path(__file__).resolve()),
        "evaluator_source_sha256": sha256(Path(__file__).resolve()),
        "config_path": str(paths["config"]),
        "config_sha256": sha256(paths["config"]),
        "checkpoint_path": str(paths["checkpoint"]),
        "checkpoint_sha256": summary["checkpoint_sha256"],
        "walker_dir": str(paths["walker_dir"]),
        "morphology_id": args.morphology_id,
        "morphology_xml_path": str(paths["morphology_xml"]),
        "morphology_xml_sha256": sha256(paths["morphology_xml"]),
        "morphology_metadata_path": str(paths["morphology_metadata"]),
        "morphology_metadata_sha256": sha256(paths["morphology_metadata"]),
        "seed": args.seed,
        "episodes": args.episodes,
        "device": args.device,
        "action_mode": args.action_mode,
        "restored_policy_std_mean": restored_policy_std_mean,
        "action_semantics": {
            "zero": "all policy-space action components are zero",
            "mean": "raw Normal distribution mean; no evaluator clamp",
            "sample": (
                "raw spikmorph Normal distribution sample; no clipped-Gaussian "
                "likelihood or evaluator clamp"
            ),
            "native_actuator": (
                "formal MultiUnimalNodeCentricAction followed by MuJoCo "
                "ctrllimited actuator semantics"
            ),
        },
        "observation_normalization": {
            "ob_rms_restored": True,
            "training": bool(vec_normalize.training),
            "updates_frozen": not bool(vec_normalize.training),
            "reward_normalization": False,
            "raw_reward_equals_reported_reward": True,
            "clip_observation": float(vec_normalize.clipob),
        },
        "native_reset": native_reset_metadata(
            cfg.ENV.RESET_NOISE_SCALE, args.reset_noise_scale
        ),
        "formal_torso_height": {
            "available": formal_height_count == len(transitions),
            "source": "official_termination_info",
            "available_count": formal_height_count,
            "missing_count": len(transitions) - formal_height_count,
            "reason_if_missing": (
                "formal termination info did not expose the measurement"
            ),
        },
        "state_trajectory": {
            "enabled": bool(args.record_state_trajectory),
            "evaluator_max_steps": args.max_eval_steps,
            "control_step_indexing": (
                "1-based post-control-step; reset state is not a transition"
            ),
            "capture_timing": (
                "post-physics, post-reward, post-formal-fallen-decision, "
                "pre-DummyVecEnv-auto-reset"
            ),
            "evaluator_cutoff_semantics": (
                "capture/evaluation stop only; never reported as environment "
                "terminated or truncated"
            ),
            "terminal_snapshot_note": (
                "terminal transition state is captured by an opt-in read-only "
                "TerminateOnFalling wrapper hook before DummyVecEnv reset"
            ),
            **(trajectory_metadata or {}),
        },
    }
    oracle = None
    if joint_limit_recorder is not None and joint_limit_mapping is not None:
        oracle = build_joint_limit_oracle_outputs(
            joint_limit_recorder.records,
            joint_limit_mapping,
            expected_control_steps=args.max_eval_steps,
            frame_skip=int(trajectory_metadata["frame_skip"]),
        )
        metadata["joint_limit_substep_oracle"] = {
            "enabled": True,
            "capture_timing": "pre live mj_step state, then immediate post live mj_step solver data and state",
            "extra_mj_step_calls": 0,
            "extra_mj_forward_calls": 0,
            "physics_substep_indexing": "0-based within each 1-based control_step",
            "probe_names": list(args.joint_limit_probe_names),
            "record_count": len(joint_limit_recorder.records),
        }
        if args.record_contact_generalized_response:
            metadata["contact_generalized_response_oracle"] = {
                "enabled": True,
                "probe_body_names": list(args.contact_probe_body_names),
                "body_velocity_api": "mujoco.mj_objectVelocity",
                "body_velocity_local_flag": 0,
                "body_velocity_frame": "world-oriented",
                "body_linear_velocity_point": "body origin",
                "body_state_synchronization": "copy live generalized qpos/qvel into isolated MjData, mj_forward isolated data, then read xpos/xquat and mj_objectVelocity",
                "contact_wrench_api": "mujoco.mj_contactForce",
                "contact_row_mapping": "contact.efc_address plus cone-dependent row count from contact.dim",
                "floor_detection": "compiled geom identity floor/0",
                "extra_mj_step_calls": 0,
                "extra_live_mj_forward_calls": 0,
                "isolated_body_kinematics_mj_forward_calls_per_substep": 2,
            }
        if args.record_physical_contact_projection:
            metadata["physical_contact_projection_oracle"] = {
                "enabled": True,
                "contact_wrench_api": "mujoco.mj_contactForce",
                "physical_projection_api": "mujoco.mj_applyFT",
                "contact_point": "exact mjContact.pos",
                "contact_frame_storage": "normal/tangent1/tangent2 world axes stored as rows",
                "sign_selection": "one uniform contact-side sign selected by strict full-nv total reconstruction",
                "scratch_target": "independent zero qfrc vectors; data.qfrc_applied is never passed to mj_applyFT",
                "extra_mj_step_calls": 0,
                "extra_mj_forward_calls": 0,
            }
        summary["joint_limit_substep_record_count"] = len(
            joint_limit_recorder.records
        )
    return metadata, summary, transitions, oracle


def write_outputs(
    output_dir: Path,
    metadata: dict[str, Any],
    summary: dict[str, Any],
    transitions: Sequence[dict[str, Any]],
    oracle: dict[str, Any] | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_options = {
        "sort_keys": True,
        "ensure_ascii": False,
        "allow_nan": False,
    }
    for name, payload in (
        ("metadata.json", metadata),
        ("summary.json", summary),
    ):
        (output_dir / name).write_text(
            json.dumps(payload, indent=2, **json_options) + "\n",
            encoding="utf-8",
        )
    with (output_dir / "transitions.jsonl").open(
        "x", encoding="utf-8", newline="\n"
    ) as stream:
        for transition in transitions:
            stream.write(json.dumps(transition, **json_options) + "\n")
    if oracle is not None:
        for name, payload in (
            ("joint_mapping.json", oracle["mapping"]),
            ("first_contact_and_limit_summary.json", oracle["summary"]),
            ("validation.json", oracle["validation"]),
        ):
            (output_dir / name).write_text(
                json.dumps(payload, indent=2, **json_options) + "\n",
                encoding="utf-8",
            )
        with (output_dir / "substeps.jsonl").open(
            "x", encoding="utf-8", newline="\n"
        ) as stream:
            for record in oracle["records"]:
                stream.write(json.dumps(record, **json_options) + "\n")
        (output_dir / "run.log").write_text(
            json.dumps(
                {
                    "created_at": metadata["created_at"],
                    "git_head": metadata["git_head"],
                    "evaluator_source_sha256": metadata[
                        "evaluator_source_sha256"
                    ],
                    "record_count": len(oracle["records"]),
                    "validation": oracle["validation"],
                },
                **json_options,
            )
            + "\n",
            encoding="utf-8",
        )
        if oracle.get("contact_summary") is not None:
            (output_dir / "contact_generalized_response_summary.json").write_text(
                json.dumps(
                    oracle["contact_summary"], indent=2, **json_options
                ) + "\n",
                encoding="utf-8",
            )
        physical = oracle.get("physical_outputs")
        if physical is not None:
            with (output_dir / "physical_contact_substeps.jsonl").open(
                "x", encoding="utf-8", newline="\n"
            ) as stream:
                for record in physical["records"]:
                    stream.write(json.dumps(record, **json_options) + "\n")
            for name, key in (
                ("contact_frame_validation.json", "contact_frame_validation"),
                ("physical_vs_constraint_generalized.json", "physical_vs_constraint_generalized"),
                ("selected_joint_physical_decomposition.json", "selected_joint_physical_decomposition"),
                ("unit_force_projection.json", "unit_force_projection"),
                ("joint_contact_geometry.json", "joint_contact_geometry"),
            ):
                (output_dir / name).write_text(
                    json.dumps(physical[key], indent=2, **json_options) + "\n",
                    encoding="utf-8",
                )


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    paths = validate_args(args)
    configure_runtime(args, paths)
    metadata, summary, transitions, oracle = evaluate(args, paths)
    write_outputs(paths["output_dir"], metadata, summary, transitions, oracle)
    print(
        json.dumps(
            {
                "output_dir": str(paths["output_dir"]),
                "action_mode": args.action_mode,
                "completed_episode_count": summary["completed_episode_count"],
                "all_values_finite": summary["all_values_finite"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

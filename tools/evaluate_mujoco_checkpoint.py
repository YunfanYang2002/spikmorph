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
from typing import Any, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
ACTION_MODES = ("zero", "mean", "sample")
OUTPUT_FILENAMES = ("metadata.json", "summary.json", "transitions.jsonl")


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
    existing = [output_dir / name for name in OUTPUT_FILENAMES]
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
    return {
        "physics_timestep": float(model.opt.timestep),
        "frame_skip": int(base_env.frame_skip),
        "control_dt": float(model.opt.timestep * base_env.frame_skip),
        "root_free_joint": root,
        "root_qpos_convention": "[world_x, world_y, world_z, quat_w, quat_x, quat_y, quat_z]",
        "root_qvel_convention": "native MuJoCo generalized free-joint [linear_xyz, angular_xyz]",
        "torso_velocity_convention": "sim.data.body_xvelp/body_xvelr, world-frame Cartesian",
        "ordinary_joint_mapping": ordinary,
        "ordered_joint_names": [joint["joint_name"] for joint in ordinary],
        "generalized_dof_mapping": dof_mapping,
        "actuator_mapping": actuators,
        "body_mapping": [
            {"body_id": index, "body_name": str(name)}
            for index, name in enumerate(model.body_names)
        ],
        "torso_body_id": torso_id,
        "torso_body_name": "torso/0",
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
        "coordinate_conventions": {
            "positions": "MuJoCo world frame, metres",
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
    torso_id = metadata["torso_body_id"]
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
    return {
        "simulation_time": tracker.scalar(data.time),
        "full_qpos": tracker.vector(qpos),
        "full_qvel": tracker.vector(qvel),
        "torso_world_position": tracker.vector(data.body_xpos[torso_id]),
        "root_free_joint_position": tracker.vector(root_qpos[:3]),
        "root_free_joint_orientation_wxyz": tracker.vector(root_qpos[3:7]),
        "root_linear_velocity_world": tracker.vector(data.body_xvelp[torso_id]),
        "root_angular_velocity_world": tracker.vector(data.body_xvelr[torso_id]),
        "torso_linear_velocity_world": tracker.vector(data.body_xvelp[torso_id]),
        "torso_angular_velocity_world": tracker.vector(data.body_xvelr[torso_id]),
        "root_generalized_qvel": tracker.vector(root_qvel),
        "ordered_joint_qpos": [tracker.scalar(qpos[joint["qpos_indices"][0]]) for joint in ordinary],
        "ordered_joint_qvel": [tracker.scalar(qvel[joint["qvel_indices"][0]]) for joint in ordinary],
        "actuator_ctrl": tracker.vector(data.ctrl),
        "actuator_force": tracker.vector(getattr(data, "actuator_force", [])),
        "qfrc_actuator": tracker.vector(data.qfrc_actuator),
        "qfrc_passive": tracker.vector(data.qfrc_passive),
        "contact_count": int(data.ncon),
        "contacts": contacts,
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
    try:
        observation = envs.reset()
        if args.record_state_trajectory:
            base_env = unwrap_single_mujoco_env(envs)
            trajectory_metadata = build_state_trajectory_metadata(base_env)
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
                total_valid_actions += action_stats.pop("_valid_action_count")
                total_out_of_bounds_actions += action_stats.pop(
                    "_out_of_bounds_count"
                )
                all_action_values_finite &= action_stats.pop(
                    "_action_values_finite"
                )

                next_observation, reward_tensor, done_array, infos = envs.step(
                    raw_action
                )
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
                if trajectory_metadata is not None and base_env is not None:
                    record["control_step"] = episode_length - 1
                    record["elapsed_control_time"] = episode_length * float(
                        trajectory_metadata["control_dt"]
                    )
                    if done:
                        record["state_snapshot_status"] = (
                            "unavailable_after_dummy_vec_env_auto_reset"
                        )
                    else:
                        record.update(
                            capture_state_trajectory(
                                base_env, trajectory_metadata, tracker
                            )
                        )
                        record["state_snapshot_status"] = "post_control_step"
                transitions.append(record)

                observation = next_observation
                if done:
                    episode_fallen = fallen
                    episode_terminated = terminated
                    episode_truncated = truncated
                    break
                if evaluator_cutoff_reached(episode_length, args.max_eval_steps):
                    evaluator_cutoff = True
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
    metadata = {
        "schema_version": "spikmorph-mujoco-checkpoint-evaluation-metadata-v1",
        "created_at": datetime.now().astimezone().isoformat(),
        "git_head": git_output("rev-parse", "HEAD"),
        "git_status_short": git_output("status", "--short").splitlines(),
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
        "native_reset": {
            "reset_noise_scale": float(cfg.ENV.RESET_NOISE_SCALE),
            "requested_reset_noise_scale": args.reset_noise_scale,
            "qpos_qvel_noise_preserved": bool(cfg.ENV.RESET_NOISE_SCALE != 0.0),
            "deterministic_reset_effective": bool(cfg.ENV.RESET_NOISE_SCALE == 0.0),
            "deterministic_reset_forced": bool(args.reset_noise_scale == 0.0),
        },
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
            "terminal_snapshot_note": (
                "DummyVecEnv auto-resets before evaluator readback; terminal state "
                "snapshots are marked unavailable"
            ),
            **(trajectory_metadata or {}),
        },
    }
    return metadata, summary, transitions


def write_outputs(
    output_dir: Path,
    metadata: dict[str, Any],
    summary: dict[str, Any],
    transitions: Sequence[dict[str, Any]],
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


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    paths = validate_args(args)
    configure_runtime(args, paths)
    metadata, summary, transitions = evaluate(args, paths)
    write_outputs(paths["output_dir"], metadata, summary, transitions)
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

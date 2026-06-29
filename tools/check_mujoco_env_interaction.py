"""Run action-only MuJoCo environment interaction smoke tests.

This intentionally uses the same environment factory and wrappers as PPO, but
steps the individual environments directly so diagnostics can be captured
before a completed environment is reset.
"""

import argparse
import json
import math
import os
import sys
from collections import Counter

import numpy as np


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from metamorph.config import cfg  # noqa: E402
from metamorph.utils import gym_compat as gu  # noqa: E402
from metamorph.utils import sample as su  # noqa: E402


ISAAC_REFERENCE = {
    "small_random": 14.9828,
    "random_uniform": 299.7234,
}


class RunningStats:
    def __init__(self):
        self.count = 0
        self.total = 0.0
        self.minimum = math.inf
        self.maximum = -math.inf

    def add(self, value):
        values = np.asarray(value, dtype=np.float64).reshape(-1)
        values = values[np.isfinite(values)]
        if values.size == 0:
            return
        self.count += int(values.size)
        self.total += float(values.sum())
        self.minimum = min(self.minimum, float(values.min()))
        self.maximum = max(self.maximum, float(values.max()))

    def result(self):
        if self.count == 0:
            return {"min": None, "max": None, "mean": None}
        return {
            "min": self.minimum,
            "max": self.maximum,
            "mean": self.total / self.count,
        }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Smoke-test MuJoCo environment interaction without a policy."
    )
    parser.add_argument("--cfg", dest="cfg_file", required=True)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--num-envs", type=int, default=3)
    parser.add_argument(
        "--action-modes",
        nargs="+",
        default=["zero", "small_random", "random_uniform"],
        choices=["zero", "small_random", "random_uniform"],
    )
    parser.add_argument("--small-random-scale", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--out", default="output/mujoco_env_interaction_200.json"
    )
    parser.add_argument(
        "opts",
        nargs=argparse.REMAINDER,
        help="Additional config KEY VALUE pairs, as accepted by train_ppo.py",
    )
    args = parser.parse_args()
    if args.steps <= 0:
        parser.error("--steps must be positive")
    if args.num_envs <= 0:
        parser.error("--num-envs must be positive")
    if args.small_random_scale < 0:
        parser.error("--small-random-scale must be non-negative")
    return args


def configure(args):
    cfg.merge_from_file(args.cfg_file)
    cfg.merge_from_list(args.opts)
    cfg.RNG_SEED = args.seed
    cfg.PPO.NUM_ENVS = args.num_envs
    cfg.RANK = 0
    cfg.LOCAL_RANK = 0
    cfg.WORLD_SIZE = 1
    cfg.DISTRIBUTED = False

    if cfg.ENV_NAME == "Unimal-v0" and not cfg.ENV.WALKERS:
        xml_dir = os.path.join(cfg.ENV.WALKER_DIR, "xml")
        if not os.path.isdir(xml_dir):
            raise FileNotFoundError(
                "No ENV.WALKERS were configured and walker XML directory does "
                "not exist: {}".format(xml_dir)
            )
        cfg.ENV.WALKERS = sorted(
            os.path.splitext(name)[0]
            for name in os.listdir(xml_dir)
            if name.endswith(".xml")
        )

    if not cfg.ENV.WALKERS:
        raise ValueError("The smoke test requires at least one ENV.WALKERS entry")

    metadata = []
    for walker in cfg.ENV.WALKERS:
        path = os.path.join(cfg.ENV.WALKER_DIR, "metadata", walker + ".json")
        with open(path, "r", encoding="utf-8") as handle:
            metadata.append(json.load(handle))
    # This is the same inference performed by tools/train_ppo.py.
    cfg.MODEL.MAX_JOINTS = max(item["dof"] for item in metadata) + 1
    cfg.MODEL.MAX_LIMBS = max(item["num_limbs"] + 1 for item in metadata) + 1


def wrapper_chain(env):
    current = env
    seen = set()
    while id(current) not in seen:
        yield current
        seen.add(id(current))
        if hasattr(current, "env"):
            current = current.env
        elif hasattr(current, "_env"):
            current = current._env
        else:
            break


def find_wrapper(env, class_name):
    for current in wrapper_chain(env):
        if current.__class__.__name__ == class_name:
            return current
    return None


def morphology_id(env):
    metadata = getattr(env.unwrapped, "metadata", {})
    return metadata.get("unimal_id", getattr(env.unwrapped, "unimal_id", None))


def active_action(env, action):
    action_wrapper = find_wrapper(env, "MultiUnimalNodeCentricAction")
    if action_wrapper is None:
        return np.asarray(action, dtype=np.float64).reshape(-1)
    return np.asarray(
        action_wrapper.action(np.asarray(action).copy()), dtype=np.float64
    ).reshape(-1)


def torso_height(env):
    """Read height by calling the existing UnimalHeightObs implementation."""
    height_wrapper = find_wrapper(env, "UnimalHeightObs")
    base = env.unwrapped
    if height_wrapper is None or not hasattr(base, "_get_obs"):
        return None
    raw_obs = base._get_obs()
    height_obs = height_wrapper.observation(raw_obs)
    value = np.asarray(height_obs["torso_height"]).reshape(-1)[0]
    return float(value)


def fall_threshold(env):
    value = getattr(env.unwrapped, "metadata", {}).get("fall_threshold")
    return None if value is None else float(value)


def torso_x(env):
    return float(env.unwrapped.sim.data.get_body_xpos("torso/0")[0])


def sim_values(env):
    sim = env.unwrapped.sim
    gear = np.asarray(sim.model.actuator_gear[:, 0], dtype=np.float64).copy()
    ctrl = np.asarray(sim.data.ctrl, dtype=np.float64).copy()
    qpos = np.asarray(sim.data.qpos, dtype=np.float64).reshape(-1)
    qvel = np.asarray(sim.data.qvel, dtype=np.float64).reshape(-1)
    body_linear = np.asarray(sim.data.body_xvelp, dtype=np.float64)
    body_angular = np.asarray(sim.data.body_xvelr, dtype=np.float64)
    return {
        "gear": gear,
        "ctrl": ctrl,
        "joint_pos": qpos[7:].copy(),
        "joint_vel": qvel[6:].copy(),
        "body_linear_vel": body_linear.copy(),
        "body_angular_vel": body_angular.copy(),
    }


def scalar(info, *keys):
    for key in keys:
        if key in info:
            value = np.asarray(info[key]).reshape(-1)
            if value.size:
                return float(value[0])
    return None


def sample_action(mode, shape, rng, small_scale):
    if mode == "zero":
        return np.zeros(shape, dtype=np.float32)
    if mode == "small_random":
        return rng.uniform(-small_scale, small_scale, size=shape).astype(np.float32)
    return rng.uniform(-1.0, 1.0, size=shape).astype(np.float32)


def add_optional(stats, key, value):
    if value is not None:
        stats[key].add(value)


def contains_non_finite(*values):
    for value in values:
        if value is None:
            continue
        if isinstance(value, dict):
            if contains_non_finite(*value.values()):
                return True
            continue
        if isinstance(value, (list, tuple)):
            if contains_non_finite(*value):
                return True
            continue
        try:
            if not np.all(np.isfinite(np.asarray(value, dtype=np.float64))):
                return True
        except (TypeError, ValueError):
            continue
    return False


def make_diagnostic_envs(num_envs, seed):
    # Lazy import keeps --help usable when optional MuJoCo dependencies are not
    # installed, while still reusing the exact PPO environment factory.
    from metamorph.algos.ppo.envs import make_env

    envs = []
    walkers = list(cfg.ENV.WALKERS)
    try:
        for index in range(num_envs):
            walker = walkers[index % len(walkers)]
            env = make_env(cfg.ENV_NAME, seed, index, xml_file=walker)()
            gu.normalize_reset(env.reset())
            envs.append(env)
    except Exception:
        for env in envs:
            env.close()
        raise
    return envs


def run_mode(mode, args):
    # Recreate and reseed environments for each mode so initial states match.
    su.set_seed(args.seed)
    rng = np.random.RandomState(args.seed)
    envs = make_diagnostic_envs(args.num_envs, args.seed)
    stat_names = [
        "action",
        "action_norm",
        "actuator_gear",
        "actuator_gear_abs",
        "action_times_actuator_gear",
        "action_times_actuator_gear_abs",
        "applied_ctrl",
        "applied_ctrl_abs",
        "initial_torso_height",
        "torso_height",
        "fall_threshold",
        "reward",
        "forward_reward",
        "control_cost",
        "x_delta",
        "x_velocity",
        "joint_pos",
        "joint_vel",
        "body_linear_vel",
        "body_angular_vel",
    ]
    stats = {name: RunningStats() for name in stat_names}
    episode_lengths = [0] * args.num_envs
    completed_lengths = []
    morphology_ids = []
    done_count = 0
    fallen_count = 0
    timeout_count = 0
    bad_env_reset_count = 0
    non_finite_count = 0
    reasons = Counter()
    samples = []

    try:
        for env in envs:
            morphology_ids.append(morphology_id(env))
            add_optional(stats, "initial_torso_height", torso_height(env))
            add_optional(stats, "fall_threshold", fall_threshold(env))

        for step in range(1, args.steps + 1):
            for env_index, env in enumerate(envs):
                raw_action = sample_action(
                    mode, env.action_space.shape, rng, args.small_random_scale
                )
                effective_action = active_action(env, raw_action)
                x_before = torso_x(env)
                obs, reward, done, info = gu.normalize_step(env.step(raw_action))
                episode_lengths[env_index] += 1

                height = torso_height(env)
                threshold = fall_threshold(env)
                fallen = (
                    height is not None
                    and threshold is not None
                    and height <= threshold
                )
                values = sim_values(env)
                ctrl = values["ctrl"]
                gear = values["gear"]
                action_for_gear = effective_action
                if action_for_gear.size != gear.size:
                    # sim.data.ctrl is authoritative after action wrappers.
                    action_for_gear = ctrl
                action_times_gear = action_for_gear * gear
                x_after = torso_x(env)
                forward_reward = scalar(info, "__reward__forward", "forward_reward")
                control_cost = scalar(info, "__reward__ctrl", "control_cost")
                x_velocity = scalar(info, "x_vel", "x_velocity")

                stats["action"].add(effective_action)
                stats["action_norm"].add(np.linalg.norm(effective_action))
                stats["actuator_gear"].add(gear)
                stats["actuator_gear_abs"].add(np.abs(gear))
                stats["action_times_actuator_gear"].add(action_times_gear)
                stats["action_times_actuator_gear_abs"].add(
                    np.abs(action_times_gear)
                )
                stats["applied_ctrl"].add(ctrl)
                stats["applied_ctrl_abs"].add(np.abs(ctrl))
                add_optional(stats, "torso_height", height)
                add_optional(stats, "fall_threshold", threshold)
                stats["reward"].add(reward)
                add_optional(stats, "forward_reward", forward_reward)
                add_optional(stats, "control_cost", control_cost)
                stats["x_delta"].add(x_after - x_before)
                add_optional(stats, "x_velocity", x_velocity)
                stats["joint_pos"].add(values["joint_pos"])
                stats["joint_vel"].add(values["joint_vel"])
                stats["body_linear_vel"].add(values["body_linear_vel"])
                stats["body_angular_vel"].add(values["body_angular_vel"])

                if contains_non_finite(
                    obs,
                    reward,
                    effective_action,
                    ctrl,
                    gear,
                    height,
                    x_after,
                    *values.values(),
                ):
                    non_finite_count += 1

                timeout = bool(info.get("timeout", False))
                bad_reset = bool(
                    info.get("bad_env_reset", False)
                    or info.get("bad_reset", False)
                    or info.get("reset_error", False)
                )
                mj_step_error = bool(info.get("mj_step_error", False))

                if env_index == 0 and step <= 20:
                    sample = {
                        "step": step,
                        "action_norm": float(np.linalg.norm(effective_action)),
                        "action_times_actuator_gear_norm": float(
                            np.linalg.norm(action_times_gear)
                        ),
                        "applied_ctrl_norm": float(np.linalg.norm(ctrl)),
                        "torso_height": height,
                        "fall_threshold": threshold,
                        "fallen": bool(fallen),
                        "reward": float(reward),
                        "forward_reward": forward_reward,
                        "x_velocity": x_velocity,
                    }
                    samples.append(sample)

                if done:
                    done_count += 1
                    completed_lengths.append(episode_lengths[env_index])
                    if fallen:
                        fallen_count += 1
                        reasons["fallen"] += 1
                    elif timeout:
                        reasons["timeout"] += 1
                    elif mj_step_error:
                        reasons["mj_step_error"] += 1
                    else:
                        reasons["done_unknown"] += 1
                    timeout_count += int(timeout)
                    bad_env_reset_count += int(bad_reset)
                    episode_lengths[env_index] = 0
                    gu.normalize_reset(env.reset())
                    morphology_ids.append(morphology_id(env))
                    add_optional(stats, "initial_torso_height", torso_height(env))
                    add_optional(stats, "fall_threshold", fall_threshold(env))
                elif bad_reset:
                    bad_env_reset_count += 1
                    reasons["bad_env_reset"] += 1
    finally:
        for env in envs:
            env.close()

    result = {
        "action_mode": mode,
        "requested_steps": args.steps,
        "num_envs": args.num_envs,
        "transition_count": args.steps * args.num_envs,
        "morphology_ids": list(dict.fromkeys(morphology_ids)),
        "action_scale": (
            0.0
            if mode == "zero"
            else args.small_random_scale
            if mode == "small_random"
            else 1.0
        ),
    }
    for name, running in stats.items():
        values = running.result()
        result[name + "_min"] = values["min"]
        result[name + "_max"] = values["max"]
        result[name + "_mean"] = values["mean"]
    result.update(
        {
            "mean_episode_length": (
                float(np.mean(completed_lengths)) if completed_lengths else None
            ),
            "completed_episode_count": len(completed_lengths),
            "done_count": done_count,
            "fallen_count": fallen_count,
            "timeout_count": timeout_count,
            "bad_env_reset_count": bad_env_reset_count,
            "non_finite_count": non_finite_count,
            "termination_reason_histogram": dict(sorted(reasons.items())),
            "incomplete_episode_lengths": episode_lengths,
            "first_20_step_samples_env_0": samples,
        }
    )
    return result


def same_order_of_magnitude(value, reference):
    if value is None or value <= 0:
        return False
    ratio = value / reference
    return 0.1 <= ratio <= 10.0


def print_conclusion(summaries):
    by_mode = {item["action_mode"]: item for item in summaries}
    print("\nMuJoCo smoke test conclusion")
    for mode in ("zero", "small_random", "random_uniform"):
        if mode not in by_mode:
            continue
        item = by_mode[mode]
        print(
            "- {} mean_episode_length: {} (done={}, fallen={})".format(
                mode,
                item["mean_episode_length"],
                item["done_count"],
                item["fallen_count"],
            )
        )

    all_fast = all(
        by_mode.get(mode, {}).get("mean_episode_length") is not None
        and by_mode[mode]["mean_episode_length"] <= 30.0
        for mode in ("zero", "small_random", "random_uniform")
        if mode in by_mode
    ) and all(mode in by_mode for mode in ("zero", "small_random", "random_uniform"))
    print("- all three modes done quickly (mean <= 30 steps): {}".format(all_fast))

    zero_length = by_mode.get("zero", {}).get("mean_episode_length")
    small_length = by_mode.get("small_random", {}).get("mean_episode_length")
    random_length = by_mode.get("random_uniform", {}).get("mean_episode_length")
    small_more_stable = (
        zero_length is not None
        and small_length is not None
        and small_length > zero_length
    )
    random_more_unstable = (
        random_length is not None
        and zero_length is not None
        and small_length is not None
        and random_length < min(zero_length, small_length)
    )
    print("- small_random more stable than zero: {}".format(small_more_stable))
    print(
        "- random_uniform more unstable than zero and small_random: {}".format(
            random_more_unstable
        )
    )

    comparable = []
    for mode in ("small_random", "random_uniform"):
        if mode not in by_mode:
            continue
        value = by_mode[mode]["action_times_actuator_gear_abs_max"]
        ctrl = by_mode[mode]["applied_ctrl_abs_max"]
        same_order = same_order_of_magnitude(value, ISAAC_REFERENCE[mode])
        comparable.append(same_order)
        print(
            "- {}: |action*gear|max={}, |ctrl|max={}, "
            "same order as Isaac {}: {}".format(
                mode, value, ctrl, ISAAC_REFERENCE[mode], same_order
            )
        )

    if all_fast:
        print(
            "- Interpretation: MuJoCo is also short-lived; Isaac zero-action "
            "collapse is less likely to be the primary migration bug. Check "
            "PPO/action distribution/reward design next."
        )
    elif any(
        by_mode.get(mode, {}).get("mean_episode_length") is not None
        and by_mode[mode]["mean_episode_length"] > 30.0
        for mode in ("zero", "small_random")
    ):
        print(
            "- Interpretation: MuJoCo zero/small_random survives longer; inspect "
            "Isaac reset, passive dynamics, damping, collision, and termination."
        )
    if small_more_stable and random_more_unstable:
        print(
            "- Interpretation: the MuJoCo stability ordering suggests checking "
            "Isaac effort scale and passive dynamics."
        )
    if comparable and not all(comparable):
        print(
            "- Interpretation: action*gear differs materially from the Isaac "
            "effort reference; continue checking action-to-effort scaling."
        )


def main():
    args = parse_args()
    configure(args)
    summaries = [run_mode(mode, args) for mode in args.action_modes]
    payload = {
        "event": "mujoco_env_interaction_smoke",
        "summaries": summaries,
    }
    output_path = os.path.abspath(args.out)
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
    print("Wrote {}".format(output_path))
    print_conclusion(summaries)


if __name__ == "__main__":
    main()

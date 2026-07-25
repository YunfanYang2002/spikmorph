"""Orchestrate independent formal MuJoCo PPO control jobs.

This module deliberately delegates training to ``tools/train_ppo.py``.  It
adds experiment-level parallelism and provenance capture without changing the
repository's MuJoCo task, reset, physics, or PPO implementation.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Sequence
import zipfile


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MORPHOLOGY = "floor-1409-0-3-01-15-56-55"


@dataclass(frozen=True)
class JobSpec:
    job_id: str
    seed: int
    base_lr: float
    output_dir: str


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description=(
            "Run independent formal MuJoCo PPO jobs with bounded "
            "experiment-level parallelism."
        )
    )
    result.add_argument("--cfg", default="configs/ft.yaml")
    result.add_argument("--walker-dir", required=True)
    result.add_argument("--morphology", default=DEFAULT_MORPHOLOGY)
    result.add_argument("--seeds", default="1409")
    result.add_argument("--base-lrs", default="0.00015")
    result.add_argument("--max-parallel", type=int, default=1)
    result.add_argument(
        "--devices",
        default="cpu",
        help=(
            "Comma-separated PyTorch policy devices assigned round-robin "
            "(for example cpu or cuda:0,cuda:1). MuJoCo physics remains CPU."
        ),
    )
    result.add_argument("--action-std", type=float, default=0.3)
    result.add_argument("--num-envs", type=int, default=4)
    result.add_argument("--timesteps", type=int, default=128)
    result.add_argument("--max-state-action-pairs", type=int, default=51_200)
    result.add_argument(
        "--batch-size",
        type=int,
        default=512,
        help=(
            "Configured MuJoCo minibatch size. The original sampler requires "
            "this to be no larger than the rollout; 512 matches the corrected "
            "Isaac effective geometry but not its configured value 5120."
        ),
    )
    result.add_argument(
        "--target-kl",
        type=float,
        default=0.02,
        help=(
            "Effective KL early-stop threshold. It is mapped to this "
            "repository's PPO.KL_TARGET_COEF field."
        ),
    )
    result.add_argument("--output-root", default="output/diagnostics")
    result.add_argument("--tag", default="mujoco_control_51k")
    result.add_argument("--timeout-seconds", type=float, default=0.0)
    result.add_argument(
        "--archive", action=argparse.BooleanOptionalAction, default=False
    )
    result.add_argument("--archive-dir", default="./tmp")
    result.add_argument("--dry-run", action="store_true")
    result.add_argument("--timestamp", help=argparse.SUPPRESS)
    return result


def csv_ints(raw: str, label: str) -> tuple[int, ...]:
    try:
        values = tuple(int(item.strip()) for item in raw.split(",") if item.strip())
    except ValueError as error:
        raise ValueError(f"{label} must be comma-separated integers") from error
    if not values:
        raise ValueError(f"{label} must not be empty")
    return values


def csv_floats(raw: str, label: str) -> tuple[float, ...]:
    try:
        values = tuple(
            float(item.strip()) for item in raw.split(",") if item.strip()
        )
    except ValueError as error:
        raise ValueError(f"{label} must be comma-separated numbers") from error
    if not values or any(not math.isfinite(value) or value <= 0 for value in values):
        raise ValueError(f"{label} values must be finite and positive")
    return values


def csv_strings(raw: str, label: str) -> tuple[str, ...]:
    values = tuple(item.strip() for item in raw.split(",") if item.strip())
    if not values:
        raise ValueError(f"{label} must not be empty")
    return values


def build_job_matrix(
    *,
    seeds: Sequence[int],
    base_lrs: Sequence[float],
    batch_root: Path,
) -> tuple[JobSpec, ...]:
    jobs = []
    for index, (seed, base_lr) in enumerate(
        (seed, base_lr) for seed in seeds for base_lr in base_lrs
    ):
        lr_label = format(base_lr, ".8g").replace(".", "p").replace("-", "m")
        job_id = f"job_{index:03d}_seed{seed}_lr{lr_label}"
        jobs.append(
            JobSpec(
                job_id=job_id,
                seed=int(seed),
                base_lr=float(base_lr),
                output_dir=str((batch_root / "jobs" / job_id).resolve()),
            )
        )
    return tuple(jobs)


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


def resolve_source_audit(args: argparse.Namespace) -> dict[str, Any]:
    config = (REPO_ROOT / args.cfg).resolve()
    walker_dir = Path(args.walker_dir).resolve()
    morphology_xml = walker_dir / "xml" / f"{args.morphology}.xml"
    morphology_metadata = walker_dir / "metadata" / f"{args.morphology}.json"
    tracked_sources = (
        REPO_ROOT / "tools" / "train_ppo.py",
        REPO_ROOT / "metamorph" / "algos" / "ppo" / "ppo.py",
        REPO_ROOT / "metamorph" / "envs" / "tasks" / "locomotion.py",
        REPO_ROOT / "metamorph" / "envs" / "tasks" / "unimal.py",
        REPO_ROOT / "metamorph" / "envs" / "assets" / "unimal.xml",
    )
    required = (config, morphology_xml, morphology_metadata, *tracked_sources)
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(f"source audit input is missing: {path}")
    return {
        "git_head": git_output("rev-parse", "HEAD"),
        "git_status_short": git_output("status", "--short").splitlines(),
        "config_path": str(config),
        "config_sha256": sha256(config),
        "walker_dir": str(walker_dir),
        "morphology": args.morphology,
        "morphology_xml_path": str(morphology_xml),
        "morphology_xml_sha256": sha256(morphology_xml),
        "morphology_metadata_path": str(morphology_metadata),
        "morphology_metadata_sha256": sha256(morphology_metadata),
        "relevant_source_sha256": {
            str(path.relative_to(REPO_ROOT)): sha256(path)
            for path in tracked_sources
        },
    }


def build_training_command(
    args: argparse.Namespace, job: JobSpec, device: str
) -> list[str]:
    # The original implementation uses KL_TARGET_COEF * 0.01.
    kl_target_coef = args.target_kl / 0.01
    return [
        sys.executable,
        "-B",
        str(REPO_ROOT / "tools" / "train_ppo.py"),
        "--cfg",
        str((REPO_ROOT / args.cfg).resolve()),
        "OUT_DIR",
        job.output_dir,
        "ENV.WALKER_DIR",
        str(Path(args.walker_dir).resolve()),
        "ENV.WALKERS",
        json.dumps([args.morphology], separators=(",", ":")),
        "RNG_SEED",
        str(job.seed),
        "DEVICE",
        device,
        "PPO.BASE_LR",
        repr(job.base_lr),
        "PPO.MAX_STATE_ACTION_PAIRS",
        str(args.max_state_action_pairs),
        "PPO.NUM_ENVS",
        str(args.num_envs),
        "PPO.TIMESTEPS",
        str(args.timesteps),
        "PPO.BATCH_SIZE",
        str(args.batch_size),
        "PPO.GAMMA",
        "0.99",
        "PPO.GAE_LAMBDA",
        "0.95",
        "PPO.EPOCHS",
        "8",
        "PPO.KL_TARGET_COEF",
        repr(kl_target_coef),
        "PPO.CLIP_EPS",
        "0.2",
        "PPO.VALUE_COEF",
        "0.5",
        "PPO.ENTROPY_COEF",
        "0.0",
        "PPO.MAX_GRAD_NORM",
        "0.5",
        "PPO.LR_POLICY",
        "cos",
        "PPO.MIN_LR",
        "0.0",
        "PPO.WARMUP_FACTOR",
        "0.1",
        "PPO.WARMUP_ITERS",
        "5",
        "MODEL.ACTION_STD_FIXED",
        "True",
        "MODEL.ACTION_STD",
        repr(args.action_std),
        "MODEL.TRANSFORMER.DROPOUT",
        "0.0",
        "PPO.CHECKPOINT_PATH",
        "",
        "PPO.EARLY_EXIT",
        "False",
        "VIDEO.SAVE",
        "False",
        "LOG_PERIOD",
        "1",
    ]


def training_profile(args: argparse.Namespace) -> dict[str, Any]:
    transitions_per_iteration = args.num_envs * args.timesteps
    return {
        "fresh_training": True,
        "action_std": args.action_std,
        "action_std_fixed": True,
        "transformer_dropout": 0.0,
        "num_envs_per_job": args.num_envs,
        "rollout_steps": args.timesteps,
        "transitions_per_iteration": transitions_per_iteration,
        "iterations": args.max_state_action_pairs // transitions_per_iteration,
        "max_state_action_pairs": args.max_state_action_pairs,
        "batch_size": args.batch_size,
        "isaac_reference_configured_batch_size": 5120,
        "configured_batch_size_match": args.batch_size == 5120,
        "effective_update_geometry_match": args.batch_size == 512,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "epochs": 8,
        "target_kl": args.target_kl,
        "clip_eps": 0.2,
        "value_loss_coef": 0.5,
        "entropy_coef": 0.0,
        "max_grad_norm": 0.5,
        "lr_schedule": "cos",
        "min_lr_factor": 0.0,
        "warmup_factor": 0.1,
        "warmup_iterations": 5,
    }


def validate_args(args: argparse.Namespace) -> None:
    for name in (
        "max_parallel",
        "num_envs",
        "timesteps",
        "max_state_action_pairs",
        "batch_size",
    ):
        if int(getattr(args, name)) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if not math.isfinite(args.action_std) or args.action_std <= 0:
        raise ValueError("--action-std must be finite and positive")
    if not math.isfinite(args.target_kl) or args.target_kl <= 0:
        raise ValueError("--target-kl must be finite and positive")
    rollout_size = args.num_envs * args.timesteps
    if args.batch_size > rollout_size:
        raise ValueError(
            f"--batch-size {args.batch_size} exceeds rollout size {rollout_size}"
        )
    if args.max_state_action_pairs % rollout_size:
        raise ValueError(
            "--max-state-action-pairs must be exactly divisible by "
            "--num-envs * --timesteps"
        )


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def dry_run_plan(
    *,
    args: argparse.Namespace,
    jobs: Sequence[JobSpec],
    devices: Sequence[str],
    source_audit: dict[str, Any],
) -> dict[str, Any]:
    planned_jobs = []
    for index, job in enumerate(jobs):
        device = devices[index % len(devices)]
        planned_jobs.append(
            {
                **asdict(job),
                "device": device,
                "command": build_training_command(args, job, device),
            }
        )
    return {
        "event": "mujoco_control_experiment_plan",
        "dry_run": True,
        "max_parallel": args.max_parallel,
        "devices": list(devices),
        "job_count": len(jobs),
        "training_profile": training_profile(args),
        "source_audit": source_audit,
        "jobs": planned_jobs,
    }


def checkpoint_metadata(job: JobSpec) -> dict[str, Any]:
    path = Path(job.output_dir) / "Unimal-v0.pt"
    if not path.is_file():
        return {"exists": False, "path": str(path)}
    return {
        "exists": True,
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def archive_batch(batch_root: Path, archive_dir: Path) -> Path:
    archive_dir.mkdir(parents=True, exist_ok=True)
    archive_path = archive_dir / f"{batch_root.name}_diagnostics.zip"
    if archive_path.exists():
        raise FileExistsError(f"archive already exists: {archive_path}")
    with zipfile.ZipFile(
        archive_path, "x", compression=zipfile.ZIP_DEFLATED
    ) as archive:
        for path in sorted(batch_root.rglob("*")):
            if not path.is_file():
                continue
            if path.suffix == ".pt" or "tensorboard" in path.parts:
                continue
            archive.write(path, path.relative_to(batch_root.parent))
    return archive_path


def run_jobs(
    *,
    args: argparse.Namespace,
    jobs: Sequence[JobSpec],
    devices: Sequence[str],
    batch_root: Path,
    manifest: dict[str, Any],
) -> int:
    pending = list(enumerate(jobs))
    active: list[dict[str, Any]] = []
    results = []
    while pending or active:
        while pending and len(active) < args.max_parallel:
            index, job = pending.pop(0)
            device = devices[index % len(devices)]
            output_dir = Path(job.output_dir)
            output_dir.mkdir(parents=True, exist_ok=False)
            log_path = output_dir / "train.log"
            command = build_training_command(args, job, device)
            write_json(
                output_dir / "job.json",
                {
                    **asdict(job),
                    "device": device,
                    "command": command,
                    "transition_budget": args.max_state_action_pairs,
                    "source_commit": manifest["source_audit"]["git_head"],
                    "morphology_sha256": manifest["source_audit"][
                        "morphology_xml_sha256"
                    ],
                },
            )
            log_stream = log_path.open("w", encoding="utf-8")
            started = time.monotonic()
            process = subprocess.Popen(
                command,
                cwd=REPO_ROOT,
                stdout=log_stream,
                stderr=subprocess.STDOUT,
                text=True,
            )
            active.append(
                {
                    "job": job,
                    "device": device,
                    "process": process,
                    "log_stream": log_stream,
                    "log_path": log_path,
                    "started": started,
                    "started_at": datetime.now().astimezone().isoformat(),
                }
            )

        for state in list(active):
            process = state["process"]
            timed_out = bool(
                args.timeout_seconds > 0
                and time.monotonic() - state["started"] > args.timeout_seconds
            )
            if timed_out and process.poll() is None:
                process.terminate()
            return_code = process.poll()
            if return_code is None:
                continue
            state["log_stream"].close()
            job = state["job"]
            checkpoint = checkpoint_metadata(job)
            status = (
                "succeeded"
                if return_code == 0 and checkpoint["exists"] and not timed_out
                else ("timed_out" if timed_out else "failed")
            )
            result = {
                **asdict(job),
                "device": state["device"],
                "pid": process.pid,
                "status": status,
                "return_code": return_code,
                "started_at": state["started_at"],
                "finished_at": datetime.now().astimezone().isoformat(),
                "elapsed_seconds": time.monotonic() - state["started"],
                "log_path": str(state["log_path"]),
                "checkpoint": checkpoint,
            }
            write_json(Path(job.output_dir) / "final_status.json", result)
            results.append(result)
            active.remove(state)
        if active:
            time.sleep(0.2)

    manifest["dry_run"] = False
    manifest["jobs"] = results
    manifest["status"] = (
        "succeeded"
        if all(result["status"] == "succeeded" for result in results)
        else "failed"
    )
    write_json(batch_root / "manifest.json", manifest)
    return 0 if manifest["status"] == "succeeded" else 1


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    validate_args(args)
    seeds = csv_ints(args.seeds, "--seeds")
    base_lrs = csv_floats(args.base_lrs, "--base-lrs")
    devices = csv_strings(args.devices, "--devices")
    timestamp = args.timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_root = (Path(args.output_root) / f"{args.tag}_{timestamp}").resolve()
    jobs = build_job_matrix(
        seeds=seeds, base_lrs=base_lrs, batch_root=batch_root
    )
    source_audit = resolve_source_audit(args)
    plan = dry_run_plan(
        args=args,
        jobs=jobs,
        devices=devices,
        source_audit=source_audit,
    )
    if args.dry_run:
        print(json.dumps(plan, indent=2, sort_keys=True, ensure_ascii=False))
        return 0

    batch_root.mkdir(parents=True, exist_ok=False)
    manifest = {
        **plan,
        "dry_run": False,
        "created_at": datetime.now().astimezone().isoformat(),
        "jobs": [asdict(job) for job in jobs],
        "status": "running",
    }
    write_json(batch_root / "manifest.json", manifest)
    return_code = run_jobs(
        args=args,
        jobs=jobs,
        devices=devices,
        batch_root=batch_root,
        manifest=manifest,
    )
    if args.archive:
        archive_path = archive_batch(batch_root, Path(args.archive_dir).resolve())
        print(f"Archive: {archive_path}")
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())

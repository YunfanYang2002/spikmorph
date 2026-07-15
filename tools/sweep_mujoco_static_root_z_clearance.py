"""Statically sweep free-root height using reset, state injection, and mj_forward only."""

from __future__ import annotations

import argparse
import json
import math
import sys
from decimal import Decimal
from pathlib import Path

try:
    from tools import dump_mujoco_transition as transition
except ImportError:  # Direct execution as ``python tools/<script>.py``.
    import dump_mujoco_transition as transition


SCHEMA_VERSION = "metamorph-static-root-z-clearance-v1"
BACKEND = "mujoco"
NOT_AVAILABLE = transition.NOT_AVAILABLE
REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Sweep static model-default root height without calling mj_step."
    )
    parser.add_argument("--cfg", required=True, type=Path)
    parser.add_argument("--walker-dir", required=True, type=Path)
    parser.add_argument("--morphology", default=transition.DEFAULT_MORPHOLOGY)
    parser.add_argument("--source-xml", required=True, type=Path)
    parser.add_argument(
        "--expected-source-xml-sha256", required=True, type=transition.parse_sha256
    )
    parser.add_argument(
        "--canonical-qpos-mode",
        choices=("model-default",),
        default="model-default",
    )
    parser.add_argument("--root-z-min", required=True, type=float)
    parser.add_argument("--root-z-max", required=True, type=float)
    parser.add_argument("--root-z-step", required=True, type=float)
    parser.add_argument("--penetration-tolerance", default=0.001, type=float)
    parser.add_argument("--safety-margin", default=0.02, type=float)
    parser.add_argument("--refine-resolution", default=0.001, type=float)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    finite_fields = (
        "root_z_min", "root_z_max", "root_z_step", "penetration_tolerance",
        "safety_margin", "refine_resolution",
    )
    for field in finite_fields:
        if not math.isfinite(getattr(args, field)):
            parser.error("--{} must be finite".format(field.replace("_", "-")))
    if args.root_z_max < args.root_z_min:
        parser.error("--root-z-max must be greater than or equal to --root-z-min")
    if args.root_z_step <= 0.0:
        parser.error("--root-z-step must be positive")
    if args.penetration_tolerance < 0.0:
        parser.error("--penetration-tolerance must be non-negative")
    if args.safety_margin < 0.0:
        parser.error("--safety-margin must be non-negative")
    if args.refine_resolution <= 0.0:
        parser.error("--refine-resolution must be positive")
    return args


def decimal_grid(start, stop, step, include_start=True):
    start_d, stop_d, step_d = Decimal(str(start)), Decimal(str(stop)), Decimal(str(step))
    value = start_d if include_start else start_d + step_d
    values = []
    while value <= stop_d:
        values.append(float(value))
        value += step_d
    return values


def strict_json_for_root_z(value, root_z, field="root"):
    if hasattr(value, "tolist"):
        return strict_json_for_root_z(value.tolist(), root_z, field)
    if isinstance(value, dict):
        return {
            str(key): strict_json_for_root_z(item, root_z, "{}.{}".format(field, key))
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [
            strict_json_for_root_z(item, root_z, "{}[{}]".format(field, index))
            for index, item in enumerate(value)
        ]
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return value
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(
                "non-finite field={} root_z={} value={}".format(field, root_z, value)
            )
        return float(value)
    if hasattr(value, "item"):
        return strict_json_for_root_z(value.item(), root_z, field)
    raise TypeError("unsupported JSON value at {}: {}".format(field, type(value).__name__))


def static_ground_contacts(sim, names, ground, robot_geoms, np):
    model, data = sim.model, sim.data
    records = []
    for contact_index in range(int(data.ncon)):
        contact = data.contact[contact_index]
        geom1_id, geom2_id = int(contact.geom1), int(contact.geom2)
        if not transition.is_robot_ground_contact(
            geom1_id, geom2_id, ground["geom_id"], robot_geoms
        ):
            continue
        body1_id = int(model.geom_bodyid[geom1_id])
        body2_id = int(model.geom_bodyid[geom2_id])
        robot_geom_id = geom2_id if geom1_id == ground["geom_id"] else geom1_id
        robot_body_id = int(model.geom_bodyid[robot_geom_id])
        frame = np.asarray(contact.frame, dtype=np.float64).reshape(3, 3)
        records.append(
            {
                "contact_index": contact_index,
                "geom1_id": geom1_id,
                "geom1_name": names["geom"][geom1_id],
                "geom1_body_name": names["body"][body1_id],
                "geom2_id": geom2_id,
                "geom2_name": names["geom"][geom2_id],
                "geom2_body_name": names["body"][body2_id],
                "robot_body_name": names["body"][robot_body_id],
                "robot_geom_name": names["geom"][robot_geom_id],
                "ground_geom_name": ground["geom_name"],
                "distance": float(contact.dist),
                "penetration_depth": transition.penetration_depth(contact.dist),
                "contact_position": np.asarray(contact.pos, dtype=np.float64).copy(),
                "contact_normal": frame[0].copy(),
            }
        )
    return records


def evaluate_root_z(env, root_z, args, names, ground, robot_geoms, torso_body_id, np):
    base = transition.prepare_first_ground_contact_env(env)
    sim, model = base.sim, base.sim.model
    target_joint_id = next(
        (
            joint_id for joint_id in range(int(model.njnt))
            if int(model.jnt_type[joint_id]) in (2, 3)
        ),
        None,
    )
    if target_joint_id is None:
        raise ValueError("compiled model has no scalar policy joint")
    canonical = transition.build_canonical_state(
        model,
        root_z,
        names["joint"],
        args.canonical_qpos_mode,
        np,
        target_joint_id=target_joint_id,
        target_joint_initial_position="default",
    )
    transition.apply_and_verify_canonical_state(
        base, canonical, np.zeros(env.action_space.shape, dtype=np.float32), np
    )
    contacts = static_ground_contacts(sim, names, ground, robot_geoms, np)
    root_address = canonical["root_qpos_adr"]
    torso = transition.runtime_torso_fields(
        sim.data, torso_body_id, names["body"][torso_body_id], 0, np
    )
    depths = [float(item["penetration_depth"]) for item in contacts]
    distances = [float(item["distance"]) for item in contacts]
    record = {
        "root_z": float(root_z),
        "root_position": np.asarray(
            sim.data.qpos[root_address:root_address + 3], dtype=np.float64
        ).copy(),
        "torso_position": torso["torso_position"],
        "torso_height": torso["torso_height"],
        "qvel_abs_max": float(np.max(np.abs(np.asarray(sim.data.qvel)))) if int(model.nv) else 0.0,
        "contact_count_total": int(sim.data.ncon),
        "ground_contact_count": len(contacts),
        "max_penetration_depth": max(depths, default=0.0),
        "minimum_contact_distance": min(distances) if distances else NOT_AVAILABLE,
        "contacting_robot_body_names": sorted({item["robot_body_name"] for item in contacts}),
        "contacting_robot_geom_names": sorted({item["robot_geom_name"] for item in contacts}),
        "ground_contacts": contacts,
        "is_clear": len(contacts) == 0,
        "within_penetration_tolerance": max(depths, default=0.0) <= args.penetration_tolerance,
    }
    return strict_json_for_root_z(record, root_z)


def execute_sweep(evaluator, args):
    records = []
    coarse = []
    for root_z in decimal_grid(args.root_z_min, args.root_z_max, args.root_z_step):
        record = dict(evaluator(root_z))
        record["phase"] = "coarse"
        records.append(record)
        coarse.append(record)
    first_clear_index = next(
        (index for index, record in enumerate(coarse) if record["is_clear"]), None
    )
    coarse_first_clear = (
        coarse[first_clear_index]["root_z"] if first_clear_index is not None else None
    )
    last_non_clear = (
        coarse[first_clear_index - 1]["root_z"]
        if first_clear_index is not None and first_clear_index > 0
        else None
    )
    refined = []
    refined_minimum = coarse_first_clear
    refined_last_non_clear = last_non_clear
    if coarse_first_clear is not None and last_non_clear is not None:
        for root_z in decimal_grid(
            last_non_clear, coarse_first_clear, args.refine_resolution, include_start=False
        ):
            record = dict(evaluator(root_z))
            record["phase"] = "refined"
            records.append(record)
            refined.append(record)
            if record["is_clear"]:
                refined_minimum = record["root_z"]
                break
            refined_last_non_clear = record["root_z"]
    recommended = (
        float(Decimal(str(refined_minimum)) + Decimal(str(args.safety_margin)))
        if refined_minimum is not None
        else None
    )
    root_min_record = coarse[0]
    summary = {
        "ok": refined_minimum is not None,
        "tested_count": len(records),
        "coarse_tested_count": len(coarse),
        "refined_tested_count": len(refined),
        "coarse_first_clear_root_z": coarse_first_clear,
        "refined_minimum_clear_root_z": refined_minimum,
        "recommended_root_z": recommended,
        "last_non_clear_root_z": refined_last_non_clear,
        "maximum_penetration_at_root_z_min": root_min_record["max_penetration_depth"],
        "contacting_bodies_at_root_z_min": root_min_record["contacting_robot_body_names"],
        "highest_tested_root_z_state": coarse[-1] if refined_minimum is None else NOT_AVAILABLE,
        "suggestion": (
            "increase --root-z-max" if refined_minimum is None else NOT_AVAILABLE
        ),
    }
    return records, summary


def run(args):
    import numpy as np

    cfg_path = transition.require_file(args.cfg, "config file")
    args.cfg = cfg_path
    walker_dir = args.walker_dir.expanduser().resolve()
    if not walker_dir.is_dir():
        raise FileNotFoundError("walker directory does not exist: {}".format(walker_dir))
    source_xml = transition.require_file(args.source_xml, "source XML")
    source_hash = transition.sha256_file(source_xml)
    if source_hash != args.expected_source_xml_sha256:
        raise ValueError(
            "source XML SHA-256 mismatch: expected {}, calculated {}".format(
                args.expected_source_xml_sha256, source_hash
            )
        )
    morphology_xml = transition.require_file(
        walker_dir / "xml" / "{}.xml".format(args.morphology), "morphology XML"
    )
    metadata_path = transition.require_file(
        walker_dir / "metadata" / "{}.json".format(args.morphology),
        "morphology metadata",
    )
    output_dir = args.output.expanduser().resolve()
    transition.prepare_output_directory(output_dir)
    cfg = transition.configure_environment(args, walker_dir, metadata_path)
    from metamorph.algos.ppo.envs import make_env

    env = make_env(cfg.ENV_NAME, int(cfg.RNG_SEED), 0, xml_file=args.morphology)()
    try:
        base = transition.prepare_first_ground_contact_env(env)
        model = base.sim.model
        names = {
            "joint": transition.model_names(model, "joint", int(model.njnt)),
            "body": transition.model_names(model, "body", int(model.nbody)),
            "geom": transition.model_names(model, "geom", int(model.ngeom)),
        }
        ground = transition.identify_unique_ground_geom(
            model, names["geom"], names["body"]
        )
        robot_geoms = transition.robot_geom_ids(model, names["geom"], names["body"])
        torso_body_id = transition.unique_named_body(names["body"], "torso/0")

        def evaluator(root_z):
            return evaluate_root_z(
                env, root_z, args, names, ground, robot_geoms, torso_body_id, np
            )

        records, summary = execute_sweep(evaluator, args)
        metadata = {
            "schema_version": SCHEMA_VERSION,
            "backend": BACKEND,
            "repository_path": str(REPO_ROOT),
            "evaluation_mode": "static_mj_forward_no_mj_step",
            "morphology_id": args.morphology,
            "source_xml": str(source_xml),
            "source_xml_sha256": source_hash,
            "source_xml_matches_environment_entry_path": source_xml == morphology_xml,
            "canonical_qpos_mode": args.canonical_qpos_mode,
            "qvel_mode": "zero",
            "action_mode": "zero",
            "root_z_min": args.root_z_min,
            "root_z_max": args.root_z_max,
            "root_z_step": args.root_z_step,
            "refine_resolution": args.refine_resolution,
            "penetration_tolerance": args.penetration_tolerance,
            "safety_margin": args.safety_margin,
            "ground_geom_id": ground["geom_id"],
            "ground_geom_name": ground["geom_name"],
            "ground_body_id": ground["body_id"],
            "ground_body_name": ground["body_name"],
            "record_timing": "static_after_mj_forward",
            "mj_step_called": False,
        }
        (output_dir / "metadata.json").write_text(
            json.dumps(strict_json_for_root_z(metadata, NOT_AVAILABLE), indent=2, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        with (output_dir / "root_z_sweep.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        (output_dir / "summary.json").write_text(
            json.dumps(strict_json_for_root_z(summary, NOT_AVAILABLE), indent=2, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        return 0 if summary["ok"] else 2
    finally:
        env.close()


def main(argv=None):
    return run(parse_args(argv))


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print("error: {}".format(error), file=sys.stderr)
        raise SystemExit(1)

"""Dump deterministic, headless MuJoCo transitions for one morphology.

MuJoCo imports are intentionally lazy so ``--help`` and the pure helpers in
this module remain usable on machines without a MuJoCo runtime.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path


SCHEMA_VERSION = "metamorph-transition-v1"
BACKEND = "mujoco"
DEFAULT_MORPHOLOGY = "floor-1409-0-3-01-15-56-55"
VALID_CASES = ("zero", "positive", "negative")
NOT_AVAILABLE = "not_available"
REPO_ROOT = Path(__file__).resolve().parents[1]
TRANSITION_CORE_FIELDS = (
    "schema_version",
    "backend",
    "morphology_id",
    "case",
    "step",
    "joint_names",
    "policy_action",
    "native_action_or_ctrl",
    "requested_torque",
    "applied_torque_if_available",
    "root_position",
    "root_quaternion",
    "root_quaternion_order",
    "root_linear_velocity",
    "root_angular_velocity",
    "joint_qpos",
    "joint_qvel",
)
METADATA_GATE0_FIELDS = (
    "schema_version",
    "backend",
    "morphology_id",
    "source_xml_path",
    "source_xml_sha256",
    "cases",
    "requested_joint_name",
    "physics_timestep_actual",
    "control_timestep",
    "frame_skip",
    "joint_names",
    "actuator_names",
    "actuator_trnid",
    "joint_range",
    "joint_axis",
    "dof_damping",
    "dof_armature",
    "actuator_gear",
    "gravity",
    "canonical_initial_state_spec",
    "policy_action_semantics",
    "native_action_or_ctrl_semantics",
    "requested_torque_semantics",
    "applied_torque_if_available_semantics",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_sha256(value: str) -> str:
    normalized = value.strip().lower()
    if len(normalized) != 64 or any(character not in "0123456789abcdef" for character in normalized):
        raise argparse.ArgumentTypeError("expected a 64-character hexadecimal SHA-256")
    return normalized


def parse_cases(value: str) -> list[str]:
    cases = [item.strip().lower() for item in value.split(",") if item.strip()]
    if not cases:
        raise argparse.ArgumentTypeError("--cases must contain at least one case")
    invalid = [item for item in cases if item not in VALID_CASES]
    if invalid:
        raise argparse.ArgumentTypeError(
            "unknown case(s): {}; expected comma-separated {}".format(
                ", ".join(invalid), ",".join(VALID_CASES)
            )
        )
    duplicates = sorted({item for item in cases if cases.count(item) > 1})
    if duplicates:
        raise argparse.ArgumentTypeError(
            "duplicate case(s): {}".format(", ".join(duplicates))
        )
    return cases


def prepare_output_directory(path: Path) -> None:
    """Create an empty output directory without deleting or overwriting files."""
    if path.exists():
        if not path.is_dir():
            raise NotADirectoryError("--output is not a directory: {}".format(path))
        existing = list(path.iterdir())
        if existing:
            raise FileExistsError(
                "refusing to write into non-empty output directory: {}".format(path)
            )
    else:
        try:
            path.mkdir(parents=True, exist_ok=False)
        except OSError as error:
            raise PermissionError(
                "cannot create output directory: {}".format(path)
            ) from error
    probe = path / ".dump_mujoco_transition_write_probe"
    try:
        probe.write_text("probe", encoding="utf-8")
        probe.unlink()
    except OSError as error:
        raise PermissionError("output directory is not writable: {}".format(path)) from error


def jsonable(value):
    """Convert NumPy-like values to strict JSON-compatible Python objects."""
    if hasattr(value, "tolist"):
        return jsonable(value.tolist())
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return value
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("non-finite value cannot be serialized to strict JSON")
        return float(value)
    if hasattr(value, "item"):
        return jsonable(value.item())
    raise TypeError("unsupported JSON value type: {}".format(type(value).__name__))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Dump canonical MuJoCo transitions without rendering or a policy."
    )
    parser.add_argument("--cfg", required=True, type=Path)
    parser.add_argument("--walker-dir", required=True, type=Path)
    parser.add_argument("--morphology", default=DEFAULT_MORPHOLOGY)
    parser.add_argument("--source-xml", required=True, type=Path)
    parser.add_argument(
        "--expected-source-xml-sha256",
        type=parse_sha256,
        help="Optional Gate 0 hash; mismatch aborts before environment creation.",
    )
    parser.add_argument("--root-z", required=True, type=float)
    parser.add_argument("--steps", default=10, type=int)
    parser.add_argument("--joint-name", default="limbx/0")
    parser.add_argument("--action-magnitude", default=0.25, type=float)
    parser.add_argument("--cases", default=list(VALID_CASES), type=parse_cases)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.steps <= 0:
        parser.error("--steps must be positive")
    if not math.isfinite(args.action_magnitude):
        parser.error("--action-magnitude must be finite")
    if args.action_magnitude < 0.0:
        parser.error("--action-magnitude must be non-negative")
    if not math.isfinite(args.root_z):
        parser.error("--root-z must be finite")
    if not args.joint_name.strip():
        parser.error("--joint-name must not be empty")
    if not args.morphology.strip():
        parser.error("--morphology must not be empty")
    return args


def require_file(path: Path, description: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError("{} does not exist: {}".format(description, resolved))
    return resolved


def optional_array(owner, name):
    try:
        return jsonable(getattr(owner, name))
    except (AttributeError, TypeError, ValueError):
        return NOT_AVAILABLE


def selected_array(owner, name, indices, np):
    try:
        values = np.asarray(getattr(owner, name))
        return jsonable(values[list(indices)])
    except (AttributeError, IndexError, TypeError, ValueError):
        return NOT_AVAILABLE


def validate_transition_schema(record) -> None:
    missing = [field for field in TRANSITION_CORE_FIELDS if field not in record]
    if missing:
        raise ValueError("transition is missing core fields: {}".format(", ".join(missing)))
    if record["schema_version"] != SCHEMA_VERSION:
        raise ValueError("transition schema_version does not match {}".format(SCHEMA_VERSION))
    if record["root_quaternion_order"] != "wxyz":
        raise ValueError("root_quaternion_order must be 'wxyz'")
    joint_count = len(record["joint_names"])
    for field in ("requested_torque", "joint_qpos", "joint_qvel"):
        value = record[field]
        if not isinstance(value, str) and len(value) != joint_count:
            raise ValueError("{} length does not match joint_names".format(field))


def validate_canonical_step0(records, cases) -> None:
    state_fields = (
        "root_position",
        "root_quaternion",
        "root_linear_velocity",
        "root_angular_velocity",
        "joint_qpos",
        "joint_qvel",
    )
    signatures = []
    for case in cases:
        matches = [
            record
            for record in records
            if record.get("case") == case and record.get("step") == 0
        ]
        if len(matches) != 1:
            raise ValueError("case {!r} must have exactly one step 0 record".format(case))
        signatures.append(
            {field: jsonable(matches[0][field]) for field in state_fields}
        )
    if any(signature != signatures[0] for signature in signatures[1:]):
        raise ValueError("canonical step 0 state differs between action cases")


def validate_metadata_gate0(metadata) -> None:
    missing = [field for field in METADATA_GATE0_FIELDS if field not in metadata]
    if missing:
        raise ValueError("metadata is missing Gate 0 fields: {}".format(", ".join(missing)))
    if metadata["schema_version"] != SCHEMA_VERSION:
        raise ValueError("metadata schema_version does not match {}".format(SCHEMA_VERSION))
    if metadata["policy_action_semantics"] != "actuator_order_ctrl":
        raise ValueError("unexpected policy_action_semantics")
    if metadata["native_action_or_ctrl_semantics"] != "mujoco_actuator_order_ctrl":
        raise ValueError("unexpected native_action_or_ctrl_semantics")
    expected_control_timestep = (
        float(metadata["physics_timestep_actual"]) * int(metadata["frame_skip"])
    )
    if not math.isclose(
        float(metadata["control_timestep"]),
        expected_control_timestep,
        rel_tol=0.0,
        abs_tol=1e-15,
    ):
        raise ValueError("control_timestep does not equal physics_timestep_actual * frame_skip")


def model_names(model, kind: str, count: int) -> list[str]:
    plural = "{}s".format(kind)
    try:
        values = list(getattr(model, "{}_names".format(kind)))
        if len(values) == count:
            return ["" if value is None else str(value) for value in values]
    except (AttributeError, TypeError):
        pass

    # The repository's modern MuJoCo adapter exposes the underlying MjModel;
    # mujoco-py exposes the plural name arrays directly above.
    raw_model = getattr(model, "_model", model)
    from metamorph.utils import mujoco_compat

    if mujoco_compat.BACKEND == "mujoco":
        enum_name = "mjOBJ_{}".format(kind.upper())
        obj_type = getattr(mujoco_compat.mujoco.mjtObj, enum_name)
        return [
            mujoco_compat.mujoco.mj_id2name(raw_model, obj_type, index) or ""
            for index in range(count)
        ]
    raise RuntimeError("unable to read {} names from compiled model".format(plural))


def joint_widths(model, joint_id: int) -> tuple[int, int]:
    qpos_start = int(model.jnt_qposadr[joint_id])
    dof_start = int(model.jnt_dofadr[joint_id])
    qpos_end = int(model.jnt_qposadr[joint_id + 1]) if joint_id + 1 < model.njnt else int(model.nq)
    dof_end = int(model.jnt_dofadr[joint_id + 1]) if joint_id + 1 < model.njnt else int(model.nv)
    return qpos_end - qpos_start, dof_end - dof_start


def build_canonical_state(model, root_z: float, joint_names, np):
    reference_qpos = np.asarray(model.qpos0, dtype=np.float64).reshape(-1).copy()
    if reference_qpos.shape != (int(model.nq),):
        raise ValueError("model qpos0 length does not match model.nq")
    qpos = reference_qpos.copy()
    qvel = np.zeros(int(model.nv), dtype=np.float64)
    free_joint_ids = [
        joint_id for joint_id in range(int(model.njnt))
        if int(model.jnt_type[joint_id]) == 0
    ]
    if len(free_joint_ids) != 1:
        raise ValueError(
            "expected exactly one free root joint, found {}".format(len(free_joint_ids))
        )
    root_joint_id = free_joint_ids[0]
    root_qpos_adr = int(model.jnt_qposadr[root_joint_id])
    root_dof_adr = int(model.jnt_dofadr[root_joint_id])
    qwidth, dwidth = joint_widths(model, root_joint_id)
    if (qwidth, dwidth) != (7, 6):
        raise ValueError("free root joint does not have 7 qpos and 6 qvel entries")
    qpos[root_qpos_adr:root_qpos_adr + 7] = [0.0, 0.0, root_z, 1.0, 0.0, 0.0, 0.0]
    qvel[root_dof_adr:root_dof_adr + 6] = 0.0

    fallback_joints = []
    non_root_joint_ids = []
    policy_joint_ids = []
    for joint_id in range(int(model.njnt)):
        if joint_id == root_joint_id:
            continue
        qwidth, dwidth = joint_widths(model, joint_id)
        if (qwidth, dwidth) != (1, 1):
            raise ValueError(
                "non-root joint {!r} is not a 1-DOF joint".format(joint_names[joint_id])
            )
        non_root_joint_ids.append(joint_id)
        qpos_adr = int(model.jnt_qposadr[joint_id])
        joint_type = int(model.jnt_type[joint_id])
        if joint_type == 3:
            policy_joint_ids.append(joint_id)
        limited = bool(model.jnt_limited[joint_id])
        limits = np.asarray(model.jnt_range[joint_id], dtype=np.float64)
        if joint_type == 3 and limited and np.all(np.isfinite(limits)):
            qpos[qpos_adr] = float((limits[0] + limits[1]) / 2.0)
        else:
            qpos[qpos_adr] = reference_qpos[qpos_adr]
            fallback_joints.append(
                {
                    "joint_id": joint_id,
                    "joint_name": joint_names[joint_id],
                    "reason": "unlimited_hinge" if joint_type == 3 and not limited else "non_hinge_or_non_finite_range",
                    "reference_qpos": float(reference_qpos[qpos_adr]),
                }
            )
    if qpos.shape != (int(model.nq),) or qvel.shape != (int(model.nv),):
        raise ValueError("canonical qpos/qvel length does not match compiled model")
    return {
        "qpos": qpos,
        "qvel": qvel,
        "root_joint_id": root_joint_id,
        "root_qpos_adr": root_qpos_adr,
        "root_dof_adr": root_dof_adr,
        "non_root_joint_ids": non_root_joint_ids,
        "policy_joint_ids": policy_joint_ids,
        "fallback_joints": fallback_joints,
    }


def resolve_target(model, requested_name, joint_names, actuator_names, np):
    matches = [index for index, name in enumerate(joint_names) if name == requested_name]
    if len(matches) != 1:
        raise ValueError(
            "target joint {!r} matched {} compiled joint names; no index fallback is allowed".format(
                requested_name, len(matches)
            )
        )
    joint_id = matches[0]
    if int(model.jnt_type[joint_id]) != 3 or joint_widths(model, joint_id) != (1, 1):
        raise ValueError("target joint {!r} is not a scalar hinge joint".format(requested_name))
    trnid = np.asarray(model.actuator_trnid, dtype=np.int64)
    try:
        trntype = np.asarray(model.actuator_trntype, dtype=np.int64).reshape(-1)
    except (AttributeError, TypeError, ValueError):
        trntype = None
    actuator_matches = [
        index
        for index in range(int(model.nu))
        if int(trnid[index, 0]) == joint_id
        and (trntype is None or int(trntype[index]) in (0, 1))
    ]
    if len(actuator_matches) != 1:
        raise ValueError(
            "target joint {!r} is referenced by {} actuators; expected exactly one".format(
                requested_name, len(actuator_matches)
            )
        )
    actuator_id = actuator_matches[0]
    gear = np.asarray(model.actuator_gear[actuator_id], dtype=np.float64)
    if not np.any(np.abs(gear) > 0.0) or gear.size == 0 or gear[0] == 0.0:
        raise ValueError("target actuator gear is zero for joint {!r}".format(requested_name))
    if not 0 <= actuator_id < int(model.nu):
        raise ValueError("target actuator ctrl index is outside model.nu")
    return {
        "joint_id": joint_id,
        "runtime_name": joint_names[joint_id],
        "actuator_id": actuator_id,
        "actuator_name": actuator_names[actuator_id],
        "ctrl_index": actuator_id,
    }


def non_root_dof_indices(model, joint_ids):
    indices = []
    for joint_id in joint_ids:
        start = int(model.jnt_dofadr[joint_id])
        _, width = joint_widths(model, joint_id)
        indices.extend(range(start, start + width))
    return indices


def requested_generalized_torque(model, ctrl, non_root_ids, np):
    torque = np.zeros(int(model.nv), dtype=np.float64)
    trnid = np.asarray(model.actuator_trnid, dtype=np.int64)
    gear = np.asarray(model.actuator_gear, dtype=np.float64)
    try:
        trntype = np.asarray(model.actuator_trntype, dtype=np.int64).reshape(-1)
    except (AttributeError, TypeError, ValueError):
        trntype = None
    for actuator_id in range(int(model.nu)):
        if trntype is not None and int(trntype[actuator_id]) not in (0, 1):
            raise ValueError(
                "requested torque calculation encountered a non-joint transmission"
            )
        joint_id = int(trnid[actuator_id, 0])
        if joint_id < 0:
            continue
        _, dof_width = joint_widths(model, joint_id)
        if dof_width != 1:
            raise ValueError("requested torque calculation only supports 1-DOF actuated joints")
        torque[int(model.jnt_dofadr[joint_id])] += float(ctrl[actuator_id]) * float(gear[actuator_id, 0])
    dof_indices = non_root_dof_indices(model, non_root_ids)
    return torque[dof_indices]


def contact_details(sim, geom_names, body_names):
    model, data = sim.model, sim.data
    count = int(data.ncon)
    details = []
    try:
        geom_bodyid = model.geom_bodyid
        for index in range(count):
            contact = data.contact[index]
            geom1, geom2 = int(contact.geom1), int(contact.geom2)
            body1, body2 = int(geom_bodyid[geom1]), int(geom_bodyid[geom2])
            details.append(
                {
                    "geom1": geom_names[geom1],
                    "geom2": geom_names[geom2],
                    "body1": body_names[body1],
                    "body2": body_names[body2],
                }
            )
    except (AttributeError, IndexError, TypeError):
        return count, NOT_AVAILABLE
    return count, details


def force_slice(data, name, dof_indices, np):
    try:
        values = np.asarray(getattr(data, name), dtype=np.float64).reshape(-1)
        return values[dof_indices]
    except (AttributeError, IndexError, TypeError, ValueError):
        return NOT_AVAILABLE


def transition_record(base_env, morphology, case, step, ctrl, canonical, names, np):
    sim, model, data = base_env.sim, base_env.sim.model, base_env.sim.data
    root_qpos = canonical["root_qpos_adr"]
    root_dof = canonical["root_dof_adr"]
    joint_ids = canonical["policy_joint_ids"]
    qpos_indices = [int(model.jnt_qposadr[joint_id]) for joint_id in joint_ids]
    dof_indices = non_root_dof_indices(model, joint_ids)
    contact_count, contacts = contact_details(sim, names["geom"], names["body"])
    try:
        actuator_force = np.asarray(data.actuator_force, dtype=np.float64).reshape(-1).copy()
    except (AttributeError, TypeError, ValueError):
        actuator_force = NOT_AVAILABLE
    applied_torque = force_slice(data, "qfrc_actuator", dof_indices, np)
    record = {
        "schema_version": SCHEMA_VERSION,
        "backend": BACKEND,
        "morphology_id": morphology,
        "case": case,
        "step": step,
        "joint_names": [names["joint"][joint_id] for joint_id in joint_ids],
        "policy_action": ctrl.copy(),
        "native_action_or_ctrl": np.asarray(data.ctrl, dtype=np.float64).reshape(-1).copy(),
        "requested_torque": requested_generalized_torque(model, ctrl, joint_ids, np),
        "applied_torque_if_available": applied_torque,
        "actuator_force_if_available": actuator_force,
        "joint_actuator_torque_if_available": applied_torque,
        "passive_force_if_available": force_slice(data, "qfrc_passive", dof_indices, np),
        "root_position": np.asarray(data.qpos[root_qpos:root_qpos + 3], dtype=np.float64).copy(),
        "root_quaternion": np.asarray(data.qpos[root_qpos + 3:root_qpos + 7], dtype=np.float64).copy(),
        "root_quaternion_order": "wxyz",
        "root_linear_velocity": np.asarray(data.qvel[root_dof:root_dof + 3], dtype=np.float64).copy(),
        "root_angular_velocity": np.asarray(data.qvel[root_dof + 3:root_dof + 6], dtype=np.float64).copy(),
        "joint_qpos": np.asarray(data.qpos, dtype=np.float64)[qpos_indices].copy(),
        "joint_qvel": np.asarray(data.qvel, dtype=np.float64)[dof_indices].copy(),
        "contact_count_if_available": contact_count,
        "contacts_if_available": contacts,
        "done": False,
    }
    validate_transition_schema(record)
    return record


def build_ctrl(case, magnitude, size, target_index, np):
    ctrl = np.zeros(size, dtype=np.float64)
    if case == "positive":
        ctrl[target_index] = magnitude
    elif case == "negative":
        ctrl[target_index] = -magnitude
    return ctrl


def configure_environment(args, walker_dir, metadata_path):
    from metamorph.config import cfg

    cfg.merge_from_file(str(args.cfg.resolve()))
    cfg.ENV.WALKER_DIR = str(walker_dir)
    cfg.ENV.WALKERS = [args.morphology]
    with metadata_path.open("r", encoding="utf-8") as handle:
        walker_metadata = json.load(handle)
    cfg.MODEL.MAX_JOINTS = int(walker_metadata["dof"]) + 1
    cfg.MODEL.MAX_LIMBS = int(walker_metadata["num_limbs"]) + 2
    return cfg


def run(args) -> None:
    import numpy as np

    cfg_path = require_file(args.cfg, "config file")
    args.cfg = cfg_path
    walker_dir = args.walker_dir.expanduser().resolve()
    if not walker_dir.is_dir():
        raise FileNotFoundError("walker directory does not exist: {}".format(walker_dir))
    source_xml = require_file(args.source_xml, "source XML")
    source_xml_sha256 = sha256_file(source_xml)
    if (
        args.expected_source_xml_sha256 is not None
        and source_xml_sha256 != args.expected_source_xml_sha256
    ):
        raise ValueError(
            "source XML SHA-256 mismatch: expected {}, calculated {}".format(
                args.expected_source_xml_sha256, source_xml_sha256
            )
        )
    morphology_xml = require_file(
        walker_dir / "xml" / "{}.xml".format(args.morphology),
        "morphology XML",
    )
    metadata_path = require_file(
        walker_dir / "metadata" / "{}.json".format(args.morphology),
        "morphology metadata",
    )
    output_dir = args.output.expanduser().resolve()
    prepare_output_directory(output_dir)
    cfg = configure_environment(args, walker_dir, metadata_path)

    # Lazy import avoids importing a MuJoCo backend during local static tests.
    from metamorph.algos.ppo.envs import make_env

    env = make_env(cfg.ENV_NAME, int(cfg.RNG_SEED), 0, xml_file=args.morphology)()
    try:
        base = env.unwrapped
        sim, model = base.sim, base.sim.model
        names = {
            "joint": model_names(model, "joint", int(model.njnt)),
            "actuator": model_names(model, "actuator", int(model.nu)),
            "body": model_names(model, "body", int(model.nbody)),
            "geom": model_names(model, "geom", int(model.ngeom)),
        }
        target = resolve_target(
            model, args.joint_name, names["joint"], names["actuator"], np
        )
        if target["ctrl_index"] >= np.asarray(sim.data.ctrl).size:
            raise ValueError("target ctrl index is outside sim.data.ctrl")
        canonical = build_canonical_state(model, args.root_z, names["joint"], np)
        records = []
        contact_summary = {}
        for case in args.cases:
            ctrl = build_ctrl(
                case, args.action_magnitude, int(model.nu), target["ctrl_index"], np
            )
            sim.reset()
            base.set_state(canonical["qpos"].copy(), canonical["qvel"].copy())
            sim.data.ctrl[:] = ctrl
            sim.forward()
            record = transition_record(base, args.morphology, case, 0, ctrl, canonical, names, np)
            records.append(record)
            max_contacts = int(record["contact_count_if_available"])
            for step in range(1, args.steps + 1):
                if base.do_simulation(ctrl):
                    raise RuntimeError(
                        "MuJoCo step failed for case {!r} at control step {}".format(case, step)
                    )
                record = transition_record(
                    base, args.morphology, case, step, ctrl, canonical, names, np
                )
                records.append(record)
                max_contacts = max(
                    max_contacts, int(record["contact_count_if_available"])
                )
            contact_summary[case] = {"max_contact_count": max_contacts}

        validate_canonical_step0(records, args.cases)

        control_timestep = float(model.opt.timestep) * int(base.frame_skip)
        physics_timestep_actual = float(model.opt.timestep)
        source_matches_entry = source_xml == morphology_xml
        policy_joint_ids = canonical["policy_joint_ids"]
        policy_dof_indices = non_root_dof_indices(model, policy_joint_ids)
        policy_joint_names = [names["joint"][joint_id] for joint_id in policy_joint_ids]
        metadata = {
            "schema_version": SCHEMA_VERSION,
            "backend": BACKEND,
            "repository_path": str(REPO_ROOT),
            "config_path": str(cfg_path),
            "morphology_id": args.morphology,
            "source_xml_path": str(source_xml),
            "source_xml_sha256": source_xml_sha256,
            "source_xml_sha256_expected": (
                args.expected_source_xml_sha256
                if args.expected_source_xml_sha256 is not None
                else NOT_AVAILABLE
            ),
            "source_xml_sha256_matches_expected": (
                source_xml_sha256 == args.expected_source_xml_sha256
                if args.expected_source_xml_sha256 is not None
                else NOT_AVAILABLE
            ),
            "source_xml_role": "user_declared_source_xml",
            "source_xml_matches_environment_entry_path": source_matches_entry,
            "source_xml_internal_equivalence_verified": False,
            "environment_morphology_entry_path": str(morphology_xml),
            "walker_dir": str(walker_dir),
            "physics_timestep": physics_timestep_actual,
            "physics_timestep_actual": physics_timestep_actual,
            "frame_skip": int(base.frame_skip),
            "control_timestep": control_timestep,
            "gravity": optional_array(model.opt, "gravity"),
            "joint_names": policy_joint_names,
            "compiled_joint_names": names["joint"],
            "actuator_names": names["actuator"],
            "actuator_trnid": optional_array(model, "actuator_trnid"),
            "requested_joint_name": args.joint_name,
            "target_joint_requested_name": args.joint_name,
            "target_joint_runtime_name": target["runtime_name"],
            "target_joint_id": target["joint_id"],
            "target_actuator_id": target["actuator_id"],
            "target_actuator_name": target["actuator_name"],
            "target_ctrl_index": target["ctrl_index"],
            "joint_axis": selected_array(model, "jnt_axis", policy_joint_ids, np),
            "joint_range": selected_array(model, "jnt_range", policy_joint_ids, np),
            "dof_damping": selected_array(model, "dof_damping", policy_dof_indices, np),
            "dof_armature": selected_array(model, "dof_armature", policy_dof_indices, np),
            "compiled_jnt_axis": optional_array(model, "jnt_axis"),
            "compiled_jnt_range": optional_array(model, "jnt_range"),
            "compiled_dof_damping": optional_array(model, "dof_damping"),
            "compiled_dof_armature": optional_array(model, "dof_armature"),
            "actuator_gear": optional_array(model, "actuator_gear"),
            "actuator_ctrlrange": optional_array(model, "actuator_ctrlrange"),
            "actuator_forcelimited": optional_array(model, "actuator_forcelimited"),
            "actuator_forcerange": optional_array(model, "actuator_forcerange"),
            "body_names": names["body"],
            "body_mass": optional_array(model, "body_mass"),
            "body_inertia": optional_array(model, "body_inertia"),
            "body_ipos": optional_array(model, "body_ipos"),
            "body_iquat": optional_array(model, "body_iquat"),
            "canonical_initial_state_spec": {
                "root_position": [0.0, 0.0, args.root_z],
                "root_quaternion": [1.0, 0.0, 0.0, 0.0],
                "root_linear_velocity": [0.0, 0.0, 0.0],
                "root_angular_velocity": [0.0, 0.0, 0.0],
                "limited_hinge_qpos": "compiled jnt_range midpoint",
                "unlimited_or_non_hinge_qpos": "compiled model qpos0 reference",
                "joint_qvel": "all zero",
                "fallback_joints": canonical["fallback_joints"],
                "reset_noise_bypassed": True,
                "reapplied_before_each_case": True,
            },
            "policy_action_semantics": "actuator_order_ctrl",
            "environment_creation_path": "metamorph.algos.ppo.envs.make_env -> gym.make -> task.make_env",
            "simulation_path": "env.unwrapped.set_state -> sim.forward -> env.unwrapped.do_simulation",
            "formal_action_wrapper_mapping": "MultiUnimalNodeCentricAction removes padded slots (and may mirror); this tool deliberately starts at its native actuator-order output",
            "native_action_or_ctrl_semantics": "mujoco_actuator_order_ctrl",
            "requested_torque_semantics": "joint_names order; sum(ctrl[actuator] * actuator_gear[actuator, 0]) via scalar joint transmissions and compiled joint/dof addresses",
            "applied_torque_if_available_semantics": "data.qfrc_actuator at compiled DOF addresses in joint_names order; not copied from requested_torque",
            "actuator_force_semantics": "data.actuator_force in actuator order when available",
            "joint_actuator_torque_semantics": "data.qfrc_actuator at compiled non-root joint DOF addresses",
            "passive_force_semantics": "data.qfrc_passive at compiled non-root joint DOF addresses",
            "cases": args.cases,
            "steps": args.steps,
            "action_magnitude": args.action_magnitude,
            "contact_summary": contact_summary,
            "contact_free_valid": all(
                item["max_contact_count"] == 0 for item in contact_summary.values()
            ),
        }
        validate_metadata_gate0(metadata)
        metadata_path_out = output_dir / "metadata.json"
        transitions_path_out = output_dir / "transitions.jsonl"
        metadata_path_out.write_text(
            json.dumps(jsonable(metadata), indent=2, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        with transitions_path_out.open("w", encoding="utf-8", newline="\n") as handle:
            for record in records:
                handle.write(
                    json.dumps(jsonable(record), ensure_ascii=False, allow_nan=False) + "\n"
                )
        print("Wrote {} transitions to {}".format(len(records), output_dir))
        print("contact_free_valid={}".format(metadata["contact_free_valid"]))
    finally:
        env.close()


def main(argv=None) -> int:
    args = parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print("error: {}".format(error), file=sys.stderr)
        raise SystemExit(1)

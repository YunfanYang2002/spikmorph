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
from collections import deque
from pathlib import Path


SCHEMA_VERSION = "metamorph-transition-v1"
RUNTIME_DYNAMICS_SCHEMA_VERSION = "metamorph-runtime-dynamics-v1"
GROUND_CONTACT_SCHEMA_VERSION = "metamorph-first-ground-contact-window-v1"
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
    "canonical_qpos_mode",
    "canonical_joint_qpos",
    "model_default_joint_qpos",
    "target_joint_initial_position_mode",
    "target_joint_initial_qpos_requested",
    "target_joint_initial_qpos_readback",
    "target_joint_range",
    "root_qpos_source",
    "joint_qpos_source",
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
    parser.add_argument(
        "--canonical-qpos-mode",
        choices=("midpoint", "model-default"),
        default="midpoint",
    )
    parser.add_argument(
        "--target-joint-initial-position",
        choices=("default", "midpoint"),
        default="default",
    )
    parser.add_argument("--steps", default=10, type=int)
    parser.add_argument("--record-physics-substeps", action="store_true")
    parser.add_argument("--record-first-ground-contact-window", action="store_true")
    parser.add_argument("--contact-window-before", default=2, type=int)
    parser.add_argument("--contact-window-after", default=3, type=int)
    parser.add_argument("--max-physics-substeps", default=400, type=int)
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
    if args.contact_window_before < 0:
        parser.error("--contact-window-before must be non-negative")
    if args.contact_window_after < 0:
        parser.error("--contact-window-after must be non-negative")
    if args.max_physics_substeps <= 0:
        parser.error("--max-physics-substeps must be positive")
    if args.record_first_ground_contact_window and args.cases != ["zero"]:
        parser.error("first ground contact window mode requires --cases zero")
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


JOINT_TYPE_NAMES = {
    0: "free",
    1: "ball",
    2: "slide",
    3: "hinge",
}
ACTUATOR_DYN_TYPE_NAMES = {
    0: "none",
    1: "integrator",
    2: "filter",
    3: "filterexact",
    4: "muscle",
    5: "user",
}
ACTUATOR_GAIN_TYPE_NAMES = {
    0: "fixed",
    1: "affine",
    2: "muscle",
    3: "user",
}
ACTUATOR_BIAS_TYPE_NAMES = {
    0: "none",
    1: "affine",
    2: "muscle",
    3: "user",
}
INTEGRATOR_NAMES = {
    0: "Euler",
    1: "RK4",
    2: "implicit",
    3: "implicitfast",
}
SOLVER_NAMES = {
    0: "PGS",
    1: "CG",
    2: "Newton",
}


def runtime_value(owner, name, np, index=None):
    try:
        value = getattr(owner, name)
        if index is not None:
            value = value[index]
        return jsonable(value)
    except (AttributeError, IndexError, TypeError, ValueError):
        return NOT_AVAILABLE


def runtime_enum(owner, name, names, np, index=None):
    value = runtime_value(owner, name, np, index=index)
    if value == NOT_AVAILABLE:
        return NOT_AVAILABLE
    try:
        enum_value = int(value)
    except (TypeError, ValueError):
        return NOT_AVAILABLE
    return names.get(enum_value, "unknown_{}".format(enum_value))


def runtime_bool(owner, name, np, index=None):
    value = runtime_value(owner, name, np, index=index)
    if value == NOT_AVAILABLE:
        return NOT_AVAILABLE
    try:
        return bool(int(value))
    except (TypeError, ValueError):
        return NOT_AVAILABLE


def scalar_dof_value(model, name, dof_address, dof_width, np):
    if dof_width != 1:
        return NOT_AVAILABLE
    return runtime_value(model, name, np, index=dof_address)


def scalar_qpos_value(model, name, qpos_address, qpos_width, np):
    if qpos_width != 1:
        return NOT_AVAILABLE
    return runtime_value(model, name, np, index=qpos_address)


def build_runtime_joint_dynamics(model, joint_names, np):
    result = []
    for joint_id in range(int(model.njnt)):
        if int(model.jnt_type[joint_id]) == 0:
            continue
        qpos_width, dof_width = joint_widths(model, joint_id)
        qpos_address = int(model.jnt_qposadr[joint_id])
        dof_address = int(model.jnt_dofadr[joint_id])
        result.append(
            {
                "joint_name": joint_names[joint_id],
                "joint_id": joint_id,
                "joint_type": JOINT_TYPE_NAMES.get(
                    int(model.jnt_type[joint_id]),
                    "unknown_{}".format(int(model.jnt_type[joint_id])),
                ),
                "qpos_address": qpos_address,
                "dof_address": dof_address,
                "axis": runtime_value(model, "jnt_axis", np, index=joint_id),
                "range": runtime_value(model, "jnt_range", np, index=joint_id),
                "limited": runtime_bool(model, "jnt_limited", np, index=joint_id),
                "damping": scalar_dof_value(
                    model, "dof_damping", dof_address, dof_width, np
                ),
                "armature": scalar_dof_value(
                    model, "dof_armature", dof_address, dof_width, np
                ),
                "stiffness": runtime_value(
                    model, "jnt_stiffness", np, index=joint_id
                ),
                "frictionloss": scalar_dof_value(
                    model, "dof_frictionloss", dof_address, dof_width, np
                ),
                "spring_reference": scalar_qpos_value(
                    model, "qpos_spring", qpos_address, qpos_width, np
                ),
            }
        )
    return result


def build_runtime_actuators(model, actuator_names, joint_names, np):
    result = []
    trnid = runtime_value(model, "actuator_trnid", np)
    trntype = runtime_value(model, "actuator_trntype", np)
    for actuator_id in range(int(model.nu)):
        target_joint_id = NOT_AVAILABLE
        target_joint_name = NOT_AVAILABLE
        if trnid != NOT_AVAILABLE and trntype != NOT_AVAILABLE:
            candidate_id = int(trnid[actuator_id][0])
            transmission_type = int(trntype[actuator_id])
            if transmission_type in (0, 1) and 0 <= candidate_id < int(model.njnt):
                target_joint_id = candidate_id
                target_joint_name = joint_names[candidate_id]
        result.append(
            {
                "actuator_name": actuator_names[actuator_id],
                "actuator_id": actuator_id,
                "target_joint_name": target_joint_name,
                "target_joint_id": target_joint_id,
                "gear": runtime_value(model, "actuator_gear", np, index=actuator_id),
                "ctrlrange": runtime_value(
                    model, "actuator_ctrlrange", np, index=actuator_id
                ),
                "ctrllimited": runtime_bool(
                    model, "actuator_ctrllimited", np, index=actuator_id
                ),
                "forcerange": runtime_value(
                    model, "actuator_forcerange", np, index=actuator_id
                ),
                "forcelimited": runtime_bool(
                    model, "actuator_forcelimited", np, index=actuator_id
                ),
                "dyntype": runtime_enum(
                    model,
                    "actuator_dyntype",
                    ACTUATOR_DYN_TYPE_NAMES,
                    np,
                    index=actuator_id,
                ),
                "gaintype": runtime_enum(
                    model,
                    "actuator_gaintype",
                    ACTUATOR_GAIN_TYPE_NAMES,
                    np,
                    index=actuator_id,
                ),
                "biastype": runtime_enum(
                    model,
                    "actuator_biastype",
                    ACTUATOR_BIAS_TYPE_NAMES,
                    np,
                    index=actuator_id,
                ),
            }
        )
    return result


def build_runtime_body_dynamics(model, body_names, np):
    result = []
    for body_id in range(int(model.nbody)):
        parent_id = runtime_value(model, "body_parentid", np, index=body_id)
        parent_name = NOT_AVAILABLE
        if parent_id != NOT_AVAILABLE:
            parent_id = int(parent_id)
            if body_id != 0 and 0 <= parent_id < len(body_names):
                parent_name = body_names[parent_id]
        result.append(
            {
                "body_name": body_names[body_id],
                "body_id": body_id,
                "parent_body_name": parent_name,
                "mass": runtime_value(model, "body_mass", np, index=body_id),
                "center_of_mass_position": runtime_value(
                    model, "body_ipos", np, index=body_id
                ),
                "inertia_diagonal": runtime_value(
                    model, "body_inertia", np, index=body_id
                ),
                "inertia_frame_quaternion": runtime_value(
                    model, "body_iquat", np, index=body_id
                ),
            }
        )
    return result


def build_runtime_solver_metadata(model, np):
    opt = model.opt
    return {
        "integrator": runtime_enum(opt, "integrator", INTEGRATOR_NAMES, np),
        "solver": runtime_enum(opt, "solver", SOLVER_NAMES, np),
        "iterations": runtime_value(opt, "iterations", np),
        "tolerance": runtime_value(opt, "tolerance", np),
        "noslip_iterations": runtime_value(opt, "noslip_iterations", np),
        "noslip_tolerance": runtime_value(opt, "noslip_tolerance", np),
        "mpr_iterations": runtime_value(opt, "mpr_iterations", np),
        "mpr_tolerance": runtime_value(opt, "mpr_tolerance", np),
    }


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


def validate_step0_joint_qpos(
    records, cases, joint_names, canonical_joint_qpos, target_joint_name
):
    target_index = joint_names.index(target_joint_name)
    readbacks = []
    for case in cases:
        record = next(
            record
            for record in records
            if record.get("case") == case and record.get("step") == 0
        )
        readback = jsonable(record["joint_qpos"])
        if len(readback) != len(canonical_joint_qpos) or any(
            not math.isclose(
                float(actual), float(expected), rel_tol=0.0, abs_tol=1e-12
            )
            for actual, expected in zip(readback, canonical_joint_qpos)
        ):
            raise ValueError(
                "case {!r} step 0 joint_qpos does not match canonical_joint_qpos".format(
                    case
                )
            )
        readbacks.append(float(readback[target_index]))
    if any(
        not math.isclose(value, readbacks[0], rel_tol=0.0, abs_tol=1e-12)
        for value in readbacks[1:]
    ):
        raise ValueError("target joint step 0 readback differs between cases")
    return readbacks[0]


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
    if metadata["canonical_qpos_mode"] not in ("midpoint", "model-default"):
        raise ValueError("unexpected canonical_qpos_mode")
    expected_joint_source = (
        "compiled_model_qpos0_with_target_joint_midpoint_override"
        if metadata["canonical_qpos_mode"] == "model-default"
        and metadata["target_joint_initial_position_mode"] == "midpoint"
        else (
            "compiled_model_qpos0"
            if metadata["canonical_qpos_mode"] == "model-default"
            else "compiled_joint_range_midpoint"
        )
    )
    if metadata["root_qpos_source"] != "explicit":
        raise ValueError("unexpected root_qpos_source")
    if metadata["joint_qpos_source"] != expected_joint_source:
        raise ValueError("unexpected joint_qpos_source")
    joint_count = len(metadata["joint_names"])
    for field in ("canonical_joint_qpos", "model_default_joint_qpos"):
        if len(metadata[field]) != joint_count:
            raise ValueError("{} length does not match joint_names".format(field))
    if metadata["target_joint_initial_position_mode"] not in ("default", "midpoint"):
        raise ValueError("unexpected target_joint_initial_position_mode")
    if len(metadata["target_joint_range"]) != 2:
        raise ValueError("target_joint_range must contain lower and upper limits")
    target_index = metadata["joint_names"].index(metadata["requested_joint_name"])
    canonical_target_qpos = float(metadata["canonical_joint_qpos"][target_index])
    for field in (
        "target_joint_initial_qpos_requested",
        "target_joint_initial_qpos_readback",
    ):
        if not math.isclose(
            float(metadata[field]),
            canonical_target_qpos,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("{} does not match canonical_joint_qpos".format(field))
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


def limited_scalar_joint_midpoint(model, joint_id, joint_names, np):
    if not 0 <= joint_id < int(model.njnt):
        raise ValueError("target joint id is outside model.njnt")
    joint_type = int(model.jnt_type[joint_id])
    if joint_type not in (2, 3) or joint_widths(model, joint_id) != (1, 1):
        raise ValueError(
            "target joint {!r} is not a scalar hinge/slide joint".format(
                joint_names[joint_id]
            )
        )
    if not bool(model.jnt_limited[joint_id]):
        raise ValueError("target joint {!r} is not limited".format(joint_names[joint_id]))
    limits = np.asarray(model.jnt_range[joint_id], dtype=np.float64).reshape(-1)
    if limits.shape != (2,) or not np.all(np.isfinite(limits)):
        raise ValueError("target joint {!r} has a non-finite range".format(joint_names[joint_id]))
    return float((limits[0] + limits[1]) / 2.0), limits.copy()


def build_canonical_state(
    model,
    root_z: float,
    joint_names,
    qpos_mode: str,
    np,
    target_joint_id=None,
    target_joint_initial_position: str = "default",
):
    if qpos_mode not in ("midpoint", "model-default"):
        raise ValueError("unsupported canonical qpos mode: {}".format(qpos_mode))
    if target_joint_initial_position not in ("default", "midpoint"):
        raise ValueError(
            "unsupported target joint initial position: {}".format(
                target_joint_initial_position
            )
        )
    if target_joint_id is None:
        raise ValueError("target_joint_id is required")
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
        if joint_type in (2, 3):
            policy_joint_ids.append(joint_id)
        if qpos_mode == "midpoint":
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
    target_qpos_address = int(model.jnt_qposadr[target_joint_id])
    target_joint_range = np.asarray(
        model.jnt_range[target_joint_id], dtype=np.float64
    ).reshape(-1).copy()
    if target_joint_initial_position == "midpoint":
        target_midpoint, target_joint_range = limited_scalar_joint_midpoint(
            model, target_joint_id, joint_names, np
        )
        qpos[target_qpos_address] = target_midpoint
    target_joint_initial_qpos_requested = float(qpos[target_qpos_address])
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
        "qpos_mode": qpos_mode,
        "target_joint_initial_position_mode": target_joint_initial_position,
        "target_joint_initial_qpos_requested": target_joint_initial_qpos_requested,
        "target_joint_range": target_joint_range,
    }


def joint_qpos_values(qpos, model, joint_ids, np):
    values = np.asarray(qpos, dtype=np.float64).reshape(-1)
    result = []
    for joint_id in joint_ids:
        qpos_width, _ = joint_widths(model, joint_id)
        if qpos_width != 1:
            raise ValueError("policy joint qpos is not scalar")
        qpos_address = int(model.jnt_qposadr[joint_id])
        result.append(float(values[qpos_address]))
    return result


def resolve_target(model, requested_name, joint_names, actuator_names, np):
    matches = [index for index, name in enumerate(joint_names) if name == requested_name]
    if len(matches) != 1:
        raise ValueError(
            "target joint {!r} matched {} compiled joint names; no index fallback is allowed".format(
                requested_name, len(matches)
            )
        )
    joint_id = matches[0]
    if int(model.jnt_type[joint_id]) not in (2, 3) or joint_widths(model, joint_id) != (1, 1):
        raise ValueError(
            "target joint {!r} is not a scalar hinge/slide joint".format(requested_name)
        )
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


def _name_has_prefix(name, prefix):
    return name == prefix or name.startswith(prefix + "/")


def classify_contact_pair(pair):
    ground = any(
        _name_has_prefix(pair[field], "floor")
        for field in ("geom1", "geom2", "body1", "body2")
    )
    morphology1 = any(
        _name_has_prefix(pair["body1"], prefix) for prefix in ("torso", "limb")
    )
    morphology2 = any(
        _name_has_prefix(pair["body2"], prefix) for prefix in ("torso", "limb")
    )
    if ground:
        return "ground"
    if morphology1 and morphology2:
        return "self"
    return "unclassified"


def contact_details(sim, geom_names, body_names):
    model, data = sim.model, sim.data
    count = int(data.ncon)
    details = []
    ground_count = 0
    self_count = 0
    try:
        geom_bodyid = model.geom_bodyid
        for index in range(count):
            contact = data.contact[index]
            geom1, geom2 = int(contact.geom1), int(contact.geom2)
            body1, body2 = int(geom_bodyid[geom1]), int(geom_bodyid[geom2])
            pair = {
                "geom1": geom_names[geom1],
                "geom2": geom_names[geom2],
                "body1": body_names[body1],
                "body2": body_names[body2],
            }
            pair["classification"] = classify_contact_pair(pair)
            if pair["classification"] == "ground":
                ground_count += 1
            elif pair["classification"] == "self":
                self_count += 1
            details.append(pair)
    except (AttributeError, IndexError, TypeError):
        return count, NOT_AVAILABLE, NOT_AVAILABLE, NOT_AVAILABLE
    return count, ground_count, self_count, details


def identify_unique_ground_geom(model, geom_names, body_names):
    """Resolve the one compiled floor geom; never guess among candidates."""
    candidates = []
    for geom_id, geom_name in enumerate(geom_names):
        body_id = int(model.geom_bodyid[geom_id])
        body_name = body_names[body_id]
        if _name_has_prefix(geom_name, "floor") or _name_has_prefix(body_name, "floor"):
            candidates.append(
                {
                    "geom_id": geom_id,
                    "geom_name": geom_name,
                    "body_id": body_id,
                    "body_name": body_name,
                }
            )
    if len(candidates) != 1:
        raise ValueError(
            "expected exactly one compiled ground geom, found {}: {}".format(
                len(candidates), candidates
            )
        )
    return candidates[0]


def robot_geom_ids(model, geom_names, body_names):
    result = set()
    for geom_id, geom_name in enumerate(geom_names):
        body_name = body_names[int(model.geom_bodyid[geom_id])]
        if any(
            _name_has_prefix(name, prefix)
            for name in (geom_name, body_name)
            for prefix in ("torso", "limb")
        ):
            result.add(geom_id)
    return result


def is_robot_ground_contact(geom1_id, geom2_id, ground_geom_id, robot_geoms):
    return (
        geom1_id == ground_geom_id and geom2_id in robot_geoms
    ) or (
        geom2_id == ground_geom_id and geom1_id in robot_geoms
    )


def penetration_depth(distance):
    value = float(distance)
    if not math.isfinite(value):
        raise ValueError("contact distance is non-finite: {}".format(value))
    return max(0.0, -value)


class FirstGroundContactWindow:
    """Keep only the requested records around the first ground contact."""

    def __init__(self, before, after):
        if before < 0 or after < 0:
            raise ValueError("contact window sizes must be non-negative")
        self.before = int(before)
        self.after = int(after)
        self._before = deque(maxlen=self.before)
        self.records = []
        self.first_step = None
        self._remaining_after = None

    @property
    def found(self):
        return self.first_step is not None

    @property
    def complete(self):
        return self.found and self._remaining_after == 0

    def observe(self, record, has_ground_contact):
        if not self.found:
            if not has_ground_contact:
                self._before.append(record)
                return False
            self.first_step = int(record["global_physics_step"])
            self.records = list(self._before) + [record]
            self._remaining_after = self.after
            return self.complete
        if self._remaining_after > 0:
            self.records.append(record)
            self._remaining_after -= 1
        return self.complete

    def summary(self):
        before_available = (
            sum(int(record["global_physics_step"]) < self.first_step for record in self.records)
            if self.found
            else 0
        )
        after_available = (
            sum(int(record["global_physics_step"]) > self.first_step for record in self.records)
            if self.found
            else 0
        )
        return {
            "first_ground_contact_found": self.found,
            "first_ground_contact_step": self.first_step,
            "requested_before": self.before,
            "available_before": before_available,
            "requested_after": self.after,
            "available_after": after_available,
            "window_complete": self.complete,
        }


def contact_window_exit_code(summary):
    return 0 if (
        summary["first_ground_contact_found"] and summary["window_complete"]
    ) else 2


def strict_json_value(value, field="root", global_physics_step=NOT_AVAILABLE):
    """Convert to JSON and identify the exact non-finite field and substep."""
    if hasattr(value, "tolist"):
        return strict_json_value(value.tolist(), field, global_physics_step)
    if isinstance(value, dict):
        return {
            str(key): strict_json_value(
                item, "{}.{}".format(field, key), global_physics_step
            )
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [
            strict_json_value(item, "{}[{}]".format(field, index), global_physics_step)
            for index, item in enumerate(value)
        ]
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return value
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(
                "non-finite field={} global_physics_step={} value={}".format(
                    field, global_physics_step, value
                )
            )
        return float(value)
    if hasattr(value, "item"):
        return strict_json_value(value.item(), field, global_physics_step)
    raise TypeError("unsupported JSON value type at {}: {}".format(field, type(value).__name__))


def max_contact_count(current, value):
    if current == NOT_AVAILABLE or value == NOT_AVAILABLE:
        return NOT_AVAILABLE
    return max(int(current), int(value))


def contact_free_from_summary(summary, field):
    values = [item[field] for item in summary.values()]
    if any(value == NOT_AVAILABLE for value in values):
        return NOT_AVAILABLE
    return all(int(value) == 0 for value in values)


def physics_substep_sequence(control_steps, frame_skip):
    if control_steps <= 0 or frame_skip <= 0:
        raise ValueError("control_steps and frame_skip must be positive")
    sequence = [(0, 0, 0)]
    for control_step in range(1, control_steps + 1):
        for physics_substep in range(1, frame_skip + 1):
            global_physics_step = (control_step - 1) * frame_skip + physics_substep
            sequence.append(
                (control_step, physics_substep, global_physics_step)
            )
    return sequence


def physics_substep_fields(control_step, physics_substep, frame_skip):
    if control_step == 0 and physics_substep == 0:
        global_physics_step = 0
    elif control_step > 0 and 1 <= physics_substep <= frame_skip:
        global_physics_step = (control_step - 1) * frame_skip + physics_substep
    else:
        raise ValueError("invalid control_step/physics_substep coordinates")
    return {
        "control_step": control_step,
        "physics_substep": physics_substep,
        "global_physics_step": global_physics_step,
        "record_level": "physics_substep",
    }


def summarize_physics_contacts(records, cases, total_physics_steps, enabled):
    first_contact = {}
    first_self_contact = {}
    clean_prefix = {}
    for case in cases:
        if not enabled:
            first_contact[case] = NOT_AVAILABLE
            first_self_contact[case] = NOT_AVAILABLE
            clean_prefix[case] = NOT_AVAILABLE
            continue
        case_records = sorted(
            (record for record in records if record["case"] == case),
            key=lambda record: record["global_physics_step"],
        )
        expected_steps = list(range(total_physics_steps + 1))
        actual_steps = [record["global_physics_step"] for record in case_records]
        if actual_steps != expected_steps:
            raise ValueError("physics substep records are incomplete for case {!r}".format(case))

        total_contacts = [
            int(record["contact_count_if_available"]) for record in case_records
        ]
        first_contact[case] = next(
            (
                record["global_physics_step"]
                for record, count in zip(case_records, total_contacts)
                if count > 0
            ),
            None,
        )

        self_counts = [
            record["self_contact_count_if_available"] for record in case_records
        ]
        if any(value == NOT_AVAILABLE for value in self_counts):
            first_self_contact[case] = NOT_AVAILABLE
        else:
            first_self_contact[case] = next(
                (
                    record["global_physics_step"]
                    for record, count in zip(case_records, self_counts)
                    if int(count) > 0
                ),
                None,
            )

        prefix_length = 0
        if total_contacts[0] == 0:
            for count in total_contacts[1:]:
                if count > 0:
                    break
                prefix_length += 1
        clean_prefix[case] = prefix_length
    return {
        "first_contact_global_physics_step_by_case": first_contact,
        "first_self_contact_global_physics_step_by_case": first_self_contact,
        "contact_free_prefix_length_by_case": clean_prefix,
    }


def force_slice(data, name, dof_indices, np):
    try:
        values = np.asarray(getattr(data, name), dtype=np.float64).reshape(-1)
        return values[dof_indices]
    except (AttributeError, IndexError, TypeError, ValueError):
        return NOT_AVAILABLE


def transition_record(
    base_env,
    morphology,
    case,
    step,
    ctrl,
    canonical,
    names,
    np,
    record_fields=None,
):
    sim, model, data = base_env.sim, base_env.sim.model, base_env.sim.data
    root_qpos = canonical["root_qpos_adr"]
    root_dof = canonical["root_dof_adr"]
    joint_ids = canonical["policy_joint_ids"]
    qpos_indices = [int(model.jnt_qposadr[joint_id]) for joint_id in joint_ids]
    dof_indices = non_root_dof_indices(model, joint_ids)
    contact_count, ground_count, self_count, contacts = contact_details(
        sim, names["geom"], names["body"]
    )
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
        "ground_contact_count_if_available": ground_count,
        "self_contact_count_if_available": self_count,
        "contact_pairs_if_available": contacts,
        "contacts_if_available": contacts,
        "done": False,
    }
    if record_fields is not None:
        record.update(record_fields)
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


class StepRecordingSimProxy:
    """Delegate the simulator API and notify after each real physics step."""

    def __init__(self, sim, callback):
        self._sim = sim
        self._callback = callback
        self.callback_error = None

    def __getattr__(self, name):
        return getattr(self._sim, name)

    def step(self):
        result = self._sim.step()
        if self.callback_error is None:
            try:
                self._callback()
            except Exception as error:
                # The formal do_simulation catches sim.step exceptions. Preserve
                # the diagnostic failure and re-raise it after env.step returns.
                self.callback_error = error
        return result


def wrapper_chain(env):
    current = env
    seen = set()
    while current is not None and id(current) not in seen:
        yield current
        seen.add(id(current))
        current = getattr(current, "env", None)


def find_wrapper_with_method(env, method_name):
    return next(
        (item for item in wrapper_chain(env) if callable(getattr(item, method_name, None))),
        None,
    )


def contact_force_6d(sim, contact_index, np):
    raw_sim = getattr(sim, "_sim", sim)
    force = np.zeros(6, dtype=np.float64)
    try:
        from metamorph.utils import mujoco_compat as mjc

        if getattr(mjc, "BACKEND", None) == "mujoco":
            mjc.mujoco.mj_contactForce(raw_sim._model, raw_sim._data, contact_index, force)
        elif getattr(mjc, "BACKEND", None) == "mujoco_py":
            mjc.mujoco_py.functions.mj_contactForce(
                raw_sim.model, raw_sim.data, contact_index, force
            )
        else:
            return NOT_AVAILABLE
    except (AttributeError, ImportError, TypeError, ValueError):
        return NOT_AVAILABLE
    return force


def unique_named_body(body_names, requested_name):
    matches = [index for index, name in enumerate(body_names) if name == requested_name]
    if len(matches) != 1:
        raise ValueError(
            "body {!r} matched {} compiled bodies; available body names: {}".format(
                requested_name, len(matches), body_names
            )
        )
    return matches[0]


def runtime_torso_fields(data, torso_body_id, torso_body_name, global_step, np):
    try:
        position = np.asarray(
            data.body_xpos[torso_body_id], dtype=np.float64
        ).reshape(-1).copy()
        quaternion = np.asarray(
            data.body_xquat[torso_body_id], dtype=np.float64
        ).reshape(-1).copy()
        linear_velocity = np.asarray(
            data.body_xvelp[torso_body_id], dtype=np.float64
        ).reshape(-1).copy()
        angular_velocity = np.asarray(
            data.body_xvelr[torso_body_id], dtype=np.float64
        ).reshape(-1).copy()
    except (AttributeError, IndexError, TypeError, ValueError) as error:
        raise ValueError(
            "runtime torso data unavailable at global_physics_step={}: {}".format(
                global_step, error
            )
        ) from error
    if position.size != 3 or quaternion.size != 4:
        raise ValueError(
            "invalid torso position/quaternion shape at global_physics_step={}: {}/{}".format(
                global_step, position.shape, quaternion.shape
            )
        )
    if linear_velocity.size != 3 or angular_velocity.size != 3:
        raise ValueError(
            "invalid torso velocity shape at global_physics_step={}: {}/{}".format(
                global_step, linear_velocity.shape, angular_velocity.shape
            )
        )
    return {
        "torso_body_name": torso_body_name,
        "torso_position": position,
        "torso_quaternion": quaternion,
        "torso_linear_velocity": linear_velocity,
        "torso_angular_velocity": angular_velocity,
        "torso_height": float(position[2]),
    }


class MethodResultCapture:
    """Capture a formal wrapper method result without changing its semantics."""

    def __init__(self, owner, method_name):
        self.owner = owner
        self.method_name = method_name
        self.original = getattr(owner, method_name)
        self.had_instance_attribute = method_name in getattr(owner, "__dict__", {})
        self.instance_attribute = getattr(owner, "__dict__", {}).get(method_name)
        self.value = NOT_AVAILABLE

    def _call(self, *args, **kwargs):
        result = self.original(*args, **kwargs)
        self.value = bool(result)
        return result

    def install(self):
        setattr(self.owner, self.method_name, self._call)

    def reset(self):
        self.value = NOT_AVAILABLE

    def restore(self):
        if self.had_instance_attribute:
            setattr(self.owner, self.method_name, self.instance_attribute)
        else:
            delattr(self.owner, self.method_name)


def ground_contacts_at_substep(sim, names, ground, robot_geoms, np):
    model, data = sim.model, sim.data
    contacts = []
    for contact_index in range(int(data.ncon)):
        contact = data.contact[contact_index]
        geom1_id, geom2_id = int(contact.geom1), int(contact.geom2)
        if not is_robot_ground_contact(
            geom1_id, geom2_id, ground["geom_id"], robot_geoms
        ):
            continue
        geom1_body_id = int(model.geom_bodyid[geom1_id])
        geom2_body_id = int(model.geom_bodyid[geom2_id])
        robot_geom_id = geom2_id if geom1_id == ground["geom_id"] else geom1_id
        robot_body_id = int(model.geom_bodyid[robot_geom_id])
        force = contact_force_6d(sim, contact_index, np)
        normal = np.asarray(contact.frame, dtype=np.float64).reshape(3, 3)[0].copy()
        force_available = not isinstance(force, str)
        contacts.append(
            {
                "contact_index": contact_index,
                "geom1_id": geom1_id,
                "geom1_name": names["geom"][geom1_id],
                "geom1_body_id": geom1_body_id,
                "geom1_body_name": names["body"][geom1_body_id],
                "geom2_id": geom2_id,
                "geom2_name": names["geom"][geom2_id],
                "geom2_body_id": geom2_body_id,
                "geom2_body_name": names["body"][geom2_body_id],
                "robot_body_name": names["body"][robot_body_id],
                "ground_geom_name": ground["geom_name"],
                "contact_position": np.asarray(contact.pos, dtype=np.float64).copy(),
                "contact_frame": np.asarray(contact.frame, dtype=np.float64).copy(),
                "contact_normal": normal,
                "distance": float(contact.dist),
                "penetration_depth": penetration_depth(contact.dist),
                "normal_force": float(force[0]) if force_available else NOT_AVAILABLE,
                "tangent_force_1": float(force[1]) if force_available else NOT_AVAILABLE,
                "tangent_force_2": float(force[2]) if force_available else NOT_AVAILABLE,
                "contact_force_6d": force,
            }
        )
    return contacts


def first_ground_contact_record(
    base, args, canonical, names, ground, robot_geoms, torso_body_id,
    policy_action, clipped_action, control_step, physics_substep, global_step, np,
):
    sim, model, data = base.sim, base.sim.model, base.sim.data
    joint_ids = canonical["policy_joint_ids"]
    qpos_indices = [int(model.jnt_qposadr[joint_id]) for joint_id in joint_ids]
    dof_indices = non_root_dof_indices(model, joint_ids)
    joint_names = [names["joint"][joint_id] for joint_id in joint_ids]
    debug_index = joint_names.index("limby/9") if "limby/9" in joint_names else NOT_AVAILABLE
    ground_contacts = ground_contacts_at_substep(
        sim, names, ground, robot_geoms, np
    )
    root_qpos, root_dof = canonical["root_qpos_adr"], canonical["root_dof_adr"]
    ctrl = np.asarray(data.ctrl, dtype=np.float64).reshape(-1).copy()
    joint_qpos = np.asarray(data.qpos, dtype=np.float64)[qpos_indices].copy()
    joint_qvel = np.asarray(data.qvel, dtype=np.float64)[dof_indices].copy()
    requested = requested_generalized_torque(model, ctrl, joint_ids, np)
    joint_passive = force_slice(data, "qfrc_passive", dof_indices, np)
    joint_actuator = force_slice(data, "qfrc_actuator", dof_indices, np)
    try:
        actuator_force = np.asarray(data.actuator_force, dtype=np.float64).reshape(-1).copy()
    except (AttributeError, TypeError, ValueError):
        actuator_force = NOT_AVAILABLE
    torso = runtime_torso_fields(
        data, torso_body_id, names["body"][torso_body_id], global_step, np
    )
    record = {
        "schema_version": GROUND_CONTACT_SCHEMA_VERSION,
        "backend": BACKEND,
        "morphology_id": args.morphology,
        "episode_index": 0,
        "control_step": control_step,
        "physics_substep_in_control_step": physics_substep,
        "global_physics_step": global_step,
        "time": float(data.time),
        "relative_to_first_contact": NOT_AVAILABLE,
        "is_first_ground_contact": False,
        "root_position": np.asarray(data.qpos[root_qpos:root_qpos + 3], dtype=np.float64).copy(),
        "root_quaternion": np.asarray(data.qpos[root_qpos + 3:root_qpos + 7], dtype=np.float64).copy(),
        "root_linear_velocity": np.asarray(data.qvel[root_dof:root_dof + 3], dtype=np.float64).copy(),
        "root_angular_velocity": np.asarray(data.qvel[root_dof + 3:root_dof + 6], dtype=np.float64).copy(),
        "joint_names": joint_names,
        "joint_qpos": joint_qpos,
        "joint_qvel": joint_qvel,
        "joint_passive_force": joint_passive,
        "joint_actuator_force": joint_actuator,
        "joint_total_generalized_force": NOT_AVAILABLE,
        "debug_joint_name": "limby/9",
        "debug_joint_index": debug_index,
        "debug_joint_qpos": (
            float(joint_qpos[debug_index]) if isinstance(debug_index, int) else NOT_AVAILABLE
        ),
        "debug_joint_qvel": (
            float(joint_qvel[debug_index]) if isinstance(debug_index, int) else NOT_AVAILABLE
        ),
        "policy_action": policy_action,
        "clipped_action": clipped_action,
        "native_action_or_ctrl": ctrl,
        "requested_torque": requested,
        "applied_actuator_force": actuator_force,
        "contact_count_total": int(data.ncon),
        "ground_contact_count": len(ground_contacts),
        "ground_contacts": ground_contacts,
        "reward_components_if_available": NOT_AVAILABLE,
        "reward_if_available": NOT_AVAILABLE,
        "fallen": NOT_AVAILABLE,
        "done": NOT_AVAILABLE,
        "termination_reason": NOT_AVAILABLE,
        "auto_reset_triggered": False,
        "evaluated_at_control_boundary": False,
    }
    record.update(torso)
    return strict_json_value(record, global_physics_step=global_step)


def write_strict_json(path, value):
    path.write_text(
        json.dumps(strict_json_value(value), indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def prepare_first_ground_contact_env(env):
    """Satisfy the outer Gym wrapper lifecycle before canonical injection."""
    env.reset()
    return env.unwrapped


def prepare_env_for_probe_mode(env, record_first_ground_contact_window):
    if record_first_ground_contact_window:
        return prepare_first_ground_contact_env(env)
    return env.unwrapped


def apply_and_verify_canonical_state(base, canonical, policy_action, np):
    """Inject canonical state after reset, synchronize, and verify readback."""
    sim = base.sim
    expected_qpos = np.asarray(canonical["qpos"], dtype=np.float64).reshape(-1)
    expected_qvel = np.asarray(canonical["qvel"], dtype=np.float64).reshape(-1)
    base.set_state(expected_qpos.copy(), expected_qvel.copy())
    sim.data.ctrl[:] = 0.0
    sim.forward()
    actual_qpos = np.asarray(sim.data.qpos, dtype=np.float64).reshape(-1)
    actual_qvel = np.asarray(sim.data.qvel, dtype=np.float64).reshape(-1)
    actual_ctrl = np.asarray(sim.data.ctrl, dtype=np.float64).reshape(-1)
    if actual_qpos.shape != expected_qpos.shape or not np.allclose(
        actual_qpos, expected_qpos, rtol=0.0, atol=1e-10
    ):
        raise ValueError(
            "canonical qpos readback mismatch after reset: expected={}, actual={}".format(
                expected_qpos.tolist(), actual_qpos.tolist()
            )
        )
    if actual_qvel.shape != expected_qvel.shape or not np.allclose(
        actual_qvel, expected_qvel, rtol=0.0, atol=1e-12
    ):
        raise ValueError(
            "canonical qvel readback mismatch after reset: expected={}, actual={}".format(
                expected_qvel.tolist(), actual_qvel.tolist()
            )
        )
    if np.any(actual_ctrl != 0.0) or np.any(np.asarray(policy_action) != 0.0):
        raise ValueError(
            "zero-action initialization mismatch: policy_action={}, data.ctrl={}".format(
                np.asarray(policy_action).tolist(), actual_ctrl.tolist()
            )
        )
    return {
        "root_z_expected": float(expected_qpos[canonical["root_qpos_adr"] + 2]),
        "root_z_readback": float(actual_qpos[canonical["root_qpos_adr"] + 2]),
        "qpos_verified": True,
        "qvel_verified": True,
        "zero_action_verified": True,
    }


def write_first_ground_contact_failure(output_dir, args, stage, error, metadata=None):
    summary = {
        "ok": False,
        "stage": stage,
        "first_ground_contact_found": False,
        "first_ground_contact_step": None,
        "error": str(error),
    }
    if metadata and "torso_body_candidates" in metadata:
        summary["torso_body_candidates"] = metadata["torso_body_candidates"]
    failure_metadata = {
        "schema_version": GROUND_CONTACT_SCHEMA_VERSION,
        "backend": BACKEND,
        "morphology_id": args.morphology,
        "root_z": args.root_z,
        "reset_qpos_mode": args.canonical_qpos_mode,
        "action_mode": "zero",
        "ok": False,
        "failure_stage": stage,
        "error": str(error),
    }
    if metadata:
        failure_metadata.update(metadata)
    write_strict_json(output_dir / "metadata.json", failure_metadata)
    write_strict_json(output_dir / "summary.json", summary)


def run_first_ground_contact_window(
    env, base, args, canonical, names, output_dir, source_xml, source_hash,
    walker_dir, morphology_xml, np,
):
    model, original_sim = base.sim.model, base.sim
    frame_skip = int(base.frame_skip)
    failure_context = {
        "source_xml": str(source_xml),
        "source_xml_sha256": source_hash,
        "walker_dir": str(walker_dir),
        "physics_timestep": float(model.opt.timestep),
        "control_timestep": float(model.opt.timestep) * frame_skip,
        "decimation": frame_skip,
    }
    ground = identify_unique_ground_geom(model, names["geom"], names["body"])
    robot_geoms = robot_geom_ids(model, names["geom"], names["body"])
    try:
        torso_body_id = unique_named_body(names["body"], "torso/0")
    except Exception as error:
        failure_context["torso_body_candidates"] = names["body"]
        write_first_ground_contact_failure(
            output_dir, args, "torso_mapping", error, failure_context
        )
        raise
    policy_action = np.zeros(env.action_space.shape, dtype=np.float32)
    clipped_action = policy_action.copy()  # No explicit clipping occurs in the formal wrapper path.
    window = FirstGroundContactWindow(args.contact_window_before, args.contact_window_after)
    global_step = 0
    current_control_step = 0
    current_substep = 0
    try:
        initialization = apply_and_verify_canonical_state(
            base, canonical, policy_action, np
        )
    except Exception as error:
        write_first_ground_contact_failure(
            output_dir, args, "initialization", error, failure_context
        )
        raise

    def capture():
        nonlocal global_step, current_substep
        current_substep += 1
        global_step += 1
        if global_step > args.max_physics_substeps:
            return
        record = first_ground_contact_record(
            base, args, canonical, names, ground, robot_geoms, torso_body_id,
            policy_action, clipped_action, current_control_step, current_substep,
            global_step, np,
        )
        window.observe(record, record["ground_contact_count"] > 0)

    proxy = StepRecordingSimProxy(original_sim, capture)
    base.sim = proxy
    falling_wrapper = find_wrapper_with_method(env, "has_fallen")
    fallen_capture = (
        MethodResultCapture(falling_wrapper, "has_fallen")
        if falling_wrapper is not None
        else None
    )
    if fallen_capture is not None:
        fallen_capture.install()
    try:
        try:
            while global_step < args.max_physics_substeps and not window.complete:
                current_control_step += 1
                current_substep = 0
                if fallen_capture is not None:
                    fallen_capture.reset()
                obs, reward, done, info = env.step(policy_action.copy())
                if proxy.callback_error is not None:
                    raise RuntimeError(
                        "physics-substep record failed: {}".format(proxy.callback_error)
                    ) from proxy.callback_error
                if info.get("mj_step_error"):
                    raise RuntimeError("formal MuJoCo do_simulation reported a step error")
                boundary_record = next(
                    (
                        record for record in reversed(window.records)
                        if global_step <= args.max_physics_substeps
                        and record["global_physics_step"] == global_step
                    ),
                    None,
                )
                if boundary_record is not None:
                    reward_components = {
                        key: value for key, value in info.items() if "__reward__" in key
                    }
                    fallen = (
                        fallen_capture.value
                        if fallen_capture is not None
                        else NOT_AVAILABLE
                    )
                    boundary_record.update(
                        strict_json_value(
                            {
                                "reward_components_if_available": reward_components,
                                "reward_if_available": float(reward),
                                "fallen": fallen,
                                "done": bool(done),
                                "termination_reason": (
                                    "fallen" if fallen is True else ("other" if done else None)
                                ),
                                "auto_reset_triggered": False,
                                "evaluated_at_control_boundary": True,
                            },
                            global_physics_step=boundary_record["global_physics_step"],
                        )
                    )
        except Exception as error:
            write_first_ground_contact_failure(
                output_dir, args, "rollout", error, failure_context
            )
            raise
    finally:
        if fallen_capture is not None:
            fallen_capture.restore()
        base.sim = original_sim

    summary = window.summary()
    if not window.found:
        summary.update(
            {
                "ok": False,
                "stage": "contact_not_found",
                "error": "no first ground contact within max_physics_substeps",
            }
        )
    elif not window.complete:
        summary.update(
            {
                "ok": False,
                "stage": "rollout",
                "error": "first ground contact found but requested after-window is incomplete",
            }
        )
    else:
        summary.update({"ok": True, "stage": "completed", "error": None})
    if window.found:
        for record in window.records:
            relative = int(record["global_physics_step"]) - int(window.first_step)
            record["relative_to_first_contact"] = relative
            record["is_first_ground_contact"] = relative == 0
    metadata = {
        "schema_version": GROUND_CONTACT_SCHEMA_VERSION,
        "backend": BACKEND,
        "repository_path": str(REPO_ROOT),
        "morphology_id": args.morphology,
        "source_xml": str(source_xml),
        "source_xml_sha256": source_hash,
        "source_xml_role": "user_declared_source_xml",
        "source_xml_matches_environment_entry_path": source_xml == morphology_xml,
        "walker_dir": str(walker_dir),
        "physics_timestep": float(model.opt.timestep),
        "control_timestep": float(model.opt.timestep) * frame_skip,
        "decimation": frame_skip,
        "root_z": args.root_z,
        "reset_qpos_mode": args.canonical_qpos_mode,
        "action_mode": "zero",
        "ground_geom": ground,
        "torso_body_name": names["body"][torso_body_id],
        "torso_body_id": torso_body_id,
        "torso_height_source": "data.xpos[torso_body_id, 2]",
        "torso_height_frame": "world",
        "joint_names": [names["joint"][item] for item in canonical["policy_joint_ids"]],
        "body_names": names["body"],
        "geom_names": names["geom"],
        "root_quaternion_order": "wxyz",
        "torso_quaternion_order": "wxyz",
        "torso_height_semantics": "world-frame z coordinate from data.xpos[torso_body_id, 2] (exposed by the compatibility adapter as body_xpos)",
        "contact_frame_semantics": "MuJoCo contact.frame; first row is normal from geom1 toward geom2",
        "contact_force_convention": "mj_contactForce 6D force/torque in contact frame; force, not impulse",
        "first_ground_contact_definition": "first physics substep with exactly one pair side equal to the unique ground geom and the other side a torso/limb robot geom",
        "policy_action_semantics": "zero action in the outer node-centric policy action space",
        "clipped_action_semantics": "formal environment performs no explicit clip; zero action is unchanged",
        "native_action_or_ctrl_semantics": "actual data.ctrl in MuJoCo actuator order after the formal action wrapper",
        "requested_torque_semantics": "joint_names order; data.ctrl times actuator gear mapped through compiled actuator_trnid and joint DOF addresses",
        "applied_actuator_force_semantics": "data.actuator_force in actuator order",
        "joint_passive_force_semantics": "data.qfrc_passive at compiled policy-joint DOF addresses",
        "joint_actuator_force_semantics": "data.qfrc_actuator at compiled policy-joint DOF addresses",
        "joint_total_generalized_force_semantics": NOT_AVAILABLE,
        "termination_evaluation_timing": "formal wrappers evaluate reward/fallen/done only at control boundary after frame_skip physics steps",
        "auto_reset": False,
        "continue_after_done": True,
        "contact_window_before": args.contact_window_before,
        "contact_window_after": args.contact_window_after,
        "max_physics_substeps": args.max_physics_substeps,
        "initialization_verification": initialization,
    }
    metadata.update(summary)
    try:
        write_strict_json(output_dir / "metadata.json", metadata)
        write_strict_json(output_dir / "summary.json", summary)
        with (output_dir / "first_ground_contact_window.jsonl").open(
            "w", encoding="utf-8", newline="\n"
        ) as handle:
            for record in window.records:
                handle.write(
                    json.dumps(
                        strict_json_value(
                            record, global_physics_step=record["global_physics_step"]
                        ),
                        ensure_ascii=False,
                        allow_nan=False,
                    ) + "\n"
                )
    except Exception as error:
        write_first_ground_contact_failure(
            output_dir, args, "serialization", error, failure_context
        )
        raise
    print("first_ground_contact_found={}".format(summary["first_ground_contact_found"]))
    print("first_ground_contact_step={}".format(summary["first_ground_contact_step"]))
    return contact_window_exit_code(summary)


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
        if args.record_first_ground_contact_window:
            try:
                base = prepare_env_for_probe_mode(env, True)
            except Exception as error:
                write_first_ground_contact_failure(
                    output_dir,
                    args,
                    "initialization",
                    error,
                    {
                        "source_xml": str(source_xml),
                        "source_xml_sha256": source_xml_sha256,
                        "walker_dir": str(walker_dir),
                    },
                )
                raise
        else:
            base = prepare_env_for_probe_mode(env, False)
        sim, model = base.sim, base.sim.model
        names = {
            "joint": model_names(model, "joint", int(model.njnt)),
            "actuator": model_names(model, "actuator", int(model.nu)),
            "body": model_names(model, "body", int(model.nbody)),
            "geom": model_names(model, "geom", int(model.ngeom)),
        }
        runtime_joint_dynamics = build_runtime_joint_dynamics(
            model, names["joint"], np
        )
        runtime_actuators = build_runtime_actuators(
            model, names["actuator"], names["joint"], np
        )
        runtime_body_dynamics = build_runtime_body_dynamics(
            model, names["body"], np
        )
        runtime_solver_metadata = build_runtime_solver_metadata(model, np)
        target = resolve_target(
            model, args.joint_name, names["joint"], names["actuator"], np
        )
        if target["ctrl_index"] >= np.asarray(sim.data.ctrl).size:
            raise ValueError("target ctrl index is outside sim.data.ctrl")
        canonical = build_canonical_state(
            model,
            args.root_z,
            names["joint"],
            args.canonical_qpos_mode,
            np,
            target_joint_id=target["joint_id"],
            target_joint_initial_position=args.target_joint_initial_position,
        )
        if args.record_first_ground_contact_window:
            return run_first_ground_contact_window(
                env,
                base,
                args,
                canonical,
                names,
                output_dir,
                source_xml,
                source_xml_sha256,
                walker_dir,
                morphology_xml,
                np,
            )
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
            initial_record_fields = (
                physics_substep_fields(0, 0, int(base.frame_skip))
                if args.record_physics_substeps
                else None
            )
            case_records = [
                transition_record(
                    base,
                    args.morphology,
                    case,
                    0,
                    ctrl,
                    canonical,
                    names,
                    np,
                    record_fields=initial_record_fields,
                )
            ]
            for control_step in range(1, args.steps + 1):
                if args.record_physics_substeps:
                    base.step_count += 1
                    sim.data.ctrl[:] = ctrl
                    for physics_substep in range(1, int(base.frame_skip) + 1):
                        try:
                            sim.step()
                        except Exception as error:
                            raise RuntimeError(
                                "MuJoCo step failed for case {!r}, control step {}, physics substep {}".format(
                                    case, control_step, physics_substep
                                )
                            ) from error
                        case_records.append(
                            transition_record(
                                base,
                                args.morphology,
                                case,
                                control_step,
                                ctrl,
                                canonical,
                                names,
                                np,
                                record_fields=physics_substep_fields(
                                    control_step,
                                    physics_substep,
                                    int(base.frame_skip),
                                ),
                            )
                        )
                else:
                    if base.do_simulation(ctrl):
                        raise RuntimeError(
                            "MuJoCo step failed for case {!r} at control step {}".format(
                                case, control_step
                            )
                        )
                    case_records.append(
                        transition_record(
                            base,
                            args.morphology,
                            case,
                            control_step,
                            ctrl,
                            canonical,
                            names,
                            np,
                        )
                    )
            records.extend(case_records)
            max_contacts = max(
                int(record["contact_count_if_available"])
                for record in case_records
            )
            max_ground_contacts = 0
            max_self_contacts = 0
            for record in case_records:
                max_ground_contacts = max_contact_count(
                    max_ground_contacts,
                    record["ground_contact_count_if_available"],
                )
                max_self_contacts = max_contact_count(
                    max_self_contacts,
                    record["self_contact_count_if_available"],
                )
            contact_summary[case] = {
                "max_contact_count": max_contacts,
                "max_ground_contact_count": max_ground_contacts,
                "max_self_contact_count": max_self_contacts,
            }

        validate_canonical_step0(records, args.cases)

        control_timestep = float(model.opt.timestep) * int(base.frame_skip)
        physics_timestep_actual = float(model.opt.timestep)
        source_matches_entry = source_xml == morphology_xml
        policy_joint_ids = canonical["policy_joint_ids"]
        policy_dof_indices = non_root_dof_indices(model, policy_joint_ids)
        policy_joint_names = [names["joint"][joint_id] for joint_id in policy_joint_ids]
        canonical_joint_qpos = joint_qpos_values(
            canonical["qpos"], model, policy_joint_ids, np
        )
        model_default_joint_qpos = joint_qpos_values(
            model.qpos0, model, policy_joint_ids, np
        )
        joint_qpos_source = (
            "compiled_model_qpos0_with_target_joint_midpoint_override"
            if args.canonical_qpos_mode == "model-default"
            and args.target_joint_initial_position == "midpoint"
            else (
                "compiled_model_qpos0"
                if args.canonical_qpos_mode == "model-default"
                else "compiled_joint_range_midpoint"
            )
        )
        target_joint_initial_qpos_readback = validate_step0_joint_qpos(
            records,
            args.cases,
            policy_joint_names,
            canonical_joint_qpos,
            target["runtime_name"],
        )
        ground_contact_free_valid = contact_free_from_summary(
            contact_summary, "max_ground_contact_count"
        )
        self_contact_free_valid = contact_free_from_summary(
            contact_summary, "max_self_contact_count"
        )
        classified_contact_free_valid = (
            ground_contact_free_valid and self_contact_free_valid
            if isinstance(ground_contact_free_valid, bool)
            and isinstance(self_contact_free_valid, bool)
            else NOT_AVAILABLE
        )
        physics_contact_summary = summarize_physics_contacts(
            records,
            args.cases,
            args.steps * int(base.frame_skip),
            args.record_physics_substeps,
        )
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
            "integrator": runtime_solver_metadata["integrator"],
            "solver": runtime_solver_metadata["solver"],
            "iterations": runtime_solver_metadata["iterations"],
            "runtime_solver_settings": runtime_solver_metadata,
            "runtime_dynamics_schema_version": RUNTIME_DYNAMICS_SCHEMA_VERSION,
            "runtime_joint_dynamics": runtime_joint_dynamics,
            "runtime_actuators": runtime_actuators,
            "runtime_body_dynamics": runtime_body_dynamics,
            "inertia_quaternion_order": "wxyz",
            "center_of_mass_position_semantics": "body-local inertial-frame position",
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
            "canonical_qpos_mode": args.canonical_qpos_mode,
            "canonical_joint_qpos": canonical_joint_qpos,
            "model_default_joint_qpos": model_default_joint_qpos,
            "target_joint_initial_position_mode": args.target_joint_initial_position,
            "target_joint_initial_qpos_requested": canonical[
                "target_joint_initial_qpos_requested"
            ],
            "target_joint_initial_qpos_readback": target_joint_initial_qpos_readback,
            "target_joint_range": canonical["target_joint_range"],
            "root_qpos_source": "explicit",
            "joint_qpos_source": joint_qpos_source,
            "canonical_initial_state_spec": {
                "root_position": [0.0, 0.0, args.root_z],
                "root_quaternion": [1.0, 0.0, 0.0, 0.0],
                "root_linear_velocity": [0.0, 0.0, 0.0],
                "root_angular_velocity": [0.0, 0.0, 0.0],
                "canonical_qpos_mode": args.canonical_qpos_mode,
                "target_joint_initial_position_mode": args.target_joint_initial_position,
                "limited_hinge_qpos": (
                    "compiled model qpos0 reference"
                    if args.canonical_qpos_mode == "model-default"
                    else "compiled jnt_range midpoint"
                ),
                "unlimited_or_non_hinge_qpos": "compiled model qpos0 reference",
                "joint_qvel": "all zero",
                "fallback_joints": canonical["fallback_joints"],
                "reset_noise_bypassed": True,
                "reapplied_before_each_case": True,
            },
            "policy_action_semantics": "actuator_order_ctrl",
            "environment_creation_path": "metamorph.algos.ppo.envs.make_env -> gym.make -> task.make_env",
            "simulation_path": (
                "env.unwrapped.set_state -> sim.forward -> diagnostic data.ctrl write -> frame_skip * sim.step with substep records"
                if args.record_physics_substeps
                else "env.unwrapped.set_state -> sim.forward -> env.unwrapped.do_simulation"
            ),
            "formal_action_wrapper_mapping": "MultiUnimalNodeCentricAction removes padded slots (and may mirror); this tool deliberately starts at its native actuator-order output",
            "native_action_or_ctrl_semantics": "mujoco_actuator_order_ctrl",
            "requested_torque_semantics": "joint_names order; sum(ctrl[actuator] * actuator_gear[actuator, 0]) via scalar joint transmissions and compiled joint/dof addresses",
            "applied_torque_if_available_semantics": "data.qfrc_actuator at compiled DOF addresses in joint_names order; not copied from requested_torque",
            "actuator_force_semantics": "data.actuator_force in actuator order when available",
            "joint_actuator_torque_semantics": "data.qfrc_actuator at compiled non-root joint DOF addresses",
            "passive_force_semantics": "data.qfrc_passive at compiled non-root joint DOF addresses",
            "cases": args.cases,
            "steps": args.steps,
            "record_physics_substeps": args.record_physics_substeps,
            "first_contact_global_physics_step_by_case": physics_contact_summary[
                "first_contact_global_physics_step_by_case"
            ],
            "first_self_contact_global_physics_step_by_case": physics_contact_summary[
                "first_self_contact_global_physics_step_by_case"
            ],
            "contact_free_prefix_length_by_case": physics_contact_summary[
                "contact_free_prefix_length_by_case"
            ],
            "action_magnitude": args.action_magnitude,
            "contact_summary": contact_summary,
            "ground_contact_free_valid": ground_contact_free_valid,
            "self_contact_free_valid": self_contact_free_valid,
            "contact_free_valid": classified_contact_free_valid,
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
        return 0
    finally:
        env.close()


def main(argv=None) -> int:
    args = parse_args(argv)
    return run(args)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print("error: {}".format(error), file=sys.stderr)
        raise SystemExit(1)

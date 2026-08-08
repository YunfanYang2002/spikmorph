"""Fixed-global55 Newton solver initialization and convergence diagnostic.

This tool replays the validated global55 pre-state once and evaluates four
independent Newton clones.  The only interventions are qacc_warmstart,
solver tolerance, and solver iteration limit.  Contact and constraint
physics are never edited.
"""

from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime
import hashlib
import inspect
import json
from pathlib import Path
import subprocess
import sys
import traceback
from types import SimpleNamespace
from typing import Any, Iterable, Sequence

import numpy as np


def _ensure_numpy_bool_compatibility() -> None:
    if "bool" not in np.__dict__:
        np.__dict__["bool"] = bool


_ensure_numpy_bool_compatibility()

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools import analyze_mujoco_global55_contact_demand as oracle
from tools import audit_mujoco_global55_contact_regularization_counterfactual as regularization
from tools import audit_mujoco_global55_friction_aref_counterfactual as aref


MORPHOLOGY = oracle.MORPHOLOGY
XML_SHA256 = oracle.XML_SHA256
CHECKPOINT_SHA256 = oracle.CHECKPOINT_SHA256
GLOBAL_STEP = oracle.GLOBAL_STEP
EXPECTED_SUBSTEPS = oracle.EXPECTED_SUBSTEPS
REFERENCE_ORACLE_NAME = "mujoco_global55_contact_demand_oracle_corrected_20260804_143138"
REFERENCE_REGULARIZATION_NAME = (
    "mujoco_global55_contact_regularization_counterfactual_20260806_142049"
)
CONDITIONS = (
    ("production_warmstart_production_tolerance", "PRODUCTION_WARMSTART_PRODUCTION_TOLERANCE", False, False),
    ("zero_warmstart_production_tolerance", "ZERO_WARMSTART_PRODUCTION_TOLERANCE", True, False),
    ("production_warmstart_tight_tolerance", "PRODUCTION_WARMSTART_TIGHT_TOLERANCE", False, True),
    ("zero_warmstart_tight_tolerance", "ZERO_WARMSTART_TIGHT_TOLERANCE", True, True),
)
STATE_COPY_FIELDS = aref.cone_helper.STATE_COPY_FIELDS
REGRESSION_RTOL = 1.0e-9
REGRESSION_ATOL = 1.0e-9
SENSITIVITY_RTOL = 1.0e-6
SENSITIVITY_ATOL = 1.0e-8
PHYSICAL_IMPULSE_ATOL = 1.0e-8
TIGHT_TOLERANCE_DEFAULT = 1.0e-12
REQUIRED_SOLVER_STAT_FIELDS = (
    "improvement", "gradient", "lineslope", "nactive", "nchange",
    "neval", "nupdate",
)
FORBIDDEN_OPTION_TOKENS = (
    "cone", "friction", "solref", "solimp", "aref", "regularization",
)
CLONE_COUNT_FIELDS = (
    "nq", "nv", "na", "nu", "nbody", "njnt", "ngeom", "nsite",
    "npair", "nexclude", "ntendon", "nsensor", "nkey",
)
CLONE_ARRAY_FIELDS = (
    "geom_friction", "geom_solref", "geom_solimp", "geom_contype",
    "geom_conaffinity", "geom_bodyid", "body_mass", "body_inertia",
    "body_pos", "body_quat", "jnt_type", "jnt_bodyid", "jnt_axis",
    "jnt_range", "dof_armature", "dof_damping", "dof_frictionloss",
    "actuator_trnid", "actuator_gear", "actuator_forcelimited",
    "actuator_forcerange", "actuator_ctrllimited", "actuator_ctrlrange",
)
CLONE_NAME_OBJECTS = (
    ("geom", "mjOBJ_GEOM", "ngeom"),
    ("body", "mjOBJ_BODY", "nbody"),
    ("joint", "mjOBJ_JOINT", "njnt"),
)


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
        description="Run fixed-global55 Newton solver optimization diagnostics."
    )
    result.add_argument("--formal-server-defaults", action="store_true")
    result.add_argument("--checkpoint")
    result.add_argument("--walker-dir")
    result.add_argument("--corrected-oracle")
    result.add_argument("--regularization-reference")
    result.add_argument("--output-dir")
    result.add_argument("--zip-path")
    result.add_argument("--morphology-id", default=MORPHOLOGY)
    result.add_argument("--cfg", default="configs/ft.yaml")
    result.add_argument("--seed", type=int, default=1409)
    result.add_argument("--device", default="cpu")
    return result


def _default_output_paths() -> tuple[Path, Path]:
    stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")
    return (
        REPO_ROOT / "output/diagnostics"
        / f"mujoco_global55_solver_optimization_{stamp}",
        REPO_ROOT / "tmp"
        / f"mujoco_global55_solver_optimization_{stamp}.zip",
    )


def resolve_arguments(args: argparse.Namespace) -> argparse.Namespace:
    if args.formal_server_defaults:
        batch = REPO_ROOT / "output/diagnostics/mujoco_control_51k_20260727_091638"
        manifest = json.loads((batch / "manifest.json").read_text(encoding="utf-8"))
        args.checkpoint = str(batch / "jobs/job_000_seed1409_lr0p00015/Unimal-v0.pt")
        args.walker_dir = manifest["source_audit"]["walker_dir"]
        args.corrected_oracle = str(
            REPO_ROOT / "output/diagnostics" / REFERENCE_ORACLE_NAME
        )
        args.regularization_reference = str(
            REPO_ROOT / "output/diagnostics" / REFERENCE_REGULARIZATION_NAME
        )
    missing = [
        name for name in ("checkpoint", "walker_dir", "corrected_oracle")
        if not getattr(args, name)
    ]
    if missing:
        raise ValueError(f"missing required arguments: {', '.join(missing)}")
    default_output, default_zip = _default_output_paths()
    args.output_dir = args.output_dir or str(default_output)
    args.zip_path = args.zip_path or str(default_zip)
    if not args.regularization_reference:
        args.regularization_reference = str(
            REPO_ROOT / "output/diagnostics" / REFERENCE_REGULARIZATION_NAME
        )
    return args


def validate_paths(args: argparse.Namespace) -> dict[str, Path]:
    paths = aref.validate_paths(args)
    paths["regularization_reference"] = Path(args.regularization_reference).resolve()
    if not paths["regularization_reference"].is_dir():
        raise FileNotFoundError(
            "regularization reference artifact is missing: "
            f"{paths['regularization_reference']}"
        )
    if paths["output_dir"].exists():
        raise FileExistsError(f"refusing to overwrite output: {paths['output_dir']}")
    if paths["zip_path"].exists():
        raise FileExistsError(f"refusing to overwrite ZIP: {paths['zip_path']}")
    return paths


def _json_normalize(value: Any) -> Any:
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        numeric = float(value)
        return numeric if np.isfinite(numeric) else None
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return _json_normalize(value.item())
        return [_json_normalize(item) for item in value.tolist()]
    if isinstance(value, dict):
        return {key: _json_normalize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_normalize(item) for item in value]
    return value


def write_json(path: Path, payload: Any) -> None:
    oracle.write_json(path, _json_normalize(payload))


def _allclose(left: Any, right: Any, rtol: float = REGRESSION_RTOL, atol: float = REGRESSION_ATOL) -> bool:
    if left is None or right is None:
        return left is None and right is None
    try:
        return bool(np.allclose(np.asarray(left), np.asarray(right), rtol=rtol, atol=atol))
    except (TypeError, ValueError):
        return False


def _finite(value: Any) -> bool:
    try:
        return bool(np.isfinite(np.asarray(value, dtype=np.float64)).all())
    except (TypeError, ValueError):
        return False


def _safe_signature(value: Any) -> str | None:
    try:
        return str(inspect.signature(value))
    except (TypeError, ValueError):
        return None


def _safe_docstring(value: Any) -> str | None:
    doc = getattr(value, "__doc__", None)
    return None if doc is None else str(doc).strip()[:2000]


def _capability_discovery(mujoco: Any) -> dict[str, Any]:
    symbols = {}
    for name, owner in (
        ("mj_copyModel", mujoco),
        ("mj_saveModel", mujoco),
        ("MjModel.from_binary_path", getattr(mujoco, "MjModel", None)),
    ):
        attribute = name.rsplit(".", 1)[-1]
        value = getattr(owner, attribute, None) if owner is not None else None
        symbols[name] = {
            "available": bool(callable(value)),
            "signature": _safe_signature(value) if callable(value) else None,
            "docstring": _safe_docstring(value) if callable(value) else None,
        }
    return {
        "mujoco_version": getattr(mujoco, "__version__", None),
        "symbols": symbols,
        "formal_xml_clone_forbidden": True,
        "runtime_smoke": {},
        "EXACT_MODEL_CLONE_API": "UNAVAILABLE",
        "MODEL_CLONE_METHOD": None,
    }


def _model_counts(model: Any) -> dict[str, int | None]:
    return {
        name: (int(getattr(model, name)) if getattr(model, name, None) is not None else None)
        for name in CLONE_COUNT_FIELDS
    }


def _array_digest(value: Any) -> dict[str, Any]:
    array = np.asarray(value)
    contiguous = np.ascontiguousarray(array)
    return {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "sha256": hashlib.sha256(contiguous.tobytes()).hexdigest(),
    }


def _clone_smoke(mujoco: Any, source: Any, clone: Any) -> dict[str, Any]:
    checks: dict[str, bool] = {
        "clone_is_not_source": clone is not source,
        "counts_equal": _model_counts(source) == _model_counts(clone),
    }
    source_tolerance = float(source.opt.tolerance)
    clone_tolerance_before = float(clone.opt.tolerance)
    clone_tolerance = source_tolerance * 0.5 if source_tolerance > 0 else np.nextafter(0.0, 1.0)
    try:
        clone.opt.tolerance = clone_tolerance
        checks["clone_option_mutation_is_local"] = bool(
            float(source.opt.tolerance) == source_tolerance
        )
    except Exception:
        checks["clone_option_mutation_is_local"] = False
    finally:
        try:
            clone.opt.tolerance = clone_tolerance_before
        except Exception:
            checks["clone_restore_after_smoke"] = False
        else:
            checks["clone_restore_after_smoke"] = True
    return {
        "checks": checks,
        "source_counts": _model_counts(source),
        "clone_counts": _model_counts(clone),
        "source_tolerance_after_clone_mutation": float(source.opt.tolerance),
        "CLONE_SMOKE": "PASS" if all(checks.values()) else "FAIL",
    }


def _save_model_mjb(mujoco: Any, model: Any, path: Path) -> dict[str, Any]:
    saver = getattr(mujoco, "mj_saveModel", None)
    if not callable(saver):
        raise RuntimeError("mj_saveModel is unavailable")
    path.parent.mkdir(parents=True, exist_ok=True)
    signature = _safe_signature(saver)
    attempts = []
    if signature is not None:
        try:
            parameters = list(inspect.signature(saver).parameters.values())
            positional = [item for item in parameters if item.kind in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            )]
            has_variadic = any(item.kind == inspect.Parameter.VAR_POSITIONAL for item in parameters)
            if len(positional) == 3 or has_variadic:
                saver(model, str(path), None)
            elif len(positional) == 2:
                saver(model, str(path))
            else:
                raise RuntimeError(f"unsupported mj_saveModel signature: {signature}")
        except Exception as error:
            attempts.append(f"signature {signature}: {type(error).__name__}: {error}")
    else:
        # The documented C API has (model, filename, vfs); this path is used
        # only when the wheel does not expose an inspectable signature.
        try:
            saver(model, str(path), None)
        except Exception as error:
            attempts.append(f"documented 3-argument call: {type(error).__name__}: {error}")
    if not path.is_file() or path.stat().st_size == 0:
        attempts.append("mj_saveModel returned without a non-empty MJB")
    if attempts:
        raise RuntimeError("; ".join(attempts))
    return {
        "path": str(path),
        "signature": signature,
        "size_bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _load_model_mjb(mujoco: Any, path: Path) -> Any:
    model_type = getattr(mujoco, "MjModel", None)
    loader = getattr(model_type, "from_binary_path", None)
    if not callable(loader):
        raise RuntimeError("MjModel.from_binary_path is unavailable")
    return loader(str(path))


def _copy_model(
    mujoco: Any,
    model: Any,
    method: str,
    mjb_path: Path | None = None,
) -> tuple[Any, str, dict[str, Any]]:
    if method == "MJ_COPY_MODEL":
        copier = getattr(mujoco, "mj_copyModel", None)
        if not callable(copier):
            raise RuntimeError("selected mj_copyModel method is unavailable")
        errors = []
        for arguments in ((None, model), (model,)):
            try:
                clone = copier(*arguments)
                if clone is None:
                    raise RuntimeError("copy returned None")
                smoke = _clone_smoke(mujoco, model, clone)
                if smoke["CLONE_SMOKE"] != "PASS":
                    raise RuntimeError(json.dumps(_json_normalize(smoke), sort_keys=True))
                return clone, "MJ_COPY_MODEL", smoke
            except Exception as error:  # pragma: no cover - binding-specific
                errors.append(f"{type(error).__name__}: {error}")
        raise RuntimeError("mj_copyModel failed: " + "; ".join(errors))
    if method == "MJB_ROUNDTRIP":
        if mjb_path is None:
            raise RuntimeError("MJB roundtrip requires a condition-specific path")
        saved = _save_model_mjb(mujoco, model, mjb_path)
        clone = _load_model_mjb(mujoco, mjb_path)
        smoke = _clone_smoke(mujoco, model, clone)
        smoke["mjb"] = saved
        if smoke["CLONE_SMOKE"] != "PASS":
            raise RuntimeError(json.dumps(_json_normalize(smoke), sort_keys=True))
        return clone, "MJB_ROUNDTRIP", smoke
    raise RuntimeError(f"_copy_model does not support method {method}")


def _scalar(data: Any, name: str, default: Any = None) -> Any:
    value = getattr(data, name, None)
    if value is None:
        return default
    try:
        array = np.asarray(value)
        if array.size == 1:
            return array.reshape(-1)[0].item()
        return array.copy()
    except (TypeError, ValueError):
        return default


def _model_option_snapshot(model: Any) -> dict[str, Any]:
    snapshot = aref.cone_helper.model_option_snapshot(model)
    for name in (
        "jacobian", "impratio", "ls_tolerance", "noslip_tolerance",
        "ccd_tolerance", "noslip_iterations", "ccd_iterations",
        "sdf_initpoints", "enableflags", "o_solref", "o_solimp",
    ):
        value = getattr(model.opt, name, None)
        if value is None:
            continue
        array = np.asarray(value)
        snapshot[f"opt.{name}"] = (
            array.copy() if array.ndim else array.item()
        )
    return snapshot


def _option_difference(
    reference: dict[str, Any], candidate: dict[str, Any], allowed: set[str] = set()
) -> dict[str, Any]:
    changed = []
    details = {}
    for name, left in reference.items():
        right = candidate.get(name)
        equal = _allclose(left, right) if isinstance(left, np.ndarray) else left == right
        details[name] = {"equal": bool(equal), "allowed": name in allowed}
        if not equal:
            changed.append(name)
    unexpected = [name for name in changed if name not in allowed]
    return {
        "changed_fields": changed,
        "unexpected_changed_fields": unexpected,
        "only_allowed": not unexpected,
        "details": details,
    }


def _decode_name(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _named_sequence(mujoco: Any, model: Any, object_name: str, count_name: str) -> list[str | None]:
    enum_type = getattr(getattr(mujoco, "mjtObj", None), object_name, None)
    if enum_type is None:
        return []
    count = int(getattr(model, count_name, 0))
    return [
        _decode_name(mujoco.mj_id2name(model, enum_type, index))
        for index in range(count)
    ]


def _geom_inventory(mujoco: Any, model: Any) -> list[dict[str, Any]]:
    result = []
    for index in range(int(getattr(model, "ngeom", 0))):
        geom_name = _named_sequence(mujoco, model, "mjOBJ_GEOM", "ngeom")
        geom_name = geom_name[index] if index < len(geom_name) else None
        body_id = int(np.asarray(model.geom_bodyid)[index])
        body_names = _named_sequence(mujoco, model, "mjOBJ_BODY", "nbody")
        result.append({
            "geom_id": index,
            "geom_name": geom_name,
            "geom_type": int(np.asarray(model.geom_type)[index]),
            "body_id": body_id,
            "body_name": body_names[body_id] if body_id < len(body_names) else None,
            "friction": np.asarray(model.geom_friction)[index].copy(),
            "contype": int(np.asarray(model.geom_contype)[index]),
            "conaffinity": int(np.asarray(model.geom_conaffinity)[index]),
        })
    return result


def _runtime_source_geom_inventory(
    mujoco: Any, live_model: Any, morphology_xml: Path
) -> dict[str, Any]:
    live = _geom_inventory(mujoco, live_model)
    source = []
    source_error = None
    loader = getattr(getattr(mujoco, "MjModel", None), "from_xml_path", None)
    if callable(loader):
        try:
            source = _geom_inventory(mujoco, loader(str(morphology_xml)))
        except Exception as error:  # diagnostic only; never a formal clone path
            source_error = f"{type(error).__name__}: {error}"
    else:
        source_error = "MjModel.from_xml_path unavailable for source diagnostic"
    live_names = [item["geom_name"] for item in live]
    source_names = [item["geom_name"] for item in source]
    live_name_set, source_name_set = set(live_names), set(source_names)
    live_only_names = sorted(name for name in live_name_set - source_name_set if name is not None)
    source_only_names = sorted(name for name in source_name_set - live_name_set if name is not None)
    live_only_ids = [item["geom_id"] for item in live if item["geom_name"] in live_only_names]
    source_only_ids = [item["geom_id"] for item in source if item["geom_name"] in source_only_names]
    exact = bool(
        source_error is None
        and live_names == source_names
        and len(live) == len(source)
        and all(
            item["geom_type"] == other["geom_type"]
            and item["body_id"] == other["body_id"]
            and item["contype"] == other["contype"]
            and item["conaffinity"] == other["conaffinity"]
            and _allclose(item["friction"], other["friction"])
            for item, other in zip(live, source)
        )
    )
    if exact:
        structure = "MATCHES_SOURCE_XML"
    elif source_error is None and live_only_names and not source_only_names:
        structure = "HAS_RUNTIME_AUGMENTATION"
    else:
        structure = "OTHER_MISMATCH"
    return {
        "live_compiled_model": live,
        "source_xml_recompiled_diagnostic_model": source,
        "live_only_geom_ids": live_only_ids,
        "live_only_geom_names": live_only_names,
        "source_only_geom_ids": source_only_ids,
        "source_only_geom_names": source_only_names,
        "source_diagnostic_error": source_error,
        "RUNTIME_MODEL_STRUCTURE": structure,
        "formal_clone_path_used_source_xml": False,
    }


def _exact_value_equal(left: Any, right: Any) -> bool:
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        return isinstance(left, np.ndarray) and isinstance(right, np.ndarray) and np.array_equal(left, right)
    return left == right


def _model_clone_fidelity(mujoco: Any, source: Any, clone: Any, method: str) -> dict[str, Any]:
    count_checks = {
        name: int(getattr(source, name, -1)) == int(getattr(clone, name, -2))
        for name in CLONE_COUNT_FIELDS
    }
    source_options = _model_option_snapshot(source)
    clone_options = _model_option_snapshot(clone)
    option_checks = {
        name: _exact_value_equal(value, clone_options.get(name))
        for name, value in source_options.items()
    }
    array_checks = {}
    for name in CLONE_ARRAY_FIELDS:
        left = getattr(source, name, None)
        right = getattr(clone, name, None)
        if left is None or right is None:
            array_checks[name] = {
                "available": left is None and right is None,
                "equal": left is right,
            }
            continue
        left_array, right_array = np.asarray(left), np.asarray(right)
        same_shape = left_array.shape == right_array.shape
        equal = bool(same_shape and np.array_equal(left_array, right_array))
        array_checks[name] = {
            "available": True,
            "shape_equal": same_shape,
            "dtype_equal": str(left_array.dtype) == str(right_array.dtype),
            "max_abs_difference": float(
                np.max(np.abs(left_array.astype(np.float64) - right_array.astype(np.float64)))
            ) if same_shape and left_array.size else 0.0 if same_shape else None,
            "source_bytes": _array_digest(left_array),
            "clone_bytes": _array_digest(right_array),
            "equal": equal,
        }
    name_checks = {}
    for label, object_name, count_name in CLONE_NAME_OBJECTS:
        name_checks[label] = {
            "source": _named_sequence(mujoco, source, object_name, count_name),
            "clone": _named_sequence(mujoco, clone, object_name, count_name),
        }
        name_checks[label]["equal"] = name_checks[label]["source"] == name_checks[label]["clone"]
    valid = bool(
        method in {"MJ_COPY_MODEL", "MJB_ROUNDTRIP"}
        and all(count_checks.values())
        and all(option_checks.values())
        and all(item.get("equal", False) for item in array_checks.values())
        and all(item["equal"] for item in name_checks.values())
    )
    return {
        "method": method,
        "source_counts": _model_counts(source),
        "clone_counts": _model_counts(clone),
        "count_checks": count_checks,
        "solver_option_checks": option_checks,
        "source_options": source_options,
        "clone_options": clone_options,
        "physical_array_checks": array_checks,
        "name_order_checks": name_checks,
        "EXACT_MODEL_CLONE_FIDELITY": "PASS" if valid else "FAIL",
    }


def _clone_data_state_fidelity(
    mujoco: Any, model: Any, snapshot: Any, method: str
) -> dict[str, Any]:
    try:
        data = mujoco.MjData(model)
        mujoco.mj_copyData(data, model, snapshot)
        equality = aref.cone_helper.state_equality(snapshot, data)
        valid = bool(equality.get("STATE_COPY_EQUAL"))
        return {
            "method": method,
            "state_equality": equality,
            "CLONE_DATA_STATE_FIDELITY": "PASS" if valid else "FAIL",
        }
    except Exception as error:
        return {
            "method": method,
            "error": f"{type(error).__name__}: {error}",
            "CLONE_DATA_STATE_FIDELITY": "FAIL",
        }


def _transactional_smoke(mujoco: Any, model: Any) -> dict[str, Any]:
    before = _model_option_snapshot(model)
    checks = {}
    try:
        original_tolerance = float(model.opt.tolerance)
        original_iterations = int(model.opt.iterations)
        model.opt.tolerance = original_tolerance * 0.5
        model.opt.iterations = original_iterations + 1
        checks["tolerance_mutation_is_available"] = float(model.opt.tolerance) != original_tolerance
        checks["iterations_mutation_is_available"] = int(model.opt.iterations) == original_iterations + 1
    except Exception:
        checks["tolerance_mutation_is_available"] = False
        checks["iterations_mutation_is_available"] = False
    finally:
        model.opt.tolerance = before["opt.tolerance"]
        model.opt.iterations = before["opt.iterations"]
    after = _model_option_snapshot(model)
    checks["exact_restore"] = _option_difference(before, after)["only_allowed"]
    return {
        "checks": checks,
        "before": before,
        "after": after,
        "TRANSACTIONAL_SMOKE": "PASS" if all(checks.values()) else "FAIL",
    }


def _select_model_isolation_method(
    mujoco: Any, model: Any, discovery: dict[str, Any], mjb_root: Path
) -> tuple[str, str, Any | None]:
    if discovery["symbols"]["mj_copyModel"]["available"]:
        try:
            clone, _, smoke = _copy_model(mujoco, model, "MJ_COPY_MODEL")
            discovery["runtime_smoke"]["mj_copyModel"] = smoke
            discovery["EXACT_MODEL_CLONE_API"] = "PYTHON_MJ_COPY_MODEL"
            discovery["MODEL_CLONE_METHOD"] = "MJ_COPY_MODEL"
            return "MJ_COPY_MODEL", "PYTHON_MJ_COPY_MODEL", clone
        except Exception as error:
            discovery["runtime_smoke"]["mj_copyModel"] = {
                "error": f"{type(error).__name__}: {error}",
                "CLONE_SMOKE": "FAIL",
            }
    if (
        discovery["symbols"]["mj_saveModel"]["available"]
        and discovery["symbols"]["MjModel.from_binary_path"]["available"]
    ):
        try:
            probe_path = mjb_root / "solver_optimization_live_model_capability_probe.mjb"
            clone, _, smoke = _copy_model(mujoco, model, "MJB_ROUNDTRIP", probe_path)
            discovery["runtime_smoke"]["mjb_roundtrip"] = smoke
            discovery["EXACT_MODEL_CLONE_API"] = "MJB_ROUNDTRIP"
            discovery["MODEL_CLONE_METHOD"] = "MJB_ROUNDTRIP"
            return "MJB_ROUNDTRIP", "MJB_ROUNDTRIP", clone
        except Exception as error:
            discovery["runtime_smoke"]["mjb_roundtrip"] = {
                "error": f"{type(error).__name__}: {error}",
                "CLONE_SMOKE": "FAIL",
            }
    transaction = _transactional_smoke(mujoco, model)
    discovery["runtime_smoke"]["transactional_shared_model"] = transaction
    if transaction["TRANSACTIONAL_SMOKE"] == "PASS":
        discovery["EXACT_MODEL_CLONE_API"] = "TRANSACTIONAL_SHARED_MODEL"
        discovery["MODEL_CLONE_METHOD"] = "TRANSACTIONAL_SOLVER_OPTION_RESTORE"
        return "TRANSACTIONAL_SOLVER_OPTION_RESTORE", "TRANSACTIONAL_SHARED_MODEL", None
    discovery["EXACT_MODEL_CLONE_API"] = "UNAVAILABLE"
    raise RuntimeError("no safe compiled-model clone or transactional option path is available")


def _tight_tolerance(production: float) -> tuple[float, str]:
    if not np.isfinite(production) or production <= 0.0:
        raise RuntimeError(f"production tolerance is not a positive finite value: {production}")
    if production > TIGHT_TOLERANCE_DEFAULT:
        return TIGHT_TOLERANCE_DEFAULT, "fixed 1e-12 is strictly below production tolerance"
    candidate = float(np.nextafter(production, 0.0))
    if not np.isfinite(candidate) or candidate <= 0.0 or not candidate < production:
        raise RuntimeError("cannot select a representable stricter tolerance")
    return candidate, "production tolerance was already <= 1e-12; nextafter(production, 0) selected"


def _configure_model(
    model: Any,
    production_options: dict[str, Any],
    tight: bool,
) -> dict[str, Any]:
    original_tolerance = float(production_options["opt.tolerance"])
    original_iterations = int(production_options["opt.iterations"])
    if tight:
        tolerance, rationale = _tight_tolerance(original_tolerance)
        iterations = max(original_iterations, 100)
    else:
        tolerance, rationale = original_tolerance, "production options retained"
        iterations = original_iterations
    model.opt.tolerance = tolerance
    model.opt.iterations = iterations
    return {
        "tight": bool(tight),
        "production_tolerance": original_tolerance,
        "condition_tolerance": tolerance,
        "production_iterations": original_iterations,
        "condition_iterations": iterations,
        "tight_tolerance_rationale": rationale,
    }


def _contact_geometry(data: Any, mujoco: Any, model: Any) -> dict[str, Any]:
    contacts = []
    for index in range(int(getattr(data, "ncon", 0))):
        contact = data.contact[index]
        geom_ids = [int(contact.geom1), int(contact.geom2)]
        geom_names = []
        for geom_id in geom_ids:
            try:
                geom_names.append(str(mujoco.mj_id2name(
                    model, mujoco.mjtObj.mjOBJ_GEOM, geom_id
                )))
            except Exception:
                geom_names.append(str(geom_id))
        contacts.append({
            "contact_index": index,
            "geom_ids": geom_ids,
            "geom_names": geom_names,
            "point": np.asarray(contact.pos, dtype=np.float64).copy(),
            "frame": np.asarray(contact.frame, dtype=np.float64).copy(),
            "dist": float(contact.dist),
            "dim": int(contact.dim),
            "efc_address": int(contact.efc_address),
            "friction": np.asarray(contact.friction, dtype=np.float64).copy(),
        })
    return {"ncon": int(getattr(data, "ncon", 0)), "contacts": contacts}


def _contact_geometry_equal(left: dict[str, Any], right: dict[str, Any]) -> bool:
    if left.get("ncon") != right.get("ncon"):
        return False
    for a, b in zip(left.get("contacts", []), right.get("contacts", [])):
        if a["geom_ids"] != b["geom_ids"] or a["dim"] != b["dim"]:
            return False
        if not _allclose(a["point"], b["point"]) or not _allclose(a["frame"], b["frame"]):
            return False
        if not _allclose(a["friction"], b["friction"]):
            return False
    return True


def _state_without_warmstart(snapshot: dict[str, Any]) -> dict[str, Any]:
    return {
        name: value for name, value in snapshot.items()
        if name != "qacc_warmstart"
    }


def _state_equal_except_warmstart(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    checks = {}
    for name in _state_without_warmstart(left):
        a, b = left.get(name), right.get(name)
        checks[name] = _allclose(a, b) if isinstance(a, np.ndarray) else a == b
    return {"checks": checks, "valid": bool(all(checks.values()))}


def _stat_value(stat: Any, name: str) -> Any:
    value = getattr(stat, name, None)
    if value is None:
        return None
    try:
        array = np.asarray(value)
        return array.copy() if array.ndim else array.item()
    except (TypeError, ValueError):
        return None


def _array_metadata(value: Any) -> dict[str, Any]:
    """Return JSON-safe shape/type/value evidence for a MuJoCo array view."""
    if value is None:
        return {"available": False, "type": None, "reason": "unavailable"}
    try:
        array = np.asarray(value)
    except (TypeError, ValueError) as error:
        return {
            "available": False,
            "type": type(value).__name__,
            "error": f"{type(error).__name__}: {error}",
        }
    numeric = False
    try:
        numeric = bool(np.issubdtype(array.dtype, np.number) or np.issubdtype(array.dtype, np.bool_))
    except TypeError:
        numeric = False
    result = {
        "available": True,
        "type": type(value).__name__,
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "size": int(array.size),
        "values": array.copy() if numeric else None,
        "values_available": numeric,
    }
    if not numeric and array.size:
        try:
            result["element_type"] = type(array.reshape(-1)[0]).__name__
        except Exception:
            result["element_type"] = None
    return result


def _integer_vector(data: Any, name: str) -> tuple[list[int], str | None]:
    value = getattr(data, name, None)
    if value is None:
        return [], f"{name} is unavailable"
    try:
        array = np.asarray(value)
        if array.size == 0:
            return [], f"{name} is empty"
        numeric = np.asarray(array, dtype=np.float64).reshape(-1)
        if not np.isfinite(numeric).all() or not np.equal(numeric, np.floor(numeric)).all():
            return [], f"{name} contains non-finite or non-integer values"
        return [int(item) for item in numeric], None
    except (TypeError, ValueError) as error:
        return [], f"{name}: {type(error).__name__}: {error}"


def _safe_public_fields(value: Any) -> list[str]:
    try:
        return sorted(name for name in dir(value) if not name.startswith("_"))
    except Exception:
        return []


def _solver_layout(
    data: Any,
    mujoco: Any = None,
    active_count: int = 1,
    island_capacity: int | None = None,
) -> dict[str, Any]:
    """Discover how the Python binding exposes mjSolverStat slots."""
    stats = getattr(data, "solver", None)
    result: dict[str, Any] = {
        "available": stats is not None,
        "type": type(stats).__name__ if stats is not None else None,
        "layout": None,
        "shape": None,
        "dtype": None,
        "length": None,
        "element_type": None,
        "element_fields": [],
        "solver_slots_per_island": None,
        "errors": [],
    }
    if stats is None:
        result["errors"].append("data.solver is unavailable")
        return result
    try:
        array = np.asarray(stats)
        result["shape"] = list(array.shape)
        result["dtype"] = str(array.dtype)
        result["length"] = int(array.size)
        if array.ndim >= 2 and array.shape[0] >= max(1, active_count):
            result["layout"] = "2D"
            result["solver_slots_per_island"] = int(array.shape[1])
        else:
            result["layout"] = "FLAT"
    except (TypeError, ValueError) as error:
        result["errors"].append(f"array view: {type(error).__name__}: {error}")
        try:
            result["length"] = int(len(stats))
            result["layout"] = "FLAT"
        except Exception as length_error:
            result["errors"].append(
                f"length: {type(length_error).__name__}: {length_error}"
            )

    constant = getattr(mujoco, "mjNSOLVER", None) if mujoco is not None else None
    try:
        constant = int(constant) if constant is not None else None
    except (TypeError, ValueError):
        constant = None
    result["mjNSOLVER"] = constant
    length = result["length"]
    if result["solver_slots_per_island"] is None and length is not None:
        if constant is not None and length >= max(1, active_count) * constant:
            result["solver_slots_per_island"] = constant
        elif isinstance(island_capacity, int) and island_capacity > 0 and length % island_capacity == 0:
            result["solver_slots_per_island"] = int(length // island_capacity)
        elif active_count > 0 and length % active_count == 0:
            result["solver_slots_per_island"] = int(length // active_count)
        elif active_count == 1 and length > 0:
            result["solver_slots_per_island"] = int(length)
    try:
        first = stats[0]
        result["element_type"] = type(first).__name__
        result["element_fields"] = _safe_public_fields(first)
    except Exception as error:
        result["errors"].append(f"first element: {type(error).__name__}: {error}")
    return result


def _active_solver_island_count(
    data: Any,
    niter_values: list[int],
    nnz_values: list[int],
    fallback: int | None = None,
) -> tuple[int, str, list[str]]:
    evidence: list[str] = []
    for name, source in (
        ("solver_nisland", "SOLVER_NISLAND"),
        ("nsolver_island", "NSOLVER_ISLAND"),
        ("nisland", "DATA_NISLAND"),
    ):
        value = getattr(data, name, None)
        try:
            array = np.asarray(value)
            if value is not None and array.size == 1:
                count = int(array.reshape(-1)[0])
                if count >= 0:
                    evidence.append(f"{name}={count}")
                    return count, source, evidence
        except (TypeError, ValueError):
            pass
    nonzero = [index for index, value in enumerate(niter_values) if value > 0]
    nonzero += [index for index, value in enumerate(nnz_values) if value > 0]
    if nonzero:
        count = max(nonzero) + 1
        evidence.append(f"nonzero_solver_arrays_through_island={count - 1}")
        return count, "NONZERO_SOLVER_ARRAYS", evidence
    if fallback is not None and fallback >= 0:
        evidence.append(f"fallback={fallback}")
        return int(fallback), "FALLBACK", evidence
    return 0, "UNAVAILABLE", evidence


def _solver_stat_api_discovery(
    mujoco: Any,
    model: Any,
    snapshot: Any,
    probe_model: Any = None,
) -> dict[str, Any]:
    """Probe one ordinary staged Newton solve and record the runtime layout."""
    active_model = model if probe_model is None else probe_model
    report: dict[str, Any] = {
        "mujoco_version": getattr(mujoco, "__version__", None),
        "probe_staged_forward_count": 0,
        "probe_constraint_solve_count": 0,
        "solver_niter": {},
        "solver_nnz": {},
        "solver_nisland": {},
        "nsolver_island": {},
        "nisland": {},
        "solver": {},
        "constants": {},
        "active_solver_island_count": None,
        "active_solver_island_count_source": None,
        "solver_layout": {},
        "status": "INSUFFICIENT_EVIDENCE",
        "errors": [],
    }
    for name in ("mjNISLAND", "mjNSOLVER"):
        value = getattr(mujoco, name, None)
        try:
            report["constants"][name] = int(value) if value is not None else None
        except (TypeError, ValueError):
            report["constants"][name] = None
    try:
        data = mujoco.MjData(active_model)
        mujoco.mj_copyData(data, active_model, snapshot)
        stage_calls = aref.stage_to_constraint(mujoco, active_model, data)
        mujoco.mj_fwdConstraint(active_model, data)
        report["probe_staged_forward_count"] = 1
        report["probe_constraint_solve_count"] = 1
        report["stage_calls"] = stage_calls
        for name in ("solver_niter", "solver_nnz", "solver_nisland", "nsolver_island", "nisland"):
            present = bool(hasattr(data, name))
            report[name] = _array_metadata(getattr(data, name, None) if present else None)
            report[name]["hasattr"] = present
        solver_present = bool(hasattr(data, "solver"))
        report["solver"] = _array_metadata(
            getattr(data, "solver", None) if solver_present else None
        )
        report["solver"]["hasattr"] = solver_present
        niter_values, niter_error = _integer_vector(data, "solver_niter")
        nnz_values, nnz_error = _integer_vector(data, "solver_nnz")
        active_count, source, evidence = _active_solver_island_count(
            data, niter_values, nnz_values
        )
        report["active_solver_island_count"] = active_count
        report["active_solver_island_count_source"] = source
        report["active_solver_island_count_evidence"] = evidence
        if niter_error:
            report["errors"].append(niter_error)
        if nnz_error:
            report["errors"].append(nnz_error)
        report["solver_layout"] = _solver_layout(
            data, mujoco, max(1, active_count), island_capacity=len(niter_values)
        )
        report["solver_stat_fields"] = report["solver_layout"].get("element_fields", [])
        report["status"] = "VALID" if (
            active_count > 0
            and not niter_error
            and report["solver_layout"].get("available")
            and (report["solver_layout"].get("solver_slots_per_island") or 0) > 0
        ) else "INSUFFICIENT_EVIDENCE"
    except Exception as error:  # pragma: no cover - binding/runtime-specific
        report["errors"].append(f"{type(error).__name__}: {error}")
    return report


def _solver_stat_at(data: Any, layout: dict[str, Any], island_id: int, iteration: int) -> Any:
    stats = getattr(data, "solver", None)
    if stats is None:
        raise RuntimeError("data.solver is unavailable")
    slots = layout.get("solver_slots_per_island")
    if not isinstance(slots, int) or slots <= 0:
        raise RuntimeError("solver slots per island are unavailable")
    if layout.get("layout") == "2D":
        try:
            return stats[island_id, iteration]
        except Exception:
            return stats[island_id][iteration]
    return stats[island_id * slots + iteration]


def _solver_iteration_trace(
    data: Any, model: Any, solver_api: dict[str, Any] | None = None
) -> dict[str, Any]:
    niter_values, niter_error = _integer_vector(data, "solver_niter")
    nnz_values, nnz_error = _integer_vector(data, "solver_nnz")
    fallback_count = None
    if solver_api is not None:
        value = solver_api.get("active_solver_island_count")
        if isinstance(value, (int, np.integer)):
            fallback_count = int(value)
    active_count, count_source, count_evidence = _active_solver_island_count(
        data, niter_values, nnz_values, fallback_count
    )
    limit = int(getattr(model.opt, "iterations", -1))
    layout = (solver_api or {}).get("solver_layout") or _solver_layout(
        data, None, max(1, active_count), island_capacity=len(niter_values)
    )
    islands = []
    errors = [item for item in (niter_error, nnz_error) if item]
    for island_id in range(active_count):
        niter = niter_values[island_id] if island_id < len(niter_values) else -1
        nnz = nnz_values[island_id] if island_id < len(nnz_values) else None
        iterations = []
        island_errors = []
        for iteration in range(max(0, niter)):
            try:
                stat = _solver_stat_at(data, layout, island_id, iteration)
                row = {"island_id": island_id, "iteration_index": iteration}
                row.update({name: _stat_value(stat, name) for name in REQUIRED_SOLVER_STAT_FIELDS})
                available = [name for name in REQUIRED_SOLVER_STAT_FIELDS if row[name] is not None]
                row["available_fields"] = available
                row["statistics_available"] = bool(available)
                row["statistics_complete"] = len(available) == len(REQUIRED_SOLVER_STAT_FIELDS)
                row["statistics_finite"] = bool(available and all(_finite(row[name]) for name in available))
                iterations.append(row)
            except Exception as error:
                message = f"island {island_id} iteration {iteration}: {type(error).__name__}: {error}"
                island_errors.append(message)
                errors.append(message)
        island_available = bool(
            niter > 0 and len(iterations) == niter
            and all(row["statistics_available"] for row in iterations)
        )
        island_finite = bool(
            island_available and all(row["statistics_finite"] for row in iterations)
        )
        islands.append({
            "island_id": island_id,
            "niter": niter,
            "nnz": nnz,
            "iterations_limit": limit,
            "niter_below_limit": bool(niter >= 0 and niter < limit),
            "iterations": iterations,
            "last_iteration": iterations[-1] if iterations else {},
            "statistics_available": island_available,
            "statistics_complete": bool(
                island_available
                and all(row["statistics_complete"] for row in iterations)
            ),
            "statistics_finite": island_finite,
            "errors": island_errors,
        })
    active_valid = bool(islands and all(item["statistics_available"] for item in islands))
    active_finite = bool(active_valid and all(item["statistics_finite"] for item in islands))
    return {
        "active_solver_island_count": active_count,
        "active_solver_island_count_source": count_source,
        "active_solver_island_count_evidence": count_evidence,
        "solver_niter": niter_values,
        "solver_nnz": nnz_values,
        "iterations_limit": limit,
        "solver_layout": layout,
        "islands": islands,
        # Keep trace/last_iteration for compatibility with prior artifacts and tests.
        "trace": [row for island in islands for row in island["iterations"]],
        "last_iteration": islands[-1]["last_iteration"] if islands else {},
        "last_iterations": [island["last_iteration"] for island in islands],
        "required_fields": list(REQUIRED_SOLVER_STAT_FIELDS),
        "statistics_available": active_valid,
        "statistics_complete": bool(
            active_valid and all(item["statistics_complete"] for item in islands)
        ),
        "all_statistics_finite": active_finite,
        "errors": errors,
        "status": "VALID" if active_finite else "INSUFFICIENT_EVIDENCE",
    }


def _solver_numerics(
    data: Any,
    model: Any,
    warmstart_input: Any,
    trace: dict[str, Any],
    configuration: dict[str, Any],
) -> dict[str, Any]:
    warmstart = np.asarray(warmstart_input, dtype=np.float64).copy()
    fields = {"solver_fwdinv": _scalar(data, "solver_fwdinv")}
    finite = all(_finite(value) for value in fields.values() if value is not None)
    finite = bool(finite and _finite(warmstart) and trace["all_statistics_finite"])
    return {
        "solver_niter": trace.get("solver_niter", []),
        "solver_nnz": trace.get("solver_nnz", []),
        "active_solver_islands": [item["island_id"] for item in trace.get("islands", [])],
        "active_solver_island_count": trace.get("active_solver_island_count", 0),
        "active_solver_niter": [item["niter"] for item in trace.get("islands", [])],
        "active_solver_nnz": [item["nnz"] for item in trace.get("islands", [])],
        "solver_fwdinv": fields["solver_fwdinv"],
        "tolerance": float(model.opt.tolerance),
        "iterations_limit": int(model.opt.iterations),
        "warmstart_vector_norm": float(np.linalg.norm(warmstart)),
        "warmstart_max_abs": float(np.max(np.abs(warmstart))) if warmstart.size else 0.0,
        "warmstart_all_finite": bool(_finite(warmstart)),
        "statistics_status": trace["status"],
        "finite": finite,
        "status": "VALID" if finite else "NONFINITE" if not all(
            _finite(value) for value in fields.values() if value is not None
        ) else "INSUFFICIENT_EVIDENCE",
        "configuration": configuration,
    }


def _condition_state_validation(
    snapshot: Any,
    data: Any,
    expected_zero_warmstart: bool,
) -> dict[str, Any]:
    clone = aref.cone_helper.state_equality(snapshot, data)
    state = aref.cone_helper.state_input_snapshot(data)
    original = aref.cone_helper.state_input_snapshot(snapshot)
    warmstart = np.asarray(state["qacc_warmstart"], dtype=np.float64)
    expected = np.zeros_like(warmstart) if expected_zero_warmstart else np.asarray(
        original["qacc_warmstart"], dtype=np.float64
    )
    return {
        "clone_pre_state": clone,
        "pre_state_snapshot": state,
        "warmstart_expected": expected,
        "warmstart_exact_zero": bool(expected_zero_warmstart and np.array_equal(warmstart, np.zeros_like(warmstart))),
        "warmstart_matches_expected": bool(np.array_equal(warmstart, expected)),
        "same_complete_pre_state_before_intervention": bool(
            clone["STATE_COPY_EQUAL"] if not expected_zero_warmstart else
            _state_equal_except_warmstart(original, state)["valid"]
        ),
    }


def _restore_transactional_solver_options(
    model: Any, production_options: dict[str, Any]
) -> dict[str, Any]:
    model.opt.tolerance = production_options["opt.tolerance"]
    model.opt.iterations = production_options["opt.iterations"]
    after = _model_option_snapshot(model)
    difference = _option_difference(production_options, after)
    return {
        "after_options": after,
        "difference": difference,
        "MODEL_OPTION_RESTORE": "PASS" if difference["only_allowed"] else "FAIL",
    }


def _run_condition(
    mujoco: Any,
    base_model: Any,
    snapshot: Any,
    mapping: dict[str, Any],
    name: str,
    label: str,
    zero_warmstart: bool,
    tight: bool,
    production_options: dict[str, Any],
    isolation_method: str,
    mjb_path: Path | None,
    solver_api: dict[str, Any] | None = None,
) -> dict[str, Any]:
    transactional = isolation_method == "TRANSACTIONAL_SOLVER_OPTION_RESTORE"
    clone_smoke = None
    if transactional:
        model = base_model
        model_copy_api = "TRANSACTIONAL_SHARED_MODEL"
    else:
        model, model_copy_api, clone_smoke = _copy_model(
            mujoco, base_model, isolation_method, mjb_path
        )
    configuration = _configure_model(model, production_options, tight)
    result = None
    try:
        data = mujoco.MjData(model)
        mujoco.mj_copyData(data, model, snapshot)
        state_before_stage = aref.cone_helper.state_input_snapshot(data)
        if zero_warmstart:
            data.qacc_warmstart[:] = 0
        staged_calls = aref.stage_to_constraint(mujoco, model, data)
        state_validation = _condition_state_validation(snapshot, data, zero_warmstart)
        pre_constraint = regularization._constraint_snapshot(data, mujoco, model)
        pre_geometry = _contact_geometry(data, mujoco, model)
        warmstart_input = np.asarray(data.qacc_warmstart, dtype=np.float64).copy()
        mujoco.mj_fwdConstraint(model, data)
        solver_data = mujoco.MjData(model)
        mujoco.mj_copyData(solver_data, model, data)
        post_constraint = regularization._constraint_snapshot(
            solver_data, mujoco, model, read_physical_contact_forces=True
        )
        trace = _solver_iteration_trace(solver_data, model, solver_api)
        numerics = _solver_numerics(solver_data, model, warmstart_input, trace, configuration)
        mujoco.mj_Euler(model, data)
        capture = regularization.capture_after_integration(
            mujoco, model, data, snapshot, mapping, solver_data
        )
        demand = regularization._run_shared_demand(capture)
        excess = regularization._run_excess(capture, demand)
        result = {
            "condition_name": name,
            "condition_label": label,
            "zero_warmstart": bool(zero_warmstart),
            "tight_tolerance": bool(tight),
            "model_copy_api": model_copy_api,
            "model_clone_method": isolation_method,
            "clone_smoke": clone_smoke,
            "model_options": _model_option_snapshot(model),
            "configuration": configuration,
            "state_validation": state_validation,
            "state_before_stage": state_before_stage,
            "pre_constraint": pre_constraint,
            "pre_contact_geometry": pre_geometry,
            "post_constraint": post_constraint,
            "capture": capture,
            "shared_demand": demand,
            "excess": excess,
            "solver_iteration_trace": trace,
            "solver_numerics": numerics,
            "one_step_result": {
                "post_qpos": np.asarray(data.qpos, dtype=np.float64).copy(),
                "post_qvel": np.asarray(data.qvel, dtype=np.float64).copy(),
                "post_time": float(data.time),
                "target_post_slip": excess["post_slip"],
            },
            "counts": {
                "condition_staged_forward_count": 1,
                "constraint_solve_count": 1,
                "custom_integration_count": 1,
                "staged_calls": staged_calls,
                "solver_calls": ["mj_fwdConstraint"],
                "integration_api": "mujoco.mj_Euler",
            },
        }
    finally:
        if transactional:
            restore = _restore_transactional_solver_options(model, production_options)
        else:
            restore = {"MODEL_OPTION_RESTORE": "NOT_APPLICABLE_INDEPENDENT_CLONE"}
        if result is not None:
            result["model_option_restore"] = restore
    return result


def _relative_delta(left: float, right: float) -> float | None:
    try:
        denominator = max(abs(float(left)), np.finfo(float).eps)
        return float((float(right) - float(left)) / denominator)
    except (TypeError, ValueError):
        return None


def _physical_impulse_vectors(condition: dict[str, Any]) -> list[np.ndarray]:
    result = []
    for contact in condition["capture"].get("contacts", []):
        result.append(np.concatenate((
            np.asarray([contact["normal_impulse"]], dtype=np.float64),
            np.asarray(contact["tangential_impulse"], dtype=np.float64),
        )))
    return result


def _target_excess(condition: dict[str, Any]) -> dict[str, Any]:
    return condition["excess"]


def _pair_sensitivity(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    left_excess, right_excess = _target_excess(left), _target_excess(right)
    left_impulses = _physical_impulse_vectors(left)
    right_impulses = _physical_impulse_vectors(right)
    impulse_deltas = [
        float(np.max(np.abs(a - b)))
        for a, b in zip(left_impulses, right_impulses)
        if a.shape == b.shape
    ]
    target_impulse_delta = float(np.max(impulse_deltas)) if impulse_deltas else None
    left_actual = float(left_excess["actual_tangent_impulse_norm"])
    right_actual = float(right_excess["actual_tangent_impulse_norm"])
    left_normal = float(left_excess["normal_impulse"])
    right_normal = float(right_excess["normal_impulse"])
    left_demand = float(left_excess["rigid_demand_norm"])
    right_demand = float(right_excess["rigid_demand_norm"])
    left_excess_norm = float(left_excess["solver_excess_norm"])
    right_excess_norm = float(right_excess["solver_excess_norm"])
    qacc_delta = float(np.max(np.abs(
        np.asarray(left["post_constraint"]["qacc"], dtype=np.float64)
        - np.asarray(right["post_constraint"]["qacc"], dtype=np.float64)
    )))
    post_slip_delta = float(np.max(np.abs(
        np.asarray(left_excess["post_slip"], dtype=np.float64)
        - np.asarray(right_excess["post_slip"], dtype=np.float64)
    )))
    left_trace = left["solver_iteration_trace"].get("trace", [])
    right_trace = right["solver_iteration_trace"].get("trace", [])
    trace_equal = _json_normalize(left_trace) == _json_normalize(right_trace)
    metrics = {
        "physical_contact_impulse_max_absolute_delta": target_impulse_delta,
        "target_tangent_impulse_norm_relative_delta": _relative_delta(left_actual, right_actual),
        "normal_impulse_relative_delta": _relative_delta(left_normal, right_normal),
        "rigid_demand_relative_delta": _relative_delta(left_demand, right_demand),
        "solver_excess_relative_delta": _relative_delta(left_excess_norm, right_excess_norm),
        "qacc_max_absolute_delta": qacc_delta,
        "post_slip_max_absolute_delta": post_slip_delta,
        "solver_iteration_trace_equal": bool(trace_equal),
        "left_solver_niter": left["solver_numerics"].get("solver_niter"),
        "right_solver_niter": right["solver_numerics"].get("solver_niter"),
    }
    finite = all(
        value is not None and np.isfinite(value)
        for key, value in metrics.items()
        if key.endswith("delta") or key.endswith("_delta")
    )
    insensitive = bool(
        finite
        and float(metrics["physical_contact_impulse_max_absolute_delta"]) <= PHYSICAL_IMPULSE_ATOL
        and abs(float(metrics["target_tangent_impulse_norm_relative_delta"])) <= SENSITIVITY_RTOL
        and abs(float(metrics["normal_impulse_relative_delta"])) <= SENSITIVITY_RTOL
        and abs(float(metrics["rigid_demand_relative_delta"])) <= SENSITIVITY_RTOL
        and abs(float(metrics["solver_excess_relative_delta"])) <= SENSITIVITY_RTOL
        and float(metrics["qacc_max_absolute_delta"]) <= SENSITIVITY_ATOL
        and float(metrics["post_slip_max_absolute_delta"]) <= SENSITIVITY_ATOL
    )
    return {
        "metrics": metrics,
        "finite": finite,
        "classification": "INSENSITIVE" if insensitive else "SENSITIVE" if finite else "INSUFFICIENT_EVIDENCE",
        "thresholds": {
            "relative_output_rtol": SENSITIVITY_RTOL,
            "physical_impulse_max_abs": PHYSICAL_IMPULSE_ATOL,
            "qacc_max_abs": SENSITIVITY_ATOL,
            "post_slip_max_abs": SENSITIVITY_ATOL,
        },
    }


def _baseline_from_recent_artifact(
    condition: dict[str, Any], reference_dir: Path
) -> dict[str, Any]:
    candidates = (
        reference_dir / "conditions" / "regularization_scale_1_before" / "solver_excess.json",
        reference_dir / "conditions" / "r_scale_1_before" / "solver_excess.json",
    )
    path = next((candidate for candidate in candidates if candidate.is_file()), None)
    if path is None:
        return {"status": "INSUFFICIENT_EVIDENCE", "reason": "baseline solver_excess artifact missing"}
    try:
        reference = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return {"status": "INSUFFICIENT_EVIDENCE", "reason": str(error)}
    checks = {
        key: _allclose(condition["excess"].get(key), reference.get(key))
        for key in (
            "actual_tangent_impulse_vector", "actual_tangent_impulse_norm",
            "normal_impulse", "rigid_demand_vector", "rigid_demand_norm",
            "solver_excess_vector", "solver_excess_norm", "post_slip",
        )
    }
    return {
        "path": str(path),
        "checks": checks,
        "status": "PASS" if all(checks.values()) else "FAIL",
    }


def _all_gate_checks(value: Any) -> bool:
    """Evaluate nested artifact gates without treating a failure string as truthy."""
    if not isinstance(value, dict) or not value:
        return False
    result = []
    for item in value.values():
        if isinstance(item, dict):
            result.append(_all_gate_checks(item))
        elif isinstance(item, (bool, np.bool_)):
            result.append(bool(item))
        elif isinstance(item, str):
            result.append(item.upper() in {"PASS", "VALIDATED", "TRUE"})
        else:
            result.append(bool(item))
    return bool(result) and all(result)


def _baseline_gate_components(
    corrected_oracle: dict[str, Any],
    recent_regularization_artifact: dict[str, Any],
    sanity: dict[str, Any],
) -> dict[str, Any]:
    oracle_status = next(
        (
            corrected_oracle.get(key)
            for key in (
                "PYRAMIDAL_BASELINE_REPRODUCTION",
                "R_BASELINE_REPRODUCTION",
                "BASELINE_REPRODUCTION",
            )
            if corrected_oracle.get(key) is not None
        ),
        None,
    )
    oracle_checks = corrected_oracle.get("checks")
    recent_checks = recent_regularization_artifact.get("checks")
    components = {
        "corrected_oracle_status_pass": oracle_status == "PASS",
        "corrected_oracle_checks_all_pass": _all_gate_checks(oracle_checks),
        "recent_regularization_artifact_status_pass": (
            recent_regularization_artifact.get("status") == "PASS"
        ),
        "recent_regularization_artifact_checks_all_pass": _all_gate_checks(recent_checks),
        "sanity_all_pass": _all_gate_checks(sanity),
    }
    return {
        "oracle_status_key": (
            "PYRAMIDAL_BASELINE_REPRODUCTION"
            if corrected_oracle.get("PYRAMIDAL_BASELINE_REPRODUCTION") is not None
            else "R_BASELINE_REPRODUCTION"
            if corrected_oracle.get("R_BASELINE_REPRODUCTION") is not None
            else "BASELINE_REPRODUCTION"
            if corrected_oracle.get("BASELINE_REPRODUCTION") is not None
            else None
        ),
        "oracle_status": oracle_status,
        "components": components,
        "all_components_pass": bool(all(components.values())),
        "BASELINE_STATUS_CONSISTENCY": "PASS" if all(components.values()) else "FAIL",
    }


def _baseline_regression(
    condition: dict[str, Any],
    corrected_reference: dict[str, Any],
    regularization_reference: Path,
) -> dict[str, Any]:
    target_index = int(condition["shared_demand"]["limb_12_contact_index"])
    target = condition["capture"]["contacts"][target_index]
    demand = condition["shared_demand"]
    adapted = {
        "capture": condition["capture"],
        "shared_demand": demand,
        "budget": {"selected": {"limb/12": {
            "actual_tangential_impulse": target["tangential_impulse"],
            "actual_tangential_impulse_norm": target["tangential_impulse_norm"],
            "actual_normal_impulse": target["normal_impulse"],
            "global_normal_conditioned_sticking_impulse": demand["limb_12_tangent_impulse_2d"],
            "global_normal_conditioned_sticking_impulse_norm": float(np.linalg.norm(demand["limb_12_tangent_impulse_2d"])),
            "pre_tangential_speed": target["pre_tangential_speed"],
        }}},
        "excess": condition["excess"],
    }
    oracle_result = aref.baseline_regression(adapted, corrected_reference)
    recent_result = _baseline_from_recent_artifact(condition, regularization_reference)
    sanity = {
        "actual_tangent_norm": _allclose(condition["excess"]["actual_tangent_impulse_norm"], 3.3247363600666735),
        "normal_impulse": _allclose(condition["excess"]["normal_impulse"], 6.345240278967453),
        "rigid_demand_norm": _allclose(condition["excess"]["rigid_demand_norm"], 2.540619084288334),
        "solver_excess": _allclose(condition["excess"]["solver_excess_norm"], 0.7841172757783395),
        "solver_excess_vector_norm": _allclose(condition["excess"]["solver_excess_vector_norm"], 0.8072255552101076),
        "post_slip_speed": _allclose(np.linalg.norm(condition["excess"]["post_slip"]), 0.1713507113360867),
    }
    gate_components = _baseline_gate_components(oracle_result, recent_result, sanity)
    valid = bool(gate_components["all_components_pass"])
    return {
        "corrected_oracle": oracle_result,
        "recent_regularization_artifact": recent_result,
        "sanity": sanity,
        "baseline_gate_components": gate_components,
        "BASELINE_STATUS_CONSISTENCY": gate_components["BASELINE_STATUS_CONSISTENCY"],
        "OPTIMIZATION_BASELINE_REPRODUCTION": "PASS" if valid else "FAIL",
    }


def _pre_constraint_invariant(
    reference: dict[str, Any], candidate: dict[str, Any]
) -> dict[str, Any]:
    fields = (
        "efc_J", "efc_vel", "efc_aref", "efc_R", "efc_D", "iefc_R", "iefc_D",
    )
    return {name: _allclose(reference["pre_constraint"].get(name), candidate["pre_constraint"].get(name)) for name in fields}


def _solver_optimization_invariants(
    conditions: dict[str, dict[str, Any]],
    production_options: dict[str, Any],
) -> dict[str, Any]:
    baseline = conditions["production_warmstart_production_tolerance"]
    checks = {}
    for name, _, _, _ in CONDITIONS[1:]:
        candidate = conditions[name]
        state_checks = _state_equal_except_warmstart(
            baseline["state_validation"]["pre_state_snapshot"],
            candidate["state_validation"]["pre_state_snapshot"],
        )
        option_diff = _option_difference(
            production_options,
            candidate["model_options"],
            {"opt.tolerance", "opt.iterations"},
        )
        checks[name] = {
            "state_except_qacc_warmstart": state_checks,
            "contact_geometry": _contact_geometry_equal(
                baseline["pre_contact_geometry"], candidate["pre_contact_geometry"]
            ),
            "M_J_W": all(_allclose(
                baseline["capture"].get(field), candidate["capture"].get(field)
            ) for field in ("mass_matrix", "J_phys", "W_phys")),
            "pre_constraint": _pre_constraint_invariant(baseline, candidate),
            "model_options_only_allowed_changes": option_diff,
            "solver_type_unchanged": option_diff["details"].get("opt.solver", {}).get("equal", False),
            "cone_unchanged": option_diff["details"].get("opt.cone", {}).get("equal", False),
            "timestep_integrator_unchanged": (
                option_diff["details"].get("opt.timestep", {}).get("equal", False)
                and option_diff["details"].get("opt.integrator", {}).get("equal", False)
            ),
        }
        checks[name]["valid"] = bool(
            checks[name]["state_except_qacc_warmstart"]["valid"]
            and checks[name]["contact_geometry"]
            and checks[name]["M_J_W"]
            and all(checks[name]["pre_constraint"].values())
            and checks[name]["model_options_only_allowed_changes"]["only_allowed"]
        )
    valid = all(item["valid"] for item in checks.values())
    return {
        "checks_against_production": checks,
        "allowed_changes": ["qacc_warmstart", "opt.tolerance", "opt.iterations", "solver outputs", "post-step state"],
        "SOLVER_OPTIMIZATION_DIAGNOSTIC_ISOLATION": "VALIDATED" if valid else "FAILED",
    }


def _warmstart_activation(conditions: dict[str, dict[str, Any]]) -> dict[str, Any]:
    production = conditions["production_warmstart_production_tolerance"]
    checks = {}
    for name in ("zero_warmstart_production_tolerance", "zero_warmstart_tight_tolerance"):
        condition = conditions[name]
        warmstart = np.asarray(condition["state_validation"]["pre_state_snapshot"]["qacc_warmstart"], dtype=np.float64)
        checks[name] = {
            "warmstart_exact_zero": bool(np.array_equal(warmstart, np.zeros_like(warmstart))),
            "warmstart_all_finite": bool(_finite(warmstart)),
            "other_state_equal": _state_equal_except_warmstart(
                production["state_validation"]["pre_state_snapshot"],
                condition["state_validation"]["pre_state_snapshot"],
            ),
        }
        checks[name]["valid"] = bool(
            checks[name]["warmstart_exact_zero"]
            and checks[name]["warmstart_all_finite"]
            and checks[name]["other_state_equal"]["valid"]
        )
    activation = "VALIDATED" if all(item["valid"] for item in checks.values()) else "FAILED"
    return {
        "production_warmstart": {
            "norm": production["solver_numerics"]["warmstart_vector_norm"],
            "maximum_absolute_value": production["solver_numerics"]["warmstart_max_abs"],
            "all_finite": production["solver_numerics"]["warmstart_all_finite"],
            "full_vector": production["state_validation"]["pre_state_snapshot"]["qacc_warmstart"],
        },
        "checks": checks,
        "ZERO_WARMSTART_ACTIVATION": activation,
    }


def _tight_tolerance_activation(
    conditions: dict[str, dict[str, Any]], production_options: dict[str, Any]
) -> dict[str, Any]:
    checks = {}
    production_tolerance = float(production_options["opt.tolerance"])
    production_iterations = int(production_options["opt.iterations"])
    for name in ("production_warmstart_tight_tolerance", "zero_warmstart_tight_tolerance"):
        condition = conditions[name]
        options = condition["model_options"]
        difference = _option_difference(
            production_options, options, {"opt.tolerance", "opt.iterations"}
        )
        checks[name] = {
            "tolerance": options["opt.tolerance"],
            "iterations": options["opt.iterations"],
            "tight_tolerance_less_than_production": bool(options["opt.tolerance"] < production_tolerance),
            "tight_iterations_at_least_production": bool(options["opt.iterations"] >= production_iterations),
            "only_tolerance_iterations_changed": difference["only_allowed"],
            "solver_type_unchanged": bool(options.get("opt.solver") == production_options.get("opt.solver")),
            "difference": difference,
        }
        checks[name]["valid"] = bool(
            checks[name]["tight_tolerance_less_than_production"]
            and checks[name]["tight_iterations_at_least_production"]
            and checks[name]["only_tolerance_iterations_changed"]
            and checks[name]["solver_type_unchanged"]
        )
    return {
        "production": {
            "tolerance": production_tolerance,
            "iterations": production_iterations,
            "solver": production_options.get("opt.solver"),
        },
        "checks": checks,
        "TIGHT_TOLERANCE_ACTIVATION": "VALIDATED" if all(item["valid"] for item in checks.values()) else "FAILED",
    }


def _convergence_assessment(
    conditions: dict[str, dict[str, Any]],
    baseline: dict[str, Any],
    warmstart_sensitivity: dict[str, Any],
    tolerance_sensitivity: dict[str, Any],
) -> dict[str, Any]:
    assessments = {}
    for name, _, _, _ in CONDITIONS:
        trace = conditions[name]["solver_iteration_trace"]
        numerics = conditions[name]["solver_numerics"]
        islands = trace.get("islands", [])
        active_count = int(trace.get("active_solver_island_count", 0) or 0)
        source = trace.get("active_solver_island_count_source")
        active_count_known = bool(active_count > 0 and source not in {None, "UNAVAILABLE"})
        active_limit_checks = [
            bool(item.get("niter_below_limit"))
            for item in islands[:active_count]
        ]
        assessments[name] = {
            "active_solver_island_count": active_count,
            "active_solver_island_count_source": source,
            "active_solver_island_count_known": active_count_known,
            "statistics_finite": bool(
                numerics.get("finite") and trace.get("all_statistics_finite")
            ),
            "statistics_available": bool(trace.get("statistics_available")),
            "statistics_complete": bool(trace.get("statistics_complete")),
            "solver_niter": numerics.get("solver_niter", []),
            "solver_nnz": numerics.get("solver_nnz", []),
            "active_solver_niter": numerics.get("active_solver_niter", []),
            "active_solver_nnz": numerics.get("active_solver_nnz", []),
            "iterations_limit": trace.get("iterations_limit"),
            "niter_below_limit": bool(active_limit_checks and all(active_limit_checks)),
            "islands": islands,
            "last_iteration": trace.get("last_iteration", {}),
            "last_iterations": trace.get("last_iterations", []),
        }
    all_valid_stats = all(
        item["active_solver_island_count_known"]
        and item["statistics_finite"]
        and item["statistics_available"]
        for item in assessments.values()
    )
    any_limit_hit = any(
        not item["niter_below_limit"] for item in assessments.values()
    )
    baseline_pass = baseline["OPTIMIZATION_BASELINE_REPRODUCTION"] == "PASS"
    converged = bool(
        all_valid_stats and not any_limit_hit and baseline_pass
        and warmstart_sensitivity["SOLVER_WARMSTART_SENSITIVITY"] == "INSENSITIVE"
        and tolerance_sensitivity["SOLVER_TOLERANCE_SENSITIVITY"] == "INSENSITIVE"
    )
    if not all_valid_stats:
        status = "INSUFFICIENT_EVIDENCE"
    elif any_limit_hit:
        status = "NOT_CONVERGED"
    elif converged:
        status = "VALIDATED"
    else:
        status = "QUESTIONABLE"
    return {
        "conditions": assessments,
        "baseline_reproduction": baseline["OPTIMIZATION_BASELINE_REPRODUCTION"],
        "all_statistics_finite": all_valid_stats,
        "active_solver_island_count_known": all(
            item["active_solver_island_count_known"] for item in assessments.values()
        ),
        "active_solver_island_counts": {
            name: item["active_solver_island_count"]
            for name, item in assessments.items()
        },
        "any_iteration_limit_reached": any_limit_hit,
        "PRODUCTION_NEWTON_CONVERGENCE": status,
    }


def _active_niter_values(condition: dict[str, Any]) -> list[int]:
    numerics = condition.get("solver_numerics", {})
    value = numerics.get("active_solver_niter")
    if isinstance(value, (list, tuple, np.ndarray)):
        try:
            return [int(item) for item in np.asarray(value).reshape(-1)]
        except (TypeError, ValueError):
            return []
    value = numerics.get("solver_niter")
    if isinstance(value, (list, tuple, np.ndarray)):
        try:
            return [int(item) for item in np.asarray(value).reshape(-1)]
        except (TypeError, ValueError):
            return []
    try:
        return [int(value)] if value is not None else []
    except (TypeError, ValueError):
        return []


def _warmstart_computational_effect(
    conditions: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    pairs = {
        "production_tolerance_A_vs_B": (
            "production_warmstart_production_tolerance",
            "zero_warmstart_production_tolerance",
        ),
        "tight_tolerance_C_vs_D": (
            "production_warmstart_tight_tolerance",
            "zero_warmstart_tight_tolerance",
        ),
    }
    comparisons = {}
    classifications = []
    for name, (left_name, right_name) in pairs.items():
        left_values = _active_niter_values(conditions[left_name])
        right_values = _active_niter_values(conditions[right_name])
        if not left_values or not right_values or len(left_values) != len(right_values):
            comparisons[name] = {
                "left_active_solver_niter": left_values,
                "right_active_solver_niter": right_values,
                "classification": "INSUFFICIENT_EVIDENCE",
            }
            classifications.append("INSUFFICIENT_EVIDENCE")
            continue
        left_total, right_total = sum(left_values), sum(right_values)
        if right_total > left_total:
            classification = "REDUCES_ITERATION_COUNT"
        elif right_total < left_total:
            classification = "INCREASES_ITERATION_COUNT"
        else:
            classification = "NO_ITERATION_EFFECT"
        comparisons[name] = {
            "left_active_solver_niter": left_values,
            "right_active_solver_niter": right_values,
            "left_total_iterations": left_total,
            "right_total_iterations": right_total,
            "delta_zero_minus_production": right_total - left_total,
            "classification": classification,
        }
        classifications.append(classification)
    if not classifications or any(item == "INSUFFICIENT_EVIDENCE" for item in classifications):
        classification = "INSUFFICIENT_EVIDENCE"
    elif any(item == "REDUCES_ITERATION_COUNT" for item in classifications):
        classification = "REDUCES_ITERATION_COUNT"
    elif any(item == "INCREASES_ITERATION_COUNT" for item in classifications):
        classification = "INCREASES_ITERATION_COUNT"
    else:
        classification = "NO_ITERATION_EFFECT"
    return {
        "comparisons": comparisons,
        "WARMSTART_COMPUTATIONAL_EFFECT": classification,
    }


def _classify_final(
    invariant: dict[str, Any],
    warmstart: dict[str, Any],
    tolerance: dict[str, Any],
    convergence: dict[str, Any],
    baseline: dict[str, Any],
) -> dict[str, Any]:
    if baseline["OPTIMIZATION_BASELINE_REPRODUCTION"] != "PASS":
        status = "INSUFFICIENT_EVIDENCE"
        action = "DIAGNOSTIC_IMPLEMENTATION_FIX_REQUIRED"
    elif invariant["SOLVER_OPTIMIZATION_DIAGNOSTIC_ISOLATION"] != "VALIDATED":
        status = "NONCANONICAL"
        action = "DIAGNOSTIC_IMPLEMENTATION_FIX_REQUIRED"
    elif convergence["PRODUCTION_NEWTON_CONVERGENCE"] == "NOT_CONVERGED":
        status = "NONCONVERGED"
        action = "SOLVER_CONVERGENCE_FIX_REQUIRED"
    else:
        warm_sensitive = warmstart["SOLVER_WARMSTART_SENSITIVITY"] == "SENSITIVE"
        tolerance_sensitive = tolerance["SOLVER_TOLERANCE_SENSITIVITY"] == "SENSITIVE"
        # A VALIDATED convergence result is already the tool's explicit gate
        # that the recorded Newton statistics are finite and not truncated by
        # the iteration limit.  Keep the detailed flags as a defensive check
        # for older/reference payloads that may not contain both fields.
        stats_support_sensitivity = bool(
            baseline["OPTIMIZATION_BASELINE_REPRODUCTION"] == "PASS"
            and (
                convergence.get("PRODUCTION_NEWTON_CONVERGENCE") == "VALIDATED"
                or (
                    convergence.get("all_statistics_finite") is True
                    and convergence.get("any_iteration_limit_reached") is False
                )
            )
        )
        if warm_sensitive and tolerance_sensitive and stats_support_sensitivity:
            status = "WARMSTART_AND_TOLERANCE_SENSITIVE"
            action = "SOLVER_WARMSTART_COUNTERFACTUAL"
        elif warm_sensitive and stats_support_sensitivity:
            status = "WARMSTART_SENSITIVE"
            action = "SOLVER_WARMSTART_COUNTERFACTUAL"
        elif tolerance_sensitive and stats_support_sensitivity:
            status = "TOLERANCE_SENSITIVE"
            action = "SOLVER_TOLERANCE_COUNTERFACTUAL"
        elif convergence["PRODUCTION_NEWTON_CONVERGENCE"] == "VALIDATED":
            status = "ROBUST_CONVERGED_SOLUTION"
            action = "NORMAL_REFERENCE_ACCELERATION_COUNTERFACTUAL"
        else:
            status = "INSUFFICIENT_EVIDENCE"
            action = "INSUFFICIENT_EVIDENCE"
    return {
        "MUJOCO_SOLVER_EXCESS_OPTIMIZATION_STATUS": status,
        "NEXT_ACTION": action,
        "interpretation": (
            "The observed solver excess is not explained by Newton initialization "
            "or insufficient production convergence."
            if status == "ROBUST_CONVERGED_SOLUTION" else None
        ),
    }


def _warmstart_sensitivity(
    conditions: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    comparisons = {
        "production_tolerance_A_vs_B": _pair_sensitivity(
            conditions["production_warmstart_production_tolerance"],
            conditions["zero_warmstart_production_tolerance"],
        ),
        "tight_tolerance_C_vs_D": _pair_sensitivity(
            conditions["production_warmstart_tight_tolerance"],
            conditions["zero_warmstart_tight_tolerance"],
        ),
    }
    classification = (
        "INSUFFICIENT_EVIDENCE"
        if any(item["classification"] == "INSUFFICIENT_EVIDENCE" for item in comparisons.values())
        else "SENSITIVE"
        if any(item["classification"] == "SENSITIVE" for item in comparisons.values())
        else "INSENSITIVE"
    )
    return {
        "comparisons": comparisons,
        "SOLVER_WARMSTART_SENSITIVITY": classification,
    }


def _tolerance_sensitivity(
    conditions: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    comparisons = {
        "warmstart_production_A_vs_C": _pair_sensitivity(
            conditions["production_warmstart_production_tolerance"],
            conditions["production_warmstart_tight_tolerance"],
        ),
        "warmstart_zero_B_vs_D": _pair_sensitivity(
            conditions["zero_warmstart_production_tolerance"],
            conditions["zero_warmstart_tight_tolerance"],
        ),
    }
    classification = (
        "INSUFFICIENT_EVIDENCE"
        if any(item["classification"] == "INSUFFICIENT_EVIDENCE" for item in comparisons.values())
        else "SENSITIVE"
        if any(item["classification"] == "SENSITIVE" for item in comparisons.values())
        else "INSENSITIVE"
    )
    return {
        "comparisons": comparisons,
        "SOLVER_TOLERANCE_SENSITIVITY": classification,
    }


def _condition_payload(condition: dict[str, Any]) -> dict[str, Any]:
    capture = condition["capture"]
    return {
        "model_clone_method": condition["model_clone_method"],
        "clone_smoke": condition["clone_smoke"],
        "model_option_restore": condition["model_option_restore"],
        "state_validation": condition["state_validation"],
        "solver_iteration_trace": condition["solver_iteration_trace"],
        "solver_numerics": condition["solver_numerics"],
        "mass_jacobian_delassus": {
            "mass_matrix": capture["mass_matrix"],
            "J_phys": capture["J_phys"],
            "W_phys": capture["W_phys"],
            "mass_matrix_stats": capture.get("mass_matrix_stats"),
            "delassus_stats": capture.get("delassus_stats"),
        },
        "physical_contact_impulses": {
            "api": "mujoco.mj_contactForce",
            "parameterization_independent_readback": True,
            "contacts": [
                {
                    "contact_index": item["contact_index"],
                    "pair": [item["geom1_name"], item["geom2_name"]],
                    "normal_impulse": item["normal_impulse"],
                    "tangent_impulse": item["tangential_impulse"],
                    "tangent_impulse_norm": item["tangential_impulse_norm"],
                }
                for item in capture["contacts"]
            ],
        },
        "shared_physical_global_demand": condition["shared_demand"],
        "solver_excess": condition["excess"],
        "one_step_result": condition["one_step_result"],
        "pre_constraint": condition["pre_constraint"],
        "pre_contact_geometry": condition["pre_contact_geometry"],
        "model_options": condition["model_options"],
    }


def _write_condition(output: Path, condition: dict[str, Any]) -> None:
    target = output / "conditions" / condition["condition_name"]
    target.mkdir(parents=True, exist_ok=True)
    payload = _condition_payload(condition)
    for name, value in payload.items():
        write_json(target / f"{name}.json", value)


def _safe_sha256(path: Path) -> str | None:
    try:
        return oracle.sha256(path) if path.is_file() else None
    except (OSError, ValueError):
        return None


def _source_hashes(paths: dict[str, Path]) -> dict[str, str | None]:
    source_paths = [
        paths["morphology_xml"],
        paths["checkpoint"],
        REPO_ROOT / "tools/analyze_mujoco_global55_contact_demand.py",
        REPO_ROOT / "tools/audit_mujoco_global55_friction_aref_counterfactual.py",
        REPO_ROOT / "tools/audit_mujoco_global55_contact_regularization_counterfactual.py",
        Path(__file__).resolve(),
        paths["corrected_oracle"] / "validation.json",
    ]
    return {str(path): _safe_sha256(path) for path in source_paths}


def _write_git_identity(output: Path) -> None:
    def git(*arguments: str) -> str:
        return subprocess.run(
            ["git", *arguments], cwd=REPO_ROOT, check=True,
            capture_output=True, text=True,
        ).stdout
    (output / "git_head.txt").write_text(
        f"TOPLEVEL={git('rev-parse', '--show-toplevel').strip()}\n"
        f"HEAD={git('rev-parse', 'HEAD').strip()}\n"
        f"BRANCH={git('branch', '--show-current').strip()}\n",
        encoding="utf-8",
    )
    (output / "git_status_short.txt").write_text(
        git("status", "--short"), encoding="utf-8"
    )


def _load_reference(path: Path, filename: str) -> dict[str, Any]:
    target = path / filename
    if not target.is_file():
        raise FileNotFoundError(f"reference file is missing: {target}")
    value = json.loads(target.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"reference file is not a JSON object: {target}")
    return value


def _custom_one_step_regression(
    mujoco: Any,
    base_model: Any,
    snapshot: Any,
    mapping: dict[str, Any],
    production_options: dict[str, Any],
    isolation_method: str,
    staged_mjb_path: Path | None,
    full_mjb_path: Path | None,
) -> dict[str, Any]:
    if isolation_method == "TRANSACTIONAL_SOLVER_OPTION_RESTORE":
        staged_model = full_model = base_model
        staged_copy_api = full_copy_api = "TRANSACTIONAL_SHARED_MODEL"
    else:
        staged_model, staged_copy_api, _ = _copy_model(
            mujoco, base_model, isolation_method, staged_mjb_path
        )
        full_model, full_copy_api, _ = _copy_model(
            mujoco, base_model, isolation_method, full_mjb_path
        )
    staged_data = mujoco.MjData(staged_model)
    mujoco.mj_copyData(staged_data, staged_model, snapshot)
    aref.stage_to_constraint(mujoco, staged_model, staged_data)
    mujoco.mj_fwdConstraint(staged_model, staged_data)
    staged_solver = mujoco.MjData(staged_model)
    mujoco.mj_copyData(staged_solver, staged_model, staged_data)
    mujoco.mj_Euler(staged_model, staged_data)

    full_data = mujoco.MjData(full_model)
    mujoco.mj_copyData(full_data, full_model, snapshot)
    mujoco.mj_step(full_model, full_data)
    checks = {
        "post_qpos": _allclose(staged_data.qpos, full_data.qpos),
        "post_qvel": _allclose(staged_data.qvel, full_data.qvel),
        "post_time": bool(np.isclose(float(staged_data.time), float(full_data.time))),
    }
    staged_capture = aref.capture_after_integration(
        mujoco, staged_model, staged_data, snapshot, mapping, staged_solver
    )
    full_solver = mujoco.MjData(full_model)
    mujoco.mj_copyData(full_solver, full_model, full_data)
    full_capture = aref.capture_after_integration(
        mujoco, full_model, full_data, snapshot, mapping, full_solver
    )
    staged_target = next(item for item in staged_capture["contacts"] if item["robot_body_name"] == "limb/12")
    full_target = next(item for item in full_capture["contacts"] if item["robot_body_name"] == "limb/12")
    checks["post_slip"] = _allclose(
        staged_target["post_tangential_velocity"],
        full_target["post_tangential_velocity"],
    )
    return {
        "checks": checks,
        "staged_calls": [
            "mj_fwdPosition", "mj_fwdVelocity", "mj_fwdActuation",
            "mj_fwdAcceleration", "mj_fwdConstraint", "mj_Euler",
        ],
        "full_validation_calls": [
            "mj_step on independent model/data clone"
            if isolation_method != "TRANSACTIONAL_SOLVER_OPTION_RESTORE"
            else "mj_step on independent MjData clone using transactional shared model"
        ],
        "model_copy_api": [staged_copy_api, full_copy_api],
        "model_clone_method": isolation_method,
        "CUSTOM_PIPELINE_ONE_STEP_REPRODUCTION": "PASS" if all(checks.values()) else "FAIL",
    }


def _write_failure_placeholders(output: Path, error: Exception) -> None:
    placeholders = {
        "model_clone_api_discovery.json": {
            "EXACT_MODEL_CLONE_API": "UNAVAILABLE",
            "MODEL_CLONE_METHOD": None,
            "status": "INSUFFICIENT_EVIDENCE",
        },
        "runtime_vs_source_geom_inventory.json": {
            "RUNTIME_MODEL_STRUCTURE": "OTHER_MISMATCH",
            "status": "INSUFFICIENT_EVIDENCE",
        },
        "model_clone_fidelity.json": {
            "EXACT_MODEL_CLONE_FIDELITY": "FAIL",
        },
        "clone_data_state_fidelity.json": {
            "CLONE_DATA_STATE_FIDELITY": "FAIL",
        },
        "solver_stat_api_discovery.json": {
            "status": "INSUFFICIENT_EVIDENCE",
            "active_solver_island_count": None,
            "active_solver_island_count_source": "UNAVAILABLE",
        },
        "solver_optimization_invariant_validation.json": {
            "SOLVER_OPTIMIZATION_DIAGNOSTIC_ISOLATION": "INSUFFICIENT_EVIDENCE",
        },
        "warmstart_activation.json": {"ZERO_WARMSTART_ACTIVATION": "FAILED"},
        "tight_tolerance_activation.json": {"TIGHT_TOLERANCE_ACTIVATION": "FAILED"},
        "baseline_regression.json": {
            "BASELINE_STATUS_CONSISTENCY": "INSUFFICIENT_EVIDENCE",
            "OPTIMIZATION_BASELINE_REPRODUCTION": "FAIL",
        },
        "baseline_gate_components.json": {
            "BASELINE_STATUS_CONSISTENCY": "INSUFFICIENT_EVIDENCE",
        },
        "warmstart_sensitivity.json": {"SOLVER_WARMSTART_SENSITIVITY": "INSUFFICIENT_EVIDENCE"},
        "warmstart_computational_effect.json": {
            "WARMSTART_COMPUTATIONAL_EFFECT": "INSUFFICIENT_EVIDENCE"
        },
        "tolerance_sensitivity.json": {"SOLVER_TOLERANCE_SENSITIVITY": "INSUFFICIENT_EVIDENCE"},
        "solver_convergence_assessment.json": {"PRODUCTION_NEWTON_CONVERGENCE": "INSUFFICIENT_EVIDENCE"},
        "custom_pipeline_one_step_regression.json": {"CUSTOM_PIPELINE_ONE_STEP_REPRODUCTION": "INSUFFICIENT_EVIDENCE"},
    }
    for filename, payload in placeholders.items():
        if not (output / filename).exists():
            write_json(output / filename, payload)
    for name, _, _, _ in CONDITIONS:
        target = output / "conditions" / name
        target.mkdir(parents=True, exist_ok=True)
        for filename in (
            "state_validation.json", "solver_iteration_trace.json",
            "solver_numerics.json", "physical_contact_impulses.json",
            "mass_jacobian_delassus.json", "shared_physical_global_demand.json",
            "solver_excess.json",
            "one_step_result.json",
        ):
            path = target / filename
            if not path.exists():
                write_json(path, {"status": "INSUFFICIENT_EVIDENCE"})
    write_json(output / "failure_context.json", {
        "error_type": type(error).__name__,
        "error": str(error),
        "partial_conditions": [name for name, _, _, _ in CONDITIONS],
        "traceback_file": "traceback.txt",
    })


def _failure_payload(error: Exception) -> dict[str, Any]:
    return {
        "RUNTIME_MODEL_STRUCTURE": "OTHER_MISMATCH",
        "EXACT_MODEL_CLONE_API": "UNAVAILABLE",
        "MODEL_CLONE_METHOD": None,
        "EXACT_MODEL_CLONE_FIDELITY": "FAIL",
        "CLONE_DATA_STATE_FIDELITY": "FAIL",
        "OPTIMIZATION_BASELINE_REPRODUCTION": "FAIL",
        "BASELINE_STATUS_CONSISTENCY": "INSUFFICIENT_EVIDENCE",
        "ZERO_WARMSTART_ACTIVATION": "FAILED",
        "TIGHT_TOLERANCE_ACTIVATION": "FAILED",
        "SOLVER_OPTIMIZATION_DIAGNOSTIC_ISOLATION": "INSUFFICIENT_EVIDENCE",
        "SOLVER_WARMSTART_SENSITIVITY": "INSUFFICIENT_EVIDENCE",
        "WARMSTART_COMPUTATIONAL_EFFECT": "INSUFFICIENT_EVIDENCE",
        "SOLVER_TOLERANCE_SENSITIVITY": "INSUFFICIENT_EVIDENCE",
        "PRODUCTION_NEWTON_CONVERGENCE": "INSUFFICIENT_EVIDENCE",
        "SOLVER_STAT_API_DISCOVERY": "INSUFFICIENT_EVIDENCE",
        "SOLVER_ISLAND_COUNT_SOURCE": "UNAVAILABLE",
        "ACTIVE_SOLVER_ISLAND_COUNT": None,
        "MUJOCO_SOLVER_EXCESS_OPTIMIZATION_STATUS": "INSUFFICIENT_EVIDENCE",
        "NEXT_ACTION": "DIAGNOSTIC_IMPLEMENTATION_FIX_REQUIRED",
        "UNCONDITIONAL_ZIP_PACKAGING": "ENABLED",
        "LOCAL_IMPLEMENTATION": "INCOMPLETE",
        "COUNTERFACTUAL_VALID": False,
        "error_type": type(error).__name__,
        "error": str(error),
    }


def execute(args: argparse.Namespace, paths: dict[str, Path]) -> dict[str, Any]:
    output = paths["output_dir"]
    output.mkdir(parents=True, exist_ok=True)
    _write_git_identity(output)
    source_before = _source_hashes(paths)
    corrected_reference = aref.cone_helper.load_reference(paths["corrected_oracle"])
    mujoco = __import__("mujoco")
    discovery = _capability_discovery(mujoco)
    write_json(output / "model_clone_api_discovery.json", discovery)
    recorder, mapping = aref.cone_helper.replay_once(args, paths)
    model = recorder.raw_model
    snapshot = recorder.global55_snapshot
    if model is None or snapshot is None:
        raise RuntimeError("global55 replay did not provide native model/data/snapshot")
    if len(recorder.records) != EXPECTED_SUBSTEPS:
        raise RuntimeError(
            f"formal replay substep count mismatch: {len(recorder.records)} vs {EXPECTED_SUBSTEPS}"
        )
    if not recorder.snapshot_copy_evidence or not recorder.snapshot_copy_evidence.get(
        "live_unchanged_by_copy", False
    ):
        raise RuntimeError("formal replay snapshot copy evidence is incomplete")

    production_cone = int(mujoco.mjtCone.mjCONE_PYRAMIDAL)
    if int(model.opt.cone) != production_cone:
        raise RuntimeError("formal optimization diagnostic requires production pyramidal cone")
    production_solver = int(mujoco.mjtSolver.mjSOL_NEWTON)
    if int(model.opt.solver) != production_solver:
        raise RuntimeError("formal optimization diagnostic requires mjSOL_NEWTON")
    production_options = _model_option_snapshot(model)
    source_inventory = _runtime_source_geom_inventory(
        mujoco, model, paths["morphology_xml"]
    )
    write_json(output / "runtime_vs_source_geom_inventory.json", source_inventory)
    run_stamp = output.name.replace("mujoco_global55_solver_optimization_", "")
    mjb_root = REPO_ROOT / "tmp" / f"solver_optimization_live_model_{run_stamp}"
    isolation_method, exact_clone_api, probe_clone = _select_model_isolation_method(
        mujoco, model, discovery, mjb_root
    )
    discovery["EXACT_MODEL_CLONE_API"] = exact_clone_api
    discovery["MODEL_CLONE_METHOD"] = isolation_method
    write_json(output / "model_clone_api_discovery.json", discovery)
    if probe_clone is None:
        clone_fidelity = {
            "method": isolation_method,
            "EXACT_MODEL_CLONE_FIDELITY": "NOT_APPLICABLE_TRANSACTIONAL",
        }
        clone_data_fidelity = _clone_data_state_fidelity(
            mujoco, model, snapshot, isolation_method
        )
    else:
        clone_fidelity = _model_clone_fidelity(
            mujoco, model, probe_clone, isolation_method
        )
        clone_data_fidelity = _clone_data_state_fidelity(
            mujoco, probe_clone, snapshot, isolation_method
        )
    write_json(output / "model_clone_fidelity.json", clone_fidelity)
    write_json(output / "clone_data_state_fidelity.json", clone_data_fidelity)
    exact_fidelity_valid = clone_fidelity["EXACT_MODEL_CLONE_FIDELITY"] in {
        "PASS", "NOT_APPLICABLE_TRANSACTIONAL"
    }
    if not exact_fidelity_valid:
        raise RuntimeError("exact compiled-model clone fidelity gate failed")
    if clone_data_fidelity["CLONE_DATA_STATE_FIDELITY"] != "PASS":
        raise RuntimeError("clone-data state fidelity gate failed")
    solver_stat_api = _solver_stat_api_discovery(
        mujoco, model, snapshot, probe_model=probe_clone
    )
    write_json(output / "solver_stat_api_discovery.json", solver_stat_api)
    write_json(output / "global55_pre_state_snapshot.json", aref.cone_helper.state_input_snapshot(snapshot))
    write_json(output / "state_copy_manifest.json", {
        **aref.cone_helper.state_copy_manifest(snapshot),
        "formal_snapshot_copy_evidence": recorder.snapshot_copy_evidence,
    })

    conditions: dict[str, dict[str, Any]] = {}
    for name, label, zero_warmstart, tight in CONDITIONS:
        mjb_path = (
            REPO_ROOT / "tmp"
            / f"solver_optimization_live_model_{run_stamp}_{name}.mjb"
            if isolation_method == "MJB_ROUNDTRIP" else None
        )
        condition = _run_condition(
            mujoco, model, snapshot, mapping, name, label,
            zero_warmstart, tight, production_options, isolation_method, mjb_path,
            solver_api=solver_stat_api,
        )
        conditions[name] = condition
        _write_condition(output, condition)

    invariant = _solver_optimization_invariants(conditions, production_options)
    warmstart_activation = _warmstart_activation(conditions)
    tight_activation = _tight_tolerance_activation(conditions, production_options)
    baseline = _baseline_regression(
        conditions["production_warmstart_production_tolerance"],
        corrected_reference,
        paths["regularization_reference"],
    )
    warmstart_sensitivity = _warmstart_sensitivity(conditions)
    warmstart_computational_effect = _warmstart_computational_effect(conditions)
    tolerance_sensitivity = _tolerance_sensitivity(conditions)
    convergence = _convergence_assessment(
        conditions, baseline, warmstart_sensitivity, tolerance_sensitivity
    )
    custom_step = _custom_one_step_regression(
        mujoco, model, snapshot, mapping, production_options,
        isolation_method,
        (
            REPO_ROOT / "tmp"
            / f"solver_optimization_live_model_{run_stamp}_custom_staged.mjb"
            if isolation_method == "MJB_ROUNDTRIP" else None
        ),
        (
            REPO_ROOT / "tmp"
            / f"solver_optimization_live_model_{run_stamp}_custom_full.mjb"
            if isolation_method == "MJB_ROUNDTRIP" else None
        ),
    )
    write_json(output / "solver_optimization_invariant_validation.json", invariant)
    write_json(output / "warmstart_activation.json", warmstart_activation)
    write_json(output / "tight_tolerance_activation.json", tight_activation)
    write_json(output / "baseline_regression.json", baseline)
    write_json(
        output / "baseline_gate_components.json",
        baseline["baseline_gate_components"],
    )
    write_json(output / "warmstart_sensitivity.json", warmstart_sensitivity)
    write_json(
        output / "warmstart_computational_effect.json",
        warmstart_computational_effect,
    )
    write_json(output / "tolerance_sensitivity.json", tolerance_sensitivity)
    write_json(output / "solver_convergence_assessment.json", convergence)
    write_json(output / "custom_pipeline_one_step_regression.json", custom_step)

    source_after = _source_hashes(paths)
    source_unchanged = bool(source_before == source_after)
    model_after = _model_option_snapshot(model)
    model_restore = _option_difference(production_options, model_after)
    final = _classify_final(
        invariant, warmstart_sensitivity, tolerance_sensitivity,
        convergence, baseline,
    )
    gates = bool(
        baseline["OPTIMIZATION_BASELINE_REPRODUCTION"] == "PASS"
        and baseline["BASELINE_STATUS_CONSISTENCY"] == "PASS"
        and warmstart_activation["ZERO_WARMSTART_ACTIVATION"] == "VALIDATED"
        and tight_activation["TIGHT_TOLERANCE_ACTIVATION"] == "VALIDATED"
        and invariant["SOLVER_OPTIMIZATION_DIAGNOSTIC_ISOLATION"] == "VALIDATED"
        and convergence["PRODUCTION_NEWTON_CONVERGENCE"] == "VALIDATED"
        and solver_stat_api["status"] == "VALID"
        and custom_step["CUSTOM_PIPELINE_ONE_STEP_REPRODUCTION"] == "PASS"
        and source_unchanged
        and model_restore["only_allowed"]
        and exact_fidelity_valid
        and clone_data_fidelity["CLONE_DATA_STATE_FIDELITY"] == "PASS"
        and len(recorder.records) == EXPECTED_SUBSTEPS
    )
    if not gates and final["MUJOCO_SOLVER_EXCESS_OPTIMIZATION_STATUS"] == "ROBUST_CONVERGED_SOLUTION":
        final = {
            "MUJOCO_SOLVER_EXCESS_OPTIMIZATION_STATUS": "INSUFFICIENT_EVIDENCE",
            "NEXT_ACTION": "DIAGNOSTIC_IMPLEMENTATION_FIX_REQUIRED",
            "interpretation": None,
        }
    validation = {
        "RUNTIME_MODEL_STRUCTURE": source_inventory["RUNTIME_MODEL_STRUCTURE"],
        "EXACT_MODEL_CLONE_API": discovery["EXACT_MODEL_CLONE_API"],
        "MODEL_CLONE_METHOD": isolation_method,
        "EXACT_MODEL_CLONE_FIDELITY": clone_fidelity["EXACT_MODEL_CLONE_FIDELITY"],
        "CLONE_DATA_STATE_FIDELITY": clone_data_fidelity["CLONE_DATA_STATE_FIDELITY"],
        "OPTIMIZATION_BASELINE_REPRODUCTION": baseline["OPTIMIZATION_BASELINE_REPRODUCTION"],
        "BASELINE_STATUS_CONSISTENCY": baseline["BASELINE_STATUS_CONSISTENCY"],
        "ZERO_WARMSTART_ACTIVATION": warmstart_activation["ZERO_WARMSTART_ACTIVATION"],
        "TIGHT_TOLERANCE_ACTIVATION": tight_activation["TIGHT_TOLERANCE_ACTIVATION"],
        "SOLVER_OPTIMIZATION_DIAGNOSTIC_ISOLATION": invariant["SOLVER_OPTIMIZATION_DIAGNOSTIC_ISOLATION"],
        "SOLVER_WARMSTART_SENSITIVITY": warmstart_sensitivity["SOLVER_WARMSTART_SENSITIVITY"],
        "WARMSTART_COMPUTATIONAL_EFFECT": warmstart_computational_effect["WARMSTART_COMPUTATIONAL_EFFECT"],
        "SOLVER_TOLERANCE_SENSITIVITY": tolerance_sensitivity["SOLVER_TOLERANCE_SENSITIVITY"],
        "PRODUCTION_NEWTON_CONVERGENCE": convergence["PRODUCTION_NEWTON_CONVERGENCE"],
        "SOLVER_STAT_API_DISCOVERY": solver_stat_api["status"],
        "SOLVER_ISLAND_COUNT_SOURCE": solver_stat_api.get("active_solver_island_count_source"),
        "ACTIVE_SOLVER_ISLAND_COUNT": solver_stat_api.get("active_solver_island_count"),
        "MUJOCO_SOLVER_EXCESS_OPTIMIZATION_STATUS": final["MUJOCO_SOLVER_EXCESS_OPTIMIZATION_STATUS"],
        "NEXT_ACTION": final["NEXT_ACTION"],
        "formal_replay_physics_substeps": len(recorder.records),
        "expected_formal_replay_physics_substeps": EXPECTED_SUBSTEPS,
        "formal_replay_additional_steps": 0,
        "formal_data_mutated_by_probe": False,
        "source_hashes_unchanged": source_unchanged,
        "MODEL_OPTION_RESTORE": "PASS" if model_restore["only_allowed"] else "FAIL",
        "CUSTOM_PIPELINE_ONE_STEP_REPRODUCTION": custom_step["CUSTOM_PIPELINE_ONE_STEP_REPRODUCTION"],
        "UNCONDITIONAL_ZIP_PACKAGING": "ENABLED",
        "LOCAL_IMPLEMENTATION": "READY_FOR_SERVER_VALIDATION",
        "COUNTERFACTUAL_VALID": gates,
        "semantic_scope": (
            "Only qacc_warmstart, solver tolerance, and solver iterations were changed; "
            "conditions use independent compiled-model clones when available, otherwise "
            "transactional shared-model solver-option restore with independent MjData."
        ),
    }
    summary = {key: validation[key] for key in (
        "RUNTIME_MODEL_STRUCTURE", "EXACT_MODEL_CLONE_API", "MODEL_CLONE_METHOD",
        "EXACT_MODEL_CLONE_FIDELITY", "CLONE_DATA_STATE_FIDELITY",
        "OPTIMIZATION_BASELINE_REPRODUCTION", "ZERO_WARMSTART_ACTIVATION",
        "BASELINE_STATUS_CONSISTENCY",
        "TIGHT_TOLERANCE_ACTIVATION", "SOLVER_OPTIMIZATION_DIAGNOSTIC_ISOLATION",
        "SOLVER_WARMSTART_SENSITIVITY", "WARMSTART_COMPUTATIONAL_EFFECT",
        "SOLVER_TOLERANCE_SENSITIVITY", "PRODUCTION_NEWTON_CONVERGENCE",
        "SOLVER_STAT_API_DISCOVERY", "SOLVER_ISLAND_COUNT_SOURCE",
        "ACTIVE_SOLVER_ISLAND_COUNT", "MUJOCO_SOLVER_EXCESS_OPTIMIZATION_STATUS",
        "NEXT_ACTION", "UNCONDITIONAL_ZIP_PACKAGING", "LOCAL_IMPLEMENTATION",
    )}
    summary["interpretation"] = final.get("interpretation")
    metadata = {
        "created_at": datetime.now().astimezone().isoformat(),
        "backend": "mujoco",
        "mujoco_version": getattr(mujoco, "__version__", None),
        "morphology": MORPHOLOGY,
        "morphology_xml_sha256": source_before.get(str(paths["morphology_xml"])),
        "checkpoint_sha256": source_before.get(str(paths["checkpoint"])),
        "corrected_reference_oracle": str(paths["corrected_oracle"]),
        "regularization_reference": str(paths["regularization_reference"]),
        "formal_replay_physics_substeps": len(recorder.records),
        "formal_replay_additional_steps": 0,
        "formal_data_mutated_by_probe": False,
        "solver": "mjSOL_NEWTON",
        "cone": "mjCONE_PYRAMIDAL",
        "conditions": [label for _, label, _, _ in CONDITIONS],
        "EXACT_MODEL_CLONE_API": discovery["EXACT_MODEL_CLONE_API"],
        "MODEL_CLONE_METHOD": isolation_method,
        "EXACT_MODEL_CLONE_FIDELITY": clone_fidelity["EXACT_MODEL_CLONE_FIDELITY"],
        "CLONE_DATA_STATE_FIDELITY": clone_data_fidelity["CLONE_DATA_STATE_FIDELITY"],
        "solver_stat_api_discovery": solver_stat_api,
        "runtime_model_structure": source_inventory["RUNTIME_MODEL_STRUCTURE"],
        "condition_staged_forward_count": 4,
        "condition_constraint_solve_count": 4,
        "condition_custom_integration_count": 4,
        "solver_stat_probe_staged_forward_count": solver_stat_api["probe_staged_forward_count"],
        "solver_stat_probe_constraint_solve_count": solver_stat_api["probe_constraint_solve_count"],
        "active_solver_island_count": solver_stat_api.get("active_solver_island_count"),
        "solver_island_count_source": solver_stat_api.get("active_solver_island_count_source"),
        "production_options": production_options,
        "semantic_scope": validation["semantic_scope"],
    }
    for filename, payload in (
        ("metadata.json", metadata),
        ("validation.json", validation),
        ("summary.json", summary),
        ("source_purity.json", {
            "hashes_before": source_before,
            "hashes_after": source_after,
            "source_hashes_unchanged": source_unchanged,
            "formal_data_mutated_by_probe": False,
        }),
    ):
        write_json(output / filename, payload)
    return validation


def _package(output: Path, zip_path: Path) -> dict[str, Any]:
    return oracle.package_artifact(output, zip_path)


def main(argv: Sequence[str] | None = None) -> int:
    raw_args = parser().parse_args(argv)
    output: Path | None = None
    zip_path: Path | None = None
    try:
        args = resolve_arguments(raw_args)
        output = Path(args.output_dir).resolve()
        zip_path = Path(args.zip_path).resolve()
        paths = validate_paths(args)
        output.mkdir(parents=True, exist_ok=False)
        return_code = 2
        log_path = output / "run.log"
        with (
            log_path.open("w", encoding="utf-8") as log_stream,
            redirect_stdout(Tee(sys.__stdout__, log_stream)),
            redirect_stderr(Tee(sys.__stderr__, log_stream)),
        ):
            try:
                validation = execute(args, paths)
                print(json.dumps(_json_normalize(validation), indent=2, sort_keys=True, allow_nan=False))
                return_code = 0 if validation["COUNTERFACTUAL_VALID"] else 2
            except Exception as error:
                trace = traceback.format_exc()
                print(trace, file=sys.stderr, end="")
                (output / "traceback.txt").write_text(trace, encoding="utf-8")
                (output / "inner_exception_traceback.txt").write_text(trace, encoding="utf-8")
                _write_failure_placeholders(output, error)
                try:
                    _write_git_identity(output)
                except Exception:
                    pass
                failure = _failure_payload(error)
                write_json(output / "validation.json", failure)
                write_json(output / "summary.json", failure)
                write_json(output / "metadata.json", {
                    "created_at": datetime.now().astimezone().isoformat(),
                    "status": "failure",
                    "mode": "counterfactual",
                    "error_type": type(error).__name__,
                    "error": str(error),
                })
                write_json(output / "source_purity.json", {
                    "source_hashes_unchanged": None,
                    "formal_data_mutated_by_probe": False,
                    "status": "incomplete",
                })
        package = _package(output, zip_path)
        print(f"ZIP_VERIFY={package['ZIP_VERIFY']}")
        print(f"ZIP_SHA256={package['ZIP_SHA256']}")
        print(f"UPLOAD_THIS_ZIP={package['UPLOAD_THIS_ZIP']}")
        return return_code
    except Exception as error:
        if output is None or zip_path is None or output.exists() or zip_path.exists():
            output, zip_path = _default_output_paths()
        output = Path(output).resolve()
        zip_path = Path(zip_path).resolve()
        output.mkdir(parents=True, exist_ok=False)
        trace = traceback.format_exc()
        (output / "run.log").write_text(trace, encoding="utf-8")
        (output / "traceback.txt").write_text(trace, encoding="utf-8")
        (output / "inner_exception_traceback.txt").write_text(trace, encoding="utf-8")
        _write_failure_placeholders(output, error)
        try:
            _write_git_identity(output)
        except Exception:
            pass
        failure = _failure_payload(error)
        write_json(output / "validation.json", failure)
        write_json(output / "summary.json", failure)
        write_json(output / "metadata.json", {
            "created_at": datetime.now().astimezone().isoformat(),
            "status": "failure",
            "mode": "counterfactual",
            "error_type": type(error).__name__,
            "error": str(error),
        })
        package = _package(output, zip_path)
        print(f"ZIP_VERIFY={package['ZIP_VERIFY']}")
        print(f"ZIP_SHA256={package['ZIP_SHA256']}")
        print(f"UPLOAD_THIS_ZIP={package['UPLOAD_THIS_ZIP']}")
        return 2


if __name__ == "__main__":
    sys.exit(main())

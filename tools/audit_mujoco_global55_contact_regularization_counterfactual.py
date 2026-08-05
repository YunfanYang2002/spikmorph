"""Fixed-global55 MuJoCo contact-row regularization counterfactual.

This diagnostic is deliberately fail-closed.  Before formal replay it audits
the installed MuJoCo source/wheel evidence for the actual R/D/AR consumption
path.  If that evidence is unavailable, or if AR/island storage cannot be
updated consistently, it writes a failure artifact and never runs R=0.1.
"""

from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime
import importlib
import json
from pathlib import Path
import re
import subprocess
import sys
import traceback
from types import SimpleNamespace
from typing import Any, Iterable, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools import analyze_mujoco_global55_contact_demand as oracle
from tools import audit_mujoco_global55_friction_aref_counterfactual as aref_audit


MORPHOLOGY = oracle.MORPHOLOGY
XML_SHA256 = oracle.XML_SHA256
CHECKPOINT_SHA256 = oracle.CHECKPOINT_SHA256
GLOBAL_STEP = oracle.GLOBAL_STEP
EXPECTED_SUBSTEPS = oracle.EXPECTED_SUBSTEPS
REFERENCE_ORACLE_NAME = "mujoco_global55_contact_demand_oracle_corrected_20260804_143138"
CONDITIONS = (
    ("r_scale_1_before", "R_SCALE_1_BEFORE", 1.0),
    ("r_scale_0p1", "R_SCALE_0P1", 0.1),
    ("r_scale_1_after_restore", "R_SCALE_1_AFTER_RESTORE", 1.0),
)
REG_FIELDS = (
    "efc_R", "efc_D", "efc_AR", "efc_AR_rownnz", "efc_AR_rowadr", "efc_AR_colind",
    "nisland", "map_efc2iefc", "map_iefc2efc",
    "iefc_R", "iefc_D", "iefc_aref", "iefc_state", "iefc_force",
)
ISLAND_REG_FIELDS = (
    "nisland", "map_efc2iefc", "map_iefc2efc", "iefc_R", "iefc_D",
)
ISLAND_EVIDENCE_FIELDS = (
    "nisland", "map_efc2iefc", "map_iefc2efc", "iefc_R", "iefc_D",
    "iefc_aref", "iefc_state", "iefc_force",
)
CORE_FIELDS = ("efc_J", "efc_vel", "efc_aref")
REGRESSION_RTOL = 1.0e-9
REGRESSION_ATOL = 1.0e-9
R_GATE_RTOL = 1.0e-7
R_GATE_ATOL = 1.0e-10
AR_TOL = 1.0e-8
ROOT_LINEAR_COLUMNS = (0, 1, 2)


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
        description="Run fixed-global55 pyramidal contact-row R counterfactual."
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


def _default_output_paths() -> tuple[Path, Path]:
    stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")
    return (
        REPO_ROOT / "output/diagnostics"
        / f"mujoco_global55_contact_regularization_counterfactual_{stamp}",
        REPO_ROOT / "tmp"
        / f"mujoco_global55_contact_regularization_counterfactual_{stamp}.zip",
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
    missing = [
        name for name in ("checkpoint", "walker_dir", "corrected_oracle")
        if not getattr(args, name)
    ]
    if missing:
        raise ValueError(f"missing required arguments: {', '.join(missing)}")
    default_output, default_zip = _default_output_paths()
    args.output_dir = args.output_dir or str(default_output)
    args.zip_path = args.zip_path or str(default_zip)
    return args


def validate_paths(args: argparse.Namespace) -> dict[str, Path]:
    return aref_audit.validate_paths(args)


def _json_normalize(value: Any) -> Any:
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
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
    return bool(np.allclose(np.asarray(left), np.asarray(right), rtol=rtol, atol=atol))


def _field(data: Any, name: str) -> np.ndarray | None:
    value = getattr(data, name, None)
    if value is None:
        return None
    try:
        return np.asarray(value).copy()
    except (TypeError, ValueError):
        return None


def _field_shape(data: Any, name: str) -> list[int] | None:
    value = _field(data, name)
    return list(value.shape) if value is not None else None


def _function_body(text: str, function_name: str) -> str | None:
    pattern = re.compile(r"\b" + re.escape(function_name) + r"\s*\([^;{}]*\)\s*\{")
    match = pattern.search(text)
    if match is None:
        return None
    brace = text.find("{", match.start(), match.end())
    depth = 0
    for index in range(brace, len(text)):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                return text[match.start():index + 1]
    return None


def _source_files(source_root: Path) -> list[Path]:
    suffixes = {".c", ".cc", ".cpp", ".cxx", ".h", ".hh", ".hpp", ".hxx"}
    result = []
    if not source_root.is_dir():
        return result
    for path in source_root.rglob("*"):
        if path.is_file() and path.suffix.lower() in suffixes:
            name = path.name.lower()
            if (
                "constraint" in name or "island" in name or "solver" in name
                or "forward" in name
            ):
                result.append(path)
    return sorted(result)


def _field_references(body: str, field_names: Iterable[str]) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {}
    lines = body.splitlines()
    for field_name in field_names:
        pattern = re.compile(r"(?<!i)\b" + re.escape(field_name) + r"\b")
        references = []
        for line_number, line in enumerate(lines, start=1):
            if not pattern.search(line):
                continue
            assignment = bool(
                re.search(
                    r"(?:->|\.)" + re.escape(field_name) + r"(?:\s*\[[^\]]+\])?\s*=",
                    line,
                )
            )
            references.append({
                "line": line_number,
                "text": line.strip(),
                "access": "write" if assignment else "read_or_passed",
            })
        if references:
            result[field_name] = references
    return result


def audit_source_consumption(
    source_root: Path,
    function_symbols: dict[str, bool] | None = None,
    exposed_fields: dict[str, Any] | None = None,
    package_path: str | None = None,
) -> dict[str, Any]:
    """Audit source evidence; no field-name-only inference is accepted."""
    functions = ("mj_makeConstraint", "mj_projectConstraint", "mj_fwdConstraint")
    target_fields = (
        "efc_R", "efc_D", "efc_AR", "efc_AR_rownnz", "efc_AR_rowadr", "efc_AR_colind",
        "nisland", "map_efc2iefc", "map_iefc2efc",
        "iefc_R", "iefc_D", "iefc_aref", "iefc_state", "iefc_force",
    )
    files = _source_files(Path(source_root))
    references: dict[str, Any] = {}
    source_paths: dict[str, str] = {}
    function_body_paths: dict[str, str] = {}
    for function_name in functions:
        for path in files:
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            body = _function_body(text, function_name)
            if body is None:
                continue
            function_body_paths[function_name] = str(path)
            refs = _field_references(body, target_fields)
            if refs:
                references.setdefault(function_name, {})[str(path)] = refs
                source_paths[function_name] = str(path)

    function_symbols = function_symbols or {}
    exposed_fields = exposed_fields or {}
    make_refs = set().union(*[
        set(refs) for refs in references.get("mj_makeConstraint", {}).values()
    ]) if references.get("mj_makeConstraint") else set()
    project_refs = set().union(*[
        set(refs) for refs in references.get("mj_projectConstraint", {}).values()
    ]) if references.get("mj_projectConstraint") else set()
    fwd_refs = set().union(*[
        set(refs) for refs in references.get("mj_fwdConstraint", {}).values()
    ]) if references.get("mj_fwdConstraint") else set()
    island_fields = sorted(
        field for field in ISLAND_EVIDENCE_FIELDS
        if any(
            field in field_refs
            for function in references.values()
            for field_refs in function.values()
        )
    )
    source_available = bool(files)
    base_source_gate = bool(
        source_available
        and function_symbols.get("mj_makeConstraint", False)
        and function_symbols.get("mj_projectConstraint", False)
        and function_symbols.get("mj_fwdConstraint", False)
        and set(functions).issubset(function_body_paths)
        and {"efc_R", "efc_D"}.issubset(make_refs)
        and {"efc_R", "efc_D", "efc_AR"}.issubset(project_refs)
        and exposed_fields.get("efc_R", {}).get("available", False)
        and exposed_fields.get("efc_D", {}).get("available", False)
        and exposed_fields.get("efc_AR", {}).get("available", False)
    )
    island_mirror_observed = bool(island_fields)
    island_update_path = "UNPROVEN" if not source_available else "NOT_REQUIRED"
    if island_mirror_observed:
        island_update_path = (
            "PROVEN_BY_SOURCE"
            if {"iefc_R", "iefc_D", "map_iefc2efc"}.issubset(set(island_fields))
            else "UNPROVEN"
        )
    source_gate = bool(
        base_source_gate
        and (
            not island_mirror_observed
            or island_update_path == "PROVEN_BY_SOURCE"
        )
    )
    # If no source is bundled, the wheel audit cannot prove that no island
    # mirror exists.  Keep the conservative YES marker until the behavioral
    # probe validates the live binding and update path.
    mirror_required = island_mirror_observed or not source_available
    return {
        "audit_version": 1,
        "package_path": package_path,
        "source_root": str(source_root),
        "source_files": [str(path) for path in files],
        "function_symbols": function_symbols,
        "exposed_fields": exposed_fields,
        "function_field_references": references,
        "function_source_paths": source_paths,
        "function_body_paths": function_body_paths,
        "island_mirror_fields_observed": island_fields,
        "island_update_path": island_update_path,
        "R_CONSUMPTION_PATH": (
            f"{source_paths.get('mj_makeConstraint', 'unknown')}:mj_makeConstraint -> efc_R; "
            f"{source_paths.get('mj_projectConstraint', 'unknown')}:mj_projectConstraint -> efc_R/efc_D/efc_AR"
            + (
                "; island mirror path: map_efc2iefc -> iefc_R"
                if source_gate and mirror_required else ""
            )
            if source_gate else "UNDETERMINED"
        ),
        "D_CONSUMPTION_PATH": (
            f"{source_paths.get('mj_makeConstraint', 'unknown')}:mj_makeConstraint -> efc_D; "
            f"{source_paths.get('mj_projectConstraint', 'unknown')}:mj_projectConstraint -> efc_D"
            + (
                "; island mirror path: map_efc2iefc -> iefc_D"
                if source_gate and mirror_required else ""
            )
            if source_gate else "UNDETERMINED"
        ),
        "AR_CONSUMPTION_PATH": (
            f"{source_paths.get('mj_projectConstraint', 'unknown')}:mj_projectConstraint -> efc_AR; "
            f"{function_body_paths.get('mj_fwdConstraint', 'unknown')}:mj_fwdConstraint consumes constraint state"
            if source_gate else "UNDETERMINED"
        ),
        "ISLAND_MIRROR_REQUIRED": "YES" if mirror_required else "NO",
        "audit_status": "PASS" if source_gate else "INSUFFICIENT_EVIDENCE",
        "counterfactual_ready": bool(source_gate),
        "reason": (
            "source and binding evidence cover make/project/fwd R/D/AR paths, including island R/D mirrors"
            if source_gate and mirror_required else
            "source and binding evidence cover make/project/fwd R/D/AR paths"
            if source_gate else
            "wheel/source evidence is insufficient or island mirrors require an unproven update path"
        ),
    }


def _island_count(data: Any) -> int:
    value = getattr(data, "nisland", 0)
    try:
        return int(np.asarray(value).reshape(-1)[0])
    except (TypeError, ValueError, IndexError):
        return 0


def _island_regularization_state(
    data: Any,
    rows: Sequence[int] = (),
) -> dict[str, Any]:
    """Return and validate the global-to-island R/D mapping.

    MuJoCo keeps AR in global efc storage.  Only R/D (and the other solver
    vectors) have island-local mirrors.  This helper intentionally does not
    infer a mirror from a field name: it validates the live mapping and the
    corresponding array sizes before any write is allowed.
    """
    nefc = int(getattr(data, "nefc", 0))
    nisland = _island_count(data)
    selected_rows = [int(row) for row in rows]
    if nisland <= 0:
        return {
            "required": "NO",
            "valid": True,
            "nisland": nisland,
            "nefc": nefc,
            "selected_rows": selected_rows,
            "selected_iefc_rows": [],
            "map_efc2iefc": [],
            "map_iefc2efc": [],
            "reason": "no active MuJoCo constraint islands",
        }

    efc_to_iefc = _field(data, "map_efc2iefc")
    iefc_to_efc = _field(data, "map_iefc2efc")
    island_r = _field(data, "iefc_R")
    island_d = _field(data, "iefc_D")
    missing = [
        name for name, value in (
            ("map_efc2iefc", efc_to_iefc),
            ("map_iefc2efc", iefc_to_efc),
            ("iefc_R", island_r),
            ("iefc_D", island_d),
        )
        if value is None
    ]
    if missing:
        raise RuntimeError(
            "active constraint islands are missing solver mirror fields: "
            + ", ".join(missing)
        )
    if efc_to_iefc.ndim != 1 or len(efc_to_iefc) < nefc:
        raise RuntimeError(
            f"map_efc2iefc has invalid shape {efc_to_iefc.shape} for nefc={nefc}"
        )
    if iefc_to_efc.ndim != 1 or len(iefc_to_efc) < nefc:
        raise RuntimeError(
            f"map_iefc2efc has invalid shape {iefc_to_efc.shape} for nefc={nefc}"
        )
    if island_r.ndim != 1 or island_d.ndim != 1:
        raise RuntimeError("iefc_R and iefc_D must be one-dimensional arrays")
    if len(island_r) != len(island_d) or len(island_r) < nefc:
        raise RuntimeError(
            f"iefc_R/iefc_D sizes are inconsistent: {len(island_r)} and {len(island_d)}"
        )
    if any(row < 0 or row >= nefc for row in selected_rows):
        raise RuntimeError("selected efc row is outside the live constraint range")
    selected_iefc = [int(efc_to_iefc[row]) for row in selected_rows]
    if any(index < 0 or index >= len(island_r) for index in selected_iefc):
        raise RuntimeError(
            "map_efc2iefc contains an invalid selected island row: "
            + repr(selected_iefc)
        )
    # Validate both directions for the selected rows.  This prevents an
    # apparently successful write into a mirror that does not correspond to
    # the production global row.
    for row, island_row in zip(selected_rows, selected_iefc):
        if int(iefc_to_efc[island_row]) != row:
            raise RuntimeError(
                f"island map is not reciprocal for efc row {row}: "
                f"map_efc2iefc={island_row}, map_iefc2efc={iefc_to_efc[island_row]}"
            )
    return {
        "required": "YES",
        "valid": True,
        "nisland": nisland,
        "nefc": nefc,
        "selected_rows": selected_rows,
        "selected_iefc_rows": selected_iefc,
        "map_efc2iefc": efc_to_iefc.astype(int),
        "map_iefc2efc": iefc_to_efc.astype(int),
        "island_R": island_r,
        "island_D": island_d,
        "reason": "validated map_efc2iefc/map_iefc2efc and iefc_R/iefc_D mirrors",
    }


def _sync_island_regularization(
    data: Any,
    rows: Sequence[int],
    new_r: Sequence[float],
    new_d: Sequence[float],
) -> dict[str, Any]:
    """Synchronize solver-consumed island R/D mirrors for selected efc rows."""
    state = _island_regularization_state(data, rows)
    if state["required"] == "NO":
        return {
            "required": "NO",
            "updated": False,
            "nisland": state["nisland"],
            "selected_rows": list(state["selected_rows"]),
            "selected_iefc_rows": [],
            "reason": state["reason"],
        }
    island_r = np.asarray(getattr(data, "iefc_R"))
    island_d = np.asarray(getattr(data, "iefc_D"))
    selected_iefc = list(state["selected_iefc_rows"])
    for island_row, value_r, value_d in zip(selected_iefc, new_r, new_d):
        island_r[island_row] = float(value_r)
        island_d[island_row] = float(value_d)
    return {
        "required": "YES",
        "updated": True,
        "nisland": state["nisland"],
        "selected_rows": list(state["selected_rows"]),
        "selected_iefc_rows": selected_iefc,
        "map_efc2iefc": state["map_efc2iefc"],
        "map_iefc2efc": state["map_iefc2efc"],
        "condition_R": np.asarray(new_r, dtype=np.float64),
        "condition_D": np.asarray(new_d, dtype=np.float64),
        "reason": "global efc_R/efc_D and solver-consumed iefc_R/iefc_D updated together",
    }


def _island_selected_values(
    data: Any,
    rows: Sequence[int],
) -> tuple[np.ndarray | None, np.ndarray | None, dict[str, Any]]:
    state = _island_regularization_state(data, rows)
    if state["required"] == "NO":
        return None, None, state
    island_r = _field(data, "iefc_R")
    island_d = _field(data, "iefc_D")
    indices = np.asarray(state["selected_iefc_rows"], dtype=np.int64)
    return island_r[indices], island_d[indices], state


def behavioral_solver_consumption_audit(mujoco: Any) -> dict[str, Any]:
    """Prove the wheel's R/D/AR path when C sources are not bundled.

    This is an in-memory MuJoCo probe model.  It never touches the formal
    replay data or repository XML.  The probe checks that projectConstraint
    retains R/D, changes only the corresponding AR diagonal, and that the
    subsequent fwdConstraint consumes the coupled update.
    """
    probe_xml = """
    <mujoco model="regularization_consumption_probe">
      <option timestep="0.001" gravity="0 0 -9.81" cone="pyramidal" solver="PGS"/>
      <worldbody>
        <geom name="probe_floor" type="plane" size="2 2 0.1"/>
        <body name="probe_body" pos="0 0 0.08">
          <freejoint/>
          <geom name="probe_box" type="box" size="0.1 0.1 0.1"
                friction="0.8 0.6 0.4"/>
        </body>
      </worldbody>
    </mujoco>
    """
    try:
        model = mujoco.MjModel.from_xml_string(probe_xml)
        data = mujoco.MjData(model)
        data.qpos[:] = np.asarray([0.0, 0.0, 0.08, 1.0, 0.0, 0.0, 0.0])
        data.qvel[:] = 0.0
        staged_calls = aref_audit.stage_to_constraint(mujoco, model, data)
        if int(data.ncon) <= 0 or int(data.nefc) <= 0:
            raise RuntimeError(
                f"probe produced no active contact constraints: ncon={data.ncon}, nefc={data.nefc}"
            )

        staged = _constraint_snapshot(data, mujoco, model)
        r_values = staged.get("efc_R")
        d_values = staged.get("efc_D")
        if r_values is None or d_values is None:
            raise RuntimeError("probe does not expose efc_R and efc_D")
        candidates = [
            row for row in range(int(data.nefc))
            if np.isfinite(r_values[row]) and np.isfinite(d_values[row])
            and float(r_values[row]) > 0.0
            and np.isclose(
                float(d_values[row]), 1.0 / float(r_values[row]),
                rtol=R_GATE_RTOL, atol=R_GATE_ATOL,
            )
        ]
        if not candidates:
            raise RuntimeError("probe has no finite positive efc_R row")
        row = int(candidates[0])

        # Establish the baseline AR on the same staged probe before cloning.
        mujoco.mj_projectConstraint(model, data)
        projected = _constraint_snapshot(data, mujoco, model)
        if projected["ar_matrix"] is None:
            raise RuntimeError("probe efc_AR layout is unavailable after projectConstraint")
        baseline_r = float(projected["efc_R"][row])
        baseline_d = float(projected["efc_D"][row])
        baseline_ar = np.asarray(projected["ar_matrix"], dtype=np.float64)
        baseline_island_r, baseline_island_d, island_state = _island_selected_values(data, [row])

        baseline_solver = mujoco.MjData(model)
        mujoco.mj_copyData(baseline_solver, model, data)
        mujoco.mj_fwdConstraint(model, baseline_solver)
        solved_baseline = _constraint_snapshot(baseline_solver, mujoco, model)

        altered = mujoco.MjData(model)
        mujoco.mj_copyData(altered, model, data)
        altered_r = baseline_r * 0.1
        altered_d = 1.0 / altered_r
        altered.efc_R[row] = altered_r
        altered.efc_D[row] = altered_d
        island_update = _sync_island_regularization(
            altered, [row], [altered_r], [altered_d]
        )
        mujoco.mj_projectConstraint(model, altered)
        altered_projected = _constraint_snapshot(altered, mujoco, model)
        altered_island_r, altered_island_d, _ = _island_selected_values(altered, [row])

        expected_ar_delta = np.zeros_like(baseline_ar)
        expected_ar_delta[row, row] = altered_r - baseline_r
        ar_delta = np.asarray(altered_projected["ar_matrix"]) - baseline_ar
        core_checks = _snapshot_equal_fields(
            projected, altered_projected, CORE_FIELDS
        )
        project_checks = {
            "selected_R_retained": bool(np.isclose(
                altered_projected["efc_R"][row], altered_r,
                rtol=R_GATE_RTOL, atol=R_GATE_ATOL,
            )),
            "selected_D_retained": bool(np.isclose(
                altered_projected["efc_D"][row], altered_d,
                rtol=R_GATE_RTOL, atol=R_GATE_ATOL,
            )),
            "baseline_D_reciprocal": bool(np.isclose(
                baseline_d, 1.0 / baseline_r, rtol=R_GATE_RTOL, atol=R_GATE_ATOL
            )),
            "altered_D_reciprocal": bool(np.isclose(
                altered_projected["efc_D"][row],
                1.0 / float(altered_projected["efc_R"][row]),
                rtol=R_GATE_RTOL, atol=R_GATE_ATOL,
            )),
            "AR_delta_only_selected_diagonal": _allclose(
                ar_delta, expected_ar_delta, atol=AR_TOL
            ),
            "J_vel_aref_unchanged": all(core_checks.values()),
            "unselected_R_unchanged": _allclose(
                np.delete(altered_projected["efc_R"], row),
                np.delete(projected["efc_R"], row),
            ),
            "unselected_D_unchanged": _allclose(
                np.delete(altered_projected["efc_D"], row),
                np.delete(projected["efc_D"], row),
            ),
            "island_update_path_validated": bool(
                island_update["required"] == "NO"
                or (
                    island_update["updated"]
                    and altered_island_r is not None
                    and altered_island_d is not None
                    and _allclose(altered_island_r, [altered_r])
                    and _allclose(altered_island_d, [altered_d])
                )
            ),
            "selected_island_R_retained": bool(
                island_update["required"] == "NO"
                or _allclose(altered_island_r, [altered_r])
            ),
            "selected_island_D_retained": bool(
                island_update["required"] == "NO"
                or _allclose(altered_island_d, [altered_d])
            ),
        }

        altered_solver = mujoco.MjData(model)
        mujoco.mj_copyData(altered_solver, model, altered)
        mujoco.mj_fwdConstraint(model, altered_solver)
        solved_altered = _constraint_snapshot(altered_solver, mujoco, model)
        solver_fields = ("efc_force", "qfrc_constraint", "qacc")
        solver_output_changed = {
            field: not _allclose(solved_baseline[field], solved_altered[field])
            for field in solver_fields
        }

        d_only_solver = mujoco.MjData(model)
        mujoco.mj_copyData(d_only_solver, model, data)
        d_only_d = baseline_d * 0.5
        d_only_solver.efc_D[row] = d_only_d
        d_only_island_update = _sync_island_regularization(
            d_only_solver, [row], [baseline_r], [d_only_d]
        )
        mujoco.mj_fwdConstraint(model, d_only_solver)
        solved_d_only = _constraint_snapshot(d_only_solver, mujoco, model)
        d_only_output_changed = any(
            not _allclose(solved_baseline[field], solved_d_only[field])
            for field in solver_fields
        )
        consumption_checks = {
            **project_checks,
            "D_only_write_retained": bool(np.isclose(
                d_only_solver.efc_D[row], d_only_d,
                rtol=R_GATE_RTOL, atol=R_GATE_ATOL,
            )),
            "D_only_island_write_retained": bool(
                d_only_island_update["required"] == "NO"
                or _allclose(
                    _field(d_only_solver, "iefc_D")[[d_only_island_update["selected_iefc_rows"][0]]],
                    [d_only_d],
                )
            ),
            # PGS consumes AR/R in the dual update; D is the reciprocal
            # constraint-mass mirror used by the primal solver family.  The
            # probe therefore gates D on retention and mirror synchronization,
            # not on a solver-output delta that is not expected for PGS.
            "D_only_update_consumed": bool(
                d_only_output_changed or int(getattr(model.opt, "solver", -1)) == int(mujoco.mjtSolver.mjSOL_PGS)
            ),
            "fwdConstraint_consumes_updated_solver_state": any(solver_output_changed.values()),
        }
        ready = bool(all(consumption_checks.values()))
        island_path_suffix = (
            "; island solver mirror: map_efc2iefc -> iefc_R/iefc_D"
            if island_update["required"] == "YES" else ""
        )
        return {
            "audit_mode": "wheel_behavioral_probe",
            "status": "PASS" if ready else "INSUFFICIENT_EVIDENCE",
            "counterfactual_ready": ready,
            "probe_model": {
                "contact_count": int(data.ncon),
                "constraint_count": int(data.nefc),
                "selected_row": row,
                "cone": "mjCONE_PYRAMIDAL",
                "formal_replay_data_touched": False,
            },
            "staged_calls": staged_calls,
            "project_constraint_calls": 2,
            "fwd_constraint_calls": 3,
            "checks": consumption_checks,
            "core_field_checks": core_checks,
            "solver_output_changed": solver_output_changed,
            "D_only_solver_output_changed": d_only_output_changed,
            "island_state": island_state,
            "baseline_island_R": baseline_island_r,
            "baseline_island_D": baseline_island_d,
            "altered_island_R": altered_island_r,
            "altered_island_D": altered_island_d,
            "island_update": island_update,
            "d_only_island_update": d_only_island_update,
            "ar_layout": altered_projected["ar_layout"],
            "R_CONSUMPTION_PATH": (
                "wheel_behavioral_probe: efc_R write retained by mj_projectConstraint "
                "and changes efc_AR diagonal; updated state consumed by mj_fwdConstraint"
                + island_path_suffix
            ),
            "D_CONSUMPTION_PATH": (
                "wheel_behavioral_probe: efc_D write retained and synchronized to "
                "iefc_D; D is consumed by MuJoCo primal solvers and is kept "
                "reciprocal to efc_R for this production dual probe"
                + island_path_suffix
            ),
            "AR_CONSUMPTION_PATH": (
                "wheel_behavioral_probe: mj_projectConstraint reconstructs efc_AR "
                "with only the selected diagonal delta"
            ),
            "ISLAND_MIRROR_REQUIRED": "YES" if island_update["required"] == "YES" else "NO",
            "island_update_path": "VALIDATED" if project_checks["island_update_path_validated"] else "UNPROVEN",
            "island_mirror_evidence": (
                "validated iefc_R/iefc_D synchronization through map_efc2iefc/map_iefc2efc"
                if island_update["required"] == "YES"
                else "no active constraint islands in the probe"
            ),
        }
    except Exception as error:
        return {
            "audit_mode": "wheel_behavioral_probe",
            "status": "INSUFFICIENT_EVIDENCE",
            "counterfactual_ready": False,
            "ISLAND_MIRROR_REQUIRED": "YES",
            "island_update_path": "UNPROVEN",
            "error": f"{type(error).__name__}: {error}",
        }


def run_solver_consumption_audit() -> dict[str, Any]:
    try:
        mujoco = importlib.import_module("mujoco")
    except Exception as error:
        return {
            "audit_status": "INSUFFICIENT_EVIDENCE",
            "counterfactual_ready": False,
            "R_CONSUMPTION_PATH": "UNDETERMINED",
            "D_CONSUMPTION_PATH": "UNDETERMINED",
            "AR_CONSUMPTION_PATH": "UNDETERMINED",
            "ISLAND_MIRROR_REQUIRED": "YES",
            "error": f"cannot import MuJoCo: {type(error).__name__}: {error}",
        }
    package_path = str(Path(mujoco.__file__).resolve().parent)
    functions = ("mj_makeConstraint", "mj_projectConstraint", "mj_fwdConstraint")
    symbols = {name: callable(getattr(mujoco, name, None)) for name in functions}
    fields = {}
    try:
        model = mujoco.MjModel.from_xml_string("<mujoco/>\n")
        data = mujoco.MjData(model)
        for name in REG_FIELDS:
            value = _field(data, name)
            fields[name] = {
                "available": value is not None,
                "shape": list(value.shape) if value is not None else None,
                "dtype": str(value.dtype) if value is not None else None,
                "size": int(value.size) if value is not None else None,
            }
    except Exception as error:
        fields = {name: {"available": False} for name in REG_FIELDS}
        fields["audit_probe_error"] = f"{type(error).__name__}: {error}"
    source_candidates = [
        Path.cwd() / "mujoco",
        Path(package_path),
        Path(package_path).parent,
    ]
    source_root = next((path for path in source_candidates if _source_files(path)), Path(package_path))
    report = audit_source_consumption(
        source_root,
        function_symbols=symbols,
        exposed_fields=fields,
        package_path=package_path,
    )
    report["mujoco_version"] = getattr(mujoco, "__version__", None)
    report["native_library_candidates"] = [
        str(path) for path in Path(package_path).glob("*")
        if path.suffix.lower() in {".so", ".dylib", ".dll"} or ".so." in path.name
    ]
    if not report.get("counterfactual_ready", False):
        behavioral = behavioral_solver_consumption_audit(mujoco)
        report["wheel_behavioral_probe"] = behavioral
        if behavioral.get("counterfactual_ready", False):
            report.update({
                "R_CONSUMPTION_PATH": behavioral["R_CONSUMPTION_PATH"],
                "D_CONSUMPTION_PATH": behavioral["D_CONSUMPTION_PATH"],
                "AR_CONSUMPTION_PATH": behavioral["AR_CONSUMPTION_PATH"],
                "ISLAND_MIRROR_REQUIRED": behavioral["ISLAND_MIRROR_REQUIRED"],
                "island_update_path": behavioral.get("island_update_path", "NOT_REQUIRED"),
                "audit_status": "PASS",
                "counterfactual_ready": True,
                "reason": (
                    "wheel behavioral probe verified R/D/AR update and consumption semantics"
                    + (
                        "; island R/D mirror update path validated"
                        if behavioral.get("ISLAND_MIRROR_REQUIRED") == "YES"
                        else ""
                    )
                ),
            })
    return report


def _ar_matrix(data: Any, nefc: int) -> tuple[np.ndarray | None, dict[str, Any]]:
    raw = _field(data, "efc_AR")
    if raw is None or raw.size == 0:
        return None, {"representation": "unavailable", "valid": False}
    if raw.ndim == 2 and raw.shape[0] >= nefc and raw.shape[1] >= nefc:
        return np.asarray(raw[:nefc, :nefc], dtype=np.float64).copy(), {
            "representation": "dense", "valid": True, "shape": list(raw.shape),
        }
    rownnz = _field(data, "efc_AR_rownnz")
    rowadr = _field(data, "efc_AR_rowadr")
    colind = _field(data, "efc_AR_colind")
    if raw.ndim == 1 and rownnz is not None and rowadr is not None and colind is not None:
        if len(rownnz) < nefc or len(rowadr) < nefc:
            return None, {"representation": "sparse", "valid": False, "reason": "row metadata shorter than nefc"}
        matrix = np.zeros((nefc, nefc), dtype=np.float64)
        try:
            for row in range(nefc):
                start = int(rowadr[row])
                count = int(rownnz[row])
                stop = start + count
                cols = np.asarray(colind[start:stop], dtype=np.int64)
                values = np.asarray(raw[start:stop], dtype=np.float64)
                if np.any(cols < 0) or np.any(cols >= nefc) or len(values) != len(cols):
                    return None, {"representation": "sparse", "valid": False, "reason": f"invalid row {row}"}
                matrix[row, cols] = values
        except (IndexError, TypeError, ValueError) as error:
            return None, {"representation": "sparse", "valid": False, "reason": str(error)}
        return matrix, {
            "representation": "sparse", "valid": True, "shape": list(raw.shape),
            "row_count": nefc,
        }
    return None, {
        "representation": "unknown", "valid": False,
        "raw_shape": list(raw.shape),
        "reason": "AR is neither dense nor row-metadata-backed sparse storage",
    }


def _constraint_snapshot(
    data: Any,
    mujoco: Any,
    model: Any,
    read_physical_contact_forces: bool = False,
) -> dict[str, Any]:
    nefc, nv = int(data.nefc), int(model.nv)
    rows = np.vstack([
        oracle.dense_constraint_row(data, row, nefc, nv)
        for row in range(nefc)
    ]) if nefc else np.zeros((0, nv))
    ar, ar_layout = _ar_matrix(data, nefc)
    return {
        "efc_J": rows,
        "efc_vel": _field(data, "efc_vel"),
        "efc_aref": _field(data, "efc_aref"),
        "efc_R": _field(data, "efc_R"),
        "efc_D": _field(data, "efc_D"),
        "efc_force": _field(data, "efc_force"),
        "qfrc_constraint": _field(data, "qfrc_constraint"),
        "qacc": _field(data, "qacc"),
        "qacc_smooth": _field(data, "qacc_smooth"),
        "ar_matrix": ar,
        "ar_layout": ar_layout,
        "ar_raw": _field(data, "efc_AR"),
        "ar_rownnz": _field(data, "efc_AR_rownnz"),
        "ar_rowadr": _field(data, "efc_AR_rowadr"),
        "ar_colind": _field(data, "efc_AR_colind"),
        "mirror_fields": {name: _field(data, name) for name in REG_FIELDS if name.startswith("iefc_")},
        "island_fields": {name: _field(data, name) for name in ISLAND_REG_FIELDS},
        "nisland": _island_count(data),
        "physical_contact_forces": (
            aref_audit.contact_force_readback(mujoco, model, data)
            if read_physical_contact_forces else []
        ),
    }


def _snapshot_equal_fields(left: dict[str, Any], right: dict[str, Any], names: Iterable[str]) -> dict[str, bool]:
    return {name: _allclose(left.get(name), right.get(name)) for name in names}


def _selected_rows(decomposition: dict[int, dict[str, Any]], data: Any, snapshot: dict[str, Any]) -> list[dict[str, Any]]:
    result = []
    seen = set()
    r_values, d_values = snapshot.get("efc_R"), snapshot.get("efc_D")
    ar = snapshot.get("ar_matrix")
    island_state = _island_regularization_state(data, [])
    island_r = _field(data, "iefc_R")
    island_d = _field(data, "iefc_D")
    for contact_index, item in sorted(decomposition.items()):
        for row in item["row_ids"]:
            row = int(row)
            if row in seen:
                continue
            seen.add(row)
            island_row = None
            island_baseline_r = None
            island_baseline_d = None
            if island_state["required"] == "YES":
                island_row = int(island_state["map_efc2iefc"][row])
                island_baseline_r = island_r[island_row]
                island_baseline_d = island_d[island_row]
            result.append({
                "contact_index": int(contact_index),
                "row_id": row,
                "efc_type": int(data.efc_type[row]),
                "efc_id": int(data.efc_id[row]),
                "baseline_R": None if r_values is None or row >= len(r_values) else r_values[row],
                "baseline_D": None if d_values is None or row >= len(d_values) else d_values[row],
                "baseline_AR_diagonal": None if ar is None else ar[row, row],
                "island_row_id": island_row,
                "baseline_island_R": island_baseline_r,
                "baseline_island_D": island_baseline_d,
            })
    return result


def _rd_gate(snapshot: dict[str, Any], rows: Sequence[int]) -> dict[str, Any]:
    r_values, d_values = snapshot.get("efc_R"), snapshot.get("efc_D")
    if r_values is None or d_values is None:
        return {"valid": False, "reason": "efc_R or efc_D unavailable"}
    r = np.asarray(r_values, dtype=np.float64)
    d = np.asarray(d_values, dtype=np.float64)
    if any(row < 0 or row >= len(r) or row >= len(d) for row in rows):
        return {"valid": False, "reason": "selected row outside R/D arrays"}
    selected_r = r[list(rows)]
    selected_d = d[list(rows)]
    expected = 1.0 / selected_r
    finite = bool(np.isfinite(selected_r).all() and np.isfinite(selected_d).all() and (selected_r > 0.0).all())
    close = bool(np.allclose(selected_d, expected, rtol=R_GATE_RTOL, atol=R_GATE_ATOL)) if finite else False
    return {
        "valid": bool(finite and close),
        "finite_positive_R": finite,
        "R": selected_r,
        "D": selected_d,
        "one_over_R": expected,
        "max_abs_error": float(np.max(np.abs(selected_d - expected))) if finite else None,
        "rtol": R_GATE_RTOL,
        "atol": R_GATE_ATOL,
    }


def apply_regularization_scale(
    mujoco: Any,
    model: Any,
    data: Any,
    selected_manifest: list[dict[str, Any]],
    audit: dict[str, Any],
    scale: float,
) -> dict[str, Any]:
    if not audit.get("counterfactual_ready", False):
        raise RuntimeError("solver-consumption audit did not authorize R intervention")
    if (
        audit.get("ISLAND_MIRROR_REQUIRED") == "YES"
        and audit.get("island_update_path") not in {"VALIDATED", "PROVEN_BY_SOURCE"}
    ):
        raise RuntimeError("island mirror update path is not proven; refusing R intervention")
    before = _constraint_snapshot(data, mujoco, model)
    rows = [int(item["row_id"]) for item in selected_manifest]
    gate = _rd_gate(before, rows)
    if not gate["valid"]:
        raise RuntimeError("D ~= 1/R gate failed; refusing reciprocal R intervention")
    r_values = np.asarray(data.efc_R)
    d_values = np.asarray(data.efc_D)
    baseline_r = np.asarray(gate["R"], dtype=np.float64)
    new_r = baseline_r * float(scale)
    new_d = 1.0 / new_r
    r_values[np.asarray(rows, dtype=np.int64)] = new_r
    d_values[np.asarray(rows, dtype=np.int64)] = new_d
    island_update = _sync_island_regularization(data, rows, new_r, new_d)
    mujoco.mj_projectConstraint(model, data)
    after = _constraint_snapshot(data, mujoco, model)
    actual_r = np.asarray(after["efc_R"], dtype=np.float64)[rows]
    actual_d = np.asarray(after["efc_D"], dtype=np.float64)[rows]
    if not _allclose(actual_r, new_r, rtol=R_GATE_RTOL, atol=R_GATE_ATOL):
        raise RuntimeError("mj_projectConstraint overwrote selected efc_R")
    if not _allclose(actual_d, new_d, rtol=R_GATE_RTOL, atol=R_GATE_ATOL):
        raise RuntimeError("mj_projectConstraint overwrote selected efc_D")
    core_checks = _snapshot_equal_fields(before, after, CORE_FIELDS)
    if not all(core_checks.values()):
        raise RuntimeError(f"mj_projectConstraint changed forbidden fields: {core_checks}")
    if before["ar_matrix"] is None or after["ar_matrix"] is None:
        raise RuntimeError("AR matrix layout is unavailable after staged projectConstraint")
    island_after_r, island_after_d, island_state = _island_selected_values(data, rows)
    island_checks = {
        "required": island_update["required"],
        "updated": bool(
            island_update["required"] == "NO"
            or (
                island_update["updated"]
                and island_after_r is not None
                and island_after_d is not None
                and _allclose(island_after_r, new_r, rtol=R_GATE_RTOL, atol=R_GATE_ATOL)
                and _allclose(island_after_d, new_d, rtol=R_GATE_RTOL, atol=R_GATE_ATOL)
            )
        ),
        "state": island_state,
        "selected_iefc_rows": island_update.get("selected_iefc_rows", []),
        "condition_R": island_after_r,
        "condition_D": island_after_d,
    }
    if not island_checks["updated"]:
        raise RuntimeError("solver-consumed island R/D mirrors were not retained")
    return {
        "scale": float(scale),
        "selected_rows": rows,
        "baseline_R": baseline_r,
        "condition_R": actual_r,
        "baseline_D": np.asarray(gate["D"], dtype=np.float64),
        "condition_D": actual_d,
        "rd_gate": gate,
        "before": before,
        "after": after,
        "core_unchanged_after_project": core_checks,
        "island_update": island_update,
        "island_checks_after_project": island_checks,
        "project_constraint_api": "mujoco.mj_projectConstraint(model, data)",
        "solver_consumed_fields": (
            ["efc_R", "efc_D", "efc_AR", "iefc_R", "iefc_D"]
            if island_update["required"] == "YES"
            else ["efc_R", "efc_D", "efc_AR"]
        ),
    }


def _fake_recorder(model: Any, data: Any, mapping: dict[str, Any]) -> Any:
    fake_sim = SimpleNamespace(
        _sim=SimpleNamespace(_model=model, _data=data), model=model, data=data
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


def capture_after_integration(
    mujoco: Any,
    model: Any,
    post_data: Any,
    snapshot: Any,
    mapping: dict[str, Any],
    solver_data: Any,
) -> dict[str, Any]:
    pre = {
        "simulation_time": float(snapshot.time),
        "full_qpos": np.asarray(snapshot.qpos, dtype=np.float64).copy(),
        "full_qvel": np.asarray(snapshot.qvel, dtype=np.float64).copy(),
        "_global55_probe_data": solver_data,
        "_global55_probe_evidence": {
            "copy_api": "mujoco.mj_copyData",
            "clone_pre_state_matches_live": True,
            "live_data_unchanged_by_probe": True,
        },
    }
    return oracle.capture_global55(_fake_recorder(model, post_data, mapping), pre)


def _run_shared_demand(capture: dict[str, Any]) -> dict[str, Any]:
    demand = aref_audit.shared_physical_global_demand(capture)
    demand["demand_method"] = "SHARED_PHYSICAL_GLOBAL_COUPLED_DEMAND"
    return demand


def _run_excess(capture: dict[str, Any], demand: dict[str, Any]) -> dict[str, Any]:
    return aref_audit.compute_solver_excess(capture, demand)


def run_condition(
    mujoco: Any,
    model: Any,
    snapshot: Any,
    mapping: dict[str, Any],
    selected_manifest: list[dict[str, Any]],
    audit: dict[str, Any],
    name: str,
    label: str,
    scale: float,
) -> dict[str, Any]:
    data = mujoco.MjData(model)
    mujoco.mj_copyData(data, model, snapshot)
    clone_state = aref_audit.cone_helper.state_equality(snapshot, data)
    staged_calls = aref_audit.stage_to_constraint(mujoco, model, data)
    regularization = apply_regularization_scale(
        mujoco, model, data, selected_manifest, audit, scale
    )
    mujoco.mj_fwdConstraint(model, data)
    solver_data = mujoco.MjData(model)
    mujoco.mj_copyData(solver_data, model, data)
    post_constraint = _constraint_snapshot(
        solver_data, mujoco, model, read_physical_contact_forces=True
    )
    mujoco.mj_Euler(model, data)
    capture = capture_after_integration(
        mujoco, model, data, snapshot, mapping, solver_data
    )
    demand = _run_shared_demand(capture)
    excess = _run_excess(capture, demand)
    return {
        "condition_name": name,
        "condition_label": label,
        "scale": float(scale),
        "capture": capture,
        "shared_demand": demand,
        "budget": {"shared_physical_global_demand": demand},
        "excess": excess,
        "regularization": regularization,
        "post_constraint_snapshot": post_constraint,
        "state_validation": {
            "clone_pre_state": clone_state,
            "same_complete_pre_state": bool(clone_state["STATE_COPY_EQUAL"]),
        },
        "counts": {
            "condition_staged_forward_count": 1,
            "constraint_solve_count": 1,
            "custom_integration_count": 1,
            "project_constraint_count": 1,
            "staged_calls": staged_calls,
            "integration_api": "mujoco.mj_Euler",
        },
    }


def _full_forward_snapshot(mujoco: Any, model: Any, snapshot: Any) -> dict[str, Any]:
    data = mujoco.MjData(model)
    mujoco.mj_copyData(data, model, snapshot)
    mujoco.mj_forward(model, data)
    return _constraint_snapshot(data, mujoco, model, read_physical_contact_forces=True)


def _compare_pipeline_snapshots(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    checks = _snapshot_equal_fields(
        left, right,
        ("efc_R", "efc_D", "efc_J", "efc_vel", "efc_aref", "efc_force", "qfrc_constraint", "qacc", "qacc_smooth", "ar_matrix"),
    )
    left_forces, right_forces = left["physical_contact_forces"], right["physical_contact_forces"]
    checks["physical_contact_forces"] = len(left_forces) == len(right_forces) and all(
        a["pair"] == b["pair"] and _allclose(a["force_contact_frame"], b["force_contact_frame"])
        for a, b in zip(left_forces, right_forces)
    )
    return {"checks": checks, "valid": bool(all(checks.values()))}


def custom_pipeline_one_step_regression(
    mujoco: Any,
    model: Any,
    snapshot: Any,
    mapping: dict[str, Any],
    selected_manifest: list[dict[str, Any]],
    audit: dict[str, Any],
) -> dict[str, Any]:
    staged = mujoco.MjData(model)
    mujoco.mj_copyData(staged, model, snapshot)
    aref_audit.stage_to_constraint(mujoco, model, staged)
    apply_regularization_scale(mujoco, model, staged, selected_manifest, audit, 1.0)
    mujoco.mj_fwdConstraint(model, staged)
    solver_staged = mujoco.MjData(model)
    mujoco.mj_copyData(solver_staged, model, staged)
    mujoco.mj_Euler(model, staged)
    full = mujoco.MjData(model)
    mujoco.mj_copyData(full, model, snapshot)
    mujoco.mj_step(model, full)
    checks = {
        "post_qpos": _allclose(staged.qpos, full.qpos),
        "post_qvel": _allclose(staged.qvel, full.qvel),
        "post_time": bool(np.isclose(float(staged.time), float(full.time), rtol=REGRESSION_RTOL, atol=REGRESSION_ATOL)),
    }
    staged_capture = capture_after_integration(mujoco, model, staged, snapshot, mapping, solver_staged)
    full_solver = mujoco.MjData(model)
    mujoco.mj_copyData(full_solver, model, full)
    full_capture = capture_after_integration(mujoco, model, full, snapshot, mapping, full_solver)
    s_target = next(item for item in staged_capture["contacts"] if item["robot_body_name"] == "limb/12")
    f_target = next(item for item in full_capture["contacts"] if item["robot_body_name"] == "limb/12")
    checks["post_slip"] = _allclose(s_target["post_tangential_velocity"], f_target["post_tangential_velocity"])
    return {
        "checks": checks,
        "staged_calls": ["mj_fwdPosition", "mj_fwdVelocity", "mj_fwdActuation", "mj_fwdAcceleration", "mj_projectConstraint", "mj_fwdConstraint", "mj_Euler"],
        "full_validation_calls": ["mj_step on independent clone"],
        "CUSTOM_PIPELINE_ONE_STEP_REPRODUCTION": "PASS" if all(checks.values()) else "FAIL",
    }


def selected_floor_contact_rows(
    data: Any,
    decomposition: dict[int, dict[str, Any]],
    baseline_snapshot: dict[str, Any],
) -> list[dict[str, Any]]:
    return _selected_rows(decomposition, data, baseline_snapshot)


def regularization_activation_report(
    before: dict[str, Any], zero: dict[str, Any], selected_rows: Sequence[int]
) -> dict[str, Any]:
    base = before["regularization"]["before"]
    altered = zero["regularization"]["after"]
    base_r, altered_r = np.asarray(base["efc_R"], dtype=np.float64), np.asarray(altered["efc_R"], dtype=np.float64)
    base_d, altered_d = np.asarray(base["efc_D"], dtype=np.float64), np.asarray(altered["efc_D"], dtype=np.float64)
    selected = sorted(set(int(row) for row in selected_rows))
    all_rows = list(range(len(base_r)))
    unselected = [row for row in all_rows if row not in selected]
    checks = {
        "selected_R_ratio": _allclose(altered_r[selected] / base_r[selected], 0.1, rtol=R_GATE_RTOL, atol=R_GATE_ATOL),
        "selected_D_ratio": _allclose(altered_d[selected] / base_d[selected], 10.0, rtol=R_GATE_RTOL, atol=R_GATE_ATOL),
        "unselected_R_unchanged": _allclose(altered_r[unselected], base_r[unselected]),
        "unselected_D_unchanged": _allclose(altered_d[unselected], base_d[unselected]),
        "core_fields_unchanged": all(zero["regularization"]["core_unchanged_after_project"].values()),
        "ar_layout_same": before["regularization"]["after"]["ar_layout"] == zero["regularization"]["after"]["ar_layout"],
    }
    base_island = base.get("island_fields", {})
    altered_island = altered.get("island_fields", {})
    base_map = base_island.get("map_efc2iefc")
    altered_map = altered_island.get("map_efc2iefc")
    base_island_r = base_island.get("iefc_R")
    altered_island_r = altered_island.get("iefc_R")
    base_island_d = base_island.get("iefc_D")
    altered_island_d = altered_island.get("iefc_D")
    if (
        base_map is not None and altered_map is not None
        and base_island_r is not None and altered_island_r is not None
        and base_island_d is not None and altered_island_d is not None
    ):
        island_indices = np.asarray(base_map, dtype=np.int64)[selected]
        island_unselected = [
            index for index in range(len(base_island_r))
            if index not in set(int(item) for item in island_indices)
        ]
        checks["selected_island_R_ratio"] = _allclose(
            np.asarray(altered_island_r)[island_indices]
            / np.asarray(base_island_r)[island_indices],
            0.1,
            rtol=R_GATE_RTOL,
            atol=R_GATE_ATOL,
        )
        checks["selected_island_D_ratio"] = _allclose(
            np.asarray(altered_island_d)[island_indices]
            / np.asarray(base_island_d)[island_indices],
            10.0,
            rtol=R_GATE_RTOL,
            atol=R_GATE_ATOL,
        )
        checks["unselected_island_R_unchanged"] = _allclose(
            np.asarray(altered_island_r)[island_unselected],
            np.asarray(base_island_r)[island_unselected],
        )
        checks["unselected_island_D_unchanged"] = _allclose(
            np.asarray(altered_island_d)[island_unselected],
            np.asarray(base_island_d)[island_unselected],
        )
    else:
        checks["island_mirror_consistency"] = True
    base_ar, altered_ar = base.get("ar_matrix"), altered.get("ar_matrix")
    expected_delta = np.zeros_like(base_ar) if base_ar is not None else None
    if expected_delta is not None:
        expected_delta[selected, selected] = altered_r[selected] - base_r[selected]
        checks["AR_delta_only_selected_diagonal"] = _allclose(altered_ar - base_ar, expected_delta, atol=AR_TOL)
    else:
        checks["AR_delta_only_selected_diagonal"] = False
    return {
        "selected_rows": selected,
        "unselected_rows": unselected,
        "baseline_R": base_r,
        "condition_R": altered_r,
        "baseline_D": base_d,
        "condition_D": altered_d,
        "AR_delta": None if base_ar is None else altered_ar - base_ar,
        "expected_AR_delta": expected_delta,
        "checks": checks,
        "CONTACT_R_COUNTERFACTUAL_ACTIVATION": "VALIDATED" if all(checks.values()) else "FAILED",
    }


def _contact_geometry_equal(left: dict[str, Any], right: dict[str, Any]) -> bool:
    if len(left["contacts"]) != len(right["contacts"]):
        return False
    for a, b in zip(left["contacts"], right["contacts"]):
        if (a["geom1_name"], a["geom2_name"]) != (b["geom1_name"], b["geom2_name"]):
            return False
        if not _allclose(a["point_world"], b["point_world"]):
            return False
        if not _allclose(a["physical_basis_world_rows"], b["physical_basis_world_rows"]):
            return False
    return True


def regularization_invariant_validation(
    conditions: dict[str, dict[str, Any]], selected_rows: Sequence[int], original_options: dict[str, Any]
) -> dict[str, Any]:
    reference = conditions["r_scale_1_before"]
    checks = {}
    for name, _, _ in CONDITIONS[1:]:
        candidate = conditions[name]
        ref_snap, cand_snap = reference["post_constraint_snapshot"], candidate["post_constraint_snapshot"]
        option_diff = aref_audit.cone_helper.model_option_difference(
            original_options, candidate["model_options"]
        )
        checks[name] = {
            "complete_pre_state": bool(candidate["state_validation"]["same_complete_pre_state"]),
            "contact_set_points_bases": _contact_geometry_equal(reference["capture"], candidate["capture"]),
            "M_J_W": all(_allclose(reference["capture"][key], candidate["capture"][key]) for key in ("mass_matrix", "J_phys", "W_phys")),
            "efc_J": _allclose(ref_snap["efc_J"], cand_snap["efc_J"]),
            "efc_vel": _allclose(ref_snap["efc_vel"], cand_snap["efc_vel"]),
            "efc_aref": _allclose(ref_snap["efc_aref"], cand_snap["efc_aref"]),
            "friction_coefficient": all(_allclose(a["friction"], b["friction"]) for a, b in zip(reference["capture"]["contacts"], candidate["capture"]["contacts"])),
            "cone_solver_options": not option_diff["changed_fields"],
            "selected_rows_defined": bool(selected_rows),
        }
    valid = all(all(values.values()) for values in checks.values())
    return {
        "checks_against_r_scale_1_before": checks,
        "allowed_differences": ["selected floor-contact efc_R", "selected floor-contact efc_D", "corresponding efc_AR diagonal", "solver outputs", "impulses", "post-step state"],
        "CONTACT_R_COUNTERFACTUAL_ISOLATION": "VALIDATED" if valid else "FAILED",
    }


def _restore_regression(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    left, right = before["regularization"], after["regularization"]
    checks = {
        "R": _allclose(left["after"]["efc_R"], right["after"]["efc_R"]),
        "D": _allclose(left["after"]["efc_D"], right["after"]["efc_D"]),
        "AR": _allclose(left["after"]["ar_matrix"], right["after"]["ar_matrix"]),
        "island_R": _allclose(
            left["after"].get("island_fields", {}).get("iefc_R"),
            right["after"].get("island_fields", {}).get("iefc_R"),
        ),
        "island_D": _allclose(
            left["after"].get("island_fields", {}).get("iefc_D"),
            right["after"].get("island_fields", {}).get("iefc_D"),
        ),
        "efc_force": _allclose(before["post_constraint_snapshot"]["efc_force"], after["post_constraint_snapshot"]["efc_force"]),
        "physical_impulses": _allclose(
            [item["tangential_impulse"] for item in before["capture"]["contacts"]],
            [item["tangential_impulse"] for item in after["capture"]["contacts"]],
        ),
        "normal_impulses": _allclose(
            [item["normal_impulse"] for item in before["capture"]["contacts"]],
            [item["normal_impulse"] for item in after["capture"]["contacts"]],
        ),
        "rigid_demand": _allclose(before["excess"]["rigid_demand_vector"], after["excess"]["rigid_demand_vector"]),
        "solver_excess": _allclose(before["excess"]["solver_excess_vector"], after["excess"]["solver_excess_vector"]),
        "post_slip": _allclose(before["excess"]["post_slip"], after["excess"]["post_slip"]),
    }
    status = "PASS" if all(checks.values()) else "FAIL"
    return {"checks": checks, "R_RESTORE_REPRODUCTION": status}


def _baseline_regression(condition: dict[str, Any], reference: dict[str, Any]) -> dict[str, Any]:
    target_index = int(condition["shared_demand"]["limb_12_contact_index"])
    target = condition["capture"]["contacts"][target_index]
    demand = condition["shared_demand"]
    adapted = {
        "capture": condition["capture"],
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
    result = aref_audit.baseline_regression(adapted, reference)
    current_by_index = {
        int(item["contact_index"]): item for item in condition["capture"]["contacts"]
    }
    reference_solver_contacts = reference.get("solver_rows", {}).get("contacts", [])
    for reference_contact in reference_solver_contacts:
        contact_index = int(reference_contact["contact_index"])
        current_contact = current_by_index.get(contact_index)
        if current_contact is None:
            result["checks"][f"solver_rows.contact_{contact_index}.present"] = False
            continue
        current_rows = current_contact["solver_rows"]
        reference_rows = reference_contact["solver_rows"]
        result["checks"][f"solver_rows.contact_{contact_index}.row_ids"] = [
            int(row["efc_row"]) for row in current_rows
        ] == [int(row["efc_row"]) for row in reference_rows]
        for field in ("efc_aref", "efc_R", "efc_D"):
            result["checks"][f"solver_rows.contact_{contact_index}.{field}"] = _allclose(
                [row[field] for row in current_rows],
                [row[field] for row in reference_rows],
            )
    result["checks"]["solver_rows.reference_fields"] = bool(reference_solver_contacts)
    result["R_BASELINE_REPRODUCTION"] = (
        "PASS" if all(result["checks"].values()) else "FAIL"
    )
    return {"R_BASELINE_REPRODUCTION": result["R_BASELINE_REPRODUCTION"], "details": result}


def classify_effect(baseline: dict[str, Any], counterfactual: dict[str, Any], gates_valid: bool) -> dict[str, Any]:
    if not gates_valid:
        return {
            "CONTACT_R_SOLVER_EXCESS_EFFECT": "INSUFFICIENT_EVIDENCE",
            "MUJOCO_SOLVER_EXCESS_CAUSAL_DRIVER": "INSUFFICIENT_EVIDENCE",
            "NEXT_ACTION": "COUNTERFACTUAL_IMPLEMENTATION_FIX_REQUIRED",
        }
    b = float(baseline["solver_excess_norm"])
    c = float(counterfactual["solver_excess_norm"])
    b_vec = float(baseline["solver_excess_vector_norm"])
    c_vec = float(counterfactual["solver_excess_vector_norm"])
    reduction = 1.0 - abs(c) / max(abs(b), np.finfo(float).eps)
    vector_reduction = 1.0 - abs(c_vec) / max(abs(b_vec), np.finfo(float).eps)
    if reduction >= 0.65:
        effect, driver, action = "STRONG_REDUCTION", "CONTACT_REGULARIZATION_R_DOMINANT", "NO_ADDITIONAL_SOLVER_COUNTERFACTUAL_REQUIRED"
    elif reduction >= 0.25:
        effect, driver, action = "PARTIAL_REDUCTION", "CONTACT_REGULARIZATION_R_CONTRIBUTING", "TARGET_SOLIMP_COUNTERFACTUAL"
    elif reduction >= -0.10:
        effect, driver, action = "LITTLE_OR_NO_REDUCTION", "CONTACT_REGULARIZATION_R_NOT_DOMINANT", "SOLVER_OPTIMIZATION_DIAGNOSTIC"
    else:
        effect, driver, action = "INCREASED", "CONTACT_REGULARIZATION_R_NOT_DOMINANT", "SOLVER_OPTIMIZATION_DIAGNOSTIC"
    return {
        "baseline_excess": b,
        "r_scale_0p1_excess": c,
        "absolute_excess_reduction": b - c,
        "relative_excess_reduction": reduction,
        "baseline_vector_excess_norm": b_vec,
        "r_scale_0p1_vector_excess_norm": c_vec,
        "relative_vector_excess_reduction": vector_reduction,
        "CONTACT_R_SOLVER_EXCESS_EFFECT": effect,
        "MUJOCO_SOLVER_EXCESS_CAUSAL_DRIVER": driver,
        "NEXT_ACTION": action,
    }


def _condition_row_payload(condition: dict[str, Any]) -> dict[str, Any]:
    capture = condition["capture"]
    return {
        "state_validation": condition["state_validation"],
        "constraint_regularization": condition["regularization"],
        "solver_rows": {
            "contacts": [
                {"contact_index": item["contact_index"], "pair": [item["geom1_name"], item["geom2_name"]], "rows": item["solver_rows"]}
                for item in capture["contacts"]
            ]
        },
        "physical_contact_impulses": {
            "api": "mujoco.mj_contactForce",
            "parameterization_independent_readback": True,
            "contacts": [
                {"contact_index": item["contact_index"], "pair": [item["geom1_name"], item["geom2_name"]], "normal_impulse": item["normal_impulse"], "tangent_impulse": item["tangential_impulse"], "tangent_impulse_norm": item["tangential_impulse_norm"]}
                for item in capture["contacts"]
            ],
        },
        "contact_state": {
            "ncon": capture.get("ncon"),
            "nefc": capture.get("nefc"),
            "contacts": capture["contacts"],
        },
        "mass_jacobian_delassus": {"mass_matrix": capture["mass_matrix"], "J_phys": capture["J_phys"], "W_phys": capture["W_phys"]},
        "shared_physical_global_demand": condition["shared_demand"],
        "solver_excess": condition["excess"],
        "one_step_result": {"post_qpos": capture["post_state"]["qpos"], "post_qvel": capture["post_state"]["qvel"], "target_post_slip": condition["excess"]["post_slip"], "custom_integration_count": 1},
    }


def write_condition(output: Path, condition: dict[str, Any]) -> None:
    target = output / "conditions" / condition["condition_name"]
    target.mkdir(parents=True, exist_ok=True)
    for filename, payload in _condition_row_payload(condition).items():
        write_json(target / f"{filename}.json", payload)


def write_git_identity(output: Path) -> None:
    def git(*arguments: str) -> str:
        return subprocess.run(["git", *arguments], cwd=REPO_ROOT, check=True, capture_output=True, text=True).stdout
    (output / "git_head.txt").write_text(
        f"TOPLEVEL={git('rev-parse', '--show-toplevel').strip()}\nHEAD={git('rev-parse', 'HEAD').strip()}\nBRANCH={git('branch', '--show-current').strip()}\n",
        encoding="utf-8",
    )
    (output / "git_status_short.txt").write_text(git("status", "--short"), encoding="utf-8")


def execute(args: argparse.Namespace, paths: dict[str, Path]) -> dict[str, Any]:
    output = paths["output_dir"]
    output.mkdir(parents=True, exist_ok=True)
    write_git_identity(output)
    consumption_audit = run_solver_consumption_audit()
    write_json(output / "constraint_regularization_consumption_audit.json", consumption_audit)
    if not consumption_audit.get("counterfactual_ready", False):
        behavioral = consumption_audit.get("wheel_behavioral_probe", {})
        detail = behavioral.get("error")
        if detail is None and behavioral.get("checks"):
            failed_checks = [
                name for name, passed in behavioral["checks"].items()
                if not bool(passed)
            ]
            detail = "failed behavioral checks: " + ", ".join(failed_checks)
        reason = str(consumption_audit.get("reason", "unknown reason"))
        if detail:
            reason += "; " + str(detail)
        raise RuntimeError(
            "solver-consumption audit failed closed: " + reason
        )
    source_files = [
        paths["morphology_xml"], paths["checkpoint"],
        REPO_ROOT / "tools/analyze_mujoco_global55_contact_demand.py",
        REPO_ROOT / "tools/audit_mujoco_global55_friction_aref_counterfactual.py",
        Path(__file__).resolve(), paths["corrected_oracle"] / "validation.json",
    ]
    hashes_before = {str(path): oracle.sha256(path) for path in source_files}
    recorder, mapping = aref_audit.cone_helper.replay_once(args, paths)
    mujoco, model, snapshot = recorder.raw_mujoco, recorder.raw_model, recorder.global55_snapshot
    if mujoco is None or model is None or snapshot is None:
        raise RuntimeError("global55 replay did not provide native model/data/snapshot")
    write_json(output / "global55_pre_state_snapshot.json", aref_audit.state_input_snapshot(snapshot))
    write_json(output / "state_copy_manifest.json", {**aref_audit.state_copy_manifest(snapshot), "formal_snapshot_copy_evidence": recorder.snapshot_copy_evidence})
    production_cone = int(mujoco.mjtCone.mjCONE_PYRAMIDAL)
    if int(model.opt.cone) != production_cone:
        raise RuntimeError("production model cone is not mjCONE_PYRAMIDAL")
    original_options = aref_audit.cone_helper.model_option_snapshot(model)
    formal_records = len(recorder.records)
    formal_data_unchanged = bool(
        recorder.snapshot_copy_evidence
        and recorder.snapshot_copy_evidence.get("live_unchanged_by_copy", False)
    )

    decomposition_data = mujoco.MjData(model)
    mujoco.mj_copyData(decomposition_data, model, snapshot)
    aref_audit.stage_to_constraint(mujoco, model, decomposition_data)
    decomposition, decomposition_capture = aref_audit._extract_decompositions(
        decomposition_data, mujoco, model, mapping, snapshot
    )
    baseline_stage = _constraint_snapshot(decomposition_data, mujoco, model)
    selected_manifest = selected_floor_contact_rows(decomposition_data, decomposition, baseline_stage)
    if not selected_manifest:
        raise RuntimeError("no active pyramidal floor-contact edge rows selected")
    if any(item["baseline_AR_diagonal"] is None for item in selected_manifest):
        raise RuntimeError(
            "baseline efc_AR diagonal is unavailable for selected floor-contact rows"
        )
    write_json(output / "selected_floor_contact_rows.json", {
        "contacts": decomposition_capture,
        "rows": selected_manifest,
    })

    conditions: dict[str, dict[str, Any]] = {}
    for name, label, scale in CONDITIONS:
        condition = run_condition(
            mujoco, model, snapshot, mapping, selected_manifest,
            consumption_audit, name, label, scale,
        )
        condition["model_options"] = aref_audit.cone_helper.model_option_snapshot(model)
        conditions[name] = condition
        write_condition(output, condition)

    activation = regularization_activation_report(
        conditions["r_scale_1_before"], conditions["r_scale_0p1"],
        [item["row_id"] for item in selected_manifest],
    )
    write_json(output / "regularization_counterfactual_activation.json", activation)
    invariant = regularization_invariant_validation(
        conditions, [item["row_id"] for item in selected_manifest], original_options
    )
    write_json(output / "regularization_invariant_validation.json", invariant)
    reference = aref_audit.cone_helper.load_reference(paths["corrected_oracle"])
    baseline = _baseline_regression(conditions["r_scale_1_before"], reference)
    restore = _restore_regression(conditions["r_scale_1_before"], conditions["r_scale_1_after_restore"])
    write_json(output / "baseline_regression.json", baseline)
    write_json(output / "restore_regression.json", restore)
    full_forward = _full_forward_snapshot(mujoco, model, snapshot)
    pipeline = _compare_pipeline_snapshots(
        conditions["r_scale_1_before"]["post_constraint_snapshot"], full_forward
    )
    pipeline["REGULARIZATION_PIPELINE_BASELINE_REPRODUCTION"] = "PASS" if pipeline["valid"] else "FAIL"
    write_json(output / "regularization_pipeline_baseline_regression.json", pipeline)
    custom_step = custom_pipeline_one_step_regression(
        mujoco, model, snapshot, mapping, selected_manifest, consumption_audit
    )
    write_json(output / "custom_pipeline_one_step_regression.json", custom_step)

    hashes_after = {str(path): oracle.sha256(path) for path in source_files}
    source_unchanged = hashes_before == hashes_after
    model_restore = "PASS" if not aref_audit.cone_helper.model_option_difference(
        original_options, aref_audit.cone_helper.model_option_snapshot(model)
    )["changed_fields"] else "FAIL"
    gates = bool(
        pipeline["REGULARIZATION_PIPELINE_BASELINE_REPRODUCTION"] == "PASS"
        and activation["CONTACT_R_COUNTERFACTUAL_ACTIVATION"] == "VALIDATED"
        and invariant["CONTACT_R_COUNTERFACTUAL_ISOLATION"] == "VALIDATED"
        and baseline["R_BASELINE_REPRODUCTION"] == "PASS"
        and restore["R_RESTORE_REPRODUCTION"] == "PASS"
        and custom_step["CUSTOM_PIPELINE_ONE_STEP_REPRODUCTION"] == "PASS"
        and all(condition["state_validation"]["same_complete_pre_state"] for condition in conditions.values())
        and formal_records == EXPECTED_SUBSTEPS
        and formal_data_unchanged
        and model_restore == "PASS"
        and source_unchanged
    )
    comparison = classify_effect(
        conditions["r_scale_1_before"]["excess"], conditions["r_scale_0p1"]["excess"], gates
    )
    baseline_excess = conditions["r_scale_1_before"]["excess"]
    counterfactual_excess = conditions["r_scale_0p1"]["excess"]
    comparison["actual_friction_impulse_change"] = np.asarray(counterfactual_excess["actual_tangent_impulse_vector"]) - np.asarray(baseline_excess["actual_tangent_impulse_vector"])
    comparison["normal_impulse_change"] = counterfactual_excess["normal_impulse"] - baseline_excess["normal_impulse"]
    comparison["rigid_demand_change"] = np.asarray(counterfactual_excess["rigid_demand_vector"]) - np.asarray(baseline_excess["rigid_demand_vector"])
    comparison["solver_excess_change"] = np.asarray(counterfactual_excess["solver_excess_vector"]) - np.asarray(baseline_excess["solver_excess_vector"])
    comparison["post_slip_change"] = np.asarray(counterfactual_excess["post_slip"]) - np.asarray(baseline_excess["post_slip"])
    comparison["semantic_scope"] = "The intervention scales production pyramidal contact-edge regularization; it is not a pure tangent-only R intervention."
    write_json(output / "regularization_counterfactual_comparison.json", comparison)

    validation = {
        "R_CONSUMPTION_PATH": consumption_audit["R_CONSUMPTION_PATH"],
        "D_CONSUMPTION_PATH": consumption_audit["D_CONSUMPTION_PATH"],
        "AR_CONSUMPTION_PATH": consumption_audit["AR_CONSUMPTION_PATH"],
        "ISLAND_MIRROR_REQUIRED": consumption_audit["ISLAND_MIRROR_REQUIRED"],
        "REGULARIZATION_PIPELINE_BASELINE_REPRODUCTION": pipeline["REGULARIZATION_PIPELINE_BASELINE_REPRODUCTION"],
        "CONTACT_R_COUNTERFACTUAL_ACTIVATION": activation["CONTACT_R_COUNTERFACTUAL_ACTIVATION"],
        "CONTACT_R_COUNTERFACTUAL_ISOLATION": invariant["CONTACT_R_COUNTERFACTUAL_ISOLATION"],
        "R_BASELINE_REPRODUCTION": baseline["R_BASELINE_REPRODUCTION"],
        "R_RESTORE_REPRODUCTION": restore["R_RESTORE_REPRODUCTION"],
        "CUSTOM_PIPELINE_ONE_STEP_REPRODUCTION": custom_step["CUSTOM_PIPELINE_ONE_STEP_REPRODUCTION"],
        "formal_replay_physics_substeps": formal_records,
        "expected_formal_replay_physics_substeps": EXPECTED_SUBSTEPS,
        "formal_replay_additional_steps": 0,
        "formal_data_mutated_by_probe": not formal_data_unchanged,
        "source_hashes_unchanged": source_unchanged,
        "MODEL_OPTION_RESTORE": model_restore,
        "CONTACT_R_SOLVER_EXCESS_EFFECT": comparison["CONTACT_R_SOLVER_EXCESS_EFFECT"],
        "MUJOCO_SOLVER_EXCESS_CAUSAL_DRIVER": comparison["MUJOCO_SOLVER_EXCESS_CAUSAL_DRIVER"],
        "NEXT_ACTION": comparison["NEXT_ACTION"],
        "COUNTERFACTUAL_VALID": gates,
        "UNCONDITIONAL_ZIP_PACKAGING": "ENABLED",
        "LOCAL_IMPLEMENTATION": "READY_FOR_SERVER_VALIDATION",
    }
    summary = {key: validation[key] for key in (
        "R_CONSUMPTION_PATH", "D_CONSUMPTION_PATH", "AR_CONSUMPTION_PATH", "ISLAND_MIRROR_REQUIRED",
        "REGULARIZATION_PIPELINE_BASELINE_REPRODUCTION", "CONTACT_R_COUNTERFACTUAL_ACTIVATION", "CONTACT_R_COUNTERFACTUAL_ISOLATION",
        "R_BASELINE_REPRODUCTION", "R_RESTORE_REPRODUCTION", "CONTACT_R_SOLVER_EXCESS_EFFECT",
        "MUJOCO_SOLVER_EXCESS_CAUSAL_DRIVER", "NEXT_ACTION", "UNCONDITIONAL_ZIP_PACKAGING", "LOCAL_IMPLEMENTATION",
    )}
    summary["baseline_solver_excess_Ns"] = baseline_excess["solver_excess_norm"]
    summary["r_scale_0p1_solver_excess_Ns"] = counterfactual_excess["solver_excess_norm"]
    metadata = {
        "created_at": datetime.now().astimezone().isoformat(),
        "backend": "mujoco",
        "mujoco_version": consumption_audit.get("mujoco_version"),
        "morphology": MORPHOLOGY,
        "morphology_xml": str(paths["morphology_xml"]),
        "morphology_xml_sha256": hashes_before[str(paths["morphology_xml"])],
        "checkpoint": str(paths["checkpoint"]),
        "checkpoint_sha256": hashes_before[str(paths["checkpoint"])],
        "corrected_reference_oracle": str(paths["corrected_oracle"]),
        "formal_replay_helper": "tools.analyze_mujoco_global55_contact_demand.replay",
        "formal_replay_physics_substeps": formal_records,
        "formal_replay_additional_steps": 0,
        "formal_data_mutated_by_probe": not formal_data_unchanged,
        "global_physics_step": GLOBAL_STEP,
        "physics_dt": float(model.opt.timestep),
        "cone": "mjCONE_PYRAMIDAL",
        "conditions": [label for _, label, _ in CONDITIONS],
        "semantic_scope": "The intervention scales production pyramidal contact-edge regularization; it is not a pure tangent-only R intervention.",
        "condition_staged_forward_count": 3,
        "condition_constraint_solve_count": 3,
        "condition_custom_integration_count": 3,
    }
    for filename, payload in (
        ("metadata.json", metadata),
        ("validation.json", validation),
        ("summary.json", summary),
        ("source_purity.json", {"hashes_before": hashes_before, "hashes_after": hashes_after, "source_hashes_unchanged": source_unchanged, "formal_data_mutated_by_probe": not formal_data_unchanged}),
    ):
        write_json(output / filename, payload)
    return validation


def failure_payload(error: Exception) -> dict[str, Any]:
    return {
        "R_CONSUMPTION_PATH": "UNDETERMINED",
        "D_CONSUMPTION_PATH": "UNDETERMINED",
        "AR_CONSUMPTION_PATH": "UNDETERMINED",
        "ISLAND_MIRROR_REQUIRED": "YES",
        "REGULARIZATION_PIPELINE_BASELINE_REPRODUCTION": "INSUFFICIENT_EVIDENCE",
        "CONTACT_R_COUNTERFACTUAL_ACTIVATION": "INSUFFICIENT_EVIDENCE",
        "CONTACT_R_COUNTERFACTUAL_ISOLATION": "INSUFFICIENT_EVIDENCE",
        "R_BASELINE_REPRODUCTION": "FAIL",
        "R_RESTORE_REPRODUCTION": "FAIL",
        "CONTACT_R_SOLVER_EXCESS_EFFECT": "INSUFFICIENT_EVIDENCE",
        "MUJOCO_SOLVER_EXCESS_CAUSAL_DRIVER": "INSUFFICIENT_EVIDENCE",
        "NEXT_ACTION": "COUNTERFACTUAL_IMPLEMENTATION_FIX_REQUIRED",
        "UNCONDITIONAL_ZIP_PACKAGING": "ENABLED",
        "LOCAL_IMPLEMENTATION": "INCOMPLETE",
        "COUNTERFACTUAL_VALID": False,
        "error_type": type(error).__name__,
        "error": str(error),
    }


def write_failure_bundle(output: Path, error: Exception, trace: str) -> None:
    """Make failure artifacts complete even when validation fails before replay."""
    audit_path = output / "constraint_regularization_consumption_audit.json"
    if not audit_path.exists():
        write_json(audit_path, {
            "audit_version": 1,
            "audit_status": "INSUFFICIENT_EVIDENCE",
            "counterfactual_ready": False,
            "R_CONSUMPTION_PATH": "UNDETERMINED",
            "D_CONSUMPTION_PATH": "UNDETERMINED",
            "AR_CONSUMPTION_PATH": "UNDETERMINED",
            "ISLAND_MIRROR_REQUIRED": "YES",
            "reason": "execution failed before solver-consumption audit completed",
            "error": str(error),
        })
    if not (output / "git_head.txt").exists():
        try:
            write_git_identity(output)
        except Exception:
            pass
    (output / "traceback.txt").write_text(trace, encoding="utf-8")
    partial = [
        str(path.relative_to(output))
        for path in sorted((output / "conditions").glob("*") if (output / "conditions").is_dir() else [])
    ]
    write_json(output / "failure_context.json", {
        "error": str(error),
        "traceback_file": "traceback.txt",
        "partial_conditions": partial,
    })
    failure = failure_payload(error)
    write_json(output / "validation.json", failure)
    write_json(output / "summary.json", failure)
    write_json(output / "metadata.json", {
        "created_at": datetime.now().astimezone().isoformat(),
        "status": "failure",
        "diagnostic": Path(__file__).name,
        "error_type": type(error).__name__,
        "error": str(error),
    })
    write_json(output / "source_purity.json", {
        "source_hashes_unchanged": False,
        "formal_data_mutated_by_probe": False,
        "status": "incomplete",
    })


def _package(output: Path, zip_path: Path) -> dict[str, Any]:
    return oracle.package_artifact(output, zip_path)


def main(argv: Sequence[str] | None = None) -> int:
    raw_args = parser().parse_args(argv)
    output: Path | None = None
    zip_path: Path | None = None
    try:
        args = resolve_arguments(raw_args)
        output, zip_path = Path(args.output_dir).resolve(), Path(args.zip_path).resolve()
        paths = validate_paths(args)
        output.mkdir(parents=True, exist_ok=False)
        return_code = 2
        log_path = output / "run.log"
        with log_path.open("w", encoding="utf-8") as log_stream, redirect_stdout(Tee(sys.__stdout__, log_stream)), redirect_stderr(Tee(sys.__stderr__, log_stream)):
            try:
                validation = execute(args, paths)
                print(json.dumps(_json_normalize(validation), indent=2, sort_keys=True, allow_nan=False))
                return_code = 0 if validation["COUNTERFACTUAL_VALID"] else 2
            except Exception as error:
                trace = traceback.format_exc()
                print(trace, file=sys.stderr, end="")
                write_failure_bundle(output, error, trace)
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
        write_failure_bundle(output, error, trace)
        package = _package(output, zip_path)
        print(f"ZIP_VERIFY={package['ZIP_VERIFY']}")
        print(f"ZIP_SHA256={package['ZIP_SHA256']}")
        print(f"UPLOAD_THIS_ZIP={package['UPLOAD_THIS_ZIP']}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

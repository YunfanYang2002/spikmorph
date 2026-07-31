"""Diagnostics-only final-task-XML pair injection wrapper."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import runpy
import sys
from typing import Any, Sequence
import xml.etree.ElementTree as ET


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

TARGET_GEOMS = ("limb/11", "limb/12")
FLOOR_GEOM = "floor/0"


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--target-mu", type=float, required=True)
    result.add_argument("--pair-spec", required=True)
    result.add_argument("--audit-output", required=True)
    result.add_argument("--final-xml-output", required=True)
    result.add_argument("evaluator_args", nargs=argparse.REMAINDER)
    return result


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def format_values(values: Sequence[float]) -> str:
    return " ".join(format(float(value), ".17g") for value in values)


def pair_key(geom1: str | None, geom2: str | None) -> frozenset[str | None]:
    return frozenset((geom1, geom2))


def inject_final_pairs(
    xml_string: str, target_mu: float, production: dict[str, Any]
) -> tuple[str, dict[str, Any]]:
    root = ET.fromstring(xml_string)
    geom_names = {
        geom.get("name") for geom in root.findall(".//geom") if geom.get("name")
    }
    required = {FLOOR_GEOM, *TARGET_GEOMS}
    missing = sorted(required - geom_names)
    contact = root.find("contact")
    if contact is None:
        contact = ET.SubElement(root, "contact")
    selected_keys = {pair_key(target, FLOOR_GEOM) for target in TARGET_GEOMS}
    removed = []
    for pair in list(contact.findall("pair")):
        if pair_key(pair.get("geom1"), pair.get("geom2")) in selected_keys:
            removed.append(dict(pair.attrib))
            contact.remove(pair)
    friction = [
        float(target_mu), float(target_mu),
        *[float(value) for value in production["friction"][2:5]],
    ]
    added = []
    for target in TARGET_GEOMS:
        attributes = {
            "geom1": target,
            "geom2": FLOOR_GEOM,
            "condim": str(int(production["dim"])),
            "friction": format_values(friction),
            "solref": format_values(production["solref"]),
            "solimp": format_values(production["solimp"]),
            "margin": format(float(production["includemargin"]), ".17g"),
            "gap": "0",
        }
        ET.SubElement(contact, "pair", attributes)
        added.append(attributes)
    modified = ET.tostring(root, encoding="unicode")
    final_root = ET.fromstring(modified)
    final_pairs = [dict(pair.attrib) for pair in final_root.findall("./contact/pair")]
    target_counts = {
        target: sum(
            pair_key(pair.get("geom1"), pair.get("geom2"))
            == pair_key(target, FLOOR_GEOM)
            for pair in final_root.findall("./contact/pair")
        )
        for target in TARGET_GEOMS
    }
    limb0_explicit = any(
        pair_key(pair.get("geom1"), pair.get("geom2"))
        == pair_key("limb/0", FLOOR_GEOM)
        for pair in final_root.findall("./contact/pair")
    )
    dom_valid = (
        not missing
        and all(count == 1 for count in target_counts.values())
        and not limb0_explicit
        and all(
            [float(value) for value in pair["friction"].split()][:2]
            == [float(target_mu), float(target_mu)]
            for pair in final_pairs
            if pair_key(pair.get("geom1"), pair.get("geom2")) in selected_keys
        )
    )
    return modified, {
        "final_xml_before_injection_sha256": text_sha256(xml_string),
        "final_xml_after_injection_sha256": text_sha256(modified),
        "final_xml_before_length": len(xml_string),
        "final_xml_after_length": len(modified),
        "required_geom_presence": {
            name: name in geom_names for name in sorted(required)
        },
        "missing_required_geoms": missing,
        "removed_selected_pairs": removed,
        "added_selected_pairs": added,
        "target_pair_counts": target_counts,
        "limb_0_floor_is_explicit_pair": limb0_explicit,
        "final_contact_pairs": final_pairs,
        "final_full_xml_pair_dom_valid": dom_valid,
    }


def compiled_pair_table(mujoco: Any, model: Any) -> dict[str, Any]:
    rows = []
    for pair_id in range(int(model.npair)):
        geom1 = int(model.pair_geom1[pair_id])
        geom2 = int(model.pair_geom2[pair_id])
        rows.append({
            "pair_id": pair_id,
            "geom1_id": geom1,
            "geom1_name": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom1),
            "geom2_id": geom2,
            "geom2_name": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom2),
            "dim": int(model.pair_dim[pair_id]),
            "friction": [float(value) for value in model.pair_friction[pair_id]],
            "solref": [float(value) for value in model.pair_solref[pair_id]],
            "solimp": [float(value) for value in model.pair_solimp[pair_id]],
            "margin": float(model.pair_margin[pair_id]),
            "gap": float(model.pair_gap[pair_id]),
        })
    return {"npair": int(model.npair), "pairs": rows}


def compiled_pair_status(
    table: dict[str, Any], target_mu: float, production: dict[str, Any]
) -> str:
    wrong = False
    for target in TARGET_GEOMS:
        matches = [
            pair for pair in table["pairs"]
            if pair_key(pair["geom1_name"], pair["geom2_name"])
            == pair_key(target, FLOOR_GEOM)
        ]
        if not matches:
            return "MISSING"
        if len(matches) != 1:
            wrong = True
            continue
        pair = matches[0]
        wrong |= pair["dim"] != int(production["dim"])
        wrong |= pair["friction"][:2] != [float(target_mu), float(target_mu)]
        wrong |= pair["friction"][2:5] != [float(value) for value in production["friction"][2:5]]
        wrong |= pair["solref"] != [float(value) for value in production["solref"]]
        wrong |= pair["solimp"] != [float(value) for value in production["solimp"]]
        wrong |= pair["margin"] != float(production["includemargin"])
        wrong |= pair["gap"] != 0.0
    limb0_pair = any(
        pair_key(pair["geom1_name"], pair["geom2_name"])
        == pair_key("limb/0", FLOOR_GEOM)
        for pair in table["pairs"]
    )
    wrong |= limb0_pair
    return "WRONG_VALUES" if wrong else "PRESENT_WITH_TARGET_VALUES"


def write_audit(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


@contextmanager
def temporary_loader_wrapper(module: Any, wrapper: Any):
    original = module.load_model_from_xml
    module.load_model_from_xml = wrapper
    try:
        yield original
    finally:
        module.load_model_from_xml = original


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    evaluator_args = list(args.evaluator_args)
    if evaluator_args and evaluator_args[0] == "--":
        evaluator_args = evaluator_args[1:]
    production = json.loads(Path(args.pair_spec).read_text(encoding="utf-8"))
    audit_path = Path(args.audit_output).resolve()
    final_xml_path = Path(args.final_xml_output).resolve()
    import numpy as np
    if "bool" not in np.__dict__:
        np.bool = bool
    from metamorph.utils import mujoco_compat

    original_loader = mujoco_compat.load_model_from_xml
    calls = []

    def injecting_loader(xml_string: str) -> Any:
        modified, dom = inject_final_pairs(xml_string, args.target_mu, production)
        if not dom["final_full_xml_pair_dom_valid"]:
            raise RuntimeError(f"FINAL_FULL_XML_PAIR_DOM_VALID=false: {dom}")
        final_xml_path.parent.mkdir(parents=True, exist_ok=True)
        final_xml_path.write_text(modified, encoding="utf-8")
        model = original_loader(modified)
        table = compiled_pair_table(mujoco_compat.mujoco, model)
        status = compiled_pair_status(table, args.target_mu, production)
        call = {"dom": dom, "compiled_pair_table": table, "final_compiled_pair_status": status}
        calls.append(call)
        write_audit(audit_path, {
            "target_mu": args.target_mu,
            "activation": "diagnostics-only temporary load_model_from_xml wrapper",
            "loader_call_count": len(calls),
            "calls": calls,
            "wrapper_restored": False,
        })
        if status != "PRESENT_WITH_TARGET_VALUES":
            raise RuntimeError(f"FINAL_COMPILED_PAIR_STATUS={status}")
        return model

    return_code = 1
    caught_error = None
    try:
        with temporary_loader_wrapper(mujoco_compat, injecting_loader):
            sys.argv = ["tools/evaluate_mujoco_checkpoint.py", *evaluator_args]
            try:
                runpy.run_path("tools/evaluate_mujoco_checkpoint.py", run_name="__main__")
                return_code = 0
            except SystemExit as error:
                return_code = int(error.code or 0)
    except Exception as error:
        caught_error = error
    finally:
        restored = mujoco_compat.load_model_from_xml is original_loader
        write_audit(audit_path, {
            "target_mu": args.target_mu,
            "activation": "diagnostics-only temporary load_model_from_xml wrapper",
            "loader_call_count": len(calls),
            "calls": calls,
            "wrapper_restored": restored,
            "evaluator_return_code": return_code,
            "error": repr(caught_error) if caught_error is not None else None,
        })
        if not restored:
            raise RuntimeError("diagnostics loader wrapper was not restored")
    if caught_error is not None:
        raise caught_error
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())

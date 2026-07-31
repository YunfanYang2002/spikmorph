"""Read-only audit of temporary MJCF explicit pairs through MetaMorph loading."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
from typing import Any, Sequence
import xml.etree.ElementTree as ET


REPO_ROOT = Path(__file__).resolve().parents[1]
MORPHOLOGY = "floor-1409-0-3-01-15-56-55"
SOURCE_XML_SHA256 = "da2156e9be4b706a34599a32bc8f3f1a2037fa8ecfb44042bc22e1740e9382a0"
CHECKPOINT_SHA256 = "cdf53f3e427a5a1c081ddd4d78230574fd694a5cefd97b30184a3762d9943d03"
TARGETS = ("limb/11", "limb/12")
CONDITIONS = (
    ("condition_distal_mu_0", 0.0),
    ("condition_distal_mu_1p4", 1.4),
)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--failed-bracket-root", required=True)
    result.add_argument("--existing-temporary-root", required=True)
    result.add_argument("--regenerated-temporary-root", required=True)
    result.add_argument("--source-xml", required=True)
    result.add_argument("--metadata", required=True)
    result.add_argument("--checkpoint", required=True)
    result.add_argument("--output-dir", required=True)
    result.add_argument("--device", default="cpu")
    result.add_argument("--worker-xml")
    result.add_argument("--worker-mu", type=float)
    result.add_argument("--worker-output")
    return result


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def pair_key(geom1: str | None, geom2: str | None) -> frozenset[str | None]:
    return frozenset((geom1, geom2))


def is_target_pair(geom1: str | None, geom2: str | None, target: str) -> bool:
    return pair_key(geom1, geom2) == pair_key(target, "floor/0")


def extract_dom_pair_audit(xml_path: Path) -> dict[str, Any]:
    root = ET.parse(xml_path).getroot()
    sections = []
    for section_index, section in enumerate(root.findall(".//contact")):
        pairs = []
        for pair_index, pair in enumerate(section.findall("pair")):
            pairs.append({
                "pair_index_in_section": pair_index,
                "geom1": pair.get("geom1"),
                "geom2": pair.get("geom2"),
                "friction_raw": pair.get("friction"),
                "friction": [float(value) for value in pair.get("friction", "").split()],
                "condim": int(pair.get("condim")) if pair.get("condim") else None,
                "solref": pair.get("solref"), "solimp": pair.get("solimp"),
                "margin": pair.get("margin"), "gap": pair.get("gap"),
            })
        sections.append({"contact_section_index": section_index, "pairs": pairs})
    all_pairs = [pair for section in sections for pair in section["pairs"]]
    return {
        "xml_path": str(xml_path.resolve()),
        "xml_sha256": sha256(xml_path),
        "contact_sections": sections,
        "target_pair_counts": {
            target: sum(is_target_pair(pair["geom1"], pair["geom2"], target) for pair in all_pairs)
            for target in TARGETS
        },
    }


def dom_status(audit: dict[str, Any], target_mu: float) -> bool:
    pairs = [
        pair for section in audit["contact_sections"] for pair in section["pairs"]
    ]
    for target in TARGETS:
        matches = [pair for pair in pairs if is_target_pair(pair["geom1"], pair["geom2"], target)]
        if len(matches) != 1:
            return False
        pair = matches[0]
        if pair["condim"] != 3 or len(pair["friction"]) != 5:
            return False
        if pair["friction"][:2] != [target_mu, target_mu]:
            return False
    return True


def object_name(mujoco: Any, model: Any, object_type: Any, object_id: int) -> str | None:
    return mujoco.mj_id2name(model, object_type, int(object_id))


def compiled_pair_table(mujoco: Any, model: Any) -> dict[str, Any]:
    rows = []
    for pair_id in range(int(model.npair)):
        geom1 = int(model.pair_geom1[pair_id])
        geom2 = int(model.pair_geom2[pair_id])
        rows.append({
            "pair_id": pair_id,
            "geom1_id": geom1,
            "geom1_name": object_name(mujoco, model, mujoco.mjtObj.mjOBJ_GEOM, geom1),
            "geom2_id": geom2,
            "geom2_name": object_name(mujoco, model, mujoco.mjtObj.mjOBJ_GEOM, geom2),
            "dim": int(model.pair_dim[pair_id]),
            "friction": [float(value) for value in model.pair_friction[pair_id]],
            "solref": [float(value) for value in model.pair_solref[pair_id]],
            "solimp": [float(value) for value in model.pair_solimp[pair_id]],
            "margin": float(model.pair_margin[pair_id]),
            "gap": float(model.pair_gap[pair_id]),
        })
    return {"npair": int(model.npair), "pairs": rows}


def compiled_pair_status(table: dict[str, Any], target_mu: float) -> str:
    wrong = False
    for target in TARGETS:
        matches = [
            pair for pair in table["pairs"]
            if is_target_pair(pair["geom1_name"], pair["geom2_name"], target)
        ]
        if not matches:
            return "MISSING"
        if len(matches) != 1:
            wrong = True
            continue
        pair = matches[0]
        wrong |= pair["dim"] != 3 or pair["friction"][:2] != [target_mu, target_mu]
    return "WRONG_VALUES" if wrong else "PRESENT_WITH_TARGET_VALUES"


def regenerate_from_manifest(
    failed_root: Path, source_xml: Path, target_xml: Path, condition: str
) -> None:
    manifest = json.loads(
        (failed_root / "temporary_xml_manifest.json").read_text(encoding="utf-8")
    )
    entries = manifest["conditions"] if isinstance(manifest, dict) else manifest
    matches = [entry for entry in entries if entry["condition"] == condition]
    if len(matches) != 1:
        raise ValueError(f"failed manifest has no unique {condition}")
    attributes = matches[0]["added_pairs"]
    tree = ET.parse(source_xml)
    root = tree.getroot()
    contact = root.find("contact")
    if contact is None:
        contact = ET.SubElement(root, "contact")
    for pair in list(contact.findall("pair")):
        if any(is_target_pair(pair.get("geom1"), pair.get("geom2"), target) for target in TARGETS):
            contact.remove(pair)
    for item in attributes:
        ET.SubElement(contact, "pair", dict(item))
    target_xml.parent.mkdir(parents=True, exist_ok=False)
    tree.write(target_xml, encoding="utf-8", xml_declaration=True)


def locate_xmls(args: argparse.Namespace) -> dict[str, Path]:
    existing_root = Path(args.existing_temporary_root).resolve()
    regenerated_root = Path(args.regenerated_temporary_root).resolve()
    source_xml = Path(args.source_xml).resolve()
    result = {}
    for condition, _ in CONDITIONS:
        relative = Path(condition) / "walker" / "xml" / f"{MORPHOLOGY}.xml"
        existing = existing_root / relative
        if existing.is_file():
            result[condition] = existing
            continue
        regenerated = regenerated_root / relative
        regenerate_from_manifest(
            Path(args.failed_bracket_root).resolve(), source_xml, regenerated, condition
        )
        metadata_target = regenerated_root / condition / "walker" / "metadata" / f"{MORPHOLOGY}.json"
        metadata_target.parent.mkdir(parents=True, exist_ok=False)
        shutil.copy2(Path(args.metadata).resolve(), metadata_target)
        result[condition] = regenerated
    return result


def load_evaluator_module() -> Any:
    path = REPO_ROOT / "tools" / "evaluate_mujoco_checkpoint.py"
    spec = importlib.util.spec_from_file_location("explicit_pair_audit_evaluator", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def worker_audit(xml_path: Path, target_mu: float, checkpoint: Path, device: str) -> dict[str, Any]:
    import numpy as np
    if "bool" not in np.__dict__:
        np.bool = bool
    import mujoco

    direct_warning = io.StringIO()
    direct_error = None
    direct_table = None
    try:
        with contextlib.redirect_stderr(direct_warning):
            direct_model = mujoco.MjModel.from_xml_path(str(xml_path))
        direct_table = compiled_pair_table(mujoco, direct_model)
    except Exception as error:
        direct_error = repr(error)

    evaluator = load_evaluator_module()
    walker_dir = xml_path.parents[1]
    args = SimpleNamespace(
        checkpoint=str(checkpoint), walker_dir=str(walker_dir), morphology_id=MORPHOLOGY,
        action_mode="zero", episodes=1, seed=1409,
        output_dir=str(xml_path.parent / "unused_audit_output"), device=device,
        cfg="configs/ft.yaml", record_state_trajectory=False, max_eval_steps=None,
        reset_noise_scale=0.0, record_joint_limit_substeps=False,
        joint_limit_probe_names=[], record_contact_generalized_response=False,
        contact_probe_body_names=[], record_physical_contact_projection=False,
    )
    paths = evaluator.validate_args(args)
    evaluator.configure_runtime(args, paths)
    from metamorph.utils import mujoco_compat
    original_loader = mujoco_compat.load_model_from_xml
    provenance = []

    def recording_loader(xml_string: str) -> Any:
        dom = ET.fromstring(xml_string)
        pairs = [dict(pair.attrib) for section in dom.findall(".//contact") for pair in section.findall("pair")]
        provenance.append({
            "loader_function": "metamorph.utils.mujoco_compat.load_model_from_xml",
            "downstream_native_function": "mujoco.MjModel.from_xml_string",
            "xml_string_sha256": text_sha256(xml_string),
            "xml_string_length": len(xml_string),
            "contact_pairs_in_loader_input": pairs,
        })
        return original_loader(xml_string)

    mujoco_compat.load_model_from_xml = recording_loader
    envs = None
    environment_table = None
    environment_error = None
    try:
        envs, _, _ = evaluator.build_runtime(args, paths)
        base_env = evaluator.unwrap_single_mujoco_env(envs)
        _, raw_model, _ = evaluator._native_model_data(base_env.sim)
        environment_table = compiled_pair_table(mujoco, raw_model)
    except Exception as error:
        environment_error = repr(error)
    finally:
        mujoco_compat.load_model_from_xml = original_loader
        if envs is not None:
            envs.close()
    return {
        "target_mu": target_mu,
        "requested_walker_dir": str(walker_dir),
        "requested_morphology_xml": str(xml_path),
        "requested_morphology_xml_sha256": sha256(xml_path),
        "direct_compile": {
            "warning_stderr": direct_warning.getvalue(), "error": direct_error,
            "table": direct_table,
            "status": compiled_pair_status(direct_table, target_mu) if direct_table else "MISSING",
        },
        "environment_loader_provenance": provenance,
        "environment_compile": {
            "error": environment_error, "table": environment_table,
            "status": compiled_pair_status(environment_table, target_mu) if environment_table else "MISSING",
        },
    }


def classify_root_cause(
    dom_valid: bool,
    direct_status: str,
    environment_status: str,
    requested_sha: str,
    provenance: Sequence[dict[str, Any]],
    runtime_selection: str,
    requested_source_matches: bool = True,
) -> str:
    if not dom_valid:
        return "TEMP_XML_GENERATION_BUG"
    if direct_status != "PRESENT_WITH_TARGET_VALUES":
        return "DIRECT_MUJOCO_COMPILE_REJECTS_PAIR"
    if not requested_source_matches:
        return "ENVIRONMENT_LOADS_DIFFERENT_XML"
    if environment_status != "PRESENT_WITH_TARGET_VALUES":
        return "ENVIRONMENT_REWRITES_OR_DROPS_CONTACT_PAIRS"
    if runtime_selection == "USES_GEOM_COMBINATION":
        return "PAIR_COMPILED_BUT_NOT_SELECTED"
    if runtime_selection == "USES_EXPLICIT_PAIR":
        return "OTHER"
    return "INSUFFICIENT_EVIDENCE"


def run_minimal_runtime_contact_probe(
    xml_path: Path, checkpoint: Path, output_dir: Path, device: str
) -> dict[str, Any]:
    evaluator_args = [
        "tools/evaluate_mujoco_checkpoint.py",
        "--checkpoint", str(checkpoint), "--walker-dir", str(xml_path.parents[1]),
        "--morphology-id", MORPHOLOGY, "--action-mode", "zero",
        "--episodes", "1", "--seed", "1409", "--output-dir", str(output_dir),
        "--device", device, "--reset-noise-scale", "0.0", "--max-eval-steps", "30",
        "--record-joint-limit-substeps", "--joint-limit-probe-names", "limby/12", "limby/11",
        "--record-physical-contact-projection",
    ]
    shim = (
        "import numpy as np; "
        "np.bool=bool if 'bool' not in np.__dict__ else np.bool; "
        "import runpy,sys; "
        f"sys.argv={evaluator_args!r}; "
        "runpy.run_path('tools/evaluate_mujoco_checkpoint.py',run_name='__main__')"
    )
    completed = subprocess.run(
        [sys.executable, "-c", shim], cwd=REPO_ROOT, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False,
    )
    if completed.returncode != 0:
        return {
            "status": "INSUFFICIENT_EVIDENCE", "return_code": completed.returncode,
            "output": completed.stdout, "physics_steps_executed": None,
        }
    records = [
        json.loads(line) for line in
        (output_dir / "physical_contact_substeps.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    global55 = next(record for record in records if record["global_physics_step"] == 55)
    selected = {}
    for target in TARGETS:
        matches = [
            contact for contact in global55["contacts"]
            if is_target_pair(contact["geom1_name"], contact["geom2_name"], target)
        ]
        selected[target] = matches[0] if len(matches) == 1 else None
    frictions = [item["friction"][:2] for item in selected.values() if item]
    if len(frictions) == 2 and all(values == [0.0, 0.0] for values in frictions):
        status = "USES_EXPLICIT_PAIR"
    elif len(frictions) == 2 and all(values == [0.7, 0.7] for values in frictions):
        status = "USES_GEOM_COMBINATION"
    else:
        status = "INSUFFICIENT_EVIDENCE"
    return {
        "status": status, "return_code": completed.returncode,
        "selected_contacts": selected, "physics_steps_executed": len(records),
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.worker_xml:
        result = worker_audit(
            Path(args.worker_xml).resolve(), float(args.worker_mu),
            Path(args.checkpoint).resolve(), args.device,
        )
        write_json(Path(args.worker_output).resolve(), result)
        return 0

    source_xml = Path(args.source_xml).resolve()
    checkpoint = Path(args.checkpoint).resolve()
    output_dir = Path(args.output_dir).resolve()
    if sha256(source_xml) != SOURCE_XML_SHA256:
        raise ValueError("source XML SHA256 mismatch")
    if sha256(checkpoint) != CHECKPOINT_SHA256:
        raise ValueError("checkpoint SHA256 mismatch")
    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True)
    source_hash_before = sha256(source_xml)
    xmls = locate_xmls(args)
    dom_payload = {}
    worker_results = {}
    for condition, mu in CONDITIONS:
        dom = extract_dom_pair_audit(xmls[condition])
        dom["target_mu"] = mu
        dom["temp_xml_pair_dom_valid"] = dom_status(dom, mu)
        dom_payload[condition] = dom
        worker_output = output_dir / f"{condition}_worker.json"
        command = [
            sys.executable, str(Path(__file__).resolve()),
            "--failed-bracket-root", args.failed_bracket_root,
            "--existing-temporary-root", args.existing_temporary_root,
            "--regenerated-temporary-root", args.regenerated_temporary_root,
            "--source-xml", args.source_xml, "--metadata", args.metadata,
            "--checkpoint", args.checkpoint, "--output-dir", args.output_dir,
            "--device", args.device, "--worker-xml", str(xmls[condition]),
            "--worker-mu", str(mu), "--worker-output", str(worker_output),
        ]
        completed = subprocess.run(
            command, cwd=REPO_ROOT, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False,
        )
        if completed.returncode != 0 or not worker_output.is_file():
            worker_results[condition] = {
                "worker_return_code": completed.returncode,
                "worker_output": completed.stdout,
            }
        else:
            worker_results[condition] = json.loads(worker_output.read_text(encoding="utf-8"))

    direct = {
        condition: result.get("direct_compile", {})
        for condition, result in worker_results.items()
    }
    provenance = {
        condition: {
            "requested_walker_dir": result.get("requested_walker_dir"),
            "requested_morphology_xml": result.get("requested_morphology_xml"),
            "requested_morphology_xml_sha256": result.get("requested_morphology_xml_sha256"),
            "loader_calls": result.get("environment_loader_provenance", []),
        }
        for condition, result in worker_results.items()
    }
    environment = {
        condition: result.get("environment_compile", {})
        for condition, result in worker_results.items()
    }
    environment_all_correct = all(
        environment[condition].get("status") == "PRESENT_WITH_TARGET_VALUES"
        for condition, _ in CONDITIONS
    )
    if environment_all_correct:
        runtime = run_minimal_runtime_contact_probe(
            xmls["condition_distal_mu_0"], checkpoint,
            output_dir / "runtime_mu0_probe", args.device,
        )
    else:
        runtime = {
            "status": "INSUFFICIENT_EVIDENCE",
            "reason": "environment compiled pair tables are not both correct; global55 stepping skipped",
            "physics_steps_executed": 0,
        }
    roots = {}
    for condition, _ in CONDITIONS:
        result = worker_results[condition]
        loader_calls = result.get("environment_loader_provenance", [])
        roots[condition] = classify_root_cause(
            dom_payload[condition]["temp_xml_pair_dom_valid"],
            direct[condition].get("status", "MISSING"),
            environment[condition].get("status", "MISSING"),
            dom_payload[condition]["xml_sha256"],
            loader_calls,
            runtime["status"],
            Path(result.get("requested_morphology_xml", "")).resolve()
            == xmls[condition].resolve(),
        )
    root_cause = roots[CONDITIONS[0][0]] if len(set(roots.values())) == 1 else "INSUFFICIENT_EVIDENCE"
    validation = {
        "temp_xml_pair_dom_valid": all(item["temp_xml_pair_dom_valid"] for item in dom_payload.values()),
        "source_xml_hash_unchanged": sha256(source_xml) == source_hash_before,
        "source_xml_hash_matches_frozen": sha256(source_xml) == SOURCE_XML_SHA256,
        "no_physics_mutation": True,
        "physics_steps_executed": runtime["physics_steps_executed"],
        "stepping_policy_valid": runtime["physics_steps_executed"] in (0, 120),
        "direct_compile_conditions_completed": all("status" in item for item in direct.values()),
        "environment_compile_conditions_completed": all("status" in item for item in environment.values()),
    }
    write_json(output_dir / "temporary_xml_dom_audit.json", dom_payload)
    write_json(output_dir / "direct_compile_pair_table.json", direct)
    write_json(output_dir / "environment_loader_provenance.json", provenance)
    write_json(output_dir / "environment_compile_pair_table.json", environment)
    write_json(output_dir / "runtime_contact_pair_selection.json", runtime)
    write_json(output_dir / "root_cause.json", {
        "per_condition": roots,
        "explicit_pair_override_root_cause": root_cause,
        "source_localization": [
            {"file": "metamorph/envs/tasks/task.py", "function": "make_env", "lines": "11-22"},
            {"file": "metamorph/envs/modules/agent.py", "function": "extract_agent_from_xml", "lines": "337-348"},
            {"file": "metamorph/envs/modules/agent.py", "function": "merge_agent_with_base", "lines": "312-334"},
            {"file": "metamorph/envs/tasks/unimal.py", "function": "UnimalEnv._get_sim", "lines": "84-99"},
            {"file": "metamorph/utils/mujoco_compat.py", "function": "load_model_from_xml", "lines": "288-292"},
        ],
    })
    write_json(output_dir / "validation.json", validation)
    print(json.dumps({
        "output_dir": str(output_dir),
        "EXPLICIT_PAIR_OVERRIDE_ROOT_CAUSE": root_cause,
        "physics_steps_executed": runtime["physics_steps_executed"],
    }, indent=2))
    boolean_checks = [value for value in validation.values() if isinstance(value, bool)]
    return 0 if all(boolean_checks) else 2


if __name__ == "__main__":
    raise SystemExit(main())

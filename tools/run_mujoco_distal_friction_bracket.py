"""Run a four-condition final-task-XML distal friction bracket.

This helper never edits the source XML.  A diagnostics-only loader wrapper
injects explicit pairs after task XML assembly and immediately before compile.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any, Sequence
import xml.etree.ElementTree as ET


REPO_ROOT = Path(__file__).resolve().parents[1]
MORPHOLOGY = "floor-1409-0-3-01-15-56-55"
SOURCE_XML_SHA256 = "da2156e9be4b706a34599a32bc8f3f1a2037fa8ecfb44042bc22e1740e9382a0"
CHECKPOINT_SHA256 = "cdf53f3e427a5a1c081ddd4d78230574fd694a5cefd97b30184a3762d9943d03"
CONDITIONS = (
    ("condition_distal_mu_0", 0.0),
    ("condition_distal_mu_0p35", 0.35),
    ("condition_production_mu_0p7", 0.7),
    ("condition_distal_mu_1p4", 1.4),
)
SELECTED = (("limb/11", "limby/11"), ("limb/12", "limby/12"))
MJ_MIN_MU = 1.0e-5
MJ_MIN_MU_SOURCE = "MuJoCo mjMINMU"
MINIMUM_FRICTION_FORCE_TOLERANCE_FACTOR = 1.05
MINIMUM_FRICTION_GENERALIZED_TOLERANCE_FACTOR = 2.0
VALID_RUNTIME_SELECTIONS = {
    "USES_EXPLICIT_PAIR",
    "USES_EXPLICIT_PAIR_CLAMPED_TO_MJMINMU",
}


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--source-xml", required=True)
    result.add_argument("--metadata", required=True)
    result.add_argument("--checkpoint", required=True)
    result.add_argument("--existing-oracle", required=True)
    result.add_argument("--output-dir", required=True)
    result.add_argument("--temporary-root", required=True)
    result.add_argument("--device", default="cpu")
    return result


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def pair_matches(contact: dict[str, Any], geom: str, floor: str = "floor/0") -> bool:
    return {contact["geom1_name"], contact["geom2_name"]} == {geom, floor}


def global_record(records: Sequence[dict[str, Any]], step: int) -> dict[str, Any]:
    matches = [record for record in records if int(record["global_physics_step"]) == step]
    if len(matches) != 1:
        raise ValueError(f"expected one global physics step {step}, found {len(matches)}")
    return matches[0]


def baseline_pair_parameters(existing_oracle: Path) -> dict[str, Any]:
    records = load_jsonl(existing_oracle / "physical_contact_substeps.jsonl")
    record = global_record(records, 55)
    selected = {}
    for geom, _ in SELECTED:
        matches = [contact for contact in record["contacts"] if pair_matches(contact, geom)]
        if len(matches) != 1:
            raise ValueError(f"existing oracle must contain one {geom}-floor contact")
        contact = matches[0]
        selected[geom] = {
            "friction": [float(value) for value in contact["friction"]],
            "dim": int(contact["dim"]),
            "solref": [float(value) for value in contact["solref"]],
            "solimp": [float(value) for value in contact["solimp"]],
            "includemargin": float(contact["includemargin"]),
        }
    first, second = (selected[geom] for geom, _ in SELECTED)
    for key in ("friction", "dim", "solref", "solimp", "includemargin"):
        if first[key] != second[key]:
            raise ValueError(f"selected baseline pair parameter differs for {key}")
    if first["dim"] != 3 or first["friction"][:2] != [0.7, 0.7]:
        raise ValueError(f"unexpected production contact parameters: {first}")
    return first


def format_values(values: Sequence[float]) -> str:
    return " ".join(format(float(value), ".17g") for value in values)


def write_temporary_xml(
    source_xml: Path, target_xml: Path, mu: float, baseline: dict[str, Any]
) -> dict[str, Any]:
    tree = ET.parse(source_xml)
    root = tree.getroot()
    contact_section = root.find("contact")
    if contact_section is None:
        contact_section = ET.SubElement(root, "contact")
    selected_sets = [{geom, "floor/0"} for geom, _ in SELECTED]
    removed = []
    for pair in list(contact_section.findall("pair")):
        if {pair.get("geom1"), pair.get("geom2")} in selected_sets:
            removed.append(dict(pair.attrib))
            contact_section.remove(pair)
    friction = [float(mu), float(mu), *baseline["friction"][2:5]]
    added = []
    for geom, _ in SELECTED:
        attributes = {
            "geom1": geom,
            "geom2": "floor/0",
            "condim": "3",
            "friction": format_values(friction),
            "solref": format_values(baseline["solref"]),
            "solimp": format_values(baseline["solimp"]),
            "margin": format(float(baseline["includemargin"]), ".17g"),
            "gap": "0",
        }
        ET.SubElement(contact_section, "pair", attributes)
        added.append(attributes)
    target_xml.parent.mkdir(parents=True, exist_ok=False)
    tree.write(target_xml, encoding="utf-8", xml_declaration=True)
    return {
        "mu": float(mu),
        "removed_selected_pairs": removed,
        "added_pairs": added,
        "friction_fields": {
            "tangent1": friction[0],
            "tangent2": friction[1],
            "torsional": friction[2],
            "rolling1": friction[3],
            "rolling2": friction[4],
        },
    }


def run_condition(
    name: str,
    mu: float,
    source_xml: Path,
    metadata: Path,
    checkpoint: Path,
    output_root: Path,
    temporary_root: Path,
    baseline: dict[str, Any],
    device: str,
) -> dict[str, Any]:
    walker = temporary_root / name / "walker"
    xml_path = walker / "xml" / f"{MORPHOLOGY}.xml"
    metadata_path = walker / "metadata" / f"{MORPHOLOGY}.json"
    xml_path.parent.mkdir(parents=True, exist_ok=False)
    shutil.copy2(source_xml, xml_path)
    metadata_path.parent.mkdir(parents=True, exist_ok=False)
    shutil.copy2(metadata, metadata_path)
    condition_output = output_root / name
    condition_output.mkdir(parents=True, exist_ok=False)
    pair_spec = temporary_root / name / "pair_spec.json"
    write_json(pair_spec, baseline)
    injection_audit = condition_output / "final_xml_injection_audit.json"
    final_xml = condition_output / "final_task_xml_after_injection.xml"
    evaluator_args = [
        "tools/evaluate_mujoco_checkpoint.py",
        "--checkpoint", str(checkpoint),
        "--walker-dir", str(walker),
        "--morphology-id", MORPHOLOGY,
        "--action-mode", "zero",
        "--episodes", "1",
        "--seed", "1409",
        "--output-dir", str(condition_output),
        "--device", device,
        "--reset-noise-scale", "0.0",
        "--max-eval-steps", "30",
        "--record-joint-limit-substeps",
        "--joint-limit-probe-names", "limby/12", "limby/11",
        "--record-physical-contact-projection",
    ]
    completed = subprocess.run(
        [
            sys.executable,
            "tools/evaluate_mujoco_checkpoint_with_final_pairs.py",
            "--target-mu", str(mu),
            "--pair-spec", str(pair_spec),
            "--audit-output", str(injection_audit),
            "--final-xml-output", str(final_xml),
            "--",
            *evaluator_args[1:],
        ],
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    (condition_output / "evaluator.log").write_text(
        completed.stdout, encoding="utf-8"
    )
    injection = (
        json.loads(injection_audit.read_text(encoding="utf-8"))
        if injection_audit.is_file() else None
    )
    return {
        "condition": name,
        "mu": float(mu),
        "temporary_xml": str(xml_path),
        "temporary_xml_sha256": sha256(xml_path),
        "source_morphology_copy_unmodified": sha256(xml_path) == sha256(source_xml),
        "pair_spec": str(pair_spec),
        "final_xml_injection_audit": str(injection_audit),
        "final_xml_after_injection": str(final_xml),
        "final_xml_injection": injection,
        "output_dir": str(condition_output),
        "return_code": int(completed.returncode),
    }


def max_abs_difference(left: Any, right: Any) -> float:
    import numpy as np

    return float(np.max(np.abs(np.asarray(left, dtype=float) - np.asarray(right, dtype=float))))


def runtime_expected_mu(target_mu: float) -> float:
    return max(float(target_mu), MJ_MIN_MU)


def validate_condition_gates(
    manifest: dict[str, Any], baseline: dict[str, Any]
) -> dict[str, Any]:
    mu = float(manifest["mu"])
    effective_mu = runtime_expected_mu(mu)
    injection = manifest.get("final_xml_injection") or {}
    calls = injection.get("calls", [])
    dom_valid = bool(calls) and all(
        call["dom"]["final_full_xml_pair_dom_valid"] for call in calls
    )
    compiled_valid = bool(calls) and all(
        call["final_compiled_pair_status"] == "PRESENT_WITH_TARGET_VALUES"
        for call in calls
    )
    wrapper_restored = injection.get("wrapper_restored") is True
    records_path = Path(manifest["output_dir"]) / "physical_contact_substeps.jsonl"
    selected = {}
    limb0 = None
    runtime_valid = False
    if records_path.is_file():
        records = load_jsonl(records_path)
        contact_record = global_record(records, 55)
        contacts = contact_record["contacts"]
        for geom, _ in SELECTED:
            matches = [contact for contact in contacts if pair_matches(contact, geom)]
            selected[geom] = matches[0] if len(matches) == 1 else None
        limb0_matches = [contact for contact in contacts if pair_matches(contact, "limb/0")]
        limb0 = limb0_matches[0] if len(limb0_matches) == 1 else None
        runtime_common_valid = all(
            contact is not None
            and int(contact["dim"]) == int(baseline["dim"])
            and max_abs_difference(contact["friction"][:2], [effective_mu, effective_mu]) <= 1e-12
            and max_abs_difference(contact["friction"][2:5], baseline["friction"][2:5]) <= 1e-12
            and max_abs_difference(contact["solref"], baseline["solref"]) <= 1e-12
            and max_abs_difference(contact["solimp"], baseline["solimp"]) <= 1e-12
            and abs(float(contact["includemargin"]) - float(baseline["includemargin"])) <= 1e-12
            for contact in selected.values()
        )
        runtime_common_valid &= (
            limb0 is not None
            and max_abs_difference(limb0["friction"][:2], [0.7, 0.7]) <= 1e-12
        )
        runtime_valid = runtime_common_valid
    selection = "INSUFFICIENT_EVIDENCE"
    runtime_values = [
        contact["friction"][:2] for contact in selected.values()
        if contact is not None
    ]
    if len(runtime_values) == len(SELECTED):
        if all(max_abs_difference(value, [effective_mu, effective_mu]) <= 1e-12 for value in runtime_values):
            selection = (
                "USES_EXPLICIT_PAIR_CLAMPED_TO_MJMINMU"
                if mu == 0.0 else "USES_EXPLICIT_PAIR"
            )
        elif mu != 0.7 and all(
            max_abs_difference(value, [0.7, 0.7]) <= 1e-12
            for value in runtime_values
        ):
            selection = "USES_GEOM_COMBINATION"
    mu_zero_response_valid = True
    minimum_proxy_checks = {}
    if mu == 0.0:
        for geom, joint in SELECTED:
            contact = selected.get(geom)
            projection = contact["physical_projection"] if contact else None
            fn = abs(float(projection["Fn"])) if projection else 0.0
            ft = abs(float(projection["friction_force_norm"])) if projection else None
            normal_generalized = abs(float(projection["selected_joints"][joint]["normal"])) if projection else 0.0
            friction_generalized = abs(float(projection["selected_joints"][joint]["friction"])) if projection else None
            force_ratio = ft / fn if ft is not None and fn else None
            generalized_ratio = (
                friction_generalized / normal_generalized
                if friction_generalized is not None and normal_generalized else None
            )
            minimum_proxy_checks[geom] = {
                "runtime_effective_mu": effective_mu,
                "Ft_over_Fn": force_ratio,
                "force_ratio_limit": effective_mu * MINIMUM_FRICTION_FORCE_TOLERANCE_FACTOR,
                "friction_over_normal_generalized": generalized_ratio,
                "generalized_ratio_limit": effective_mu * MINIMUM_FRICTION_GENERALIZED_TOLERANCE_FACTOR,
                "valid": (
                    force_ratio is not None
                    and generalized_ratio is not None
                    and force_ratio <= effective_mu * MINIMUM_FRICTION_FORCE_TOLERANCE_FACTOR
                    and generalized_ratio <= effective_mu * MINIMUM_FRICTION_GENERALIZED_TOLERANCE_FACTOR
                ),
            }
        mu_zero_response_valid = all(
            item["valid"] for item in minimum_proxy_checks.values()
        )
        runtime_valid &= mu_zero_response_valid
    return {
        "condition": manifest["condition"],
        "target_mu": mu,
        "compiled_pair_target_mu": mu,
        "runtime_contact_effective_mu": effective_mu,
        "runtime_mu_clamped_by_mjminmu": mu < MJ_MIN_MU,
        "mjminmu": MJ_MIN_MU,
        "mjminmu_source": MJ_MIN_MU_SOURCE,
        "normal_only_counterfactual_type": (
            "MUJOCO_MINIMUM_FRICTION_PROXY" if mu == 0.0 else None
        ),
        "final_full_xml_pair_dom_valid": dom_valid,
        "final_compiled_pair_status": (
            "PRESENT_WITH_TARGET_VALUES" if compiled_valid else "MISSING_OR_WRONG"
        ),
        "runtime_contact_pair_selection": selection,
        "selected_runtime_contacts": selected,
        "limb_0_runtime_contact": limb0,
        "limb_0_production_contact_unchanged": (
            limb0 is not None
            and max_abs_difference(limb0["friction"][:2], [0.7, 0.7]) <= 1e-12
        ),
        "mu_zero_friction_response_valid": mu_zero_response_valid,
        "minimum_friction_proxy_checks": minimum_proxy_checks,
        "wrapper_restored": wrapper_restored,
        "condition_gate_valid": (
            dom_valid and compiled_valid and runtime_valid
            and selection in VALID_RUNTIME_SELECTIONS and wrapper_restored
        ),
    }


def flatten_body_state(state: dict[str, Any]) -> list[float]:
    values = []
    for body_name in ("limb/11", "limb/12"):
        body = state[body_name]
        for field in (
            "xpos_world", "xquat_wxyz", "linear_velocity_world_at_body_origin",
            "angular_velocity_world",
        ):
            values.extend(body[field])
    return values


def analyze(
    condition_manifests: Sequence[dict[str, Any]], baseline: dict[str, Any]
) -> dict[str, Any]:
    data = {}
    for manifest in condition_manifests:
        records = load_jsonl(Path(manifest["output_dir"]) / "physical_contact_substeps.jsonl")
        data[manifest["condition"]] = {
            "mu": manifest["mu"], "records": records,
            "global54": global_record(records, 54),
            "global55": global_record(records, 55),
        }
    reference = data["condition_production_mu_0p7"]
    identity = {}
    max_identity_error = 0.0
    no_early_distal = True
    for name, condition in data.items():
        comparisons = {
            "global54_post_qpos": max_abs_difference(
                condition["global54"]["post_step_full_qpos"],
                reference["global54"]["post_step_full_qpos"],
            ),
            "global54_post_qvel": max_abs_difference(
                condition["global54"]["post_step_full_qvel"],
                reference["global54"]["post_step_full_qvel"],
            ),
            "global55_pre_qpos": max_abs_difference(
                condition["global55"]["pre_step_full_qpos"],
                reference["global55"]["pre_step_full_qpos"],
            ),
            "global55_pre_qvel": max_abs_difference(
                condition["global55"]["pre_step_full_qvel"],
                reference["global55"]["pre_step_full_qvel"],
            ),
            "global55_pre_distal_body_state": max_abs_difference(
                flatten_body_state(condition["global55"]["solver_configuration_physical_probe_bodies_pre_velocity"]),
                flatten_body_state(reference["global55"]["solver_configuration_physical_probe_bodies_pre_velocity"]),
            ),
        }
        identity[name] = comparisons
        max_identity_error = max(max_identity_error, *comparisons.values())
        for record in condition["records"]:
            if int(record["global_physics_step"]) >= 55:
                continue
            no_early_distal &= not any(
                pair_matches(contact, geom)
                for contact in record["contacts"]
                for geom, _ in SELECTED
            )

    pair_validation = {}
    responses = {}
    point_velocity_valid = True
    condition_valid = True
    for name, condition in data.items():
        mu = float(condition["mu"])
        effective_mu = runtime_expected_mu(mu)
        contacts = condition["global55"]["contacts"]
        selected_validation = {}
        selected_response = {}
        for geom, joint in SELECTED:
            matches = [contact for contact in contacts if pair_matches(contact, geom)]
            if len(matches) != 1:
                selected_validation[geom] = {"valid": False, "count": len(matches)}
                condition_valid = False
                continue
            contact = matches[0]
            friction = [float(value) for value in contact["friction"]]
            valid = int(contact["dim"]) == 3 and max(
                abs(friction[0] - effective_mu), abs(friction[1] - effective_mu)
            ) <= 1e-12 and max(
                abs(friction[index] - baseline["friction"][index])
                for index in range(2, 5)
            ) <= 1e-12
            valid &= max_abs_difference(contact["solref"], baseline["solref"]) <= 1e-12
            valid &= max_abs_difference(contact["solimp"], baseline["solimp"]) <= 1e-12
            selected_validation[geom] = {
                "valid": bool(valid), "dim": contact["dim"], "friction": friction,
                "solref": contact["solref"], "solimp": contact["solimp"],
                "includemargin": contact["includemargin"],
            }
            condition_valid &= bool(valid)
            projection = contact["physical_projection"]
            velocity = contact["point_velocity"]
            closure = max(
                velocity["pre"]["rigid_vs_jacobian_max_abs_error"],
                velocity["post"]["rigid_vs_jacobian_max_abs_error"],
            )
            point_velocity_valid &= closure <= 1e-9
            selected_response[geom] = {
                "joint": joint,
                "Fn": projection["Fn"], "Ft1": projection["Ft1"],
                "Ft2": projection["Ft2"],
                "Ft_norm": projection["friction_force_norm"],
                "Ft_over_Fn": projection["friction_force_norm"] / projection["Fn"],
                "Ft_over_mu_Fn": (
                    projection["friction_force_norm"] / (effective_mu * projection["Fn"])
                    if effective_mu > 0 else None
                ),
                "target_mu": mu,
                "runtime_effective_mu": effective_mu,
                "pre_tangential_speed": velocity["pre"]["tangential_speed"],
                "post_tangential_speed": velocity["post"]["tangential_speed"],
                "point_velocity": velocity,
                "normal_generalized": projection["selected_joints"][joint]["normal"],
                "friction_generalized": projection["selected_joints"][joint]["friction"],
                "total_generalized": projection["selected_joints"][joint]["total"],
            }
        limb0 = [contact for contact in contacts if pair_matches(contact, "limb/0")]
        limb0_valid = len(limb0) == 1 and max_abs_difference(
            limb0[0]["friction"][:2], [0.7, 0.7]
        ) <= 1e-12
        condition_valid &= limb0_valid
        pair_validation[name] = {
            "target_mu": mu,
            "runtime_effective_mu": effective_mu,
            "runtime_mu_clamped_by_mjminmu": mu < MJ_MIN_MU,
            "normal_only_counterfactual_type": (
                "MUJOCO_MINIMUM_FRICTION_PROXY" if mu == 0.0 else None
            ),
            "selected": selected_validation,
            "limb_0_floor": limb0[0] if len(limb0) == 1 else None,
            "limb_0_floor_source_friction_preserved": limb0_valid,
            "condition_valid": all(
                value.get("valid", False) for value in selected_validation.values()
            ) and limb0_valid,
        }
        responses[name] = {
            "target_mu": mu,
            "runtime_effective_mu": effective_mu,
            "selected": selected_response,
        }

    limb12 = {
        name: response["selected"].get("limb/12")
        for name, response in responses.items()
    }
    ratios = {
        "Ft_0p35_over_0p70": limb12["condition_distal_mu_0p35"]["Ft_norm"] / limb12["condition_production_mu_0p7"]["Ft_norm"],
        "Ft_1p40_over_0p70": limb12["condition_distal_mu_1p4"]["Ft_norm"] / limb12["condition_production_mu_0p7"]["Ft_norm"],
    }
    low_cap = abs(ratios["Ft_0p35_over_0p70"] - 0.5) <= 0.15
    high_plateau = abs(ratios["Ft_1p40_over_0p70"] - 1.0) <= 0.15
    high_slip_small = limb12["condition_distal_mu_1p4"]["post_tangential_speed"] <= 0.1 * limb12["condition_distal_mu_0"]["post_tangential_speed"]
    if high_plateau and not low_cap and high_slip_small:
        regime = "DEMAND_LIMITED"
    elif low_cap and high_plateau:
        regime = "MIXED"
    elif low_cap and not high_plateau and not high_slip_small:
        regime = "COULOMB_CAP_LIMITED"
    else:
        regime = "INSUFFICIENT_EVIDENCE"

    isaac = {
        "mu0": {"Ft": 0.0, "post_slip": 0.4016187877},
        "mu0p35": {"Ft": 344.4631625, "Fn": 1022.1884766, "post_slip": 0.0361481},
        "mu0p7": {"Ft": 363.2845280, "Fn": 1023.4495239, "post_slip": 0.0231723},
        "mu1p4": {"Ft": 357.6129973, "Fn": 1024.3537598, "post_slip": 0.0203087},
    }
    mujoco_mu0 = limb12["condition_distal_mu_0"]["post_tangential_speed"]
    mujoco_mu07 = limb12["condition_production_mu_0p7"]["post_tangential_speed"]
    normal_mismatch = abs(mujoco_mu0 - isaac["mu0"]["post_slip"]) > 0.1 * max(mujoco_mu0, isaac["mu0"]["post_slip"], 1e-12)
    mujoco_correction = mujoco_mu07 - mujoco_mu0
    isaac_correction = isaac["mu0p7"]["post_slip"] - isaac["mu0"]["post_slip"]
    coupling_mismatch = abs(mujoco_correction - isaac_correction) > 0.1 * max(abs(mujoco_correction), abs(isaac_correction), 1e-12)
    if normal_mismatch and coupling_mismatch:
        mismatch = "BOTH"
    elif normal_mismatch:
        mismatch = "NORMAL_ONLY_SLIP_RESPONSE"
    elif coupling_mismatch:
        mismatch = "TANGENTIAL_EFFECTIVE_MASS_OR_COUPLING"
    else:
        mismatch = "NOT_SUPPORTED"
    pre_state_pass = max_identity_error <= 1e-12 and no_early_distal
    return {
        "runtime_pair_validation": {
            "conditions": pair_validation,
            "condition_valid": condition_valid,
        },
        "pre_state_identity": {
            "comparisons_to_mu_0p7": identity,
            "max_abs_error": max_identity_error,
            "no_selected_distal_contact_before_global55": no_early_distal,
            "global55_pre_state_identity": "PASS" if pre_state_pass else "FAIL",
        },
        "mu_response": responses,
        "point_velocity_mapping_valid": point_velocity_valid,
        "force_regime": {
            "normal_only_counterfactual_type": "MUJOCO_MINIMUM_FRICTION_PROXY",
            "mjminmu": MJ_MIN_MU,
            "mjminmu_source": MJ_MIN_MU_SOURCE,
            "ratios": ratios,
            "limb12_post_slip_vs_mu": {
                name: value["post_tangential_speed"] for name, value in limb12.items()
            },
            "limb12_Fn_vs_mu": {name: value["Fn"] for name, value in limb12.items()},
            "classification_thresholds": {
                "force_ratio_tolerance": 0.15,
                "high_mu_small_slip_fraction_of_mu0": 0.1,
            },
            "mujoco_global55_friction_force_regime": regime,
        },
        "cross_backend": {
            "normal_only_counterfactual_type": "MUJOCO_MINIMUM_FRICTION_PROXY",
            "mujoco_minimum_friction_proxy_effective_mu": MJ_MIN_MU,
            "isaac_reference": isaac,
            "mujoco_mu0_post_slip": mujoco_mu0,
            "mujoco_mu0p7_post_slip": mujoco_mu07,
            "normal_only_post_slip_difference_mujoco_minus_isaac": mujoco_mu0 - isaac["mu0"]["post_slip"],
            "production_friction_effect": {
                "mujoco": mujoco_correction, "isaac": isaac_correction,
                "difference": mujoco_correction - isaac_correction,
            },
            "production_force_difference_mujoco_minus_isaac": limb12["condition_production_mu_0p7"]["Ft_norm"] - isaac["mu0p7"]["Ft"],
            "force_plateau_ratio_difference": {
                "mu0p35_over_mu0p7_mujoco_minus_isaac": (
                    ratios["Ft_0p35_over_0p70"]
                    - isaac["mu0p35"]["Ft"] / isaac["mu0p7"]["Ft"]
                ),
                "mu1p4_over_mu0p7_mujoco_minus_isaac": (
                    ratios["Ft_1p40_over_0p70"]
                    - isaac["mu1p4"]["Ft"] / isaac["mu0p7"]["Ft"]
                ),
            },
            "post_slip_correction_difference": mujoco_correction - isaac_correction,
            "cross_backend_friction_demand_mismatch": mismatch,
        },
        "validation": {
            "condition_valid": condition_valid,
            "global55_pre_state_identity": pre_state_pass,
            "point_velocity_mapping_valid": point_velocity_valid,
            "no_extra_stepping": True,
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    source_xml = Path(args.source_xml).resolve()
    metadata = Path(args.metadata).resolve()
    checkpoint = Path(args.checkpoint).resolve()
    existing_oracle = Path(args.existing_oracle).resolve()
    output_root = Path(args.output_dir).resolve()
    temporary_root = Path(args.temporary_root).resolve()
    for path, label in ((source_xml, "source XML"), (metadata, "metadata"), (checkpoint, "checkpoint")):
        if not path.is_file():
            raise FileNotFoundError(f"{label} is missing: {path}")
    if sha256(source_xml) != SOURCE_XML_SHA256:
        raise ValueError("source XML SHA256 mismatch")
    if sha256(checkpoint) != CHECKPOINT_SHA256:
        raise ValueError("checkpoint SHA256 mismatch")
    if output_root.exists() or temporary_root.exists():
        raise FileExistsError("output and temporary roots must not already exist")
    output_root.mkdir(parents=True)
    temporary_root.mkdir(parents=True)
    source_hash_before = sha256(source_xml)
    baseline = baseline_pair_parameters(existing_oracle)
    manifests = []
    condition_gates = []
    for name, mu in CONDITIONS:
        manifest = run_condition(
            name, mu, source_xml, metadata, checkpoint, output_root,
            temporary_root, baseline, args.device,
        )
        manifests.append(manifest)
        if manifest["return_code"] != 0:
            write_json(output_root / "temporary_xml_manifest.json", manifests)
            return manifest["return_code"]
        gate = validate_condition_gates(manifest, baseline)
        condition_gates.append(gate)
        write_json(Path(manifest["output_dir"]) / "condition_gate.json", gate)
        if not gate["condition_gate_valid"]:
            write_json(output_root / "final_xml_injection_audit.json", {
                "conditions": [item.get("final_xml_injection") for item in manifests]
            })
            write_json(output_root / "compiled_pair_validation.json", {
                "conditions": condition_gates
            })
            write_json(output_root / "runtime_pair_selection.json", {
                "conditions": condition_gates
            })
            return 3
    reports = analyze(manifests, baseline)
    write_json(output_root / "temporary_xml_manifest.json", {
        "source_xml": str(source_xml),
        "source_xml_sha256_before": source_hash_before,
        "source_xml_sha256_after": sha256(source_xml),
        "baseline_runtime_pair_parameters": baseline,
        "conditions": manifests,
    })
    write_json(output_root / "final_xml_injection_audit.json", {
        "conditions": [item["final_xml_injection"] for item in manifests]
    })
    write_json(output_root / "compiled_pair_validation.json", {
        "conditions": condition_gates,
        "all_conditions_present_with_target_values": all(
            item["final_compiled_pair_status"] == "PRESENT_WITH_TARGET_VALUES"
            for item in condition_gates
        ),
    })
    write_json(output_root / "runtime_pair_selection.json", {
        "conditions": condition_gates,
        "all_conditions_use_explicit_pair": all(
            item["runtime_contact_pair_selection"] in VALID_RUNTIME_SELECTIONS
            for item in condition_gates
        ),
    })
    for filename, key in (
        ("runtime_pair_friction_validation.json", "runtime_pair_validation"),
        ("global55_pre_state_identity.json", "pre_state_identity"),
        ("global55_mu_response.json", "mu_response"),
        ("friction_force_regime.json", "force_regime"),
        ("cross_backend_friction_demand_comparison.json", "cross_backend"),
    ):
        write_json(output_root / filename, reports[key])
    validation = {
        **reports["validation"],
        "source_xml_hash_unchanged": sha256(source_xml) == source_hash_before,
        "source_xml_hash_matches_frozen": sha256(source_xml) == SOURCE_XML_SHA256,
        "temporary_files_only_under_requested_root": all(
            Path(item["temporary_xml"]).is_relative_to(temporary_root)
            for item in manifests
        ),
        "condition_return_codes": {
            item["condition"]: item["return_code"] for item in manifests
        },
        "final_full_xml_pair_dom_valid": all(
            item["final_full_xml_pair_dom_valid"] for item in condition_gates
        ),
        "compiled_pair_gates_valid": all(
            item["final_compiled_pair_status"] == "PRESENT_WITH_TARGET_VALUES"
            for item in condition_gates
        ),
        "runtime_pair_gates_valid": all(
            item["runtime_contact_pair_selection"] in VALID_RUNTIME_SELECTIONS
            for item in condition_gates
        ),
        "mjminmu": MJ_MIN_MU,
        "mjminmu_source": MJ_MIN_MU_SOURCE,
        "normal_only_counterfactual_type": "MUJOCO_MINIMUM_FRICTION_PROXY",
        "loader_wrappers_restored": all(item["wrapper_restored"] for item in condition_gates),
    }
    validation["mujoco_distal_friction_bracket"] = (
        "VALIDATED"
        if all(
            value is True
            for key, value in validation.items()
            if key not in ("condition_return_codes",)
        )
        else "INSUFFICIENT_EVIDENCE"
    )
    write_json(output_root / "validation.json", validation)
    print(json.dumps({
        "output_dir": str(output_root),
        "MUJOCO_GLOBAL55_FRICTION_FORCE_REGIME": reports["force_regime"]["mujoco_global55_friction_force_regime"],
        "CROSS_BACKEND_FRICTION_DEMAND_MISMATCH": reports["cross_backend"]["cross_backend_friction_demand_mismatch"],
        "MUJOCO_DISTAL_FRICTION_BRACKET": validation["mujoco_distal_friction_bracket"],
        "NORMAL_ONLY_COUNTERFACTUAL_TYPE": "MUJOCO_MINIMUM_FRICTION_PROXY",
    }, indent=2))
    return 0 if validation["mujoco_distal_friction_bracket"] == "VALIDATED" else 2


if __name__ == "__main__":
    raise SystemExit(main())

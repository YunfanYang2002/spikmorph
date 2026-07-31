import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "tools"
    / "run_mujoco_distal_friction_bracket.py"
)
SPEC = importlib.util.spec_from_file_location(
    "run_mujoco_distal_friction_bracket", MODULE_PATH
)
BRACKET = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = BRACKET
SPEC.loader.exec_module(BRACKET)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class MujocoDistalFrictionBracketTests(unittest.TestCase):
    def baseline(self):
        return {
            "friction": [0.7, 0.7, 0.1, 0.01, 0.02],
            "dim": 3,
            "solref": [0.02, 1.0],
            "solimp": [0.9, 0.95, 0.001, 0.5, 2.0],
            "includemargin": 0.0,
        }

    def test_temporary_pair_override_only_replaces_selected_pairs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.xml"
            source.write_text(
                "<mujoco><worldbody/><contact>"
                '<pair geom1="limb/0" geom2="floor/0" friction="0.7 0.7 0.1 0.01 0.02"/>'
                '<pair geom1="floor/0" geom2="limb/11" friction="9 9 9 9 9"/>'
                "</contact></mujoco>",
                encoding="utf-8",
            )
            source_hash = digest(source)
            target = root / "tmp" / "walker" / "xml" / "walker.xml"
            manifest = BRACKET.write_temporary_xml(
                source, target, 0.35, self.baseline()
            )
            self.assertEqual(digest(source), source_hash)
            pairs = ET.parse(target).getroot().find("contact").findall("pair")
            by_geoms = {
                frozenset((pair.get("geom1"), pair.get("geom2"))): pair
                for pair in pairs
            }
            self.assertEqual(len(pairs), 3)
            limb0 = by_geoms[frozenset(("limb/0", "floor/0"))]
            self.assertEqual(limb0.get("friction"), "0.7 0.7 0.1 0.01 0.02")
            for geom in ("limb/11", "limb/12"):
                pair = by_geoms[frozenset((geom, "floor/0"))]
                self.assertEqual(pair.get("condim"), "3")
                self.assertEqual(
                    [float(value) for value in pair.get("friction").split()],
                    [0.35, 0.35, 0.1, 0.01, 0.02],
                )
            self.assertEqual(manifest["friction_fields"]["tangent1"], 0.35)
            self.assertEqual(manifest["friction_fields"]["torsional"], 0.1)

    def test_pair_matching_is_order_independent(self):
        contact = {"geom1_name": "floor/0", "geom2_name": "limb/12"}
        self.assertTrue(BRACKET.pair_matches(contact, "limb/12"))
        self.assertFalse(BRACKET.pair_matches(contact, "limb/0"))

    def write_gate_fixture(
        self,
        output,
        target_mu,
        runtime_mu,
        *,
        force_ratio=None,
        generalized_ratio=None,
    ):
        if force_ratio is None:
            force_ratio = runtime_mu
        if generalized_ratio is None:
            generalized_ratio = runtime_mu
        contacts = []
        for geom, joint in (("limb/11", "limby/11"), ("limb/12", "limby/12")):
            fn = 1000.0
            normal_generalized = 250.0
            contacts.append({
                "geom1_name": geom, "geom2_name": "floor/0", "dim": 3,
                "friction": [runtime_mu, runtime_mu, 0.1, 0.01, 0.02],
                "solref": [0.02, 1.0],
                "solimp": [0.9, 0.95, 0.001, 0.5, 2.0],
                "includemargin": 0.0,
                "physical_projection": {
                    "Fn": fn,
                    "friction_force_norm": fn * force_ratio,
                    "selected_joints": {
                        joint: {
                            "normal": normal_generalized,
                            "friction": normal_generalized * generalized_ratio,
                        }
                    },
                },
            })
        contacts.append({
            "geom1_name": "limb/0", "geom2_name": "floor/0", "dim": 3,
            "friction": [0.7, 0.7, 0.1, 0.01, 0.02],
        })
        (output / "physical_contact_substeps.jsonl").write_text(
            json.dumps({"global_physics_step": 55, "contacts": contacts}) + "\n",
            encoding="utf-8",
        )
        return {
            "condition": f"condition_mu_{target_mu}", "mu": target_mu,
            "output_dir": str(output),
            "final_xml_injection": {
                "wrapper_restored": True,
                "calls": [{
                    "dom": {"final_full_xml_pair_dom_valid": True},
                    "final_compiled_pair_status": "PRESENT_WITH_TARGET_VALUES",
                }],
            },
        }

    def test_compiled_zero_runtime_minimum_mu_is_valid_proxy(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            manifest = self.write_gate_fixture(
                output, 0.0, BRACKET.MJ_MIN_MU,
                force_ratio=BRACKET.MJ_MIN_MU,
                generalized_ratio=1.4e-5,
            )
            gate = BRACKET.validate_condition_gates(manifest, self.baseline())
            self.assertTrue(gate["condition_gate_valid"])
            self.assertEqual(
                gate["runtime_contact_pair_selection"],
                "USES_EXPLICIT_PAIR_CLAMPED_TO_MJMINMU",
            )
            self.assertEqual(gate["compiled_pair_target_mu"], 0.0)
            self.assertEqual(
                gate["runtime_contact_effective_mu"], BRACKET.MJ_MIN_MU
            )
            self.assertTrue(gate["runtime_mu_clamped_by_mjminmu"])
            self.assertEqual(
                gate["normal_only_counterfactual_type"],
                "MUJOCO_MINIMUM_FRICTION_PROXY",
            )
            self.assertTrue(gate["limb_0_production_contact_unchanged"])

    def test_compiled_zero_runtime_geom_fallback_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            manifest = self.write_gate_fixture(output, 0.0, 0.7)
            gate = BRACKET.validate_condition_gates(manifest, self.baseline())
            self.assertFalse(gate["condition_gate_valid"])
            self.assertEqual(
                gate["runtime_contact_pair_selection"], "USES_GEOM_COMBINATION"
            )

    def test_nonzero_runtime_mu_requires_exact_target(self):
        for target_mu in (0.35, 0.7, 1.4):
            with self.subTest(target_mu=target_mu), tempfile.TemporaryDirectory() as directory:
                output = Path(directory)
                manifest = self.write_gate_fixture(output, target_mu, target_mu)
                gate = BRACKET.validate_condition_gates(manifest, self.baseline())
                self.assertTrue(gate["condition_gate_valid"])
                self.assertEqual(
                    gate["runtime_contact_pair_selection"], "USES_EXPLICIT_PAIR"
                )
                self.assertFalse(gate["runtime_mu_clamped_by_mjminmu"])

    def test_minimum_friction_proxy_rejects_excess_force_ratio(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            manifest = self.write_gate_fixture(
                output, 0.0, BRACKET.MJ_MIN_MU,
                force_ratio=BRACKET.MJ_MIN_MU * 1.051,
                generalized_ratio=BRACKET.MJ_MIN_MU,
            )
            gate = BRACKET.validate_condition_gates(manifest, self.baseline())
            self.assertFalse(gate["condition_gate_valid"])
            self.assertFalse(gate["mu_zero_friction_response_valid"])

    def test_minimum_friction_proxy_rejects_excess_generalized_ratio(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            manifest = self.write_gate_fixture(
                output, 0.0, BRACKET.MJ_MIN_MU,
                force_ratio=BRACKET.MJ_MIN_MU,
                generalized_ratio=BRACKET.MJ_MIN_MU * 2.001,
            )
            gate = BRACKET.validate_condition_gates(manifest, self.baseline())
            self.assertFalse(gate["condition_gate_valid"])
            self.assertFalse(gate["mu_zero_friction_response_valid"])


if __name__ == "__main__":
    unittest.main()

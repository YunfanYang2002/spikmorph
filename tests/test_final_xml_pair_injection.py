import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "tools"
    / "evaluate_mujoco_checkpoint_with_final_pairs.py"
)
SPEC = importlib.util.spec_from_file_location("final_xml_pair_injection", MODULE_PATH)
INJECTOR = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = INJECTOR
SPEC.loader.exec_module(INJECTOR)


class FinalXmlPairInjectionTests(unittest.TestCase):
    def production(self):
        return {
            "friction": [0.7, 0.7, 0.1, 0.1, 0.1],
            "dim": 3,
            "solref": [0.02, 1.0],
            "solimp": [0.9, 0.95, 0.001, 0.5, 2.0],
            "includemargin": 0.0,
        }

    def full_xml(self):
        return (
            "<mujoco><worldbody>"
            '<geom name="floor/0"/><body name="limb/11"><geom name="limb/11"/></body>'
            '<body name="limb/12"><geom name="limb/12"/></body><geom name="limb/0"/>'
            "</worldbody><contact>"
            '<pair geom1="floor/0" geom2="limb/11" friction="9 9 9 9 9"/>'
            '<pair geom1="limb/11" geom2="floor/0" friction="8 8 8 8 8"/>'
            "</contact></mujoco>"
        )

    def test_injection_requires_final_full_xml_and_deduplicates(self):
        modified, audit = INJECTOR.inject_final_pairs(
            self.full_xml(), 0.0, self.production()
        )
        self.assertTrue(audit["final_full_xml_pair_dom_valid"])
        self.assertEqual(audit["target_pair_counts"], {"limb/11": 1, "limb/12": 1})
        self.assertEqual(len(audit["removed_selected_pairs"]), 2)
        self.assertFalse(audit["limb_0_floor_is_explicit_pair"])
        self.assertNotEqual(
            audit["final_xml_before_injection_sha256"],
            audit["final_xml_after_injection_sha256"],
        )
        self.assertIn('friction="0 0 0.10000000000000001', modified)

    def test_mu_one_point_four_is_preserved(self):
        _, audit = INJECTOR.inject_final_pairs(
            self.full_xml(), 1.4, self.production()
        )
        for pair in audit["added_selected_pairs"]:
            self.assertEqual(
                [float(value) for value in pair["friction"].split()][:2],
                [1.4, 1.4],
            )

    def test_missing_floor_fails_dom_gate(self):
        fragment = '<mujoco><worldbody><geom name="limb/11"/><geom name="limb/12"/></worldbody></mujoco>'
        _, audit = INJECTOR.inject_final_pairs(fragment, 0.0, self.production())
        self.assertFalse(audit["final_full_xml_pair_dom_valid"])
        self.assertIn("floor/0", audit["missing_required_geoms"])

    def test_compiled_pair_table_and_limb0_gate(self):
        names = {0: "floor/0", 1: "limb/11", 2: "limb/12"}

        class Mujoco:
            class mjtObj:
                mjOBJ_GEOM = 5

            @staticmethod
            def mj_id2name(model, kind, object_id):
                return names[object_id]

        model = SimpleNamespace(
            npair=2,
            pair_geom1=np.asarray([1, 2]), pair_geom2=np.asarray([0, 0]),
            pair_dim=np.asarray([3, 3]),
            pair_friction=np.asarray([[0.0, 0.0, 0.1, 0.1, 0.1]] * 2),
            pair_solref=np.asarray([[0.02, 1.0]] * 2),
            pair_solimp=np.asarray([[0.9, 0.95, 0.001, 0.5, 2.0]] * 2),
            pair_margin=np.asarray([0.0, 0.0]), pair_gap=np.asarray([0.0, 0.0]),
        )
        table = INJECTOR.compiled_pair_table(Mujoco, model)
        self.assertEqual(
            INJECTOR.compiled_pair_status(table, 0.0, self.production()),
            "PRESENT_WITH_TARGET_VALUES",
        )

    def test_temporary_loader_wrapper_restores_after_error(self):
        original = object()
        replacement = object()
        module = SimpleNamespace(load_model_from_xml=original)
        with self.assertRaisesRegex(RuntimeError, "probe"):
            with INJECTOR.temporary_loader_wrapper(module, replacement):
                self.assertIs(module.load_model_from_xml, replacement)
                raise RuntimeError("probe")
        self.assertIs(module.load_model_from_xml, original)


if __name__ == "__main__":
    unittest.main()

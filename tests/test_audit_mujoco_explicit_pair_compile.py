import hashlib
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "tools"
    / "audit_mujoco_explicit_pair_compile.py"
)
SPEC = importlib.util.spec_from_file_location(
    "audit_mujoco_explicit_pair_compile", MODULE_PATH
)
AUDIT = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = AUDIT
SPEC.loader.exec_module(AUDIT)


class ExplicitPairCompileAuditTests(unittest.TestCase):
    def write_xml(self, path, mu):
        path.write_text(
            "<mujoco><contact>"
            f'<pair geom1="limb/11" geom2="floor/0" condim="3" friction="{mu} {mu} 0.1 0.1 0.1" solref="0.02 1" solimp="0 0.99 0.01 0.5 2" margin="0" gap="0"/>'
            f'<pair geom1="floor/0" geom2="limb/12" condim="3" friction="{mu} {mu} 0.1 0.1 0.1" solref="0.02 1" solimp="0 0.99 0.01 0.5 2" margin="0" gap="0"/>'
            "</contact></mujoco>",
            encoding="utf-8",
        )

    def test_dom_extracts_mu_zero_without_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mu0.xml"
            self.write_xml(path, 0)
            before = hashlib.sha256(path.read_bytes()).hexdigest()
            audit = AUDIT.extract_dom_pair_audit(path)
            self.assertTrue(AUDIT.dom_status(audit, 0.0))
            self.assertEqual(audit["target_pair_counts"], {"limb/11": 1, "limb/12": 1})
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), before)

    def test_dom_extracts_mu_one_point_four(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mu1p4.xml"
            self.write_xml(path, 1.4)
            self.assertTrue(AUDIT.dom_status(AUDIT.extract_dom_pair_audit(path), 1.4))

    def test_compiled_table_resolves_names_and_ignores_geom_order(self):
        names = {0: "floor/0", 1: "limb/11", 2: "limb/12"}

        class Mujoco:
            class mjtObj:
                mjOBJ_GEOM = 5

            @staticmethod
            def mj_id2name(model, object_type, object_id):
                return names[object_id]

        model = type("Model", (), {
            "npair": 2,
            "pair_geom1": np.asarray([1, 0]), "pair_geom2": np.asarray([0, 2]),
            "pair_dim": np.asarray([3, 3]),
            "pair_friction": np.asarray([[1.4, 1.4, 0.1, 0.1, 0.1]] * 2),
            "pair_solref": np.asarray([[0.02, 1.0]] * 2),
            "pair_solimp": np.asarray([[0, 0.99, 0.01, 0.5, 2]] * 2),
            "pair_margin": np.asarray([0.0, 0.0]), "pair_gap": np.asarray([0.0, 0.0]),
        })()
        table = AUDIT.compiled_pair_table(Mujoco, model)
        self.assertEqual(AUDIT.compiled_pair_status(table, 1.4), "PRESENT_WITH_TARGET_VALUES")

    def test_direct_correct_environment_missing_classifies_rewrite_drop(self):
        root = AUDIT.classify_root_cause(
            True, "PRESENT_WITH_TARGET_VALUES", "MISSING", "sha", [],
            "INSUFFICIENT_EVIDENCE", requested_source_matches=True,
        )
        self.assertEqual(root, "ENVIRONMENT_REWRITES_OR_DROPS_CONTACT_PAIRS")

    def test_different_requested_source_classifies_different_xml(self):
        root = AUDIT.classify_root_cause(
            True, "PRESENT_WITH_TARGET_VALUES", "MISSING", "sha", [],
            "INSUFFICIENT_EVIDENCE", requested_source_matches=False,
        )
        self.assertEqual(root, "ENVIRONMENT_LOADS_DIFFERENT_XML")


if __name__ == "__main__":
    unittest.main()

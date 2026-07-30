import hashlib
import importlib.util
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


if __name__ == "__main__":
    unittest.main()

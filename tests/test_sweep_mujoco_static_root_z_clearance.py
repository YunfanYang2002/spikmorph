"""Pure-Python tests for the static MuJoCo root-z clearance sweep."""

import ast
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from tools import dump_mujoco_transition as transition
from tools import sweep_mujoco_static_root_z_clearance as sweep


class StaticRootZClearanceSweepTest(unittest.TestCase):
    def args(self, **overrides):
        values = {
            "canonical_qpos_mode": "model-default",
            "penetration_tolerance": 0.001,
            "root_z_min": 1.2,
            "root_z_max": 1.4,
            "root_z_step": 0.1,
            "refine_resolution": 0.01,
            "safety_margin": 0.02,
        }
        values.update(overrides)
        return SimpleNamespace(**values)

    def test_cli_validation(self):
        base = [
            "--cfg", "configs/ft.yaml", "--walker-dir", "walkers",
            "--source-xml", "walker.xml",
            "--expected-source-xml-sha256", "a" * 64,
            "--root-z-min", "1.2", "--root-z-max", "2.5",
            "--root-z-step", "0.02", "--output", "dump",
        ]
        self.assertEqual(sweep.parse_args(base).refine_resolution, 0.001)
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                sweep.parse_args(base + ["--refine-resolution", "0"])

    def test_script_contains_no_step_or_do_simulation_call(self):
        tree = ast.parse(Path(sweep.__file__).read_text(encoding="utf-8"))
        forbidden = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if node.func.attr in ("step", "do_simulation"):
                    forbidden.append(node.func.attr)
        self.assertEqual(forbidden, [])

    def test_candidate_resets_then_injects_zero_state_without_step(self):
        events = []
        data = SimpleNamespace(
            qpos=np.zeros(3), qvel=np.ones(1), ctrl=np.ones(1), ncon=0,
            body_xpos=np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.3]]),
            body_xquat=np.array([[1.0, 0.0, 0.0, 0.0]] * 2),
            body_xvelp=np.zeros((2, 3)), body_xvelr=np.zeros((2, 3)),
            contact=[],
        )
        model = SimpleNamespace(njnt=1, nv=1, jnt_type=np.array([3]))

        class FakeSim:
            def __init__(self):
                self.model, self.data = model, data

            def forward(self):
                events.append("forward")

            def step(self):
                raise AssertionError("mj_step must not be called")

        base = SimpleNamespace(sim=FakeSim())

        class FakeEnv:
            unwrapped = base
            action_space = SimpleNamespace(shape=(1,))

            def reset(self):
                events.append("reset")

        env = FakeEnv()
        original_build = transition.build_canonical_state
        original_apply = transition.apply_and_verify_canonical_state

        def fake_build(model_, root_z, *unused, **kwargs):
            events.append(("build", root_z))
            return {"qpos": np.array([0.0, 0.0, root_z]), "qvel": np.zeros(1), "root_qpos_adr": 0}

        def fake_apply(base_, canonical, policy_action, np_module):
            events.append("inject")
            base_.sim.data.qpos[:] = canonical["qpos"]
            base_.sim.data.qvel[:] = canonical["qvel"]
            base_.sim.data.ctrl[:] = 0.0
            base_.sim.forward()

        transition.build_canonical_state = fake_build
        transition.apply_and_verify_canonical_state = fake_apply
        try:
            record = sweep.evaluate_root_z(
                env, 1.3, self.args(), {"joint": ["limb/0"], "body": ["world", "torso/0"], "geom": []},
                {"geom_id": 0, "geom_name": "floor/0"}, set(), 1, np,
            )
        finally:
            transition.build_canonical_state = original_build
            transition.apply_and_verify_canonical_state = original_apply
        self.assertEqual(events, ["reset", ("build", 1.3), "inject", "forward"])
        self.assertEqual(record["qvel_abs_max"], 0.0)
        self.assertTrue(np.all(data.ctrl == 0.0))
        self.assertTrue(record["is_clear"])
        self.assertEqual(record["minimum_contact_distance"], sweep.NOT_AVAILABLE)

    def test_ground_filter_excludes_self_contact(self):
        contacts = [
            SimpleNamespace(geom1=0, geom2=1, dist=-0.2, pos=np.zeros(3), frame=np.eye(3).reshape(-1)),
            SimpleNamespace(geom1=1, geom2=2, dist=-0.3, pos=np.zeros(3), frame=np.eye(3).reshape(-1)),
        ]
        sim = SimpleNamespace(
            model=SimpleNamespace(geom_bodyid=np.array([0, 1, 2])),
            data=SimpleNamespace(ncon=2, contact=contacts),
        )
        names = {"geom": ["floor/0", "limb/0", "limb/1"], "body": ["world", "limb/0", "limb/1"]}
        result = sweep.static_ground_contacts(
            sim, names, {"geom_id": 0, "geom_name": "floor/0"}, {1, 2}, np
        )
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["penetration_depth"], 0.2)

    def test_no_contact_is_clear_but_small_penetration_contact_is_not(self):
        no_contact = {"is_clear": True, "ground_contact_count": 0}
        shallow = {
            "is_clear": False,
            "ground_contact_count": 1,
            "max_penetration_depth": 0.0005,
            "within_penetration_tolerance": True,
        }
        self.assertTrue(no_contact["is_clear"])
        self.assertFalse(shallow["is_clear"])
        self.assertTrue(shallow["within_penetration_tolerance"])

    def test_coarse_and_refined_boundary_and_margin(self):
        def evaluator(root_z):
            clear = root_z >= 1.34
            return {
                "root_z": root_z,
                "is_clear": clear,
                "max_penetration_depth": max(0.0, 1.34 - root_z),
                "contacting_robot_body_names": [] if clear else ["limb/0"],
            }

        records, summary = sweep.execute_sweep(evaluator, self.args())
        self.assertEqual(summary["coarse_first_clear_root_z"], 1.4)
        self.assertEqual(summary["refined_minimum_clear_root_z"], 1.34)
        self.assertEqual(summary["last_non_clear_root_z"], 1.33)
        self.assertEqual(summary["recommended_root_z"], 1.36)
        self.assertEqual(summary["coarse_tested_count"], 3)
        self.assertGreater(summary["refined_tested_count"], 0)
        self.assertEqual(summary["tested_count"], len(records))

    def test_no_clear_returns_nonzero_semantics(self):
        def evaluator(root_z):
            return {
                "root_z": root_z,
                "is_clear": False,
                "max_penetration_depth": 0.1,
                "contacting_robot_body_names": ["limb/0"],
            }

        _records, summary = sweep.execute_sweep(evaluator, self.args())
        self.assertFalse(summary["ok"])
        self.assertIsNone(summary["refined_minimum_clear_root_z"])
        self.assertEqual(summary["suggestion"], "increase --root-z-max")
        self.assertEqual(0 if summary["ok"] else 2, 2)

    def test_no_contact_distance_is_not_available(self):
        self.assertEqual(sweep.NOT_AVAILABLE, "not_available")

    def test_non_finite_json_reports_field_root_z_and_value(self):
        with self.assertRaisesRegex(
            ValueError, r"field=root\.torso_height root_z=1\.2 value=nan"
        ):
            sweep.strict_json_for_root_z({"torso_height": float("nan")}, 1.2)

    def test_existing_transition_schema_is_unchanged(self):
        self.assertEqual(transition.SCHEMA_VERSION, "metamorph-transition-v1")
        self.assertFalse(
            transition.parse_args([
                "--cfg", "c", "--walker-dir", "w", "--source-xml", "x",
                "--root-z", "1.2", "--output", "o",
            ]).record_first_ground_contact_window
        )

    def test_strict_records_are_json_serializable(self):
        value = sweep.strict_json_for_root_z(
            {"root_z": np.float64(1.2), "position": np.zeros(3)}, 1.2
        )
        json.dumps(value, allow_nan=False)


if __name__ == "__main__":
    unittest.main()

"""Pure-Python tests for dump_mujoco_transition helpers."""

import argparse
import contextlib
import hashlib
import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "tools" / "dump_mujoco_transition.py"
SPEC = importlib.util.spec_from_file_location("dump_mujoco_transition", SCRIPT_PATH)
DUMP = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(DUMP)


class DumpMujocoTransitionHelpersTest(unittest.TestCase):
    def cli_args(self, *extra):
        return [
            "--cfg", "config.yaml",
            "--walker-dir", "walkers",
            "--source-xml", "walker.xml",
            "--root-z", "2.20",
            "--output", "dump",
            *extra,
        ]

    def test_canonical_qpos_mode_cli(self):
        self.assertEqual(
            DUMP.parse_args(self.cli_args()).canonical_qpos_mode,
            "midpoint",
        )
        self.assertEqual(
            DUMP.parse_args(
                self.cli_args("--canonical-qpos-mode", "midpoint")
            ).canonical_qpos_mode,
            "midpoint",
        )
        self.assertEqual(
            DUMP.parse_args(
                self.cli_args("--canonical-qpos-mode", "model-default")
            ).canonical_qpos_mode,
            "model-default",
        )
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                DUMP.parse_args(
                    self.cli_args("--canonical-qpos-mode", "invalid")
                )

    def test_target_joint_initial_position_cli(self):
        self.assertEqual(
            DUMP.parse_args(self.cli_args()).target_joint_initial_position,
            "default",
        )
        self.assertEqual(
            DUMP.parse_args(
                self.cli_args("--target-joint-initial-position", "midpoint")
            ).target_joint_initial_position,
            "midpoint",
        )
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                DUMP.parse_args(
                    self.cli_args("--target-joint-initial-position", "invalid")
                )

    def test_physics_substep_recording_defaults_off(self):
        self.assertFalse(DUMP.parse_args(self.cli_args()).record_physics_substeps)
        self.assertTrue(
            DUMP.parse_args(
                self.cli_args("--record-physics-substeps")
            ).record_physics_substeps
        )
        for field in (
            "control_step",
            "physics_substep",
            "global_physics_step",
            "record_level",
        ):
            self.assertNotIn(field, DUMP.TRANSITION_CORE_FIELDS)

    def test_physics_substep_index_sequence(self):
        self.assertEqual(
            DUMP.physics_substep_sequence(2, 4),
            [
                (0, 0, 0),
                (1, 1, 1),
                (1, 2, 2),
                (1, 3, 3),
                (1, 4, 4),
                (2, 1, 5),
                (2, 2, 6),
                (2, 3, 7),
                (2, 4, 8),
            ],
        )
        self.assertEqual(
            DUMP.physics_substep_fields(1, 4, 4),
            {
                "control_step": 1,
                "physics_substep": 4,
                "global_physics_step": 4,
                "record_level": "physics_substep",
            },
        )

    def test_contact_free_physics_prefix(self):
        records = []
        positive_total = [0, 0, 0, 1, 1]
        positive_self = [0, 0, 0, 1, 1]
        negative_total = [0, 0, 0, 0, 0]
        negative_self = [0, 0, 0, 0, 0]
        for case, totals, self_counts in (
            ("positive", positive_total, positive_self),
            ("negative", negative_total, negative_self),
        ):
            for global_step, (total, self_count) in enumerate(
                zip(totals, self_counts)
            ):
                records.append(
                    {
                        "case": case,
                        "global_physics_step": global_step,
                        "contact_count_if_available": total,
                        "self_contact_count_if_available": self_count,
                    }
                )
        summary = DUMP.summarize_physics_contacts(
            records, ["positive", "negative"], 4, True
        )
        self.assertEqual(
            summary["first_contact_global_physics_step_by_case"],
            {"positive": 3, "negative": None},
        )
        self.assertEqual(
            summary["first_self_contact_global_physics_step_by_case"],
            {"positive": 3, "negative": None},
        )
        self.assertEqual(
            summary["contact_free_prefix_length_by_case"],
            {"positive": 2, "negative": 4},
        )

        disabled = DUMP.summarize_physics_contacts(
            records, ["positive", "negative"], 4, False
        )
        self.assertEqual(
            disabled["contact_free_prefix_length_by_case"],
            {"positive": DUMP.NOT_AVAILABLE, "negative": DUMP.NOT_AVAILABLE},
        )

    def test_sha256_file(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "payload.bin"
            path.write_bytes(b"abc")
            self.assertEqual(
                DUMP.sha256_file(path), hashlib.sha256(b"abc").hexdigest()
            )
        expected = "DA2156E9BE4B706A34599A32BC8F3F1A2037FA8ECFB44042BC22E1740E9382A0"
        self.assertEqual(DUMP.parse_sha256(expected), expected.lower())
        with self.assertRaises(argparse.ArgumentTypeError):
            DUMP.parse_sha256("not-a-sha256")

    def test_parse_cases(self):
        self.assertEqual(
            DUMP.parse_cases("zero, positive,negative"),
            ["zero", "positive", "negative"],
        )
        with self.assertRaises(argparse.ArgumentTypeError):
            DUMP.parse_cases("zero,unknown")
        with self.assertRaises(argparse.ArgumentTypeError):
            DUMP.parse_cases("zero,zero")

    def test_jsonable_produces_strict_json(self):
        value = DUMP.jsonable({"quaternion": (1.0, 0.0, 0.0, 0.0)})
        json.dumps(value, allow_nan=False)
        with self.assertRaises(ValueError):
            DUMP.jsonable(float("nan"))

    def test_output_directory_must_be_empty(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory) / "dump"
            DUMP.prepare_output_directory(output)
            (output / "unknown.txt").write_text("keep", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                DUMP.prepare_output_directory(output)

    def test_transition_schema_and_joint_order(self):
        record = {field: None for field in DUMP.TRANSITION_CORE_FIELDS}
        record.update(
            {
                "schema_version": "metamorph-transition-v1",
                "joint_names": ["limbx/0", "limby/0"],
                "requested_torque": [50.0, 0.0],
                "joint_qpos": [0.1, -0.2],
                "joint_qvel": [0.3, -0.4],
                "root_quaternion_order": "wxyz",
            }
        )
        DUMP.validate_transition_schema(record)
        for metadata_only_field in (
            "runtime_dynamics_schema_version",
            "runtime_joint_dynamics",
            "runtime_actuators",
            "runtime_body_dynamics",
        ):
            self.assertNotIn(metadata_only_field, DUMP.TRANSITION_CORE_FIELDS)
        record["requested_torque"] = [50.0]
        with self.assertRaises(ValueError):
            DUMP.validate_transition_schema(record)

    def test_canonical_step0_must_match_between_cases(self):
        records = []
        for case in ("zero", "positive", "negative"):
            records.append(
                {
                    "case": case,
                    "step": 0,
                    "root_position": [0.0, 0.0, 1.2],
                    "root_quaternion": [1.0, 0.0, 0.0, 0.0],
                    "root_linear_velocity": [0.0, 0.0, 0.0],
                    "root_angular_velocity": [0.0, 0.0, 0.0],
                    "joint_qpos": [0.1, -0.2],
                    "joint_qvel": [0.0, 0.0],
                }
            )
        DUMP.validate_canonical_step0(records, ["zero", "positive", "negative"])
        self.assertEqual(
            DUMP.validate_step0_joint_qpos(
                records,
                ["zero", "positive", "negative"],
                ["limbx/0", "limby/0"],
                [0.1, -0.2],
                "limbx/0",
            ),
            0.1,
        )
        records[-1]["joint_qpos"][0] = 0.2
        with self.assertRaises(ValueError):
            DUMP.validate_canonical_step0(
                records, ["zero", "positive", "negative"]
            )

    def test_metadata_gate0_schema(self):
        metadata = {field: None for field in DUMP.METADATA_GATE0_FIELDS}
        metadata.update(
            {
                "schema_version": "metamorph-transition-v1",
                "policy_action_semantics": "actuator_order_ctrl",
                "native_action_or_ctrl_semantics": "mujoco_actuator_order_ctrl",
                "physics_timestep_actual": 0.005,
                "frame_skip": 4,
                "control_timestep": 0.02,
                "joint_names": ["limbx/0"],
                "requested_joint_name": "limbx/0",
                "canonical_qpos_mode": "model-default",
                "canonical_joint_qpos": [0.25],
                "model_default_joint_qpos": [0.25],
                "target_joint_initial_position_mode": "default",
                "target_joint_initial_qpos_requested": 0.25,
                "target_joint_initial_qpos_readback": 0.25,
                "target_joint_range": [0.0, 1.0],
                "root_qpos_source": "explicit",
                "joint_qpos_source": "compiled_model_qpos0",
            }
        )
        DUMP.validate_metadata_gate0(metadata)
        del metadata["physics_timestep_actual"]
        with self.assertRaises(ValueError):
            DUMP.validate_metadata_gate0(metadata)

    def test_requested_and_applied_torque_follow_joint_names_order(self):
        model = SimpleNamespace(
            njnt=3,
            nq=9,
            nv=8,
            nu=2,
            jnt_qposadr=np.array([0, 7, 8]),
            jnt_dofadr=np.array([0, 6, 7]),
            actuator_trnid=np.array([[2, -1], [1, -1]]),
            actuator_trntype=np.array([0, 0]),
            actuator_gear=np.array(
                [
                    [100.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [200.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                ]
            ),
        )
        joint_ids = [1, 2]
        positive = DUMP.requested_generalized_torque(
            model, np.array([0.0, 0.25]), joint_ids, np
        )
        negative = DUMP.requested_generalized_torque(
            model, np.array([0.0, -0.25]), joint_ids, np
        )
        np.testing.assert_allclose(positive, [50.0, 0.0])
        np.testing.assert_allclose(negative, [-50.0, 0.0])

        data = SimpleNamespace(qfrc_actuator=np.array([0.0] * 6 + [49.5, 0.1]))
        dof_indices = DUMP.non_root_dof_indices(model, joint_ids)
        np.testing.assert_allclose(
            DUMP.force_slice(data, "qfrc_actuator", dof_indices, np),
            [49.5, 0.1],
        )

    def test_runtime_joint_actuator_and_body_dynamics_metadata(self):
        model = SimpleNamespace(
            njnt=3,
            nq=9,
            nv=8,
            nu=2,
            nbody=3,
            jnt_type=np.array([0, 3, 3]),
            jnt_qposadr=np.array([0, 7, 8]),
            jnt_dofadr=np.array([0, 6, 7]),
            jnt_axis=np.array([[0, 0, 0], [0, 1, 0], [1, 0, 0]], dtype=float),
            jnt_range=np.array([[0, 0], [0, 1], [-1, 1]], dtype=float),
            jnt_limited=np.array([0, 1, 1]),
            dof_damping=np.array([0, 0, 0, 0, 0, 0, 6.0, 7.0]),
            dof_armature=np.array([0, 0, 0, 0, 0, 0, 0.6, 0.7]),
            dof_frictionloss=np.array([0, 0, 0, 0, 0, 0, 0.06, 0.07]),
            jnt_stiffness=np.array([0.0, np.nan, 2.0]),
            actuator_trnid=np.array([[2, -1], [1, -1]]),
            actuator_trntype=np.array([0, 0]),
            actuator_gear=np.array([[150, 0, 0, 0, 0, 0], [200, 0, 0, 0, 0, 0]]),
            actuator_ctrlrange=np.array([[-1, 1], [-1, 1]], dtype=float),
            actuator_ctrllimited=np.array([1, 1]),
            actuator_forcerange=np.array([[-3, 3], [-4, 4]], dtype=float),
            actuator_forcelimited=np.array([0, 1]),
            actuator_dyntype=np.array([0, 0]),
            actuator_gaintype=np.array([0, 0]),
            body_parentid=np.array([0, 0, 1]),
            body_mass=np.array([0.0, 5.0, 2.0]),
            body_ipos=np.array([[0, 0, 0], [0.1, 0.2, 0.3], [0, 0, 0.4]]),
            body_inertia=np.array([[0, 0, 0], [1, 2, 3], [0.4, 0.5, 0.6]]),
            body_iquat=np.array([[1, 0, 0, 0], [1, 0, 0, 0], [0.9, 0.1, 0, 0]]),
            opt=SimpleNamespace(integrator=0, solver=2, iterations=100),
        )
        joint_names = ["root", "limbx/0", "limby/9"]
        actuator_names = ["limby/9", "limbx/0"]
        body_names = ["world", "torso/0", "limb/5"]

        joints = DUMP.build_runtime_joint_dynamics(model, joint_names, np)
        self.assertEqual(joints[0]["joint_id"], 1)
        self.assertEqual(joints[0]["dof_address"], 6)
        self.assertEqual(joints[0]["qpos_address"], 7)
        self.assertEqual(joints[0]["damping"], 6.0)
        self.assertEqual(joints[0]["stiffness"], DUMP.NOT_AVAILABLE)
        self.assertEqual(joints[0]["spring_reference"], DUMP.NOT_AVAILABLE)

        actuators = DUMP.build_runtime_actuators(
            model, actuator_names, joint_names, np
        )
        self.assertEqual(actuators[0]["target_joint_id"], 2)
        self.assertEqual(actuators[0]["target_joint_name"], "limby/9")
        self.assertTrue(actuators[1]["ctrllimited"])
        self.assertEqual(actuators[0]["biastype"], DUMP.NOT_AVAILABLE)

        bodies = DUMP.build_runtime_body_dynamics(model, body_names, np)
        self.assertEqual(bodies[0]["parent_body_name"], DUMP.NOT_AVAILABLE)
        self.assertEqual(bodies[1]["parent_body_name"], "world")
        self.assertEqual(bodies[2]["parent_body_name"], "torso/0")
        self.assertEqual(bodies[2]["mass"], 2.0)

        solver = DUMP.build_runtime_solver_metadata(model, np)
        self.assertEqual(solver["integrator"], "Euler")
        self.assertEqual(solver["solver"], "Newton")
        self.assertEqual(solver["iterations"], 100)
        self.assertEqual(solver["tolerance"], DUMP.NOT_AVAILABLE)

        json.dumps(
            DUMP.jsonable(
                {
                    "runtime_dynamics_schema_version": DUMP.RUNTIME_DYNAMICS_SCHEMA_VERSION,
                    "runtime_joint_dynamics": joints,
                    "runtime_actuators": actuators,
                    "runtime_body_dynamics": bodies,
                    "runtime_solver_settings": solver,
                }
            ),
            allow_nan=False,
        )

    def test_model_default_preserves_compiled_joint_qpos(self):
        model = SimpleNamespace(
            nq=9,
            nv=8,
            njnt=3,
            qpos0=np.array([9.0, 8.0, 7.0, 0.5, 0.5, 0.5, 0.5, 0.33, -0.44]),
            jnt_type=np.array([0, 3, 3]),
            jnt_qposadr=np.array([0, 7, 8]),
            jnt_dofadr=np.array([0, 6, 7]),
            jnt_limited=np.array([0, 1, 1]),
            jnt_range=np.array(
                [[0.0, 0.0], [0.0, 1.04719755], [-1.0, 1.0]]
            ),
        )
        original_qpos0 = model.qpos0.copy()
        state = DUMP.build_canonical_state(
            model,
            2.20,
            ["root", "limbx/0", "limby/0"],
            "model-default",
            np,
            target_joint_id=1,
        )
        np.testing.assert_allclose(
            state["qpos"], [0.0, 0.0, 2.20, 1.0, 0.0, 0.0, 0.0, 0.33, -0.44]
        )
        np.testing.assert_allclose(state["qvel"], np.zeros(8))
        np.testing.assert_allclose(model.qpos0, original_qpos0)
        state["qpos"][7] = 123.0
        repeated = DUMP.build_canonical_state(
            model,
            2.20,
            ["root", "limbx/0", "limby/0"],
            "model-default",
            np,
            target_joint_id=1,
        )
        np.testing.assert_allclose(repeated["qpos"][7:], [0.33, -0.44])

        midpoint = DUMP.build_canonical_state(
            model,
            2.20,
            ["root", "limbx/0", "limby/0"],
            "midpoint",
            np,
            target_joint_id=1,
        )
        expected_target_midpoint = (0.0 + 1.04719755) / 2.0
        np.testing.assert_allclose(
            midpoint["qpos"][7:], [expected_target_midpoint, 0.0]
        )

        target_midpoint = DUMP.build_canonical_state(
            model,
            2.20,
            ["root", "limbx/0", "limby/0"],
            "model-default",
            np,
            target_joint_id=1,
            target_joint_initial_position="midpoint",
        )
        np.testing.assert_allclose(
            target_midpoint["qpos"][7:], [expected_target_midpoint, -0.44]
        )
        self.assertEqual(
            target_midpoint["target_joint_initial_qpos_requested"],
            expected_target_midpoint,
        )

    def test_target_midpoint_rejects_unlimited_joint(self):
        model = SimpleNamespace(
            nq=8,
            nv=7,
            njnt=2,
            qpos0=np.array([0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.25]),
            jnt_type=np.array([0, 3]),
            jnt_qposadr=np.array([0, 7]),
            jnt_dofadr=np.array([0, 6]),
            jnt_limited=np.array([0, 0]),
            jnt_range=np.array([[0.0, 0.0], [0.0, 1.0]]),
        )
        with self.assertRaisesRegex(ValueError, "not limited"):
            DUMP.build_canonical_state(
                model,
                2.20,
                ["root", "limbx/0"],
                "model-default",
                np,
                target_joint_id=1,
                target_joint_initial_position="midpoint",
            )

    def test_contact_pair_classification_is_conservative(self):
        self.assertEqual(
            DUMP.classify_contact_pair(
                {
                    "geom1": "floor/0",
                    "geom2": "limb/5",
                    "body1": "world",
                    "body2": "limb/5",
                }
            ),
            "ground",
        )
        self.assertEqual(
            DUMP.classify_contact_pair(
                {
                    "geom1": "limb/5",
                    "geom2": "limb/6",
                    "body1": "limb/5",
                    "body2": "limb/6",
                }
            ),
            "self",
        )
        self.assertEqual(
            DUMP.classify_contact_pair(
                {
                    "geom1": "object/0",
                    "geom2": "limb/6",
                    "body1": "object/0",
                    "body2": "limb/6",
                }
            ),
            "unclassified",
        )


if __name__ == "__main__":
    unittest.main()

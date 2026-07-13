"""Pure-Python tests for dump_mujoco_transition helpers."""

import argparse
import hashlib
import importlib.util
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


if __name__ == "__main__":
    unittest.main()

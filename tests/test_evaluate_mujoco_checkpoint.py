import argparse
import importlib.util
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from metamorph.config import cfg


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "tools"
    / "evaluate_mujoco_checkpoint.py"
)
SPEC = importlib.util.spec_from_file_location(
    "evaluate_mujoco_checkpoint", MODULE_PATH
)
EVALUATOR = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = EVALUATOR
SPEC.loader.exec_module(EVALUATOR)


class FakeDistribution:
    def __init__(self):
        self.mean = torch.tensor([[0.2, -0.4, 0.7, 0.9]])

    def sample(self):
        return torch.tensor([[1.2, -1.3, 0.5, 0.6]])


class EvaluateMujocoCheckpointTests(unittest.TestCase):
    @staticmethod
    def fake_mujoco_env():
        class Model:
            joint_names = ("root", "hip", "knee")
            jnt_type = np.asarray([0, 3, 3])
            jnt_range = np.asarray([[0, 0], [-1, 1], [-2, 2]])
            nv = 8
            nu = 2
            actuator_names = ("hip_motor", "knee_motor")
            actuator_trnid = np.asarray([[1, 0], [2, 0]])
            body_names = ("world", "torso/0", "limb/0")
            geom_bodyid = np.asarray([0, 1, 2])
            opt = SimpleNamespace(timestep=0.005)

            def get_joint_qpos_addr(self, name):
                return {"root": (0, 7), "hip": 7, "knee": 8}[name]

            def get_joint_qvel_addr(self, name):
                return {"root": (0, 6), "hip": 6, "knee": 7}[name]

            def body_name2id(self, name):
                return self.body_names.index(name)

            def body_id2name(self, index):
                return self.body_names[index]

            def geom_id2name(self, index):
                return ("floor", "torso_geom", "limb_geom")[index]

        data = SimpleNamespace(
            time=0.02,
            qpos=np.asarray([1, 2, 3, 1, 0, 0, 0, 0.25, -0.5]),
            qvel=np.asarray([1, 2, 3, 4, 5, 6, 0.7, -0.8]),
            body_xpos=np.asarray([[0, 0, 0], [1, 2, 0.55], [1, 2, 0.3]]),
            body_xvelp=np.asarray([[0, 0, 0], [0.1, 0.2, 0.3], [0, 0, 0]]),
            body_xvelr=np.asarray([[0, 0, 0], [0.4, 0.5, 0.6], [0, 0, 0]]),
            ctrl=np.asarray([0.0, 0.0]),
            actuator_force=np.asarray([0.0, 0.0]),
            qfrc_actuator=np.arange(8),
            qfrc_passive=np.arange(8) * 0.1,
            ncon=0,
            contact=[],
        )
        return SimpleNamespace(
            sim=SimpleNamespace(model=Model(), data=data), frame_skip=4
        )

    def test_free_root_and_ordinary_joint_mapping_are_separate(self):
        base_env = self.fake_mujoco_env()
        metadata = EVALUATOR.build_state_trajectory_metadata(base_env)

        self.assertEqual(metadata["root_free_joint"]["qpos_indices"], list(range(7)))
        self.assertEqual(metadata["root_free_joint"]["qvel_indices"], list(range(6)))
        self.assertEqual(metadata["ordered_joint_names"], ["hip", "knee"])
        self.assertEqual(
            [item["qpos_indices"] for item in metadata["ordinary_joint_mapping"]],
            [[7], [8]],
        )
        self.assertEqual(metadata["physics_timestep"], 0.005)
        self.assertEqual(metadata["control_dt"], 0.02)

    def test_state_snapshot_uses_joint_mapping_and_direct_force_arrays(self):
        base_env = self.fake_mujoco_env()
        metadata = EVALUATOR.build_state_trajectory_metadata(base_env)
        snapshot = EVALUATOR.capture_state_trajectory(
            base_env, metadata, EVALUATOR.FiniteTracker()
        )

        self.assertEqual(snapshot["root_free_joint_position"], [1.0, 2.0, 3.0])
        self.assertEqual(snapshot["root_free_joint_orientation_wxyz"], [1.0, 0.0, 0.0, 0.0])
        self.assertEqual(snapshot["ordered_joint_qpos"], [0.25, -0.5])
        self.assertEqual(snapshot["ordered_joint_qvel"], [0.7, -0.8])
        self.assertEqual(len(snapshot["full_qpos"]), 9)
        self.assertEqual(len(snapshot["full_qvel"]), 8)
        self.assertEqual(len(snapshot["qfrc_actuator"]), 8)
        self.assertEqual(len(snapshot["qfrc_passive"]), 8)
        self.assertEqual(snapshot["contact_count"], 0)

    def test_evaluator_cutoff_does_not_claim_environment_termination(self):
        self.assertFalse(EVALUATOR.evaluator_cutoff_reached(219, 220))
        self.assertTrue(EVALUATOR.evaluator_cutoff_reached(220, 220))
        self.assertFalse(EVALUATOR.evaluator_cutoff_reached(1000, None))

    def test_trajectory_cli_accepts_deterministic_reset_and_step_limit(self):
        args = EVALUATOR.parser().parse_args(
            [
                "--checkpoint", "checkpoint.pt",
                "--walker-dir", "walkers",
                "--morphology-id", "walker",
                "--action-mode", "zero",
                "--output-dir", "output",
                "--record-state-trajectory",
                "--max-eval-steps", "220",
                "--reset-noise-scale", "0.0",
            ]
        )
        self.assertTrue(args.record_state_trajectory)
        self.assertEqual(args.max_eval_steps, 220)
        self.assertEqual(args.reset_noise_scale, 0.0)

    def observation(self):
        return {
            "act_padding_mask": torch.tensor(
                [[False, False, True, True]]
            )
        }

    def test_zero_mean_and_sample_preserve_raw_policy_semantics(self):
        distribution = FakeDistribution()
        observation = self.observation()
        zero, mask = EVALUATOR.choose_raw_action(
            distribution, observation, "zero"
        )
        mean, _ = EVALUATOR.choose_raw_action(
            distribution, observation, "mean"
        )
        sample, _ = EVALUATOR.choose_raw_action(
            distribution, observation, "sample"
        )
        self.assertTrue(torch.equal(zero, torch.zeros_like(distribution.mean)))
        self.assertIs(mean, distribution.mean)
        self.assertAlmostEqual(sample[0, 0].item(), 1.2, places=6)
        self.assertEqual(mask.tolist(), [[True, True, False, False]])

    def test_action_diagnostics_ignore_padding_and_do_not_clamp(self):
        tracker = EVALUATOR.FiniteTracker()
        action = FakeDistribution().sample()
        diagnostics = EVALUATOR.raw_action_diagnostics(
            action, ~self.observation()["act_padding_mask"], tracker
        )
        self.assertEqual(diagnostics["_valid_action_count"], 2)
        self.assertEqual(diagnostics["_out_of_bounds_count"], 2)
        self.assertTrue(diagnostics["_action_values_finite"])
        self.assertEqual(
            diagnostics["raw_action_out_of_bounds_fraction"], 1.0
        )
        self.assertAlmostEqual(diagnostics["raw_action_max"], 1.2, places=6)
        self.assertAlmostEqual(diagnostics["raw_action_min"], -1.3, places=6)

    def test_evaluator_uses_official_fall_info_without_recomputing(self):
        tracker = EVALUATOR.FiniteTracker()
        measurement = EVALUATOR.official_fall_measurement(
            {
                "formal_torso_height": 0.42,
                "formal_fallen_threshold": 0.5,
                # Deliberately inconsistent: the evaluator must preserve
                # official fields, not derive a new fallen decision.
                "fallen": False,
            },
            tracker,
        )
        self.assertEqual(measurement["formal_torso_height"], 0.42)
        self.assertEqual(measurement["formal_fallen_threshold"], 0.5)
        self.assertEqual(
            measurement["formal_torso_height_source"],
            "official_termination_info",
        )
        self.assertNotIn("fallen", measurement)

    def test_nonfinite_values_are_json_safe_and_flagged(self):
        tracker = EVALUATOR.FiniteTracker()
        self.assertIsNone(tracker.scalar(float("nan")))
        self.assertFalse(tracker.all_values_finite)
        with tempfile.TemporaryDirectory() as temp_dir:
            EVALUATOR.write_outputs(
                Path(temp_dir),
                {"finite": False},
                {"value": None},
                [{"value": None}],
            )

    def test_native_checkpoint_restore_supports_cpu_map_location(self):
        from metamorph.algos.ppo.inherit_weight import restore_from_checkpoint

        source = torch.nn.Linear(2, 1)
        target = torch.nn.Linear(2, 1)
        with torch.no_grad():
            source.weight.fill_(0.25)
            source.bias.fill_(0.5)
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint = Path(temp_dir) / "checkpoint.pt"
            torch.save([source, {"proprioceptive": "ob-rms"}], checkpoint)
            old_checkpoint = cfg.PPO.CHECKPOINT_PATH
            old_full_model = cfg.MODEL.FINETUNE.FULL_MODEL
            try:
                cfg.PPO.CHECKPOINT_PATH = str(checkpoint)
                cfg.MODEL.FINETUNE.FULL_MODEL = True
                ob_rms = restore_from_checkpoint(
                    target, map_location=torch.device("cpu")
                )
            finally:
                cfg.PPO.CHECKPOINT_PATH = old_checkpoint
                cfg.MODEL.FINETUNE.FULL_MODEL = old_full_model
        self.assertTrue(torch.equal(source.weight, target.weight))
        self.assertTrue(torch.equal(source.bias, target.bias))
        self.assertEqual(ob_rms, {"proprioceptive": "ob-rms"})

    def test_validate_refuses_existing_outputs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            walker = root / "walkers"
            (walker / "xml").mkdir(parents=True)
            (walker / "metadata").mkdir()
            morphology = "walker"
            (walker / "xml" / f"{morphology}.xml").write_text(
                "<xml/>", encoding="utf-8"
            )
            (walker / "metadata" / f"{morphology}.json").write_text(
                "{}", encoding="utf-8"
            )
            checkpoint = root / "checkpoint.pt"
            checkpoint.write_bytes(b"checkpoint")
            output = root / "output"
            output.mkdir()
            (output / "summary.json").write_text("{}", encoding="utf-8")
            args = argparse.Namespace(
                episodes=1,
                checkpoint=str(checkpoint),
                walker_dir=str(walker),
                morphology_id=morphology,
                cfg="configs/ft.yaml",
                output_dir=str(output),
            )
            with self.assertRaisesRegex(FileExistsError, "refusing"):
                EVALUATOR.validate_args(args)


if __name__ == "__main__":
    unittest.main()

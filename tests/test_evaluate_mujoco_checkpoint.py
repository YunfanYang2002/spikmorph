import argparse
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

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

import argparse
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

from metamorph.config import cfg


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "tools"
    / "run_mujoco_control_experiments.py"
)
SPEC = importlib.util.spec_from_file_location("run_mujoco_control_experiments", MODULE_PATH)
RUNNER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = RUNNER
SPEC.loader.exec_module(RUNNER)


class MujocoControlRunnerTests(unittest.TestCase):
    def args(self):
        return argparse.Namespace(
            cfg="configs/ft.yaml",
            walker_dir="walkers",
            morphology=RUNNER.DEFAULT_MORPHOLOGY,
            action_std=0.3,
            num_envs=4,
            timesteps=128,
            max_state_action_pairs=51_200,
            batch_size=512,
            target_kl=0.02,
            max_parallel=2,
        )

    def test_job_matrix_is_seed_lr_cartesian_product(self):
        jobs = RUNNER.build_job_matrix(
            seeds=(1409, 1410, 1411),
            base_lrs=(1.0e-4, 1.5e-4, 2.0e-4),
            batch_root=Path("batch"),
        )
        self.assertEqual(len(jobs), 9)
        self.assertEqual((jobs[0].seed, jobs[0].base_lr), (1409, 1.0e-4))
        self.assertEqual((jobs[-1].seed, jobs[-1].base_lr), (1411, 2.0e-4))

    def test_command_uses_formal_entry_and_exact_control_profile(self):
        args = self.args()
        job = RUNNER.JobSpec("job", 1409, 1.5e-4, "out")
        command = RUNNER.build_training_command(args, job, "cpu")
        self.assertEqual(Path(command[2]).name, "train_ppo.py")
        expected = {
            "ENV.WALKERS": '["floor-1409-0-3-01-15-56-55"]',
            "PPO.MAX_STATE_ACTION_PAIRS": "51200",
            "PPO.NUM_ENVS": "4",
            "PPO.TIMESTEPS": "128",
            "PPO.BATCH_SIZE": "512",
            "PPO.BASE_LR": "0.00015",
            "PPO.KL_TARGET_COEF": "2.0",
            "MODEL.ACTION_STD_FIXED": "True",
            "MODEL.ACTION_STD": "0.3",
            "MODEL.TRANSFORMER.DROPOUT": "0.0",
            "PPO.CHECKPOINT_PATH": "",
        }
        for key, value in expected.items():
            self.assertEqual(command[command.index(key) + 1], value)

    def test_budget_must_be_exact_multiple_of_rollout(self):
        args = self.args()
        RUNNER.validate_args(args)
        args.max_state_action_pairs = 51_201
        with self.assertRaisesRegex(ValueError, "exactly divisible"):
            RUNNER.validate_args(args)

    def test_native_config_parser_accepts_runner_overrides(self):
        local_cfg = cfg.clone()
        local_cfg.merge_from_file("configs/ft.yaml")
        local_cfg.merge_from_list(
            [
                "ENV.WALKERS",
                '["floor-1409-0-3-01-15-56-55"]',
                "PPO.CHECKPOINT_PATH",
                "",
                "PPO.KL_TARGET_COEF",
                "2.0",
                "PPO.BATCH_SIZE",
                "512",
            ]
        )
        self.assertEqual(local_cfg.ENV.WALKERS, [RUNNER.DEFAULT_MORPHOLOGY])
        self.assertEqual(local_cfg.PPO.CHECKPOINT_PATH, "")
        self.assertEqual(local_cfg.PPO.KL_TARGET_COEF, 2.0)
        self.assertEqual(local_cfg.PPO.BATCH_SIZE, 512)

    def test_archive_excludes_checkpoint_and_tensorboard(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "batch"
            (root / "jobs" / "job" / "tensorboard").mkdir(parents=True)
            (root / "manifest.json").write_text("{}", encoding="utf-8")
            (root / "jobs" / "job" / "Unimal-v0.pt").write_bytes(b"large")
            (root / "jobs" / "job" / "train.log").write_text(
                "ok", encoding="utf-8"
            )
            (root / "jobs" / "job" / "tensorboard" / "events").write_bytes(
                b"large"
            )
            archive = RUNNER.archive_batch(root, Path(temp_dir) / "archives")
            import zipfile

            with zipfile.ZipFile(archive) as payload:
                names = payload.namelist()
            self.assertTrue(any(name.endswith("manifest.json") for name in names))
            self.assertTrue(any(name.endswith("train.log") for name in names))
            self.assertFalse(any(name.endswith(".pt") for name in names))
            self.assertFalse(any("tensorboard" in name for name in names))


if __name__ == "__main__":
    unittest.main()

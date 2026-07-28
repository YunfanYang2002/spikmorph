import argparse
import importlib.util
from pathlib import Path
import subprocess
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
    def test_direct_script_bootstraps_repository_import_path(self):
        code = (
            "import runpy, sys; "
            f"namespace = runpy.run_path({str(MODULE_PATH)!r}, "
            "run_name='evaluator_import_probe'); "
            "print(str(namespace['REPO_ROOT']) in sys.path)"
        )
        completed = subprocess.run(
            [sys.executable, "-B", "-c", code],
            cwd=MODULE_PATH.parent,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(completed.stdout.strip(), "True")

    @staticmethod
    def fake_mujoco_env():
        class Model:
            joint_names = ("root", "hip", "knee")
            njnt = 3
            nbody = 3
            ngeom = 3
            jnt_type = np.asarray([0, 3, 3])
            jnt_bodyid = np.asarray([1, 2, 2])
            jnt_range = np.asarray([[0, 0], [-1, 1], [-2, 2]])
            jnt_limited = np.asarray([False, True, True])
            jnt_qposadr = np.asarray([0, 7, 8])
            jnt_dofadr = np.asarray([0, 6, 7])
            jnt_margin = np.asarray([0.0, 0.0, 0.0])
            jnt_solref = np.asarray([[0.02, 1.0]] * 3)
            jnt_solimp = np.asarray([[0.0, 0.99, 0.01, 0.5, 2.0]] * 3)
            jnt_stiffness = np.asarray([0.0, 1.0, 1.0])
            dof_damping = np.ones(8)
            dof_armature = np.ones(8)
            dof_frictionloss = np.zeros(8)
            nv = 8
            nu = 2
            actuator_names = ("hip_motor", "knee_motor")
            actuator_trnid = np.asarray([[1, 0], [2, 0]])
            body_names = ("world", "torso/0", "limb/0")
            geom_names = ("floor", "torso_geom", "limb_geom")
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
            body_xquat=np.asarray(
                [[1, 0, 0, 0], [0.9, 0.1, 0.2, 0.3], [1, 0, 0, 0]]
            ),
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
            sim=SimpleNamespace(model=Model(), data=data), frame_skip=4,
            step_count=1,
        )

    def test_free_root_and_ordinary_joint_mapping_are_separate(self):
        base_env = self.fake_mujoco_env()
        metadata = EVALUATOR.build_state_trajectory_metadata(base_env)

        self.assertEqual(metadata["root_free_joint"]["qpos_indices"], list(range(7)))
        self.assertEqual(metadata["root_free_joint"]["qvel_indices"], list(range(6)))
        self.assertEqual(metadata["ordered_joint_names"], ["hip", "knee"])
        self.assertEqual(metadata["joint_index_map"], {"0": "hip", "1": "knee"})
        self.assertTrue(metadata["all_ordinary_joints_one_dof_hinge"])
        self.assertEqual(metadata["root_body_name"], "torso/0")
        self.assertTrue(metadata["root_body_is_torso_body"])
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
        self.assertEqual(
            snapshot["root_world_orientation_wxyz"], [0.9, 0.1, 0.2, 0.3]
        )
        self.assertEqual(
            snapshot["root_local_to_env_position_xyz"], [1.0, 2.0, 0.55]
        )
        self.assertEqual(snapshot["ordered_joint_qpos"], [0.25, -0.5])
        self.assertEqual(snapshot["ordered_joint_qvel"], [0.7, -0.8])
        self.assertEqual(snapshot["joint_qpos"], [0.25, -0.5])
        self.assertEqual(snapshot["joint_qvel"], [0.7, -0.8])
        self.assertEqual(snapshot["joint_qfrc_actuator"], [6.0, 7.0])
        np.testing.assert_allclose(
            snapshot["joint_qfrc_passive"], [0.6, 0.7]
        )
        self.assertEqual(snapshot["joint_actuator_ctrl"], [0.0, 0.0])
        self.assertEqual(len(snapshot["full_qpos"]), 9)
        self.assertEqual(len(snapshot["full_qvel"]), 8)
        self.assertEqual(len(snapshot["qfrc_actuator"]), 8)
        self.assertEqual(len(snapshot["qfrc_passive"]), 8)
        self.assertEqual(snapshot["contact_count"], 0)

    def test_joint_limit_mapping_uses_compiled_addresses_and_parameters(self):
        base_env = self.fake_mujoco_env()
        trajectory = EVALUATOR.build_state_trajectory_metadata(base_env)
        mapping = EVALUATOR.build_joint_limit_probe_mapping(
            base_env, ["hip", "knee"], trajectory
        )
        hip = mapping["joints"][0]
        self.assertEqual(hip["joint_id"], 1)
        self.assertEqual(hip["qpos_address"], 7)
        self.assertEqual(hip["dof_address"], 6)
        self.assertEqual(hip["jnt_solref"], [0.02, 1.0])
        self.assertEqual(hip["jnt_solimp"], [0.0, 0.99, 0.01, 0.5, 2.0])
        self.assertEqual(hip["dof_damping"], 1.0)
        self.assertEqual(hip["dof_armature"], 1.0)

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

    def test_trajectory_cli_is_default_off(self):
        args = EVALUATOR.parser().parse_args(
            [
                "--checkpoint", "checkpoint.pt",
                "--walker-dir", "walkers",
                "--morphology-id", "walker",
                "--action-mode", "zero",
                "--output-dir", "output",
            ]
        )
        self.assertFalse(args.record_state_trajectory)
        self.assertFalse(args.record_joint_limit_substeps)
        self.assertEqual(args.joint_limit_probe_names, [])
        self.assertIsNone(args.max_eval_steps)

    def test_joint_limit_substep_cli_is_opt_in(self):
        args = EVALUATOR.parser().parse_args(
            [
                "--checkpoint", "checkpoint.pt",
                "--walker-dir", "walkers",
                "--morphology-id", "walker",
                "--action-mode", "zero",
                "--output-dir", "output",
                "--record-joint-limit-substeps",
                "--joint-limit-probe-names", "limby/12", "limby/11",
            ]
        )
        self.assertTrue(args.record_joint_limit_substeps)
        self.assertEqual(args.joint_limit_probe_names, ["limby/12", "limby/11"])

    def test_dense_and_sparse_constraint_jacobian_rows(self):
        dense = SimpleNamespace(efc_J=np.asarray([[0.0, -1.0, 2.0], [3.0, 0.0, 0.0]]))
        self.assertEqual(
            EVALUATOR.constraint_jacobian_row(dense, 0, 2, 3),
            ([1, 2], [-1.0, 2.0]),
        )
        sparse = SimpleNamespace(
            efc_J=np.asarray([-1.0, 2.0, 3.0]),
            efc_J_rownnz=np.asarray([2, 1]),
            efc_J_rowadr=np.asarray([0, 2]),
            efc_J_colind=np.asarray([1, 2, 0]),
        )
        self.assertEqual(
            EVALUATOR.constraint_jacobian_row(sparse, 0, 2, 3),
            ([1, 2], [-1.0, 2.0]),
        )

    def test_step_proxy_calls_live_step_exactly_once(self):
        events = []

        class Sim:
            def step(self):
                events.append("mj_step")

        class Recorder:
            def capture_pre_step(self):
                events.append("pre")
                return {"pre": True}

            def capture_post_step(self, pre):
                self.pre = pre
                events.append("post")

        recorder = Recorder()
        proxy = EVALUATOR.JointLimitRecordingSimProxy(Sim(), recorder)
        proxy.step()
        self.assertEqual(events, ["pre", "mj_step", "post"])
        self.assertEqual(recorder.pre, {"pre": True})
        self.assertIsNone(proxy.callback_error)

    def test_oracle_summary_counts_four_substeps_and_missing_rows(self):
        mapping = {
            "joints": [
                {"joint_name": "limby/12", "jnt_range": [-1.57, 0.0]},
                {"joint_name": "limby/11", "jnt_range": [-1.57, 0.0]},
            ]
        }
        records = []
        for substep in range(4):
            joints = []
            for name in ("limby/12", "limby/11"):
                joints.append(
                    {
                        "joint_name": name,
                        "post_step_qpos": 0.001 if substep >= 2 else 0.0,
                        "limit_constraint_present": substep >= 2,
                        "selected_dof_limit_generalized_force": (
                            -2.0 if substep >= 2 else None
                        ),
                        "qfrc_constraint_reconstruction_error": 0.0,
                    }
                )
            records.append(
                {
                    "control_step": 1,
                    "physics_substep_in_control": substep,
                    "global_physics_step": substep + 1,
                    "contains_limb_0_floor_0_contact": substep == 1,
                    "joints": joints,
                }
            )
        oracle = EVALUATOR.build_joint_limit_oracle_outputs(records, mapping, 1, 4)
        self.assertTrue(oracle["validation"]["record_count_matches"])
        self.assertEqual(oracle["summary"]["first_ground_contact_substep"], 1)
        first = oracle["summary"]["joints"]["limby/12"]["first_limit_constraint"]
        self.assertEqual(first["physics_substep_in_control"], 2)

    def test_velocity_frame_and_quaternion_metadata_are_explicit(self):
        metadata = EVALUATOR.build_state_trajectory_metadata(
            self.fake_mujoco_env()
        )
        conventions = metadata["coordinate_conventions"]
        self.assertEqual(conventions["body_orientation"], "quaternion wxyz")
        self.assertIn("world-aligned", conventions["body_linear_velocity"])
        self.assertIn("mj_objectVelocity", metadata["body_velocity_convention"])

    def test_zero_reset_metadata_reports_effective_facts(self):
        metadata = EVALUATOR.native_reset_metadata(0.0, 0.0)
        self.assertEqual(metadata["reset_noise_scale"], 0.0)
        self.assertFalse(metadata["reset_state_noise_active"])
        self.assertFalse(metadata["qpos_qvel_noise_preserved"])
        self.assertTrue(metadata["deterministic_reset_effective"])
        self.assertTrue(metadata["deterministic_reset_forced"])

    def test_terminal_capture_precedes_auto_reset(self):
        base_env = self.fake_mujoco_env()
        metadata = EVALUATOR.build_state_trajectory_metadata(base_env)
        snapshots = {}
        tracker = EVALUATOR.FiniteTracker()

        class TerminationWrapper:
            def step(self, action):
                return "terminal_obs", 1.0, True, {"fallen": True}

        wrapper = TerminationWrapper()
        EVALUATOR.install_pre_autoreset_state_capture(
            wrapper, base_env, metadata, tracker, snapshots
        )

        class DummyAutoReset:
            def step(self, action):
                result = wrapper.step(action)
                if result[2]:
                    base_env.sim.data.qpos[:7] = np.asarray(
                        [9, 9, 9, 1, 0, 0, 0]
                    )
                    base_env.sim.data.body_xpos[1] = np.asarray([9, 9, 9])
                return result

        result = DummyAutoReset().step([0.0, 0.0])

        self.assertTrue(result[2])
        self.assertTrue(result[3]["fallen"])
        self.assertEqual(
            snapshots["latest"]["root_free_joint_position"],
            [1.0, 2.0, 3.0],
        )
        self.assertEqual(
            snapshots["latest"]["root_world_position_xyz"],
            [1.0, 2.0, 0.55],
        )

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

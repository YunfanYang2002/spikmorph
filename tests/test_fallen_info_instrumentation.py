import importlib.util
from pathlib import Path
import sys
from types import ModuleType
import unittest
from unittest import mock

import gym
import numpy as np

from metamorph.envs.vec_env.dummy_vec_env import DummyVecEnv


HFIELD_PATH = (
    Path(__file__).resolve().parents[1]
    / "metamorph"
    / "envs"
    / "wrappers"
    / "hfield.py"
)
STUBS = {
    name: ModuleType(name)
    for name in (
        "metamorph.utils.exception",
        "metamorph.utils.geom",
        "metamorph.utils.mjpy",
        "metamorph.utils.spaces",
    )
}
with mock.patch.dict(sys.modules, STUBS):
    SPEC = importlib.util.spec_from_file_location(
        "hfield_fallen_instrumentation_test", HFIELD_PATH
    )
    HFIELD = importlib.util.module_from_spec(SPEC)
    SPEC.loader.exec_module(HFIELD)
TerminateOnFalling = HFIELD.TerminateOnFalling


class HeightSequenceEnv(gym.Env):
    def __init__(self, heights, reset_height=0.9, threshold=0.5):
        self.heights = list(heights)
        self.reset_height = float(reset_height)
        self.metadata = {"fall_threshold": float(threshold)}
        self.observation_space = gym.spaces.Dict(
            {
                "torso_height": gym.spaces.Box(
                    low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32
                )
            }
        )
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32
        )
        self.current_height = self.reset_height
        self.reset_count = 0

    def reset(self):
        self.reset_count += 1
        self.current_height = self.reset_height
        return {"torso_height": np.asarray([self.current_height], dtype=np.float32)}

    def step(self, _action):
        self.current_height = float(self.heights.pop(0))
        return (
            {"torso_height": np.asarray([self.current_height], dtype=np.float32)},
            0.0,
            False,
            {},
        )


class FallenInfoInstrumentationTests(unittest.TestCase):
    def test_nonterminal_step_exposes_official_height_without_changing_fallen(self):
        wrapped = TerminateOnFalling(HeightSequenceEnv([0.7], threshold=0.5))
        _obs, _reward, done, info = wrapped.step(np.zeros(1, dtype=np.float32))
        self.assertFalse(done)
        self.assertFalse(info["fallen"])
        self.assertAlmostEqual(float(info["formal_torso_height"][0]), 0.7)
        self.assertAlmostEqual(info["formal_fallen_threshold"], 0.5)
        self.assertEqual(
            info["fallen"],
            bool(wrapped.has_fallen({"torso_height": np.asarray([0.7])})),
        )

    def test_terminal_height_survives_vecenv_autoreset(self):
        base = HeightSequenceEnv([0.4], reset_height=0.9, threshold=0.5)
        with mock.patch.dict(np.__dict__, {"bool": bool}):
            vec = DummyVecEnv([lambda: TerminateOnFalling(base)])
            try:
                vec.reset()
                vec.step_async(np.zeros((1, 1), dtype=np.float32))
                _obs, _reward, done, infos = vec.step_wait()
            finally:
                vec.close()
        self.assertTrue(done[0])
        self.assertTrue(infos[0]["fallen"])
        self.assertAlmostEqual(
            float(infos[0]["formal_torso_height"][0]), 0.4
        )
        self.assertAlmostEqual(base.current_height, 0.9)
        self.assertGreaterEqual(base.reset_count, 2)


if __name__ == "__main__":
    unittest.main()

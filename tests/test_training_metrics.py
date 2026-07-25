import math
import unittest

from metamorph.config import cfg
from metamorph.utils.meter import TrainMeter


class TrainingMetricsTests(unittest.TestCase):
    def setUp(self):
        self.previous_walkers = list(cfg.ENV.WALKERS)
        cfg.ENV.WALKERS = ["walker"]

    def tearDown(self):
        cfg.ENV.WALKERS = self.previous_walkers

    def test_transition_velocity_and_falls_do_not_require_completed_episode(self):
        meter = TrainMeter()
        meter.add_ep_info(
            [
                {"name": "walker", "x_vel": 0.1, "fallen": False},
                {"name": "walker", "x_vel": 0.3, "fallen": True},
            ]
        )
        meter.update_mean()
        stats = meter.get_stats()["__env__"]
        self.assertTrue(math.isclose(stats["transition_vel"][-1], 0.2))
        self.assertEqual(stats["fallen_count"][-1], 1)
        self.assertEqual(stats["ep_len"], [])

    def test_rollout_counters_reset_between_updates(self):
        meter = TrainMeter()
        meter.add_ep_info([{"name": "walker", "x_vel": 0.2, "fallen": True}])
        meter.update_mean()
        meter.add_ep_info([{"name": "walker", "x_vel": 0.4, "fallen": False}])
        meter.update_mean()
        stats = meter.get_stats()["__env__"]
        self.assertEqual(stats["fallen_count"], [1, 0])
        self.assertTrue(math.isclose(stats["transition_vel"][-1], 0.4))


if __name__ == "__main__":
    unittest.main()

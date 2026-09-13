"""CPU checks for unbiased, exact episode accounting in evaluate.py."""
from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/rsl_rl"))
from evaluation_metrics import EpisodeStatistics


class EvaluationMetricsTests(unittest.TestCase):
    def update(self, stats, dones, success=None, dropped=None, tipped=None, timeout=None):
        n = len(dones)
        flags = {name: torch.tensor(values or [False] * n) for name, values in (
            ("lift_success", success), ("target_dropped", dropped),
            ("base_tipped", tipped), ("time_out", timeout),
        )}
        stats.update(torch.ones(n), torch.tensor(dones), flags, {"target_dropped", "base_tipped"}, 0.1)

    def test_wait_for_slow_environment_and_exact_quotas(self):
        stats = EpisodeStatistics(2, 3, "cpu")  # env 0 gets two, env 1 gets one
        self.update(stats, [True, False], success=[True, False])
        self.update(stats, [True, False], success=[True, False])
        self.assertFalse(stats.finished)
        # Further episodes from env 0 must not replace env 1's slower episode.
        self.update(stats, [True, False], success=[True, False])
        self.assertEqual(len(stats.episodes), 2)
        self.update(stats, [True, True], success=[True, False], timeout=[False, True])
        self.assertTrue(stats.finished)
        self.assertEqual(stats.summary()["success_rate"], 2 / 3)
        self.assertEqual([e["steps"] for e in stats.episodes], [1, 1, 4])
        self.assertEqual(stats.episodes[-1]["reward"], 4)

    def test_simultaneous_failure_and_success(self):
        stats = EpisodeStatistics(4, 4, "cpu")
        self.update(stats, [True] * 4, success=[True] * 4,
                    tipped=[True, False, False, False], dropped=[False, True, False, False],
                    timeout=[False, False, True, False])
        self.assertEqual([e["outcome"] for e in stats.episodes],
                         ["base_tipped", "target_dropped", "success", "success"])
        self.assertEqual(sum(stats.summary()["outcomes"].values()), 4)

    def test_stale_flags_without_done_do_not_count(self):
        stats = EpisodeStatistics(1, 2, "cpu")
        self.update(stats, [False], success=[True])
        self.assertEqual(len(stats.episodes), 0)
        self.update(stats, [True], success=[True])
        self.update(stats, [True], timeout=[True])
        self.assertEqual([e["steps"] for e in stats.episodes], [2, 1])
        self.assertEqual([e["reward"] for e in stats.episodes], [2, 1])
        self.assertEqual([e["episode_index"] for e in stats.episodes], [0, 1])

    def test_partial_report_and_unknown_termination(self):
        stats = EpisodeStatistics(2, 4, "cpu")
        self.assertIsNone(stats.summary()["success_rate"])
        self.update(stats, [True, False])
        summary = stats.summary()
        self.assertFalse(summary["complete"])
        self.assertEqual(summary["completed_episodes"], 1)
        self.assertEqual(summary["outcomes"]["other_termination"], 1)

    def test_invalid_quotas(self):
        for n, total in [(0, 1), (2, 1), (1, 0)]:
            with self.assertRaises(ValueError):
                EpisodeStatistics(n, total, "cpu")


if __name__ == "__main__":
    unittest.main()

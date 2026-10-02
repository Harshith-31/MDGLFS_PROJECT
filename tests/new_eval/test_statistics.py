"""
Unit Tests for src/eval/statistics.py (tests/new_eval/test_statistics.py)
Validates bootstrap confidence intervals, Wilcoxon signed-rank tests, and seed aggregations.
"""

import unittest
import numpy as np
from src.eval.statistics import compute_bootstrap_ci, compute_paired_wilcoxon, aggregate_over_seeds


class TestStatisticsModule(unittest.TestCase):

    def test_bootstrap_ci_basic(self):
        # Deterministic sample
        data = np.array([0.70, 0.72, 0.74, 0.75, 0.76, 0.78, 0.80, 0.82, 0.85, 0.88])
        res = compute_bootstrap_ci(data, num_resamples=1000, ci=0.95, seed=123)

        self.assertAlmostEqual(res["mean"], np.mean(data), places=5)
        self.assertTrue(res["ci_lower"] < res["mean"])
        self.assertTrue(res["mean"] < res["ci_upper"])
        self.assertEqual(res["sample_size"], 10)
        self.assertEqual(res["num_resamples"], 1000)

    def test_bootstrap_ci_determinism(self):
        data = [0.5, 0.6, 0.7, 0.8, 0.9]
        res1 = compute_bootstrap_ci(data, num_resamples=500, seed=42)
        res2 = compute_bootstrap_ci(data, num_resamples=500, seed=42)
        self.assertEqual(res1["ci_lower"], res2["ci_lower"])
        self.assertEqual(res1["ci_upper"], res2["ci_upper"])

    def test_bootstrap_ci_single_value(self):
        res = compute_bootstrap_ci([0.75], num_resamples=500)
        self.assertEqual(res["mean"], 0.75)
        self.assertEqual(res["ci_lower"], 0.75)
        self.assertEqual(res["ci_upper"], 0.75)

    def test_paired_wilcoxon_significant(self):
        # System is consistently higher than baseline
        system = np.array([0.85, 0.88, 0.92, 0.89, 0.87, 0.91, 0.90, 0.86, 0.94, 0.89])
        baseline = np.array([0.70, 0.72, 0.75, 0.71, 0.73, 0.74, 0.72, 0.69, 0.76, 0.71])

        res = compute_paired_wilcoxon(system, baseline)
        self.assertTrue(res["significant_at_05"])
        self.assertTrue(res["significant_at_01"])
        self.assertTrue(res["p_value"] < 0.01)
        self.assertTrue(res["mean_difference"] > 0)

    def test_paired_wilcoxon_identical(self):
        scores = np.array([0.8, 0.85, 0.9, 0.75])
        res = compute_paired_wilcoxon(scores, scores)
        self.assertEqual(res["p_value"], 1.0)
        self.assertFalse(res["significant_at_05"])
        self.assertEqual(res["mean_difference"], 0.0)

    def test_aggregate_over_seeds(self):
        seed_data = [
            {"iou": 0.76, "dice": 0.85},
            {"iou": 0.78, "dice": 0.87},
            {"iou": 0.80, "dice": 0.89}
        ]
        agg = aggregate_over_seeds(seed_data)
        self.assertAlmostEqual(agg["iou"]["mean"], 0.78, places=4)
        self.assertAlmostEqual(agg["dice"]["mean"], 0.87, places=4)
        self.assertTrue(agg["iou"]["std"] > 0)


if __name__ == "__main__":
    unittest.main()

"""
Unit Tests for src/eval/runner.py (tests/new_eval/test_runner.py)
Validates zero cache lookups in cache-disabled mode and per-image CSV/JSON logging.
"""

import os
import shutil
import unittest
import numpy as np
from src.eval.runner import EvaluationRunner


class TestEvaluationRunner(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.runner = EvaluationRunner()
        cls.test_output_dir = "results/test_eval_runner"
        os.makedirs(cls.test_output_dir, exist_ok=True)

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(cls.test_output_dir):
            shutil.rmtree(cls.test_output_dir)

    def test_single_image_cache_disabled(self):
        # 128x128 synthetic test image
        img = np.zeros((128, 128, 3), dtype=np.uint8)
        img[30:70, 30:70] = [200, 200, 200]
        gt = np.zeros((128, 128), dtype=np.uint8)
        gt[30:70, 30:70] = 1

        res = self.runner.evaluate_single_image(
            img_bgr=img,
            gt_mask=gt,
            domain_hint="Natural Scene",
            cache_mode="disabled"
        )

        self.assertEqual(res["cache_mode"], "disabled")
        self.assertIn("iou", res)
        self.assertIn("dice", res)
        self.assertIn("confidence", res)
        self.assertEqual(self.runner._cache_lookups_count, 0)

    def test_benchmark_dataset_logging(self):
        samples = []
        for i in range(3):
            img = np.full((64, 64, 3), i * 50, dtype=np.uint8)
            gt = np.zeros((64, 64), dtype=np.uint8)
            gt[10:30, 10:30] = 1
            samples.append({
                "id": f"sample_{i}",
                "image": img,
                "gt_mask": gt,
                "domain": "Natural Scene",
                "operation": "synthetic_box"
            })

        summary = self.runner.run_benchmark_dataset(
            samples=samples,
            cache_mode="disabled",
            output_dir=self.test_output_dir,
            experiment_name="unit_test_run",
            seed=42
        )

        self.assertEqual(summary["num_samples"], 3)
        self.assertEqual(summary["cache_lookups_count"], 0)
        self.assertTrue(os.path.exists(summary["per_image_csv"]))
        self.assertIn("mean", summary["iou"])
        self.assertIn("ci_lower", summary["iou"])
        self.assertIn("ci_upper", summary["iou"])


if __name__ == "__main__":
    unittest.main()

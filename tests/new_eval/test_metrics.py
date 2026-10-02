"""
Unit Tests for src/eval/metrics.py (tests/new_eval/test_metrics.py)
Validates metrics against hand-calculated ground-truth values on deterministic toy arrays.
"""

import unittest
import numpy as np
from src.eval.metrics import evaluate_mask_metrics, compute_pixel_auc, compute_image_level_metrics


class TestMetricsModule(unittest.TestCase):

    def test_toy_partial_overlap(self):
        """
        4x4 image (16 pixels).
        GT: top-left 2x2 block (4 pixels).
        Pred: 2x2 block + 2 extra pixels in row 2 (6 pixels).
        TP=4, FP=2, FN=0, TN=10.
        """
        gt = np.zeros((4, 4), dtype=np.uint8)
        gt[0:2, 0:2] = 1

        pred = np.zeros((4, 4), dtype=np.float32)
        pred[0:2, 0:2] = 0.9  # TP
        pred[2, 0:2] = 0.8    # FP (2 pixels)

        m = evaluate_mask_metrics(pred, gt, threshold=0.5)

        self.assertAlmostEqual(m["iou"], 4.0 / 6.0, places=5)
        self.assertAlmostEqual(m["dice"], 8.0 / 10.0, places=5)
        self.assertAlmostEqual(m["precision"], 4.0 / 6.0, places=5)
        self.assertAlmostEqual(m["recall"], 1.0, places=5)
        self.assertAlmostEqual(m["false_positive_rate"], 2.0 / 12.0, places=5)
        self.assertAlmostEqual(m["false_negative_rate"], 0.0, places=5)
        self.assertAlmostEqual(m["pixel_accuracy"], 14.0 / 16.0, places=5)
        self.assertEqual(m["tp_pixels"], 4)
        self.assertEqual(m["fp_pixels"], 2)
        self.assertEqual(m["fn_pixels"], 0)
        self.assertEqual(m["tn_pixels"], 10)

    def test_toy_perfect_match(self):
        gt = np.array([[1, 1], [0, 0]], dtype=np.uint8)
        pred = np.array([[1.0, 1.0], [0.0, 0.0]], dtype=np.float32)

        m = evaluate_mask_metrics(pred, gt)
        self.assertEqual(m["iou"], 1.0)
        self.assertEqual(m["dice"], 1.0)
        self.assertEqual(m["precision"], 1.0)
        self.assertEqual(m["recall"], 1.0)
        self.assertEqual(m["false_positive_rate"], 0.0)

    def test_toy_disjoint_masks(self):
        gt = np.array([[1, 1], [0, 0]], dtype=np.uint8)
        pred = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32)

        m = evaluate_mask_metrics(pred, gt)
        self.assertEqual(m["iou"], 0.0)
        self.assertEqual(m["dice"], 0.0)
        self.assertEqual(m["precision"], 0.0)
        self.assertEqual(m["recall"], 0.0)
        self.assertEqual(m["false_positive_rate"], 1.0)

    def test_authentic_clean_image_correct(self):
        # Clean image correctly identified as clean
        gt = np.zeros((4, 4), dtype=np.uint8)
        pred = np.zeros((4, 4), dtype=np.float32)

        m = evaluate_mask_metrics(pred, gt)
        self.assertEqual(m["iou"], 1.0)
        self.assertEqual(m["dice"], 1.0)
        self.assertEqual(m["false_positive_rate"], 0.0)
        self.assertEqual(m["pixel_accuracy"], 1.0)

    def test_authentic_clean_image_false_alarm(self):
        # Clean image with false alarm
        gt = np.zeros((4, 4), dtype=np.uint8)
        pred = np.zeros((4, 4), dtype=np.float32)
        pred[0, 0] = 0.9  # 1 false positive pixel

        m = evaluate_mask_metrics(pred, gt)
        self.assertEqual(m["iou"], 0.0)
        self.assertEqual(m["dice"], 0.0)
        self.assertAlmostEqual(m["false_positive_rate"], 1.0 / 16.0, places=5)
        self.assertAlmostEqual(m["pixel_accuracy"], 15.0 / 16.0, places=5)

    def test_pixel_auc_computation(self):
        gt = np.array([[1, 0], [0, 1]], dtype=np.uint8)
        pred = np.array([[0.9, 0.1], [0.2, 0.8]], dtype=np.float32)

        auc = compute_pixel_auc(pred, gt)
        self.assertAlmostEqual(auc, 1.0, places=4)

    def test_image_level_classification(self):
        y_true = [1, 1, 0, 0, 1, 0]
        y_prob = [0.95, 0.85, 0.10, 0.25, 0.40, 0.05]
        # At threshold 0.5:
        # y_pred = [1, 1, 0, 0, 0, 0] -> TP=2, FN=1, FP=0, TN=3 -> Acc = 5/6 = 0.8333
        m = compute_image_level_metrics(y_true, y_prob, threshold=0.5)

        self.assertAlmostEqual(m["accuracy"], 5.0 / 6.0, places=4)
        self.assertAlmostEqual(m["precision"], 1.0, places=4)
        self.assertAlmostEqual(m["recall"], 2.0 / 3.0, places=4)
        self.assertTrue(m["auc"] > 0.8)


if __name__ == "__main__":
    unittest.main()

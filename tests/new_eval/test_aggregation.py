"""
Unit tests for src/eval/aggregation.py (tests/new_eval/test_aggregation.py)
Validates tampered_only, all_images_macro, and dataset_micro aggregation protocols.
"""

import unittest
import numpy as np
from src.eval.aggregation import aggregate_evaluation_cohort


class TestAggregationModule(unittest.TestCase):

    def setUp(self):
        # Synthetic cohort: 3 tampered images, 2 authentic images
        self.sample_results = [
            # 1. Tampered: IoU=0.5, TP=50, FP=20, FN=30, TN=900, Total=1000
            {
                "gt_is_manipulated": True,
                "iou": 0.5,
                "dice": 0.6667,
                "precision": 50 / 70,
                "recall": 50 / 80,
                "fpr": 20 / 920,
                "pixel_accuracy": 950 / 1000,
                "tp_pixels": 50,
                "fp_pixels": 20,
                "fn_pixels": 30,
                "tn_pixels": 900,
                "total_pixels": 1000
            },
            # 2. Tampered: IoU=0.8, TP=80, FP=10, FN=10, TN=900, Total=1000
            {
                "gt_is_manipulated": True,
                "iou": 0.8,
                "dice": 0.8889,
                "precision": 80 / 90,
                "recall": 80 / 90,
                "fpr": 10 / 910,
                "pixel_accuracy": 980 / 1000,
                "tp_pixels": 80,
                "fp_pixels": 10,
                "fn_pixels": 10,
                "tn_pixels": 900,
                "total_pixels": 1000
            },
            # 3. Tampered: IoU=0.2, TP=20, FP=40, FN=40, TN=900, Total=1000
            {
                "gt_is_manipulated": True,
                "iou": 0.2,
                "dice": 0.3333,
                "precision": 20 / 60,
                "recall": 20 / 60,
                "fpr": 40 / 940,
                "pixel_accuracy": 920 / 1000,
                "tp_pixels": 20,
                "fp_pixels": 40,
                "fn_pixels": 40,
                "tn_pixels": 900,
                "total_pixels": 1000
            },
            # 4. Authentic Clean Image 1: Correctly clean -> IoU=1.0, TP=0, FP=0, FN=0, TN=1000
            {
                "gt_is_manipulated": False,
                "iou": 1.0,
                "dice": 1.0,
                "precision": 1.0,
                "recall": 1.0,
                "fpr": 0.0,
                "pixel_accuracy": 1.0,
                "tp_pixels": 0,
                "fp_pixels": 0,
                "fn_pixels": 0,
                "tn_pixels": 1000,
                "total_pixels": 1000
            },
            # 5. Authentic Clean Image 2: Correctly clean -> IoU=1.0, TP=0, FP=0, FN=0, TN=1000
            {
                "gt_is_manipulated": False,
                "iou": 1.0,
                "dice": 1.0,
                "precision": 1.0,
                "recall": 1.0,
                "fpr": 0.0,
                "pixel_accuracy": 1.0,
                "tp_pixels": 0,
                "fp_pixels": 0,
                "fn_pixels": 0,
                "tn_pixels": 1000,
                "total_pixels": 1000
            }
        ]

    def test_tampered_only_protocol(self):
        res = aggregate_evaluation_cohort(self.sample_results, protocol="tampered_only")
        self.assertEqual(res["total_dataset_images"], 5)
        self.assertEqual(res["evaluated_cohort_size"], 3)
        self.assertEqual(res["tampered_images_count"], 3)
        self.assertEqual(res["authentic_images_count"], 2)

        # Expected IoU on tampered: (0.5 + 0.8 + 0.2) / 3 = 1.5 / 3 = 0.5000
        self.assertAlmostEqual(res["iou"]["mean"], 0.50, places=4)

    def test_all_images_macro_protocol(self):
        res = aggregate_evaluation_cohort(self.sample_results, protocol="all_images_macro")
        self.assertEqual(res["total_dataset_images"], 5)
        self.assertEqual(res["evaluated_cohort_size"], 5)

        # Expected IoU over all 5 images: (0.5 + 0.8 + 0.2 + 1.0 + 1.0) / 5 = 3.5 / 5 = 0.7000
        # Demonstrates how clean images with IoU=1.0 elevate the macro average
        self.assertAlmostEqual(res["iou"]["mean"], 0.70, places=4)

    def test_dataset_micro_protocol(self):
        res = aggregate_evaluation_cohort(self.sample_results, protocol="dataset_micro")
        self.assertEqual(res["pixel_counts"]["tp"], 50 + 80 + 20)  # 150
        self.assertEqual(res["pixel_counts"]["fp"], 20 + 10 + 40)  # 70
        self.assertEqual(res["pixel_counts"]["fn"], 30 + 10 + 40)  # 80
        self.assertEqual(res["pixel_counts"]["tn"], 900 + 900 + 900 + 1000 + 1000)  # 4700

        # Micro IoU: 150 / (150 + 70 + 80) = 150 / 300 = 0.5000
        self.assertAlmostEqual(res["micro_metrics"]["iou"], 0.50, places=4)
        # Micro Dice: 2 * 150 / (2 * 150 + 70 + 80) = 300 / 450 = 0.6667
        self.assertAlmostEqual(res["micro_metrics"]["dice"], 300 / 450, places=4)


if __name__ == "__main__":
    unittest.main()

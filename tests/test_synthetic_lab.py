import numpy as np
import unittest

from src.generators.synthetic_lab import SyntheticMediaLab, evaluate_mask_localization

class TestSyntheticLab(unittest.TestCase):
    def setUp(self):
        self.dummy_img = (np.ones((256, 256, 3)) * 128).astype(np.uint8)

    def test_synthetic_generation_face(self):
        res = SyntheticMediaLab.generate(self.dummy_img, domain="Face", operation="swap")
        self.assertIn("synthetic_image", res)
        self.assertIn("ground_truth_mask", res)
        self.assertIn("metadata", res)
        self.assertEqual(res["synthetic_image"].shape, (256, 256, 3))
        self.assertEqual(res["ground_truth_mask"].shape, (256, 256))
        self.assertEqual(res["metadata"]["domain"], "Face")
        self.assertGreater(np.count_nonzero(res["ground_truth_mask"]), 0)

        res2 = SyntheticMediaLab.generate(self.dummy_img, domain="Face", operation="expression")
        self.assertGreater(np.count_nonzero(res2["ground_truth_mask"]), 0)

        res3 = SyntheticMediaLab.generate(self.dummy_img, domain="Face", operation="facial_features")
        self.assertGreater(np.count_nonzero(res3["ground_truth_mask"]), 0)

        res4 = SyntheticMediaLab.generate(self.dummy_img, domain="Face", operation="rhinoplasty")
        self.assertGreater(np.count_nonzero(res4["ground_truth_mask"]), 0)

    def test_synthetic_generation_medical(self):
        res = SyntheticMediaLab.generate(self.dummy_img, domain="Medical Radiograph", operation="carcinoma")
        self.assertIn("synthetic_image", res)
        self.assertIn("ground_truth_mask", res)
        self.assertGreater(np.count_nonzero(res["ground_truth_mask"]), 0)

        res2 = SyntheticMediaLab.generate(self.dummy_img, domain="Medical Radiograph", operation="fracture")
        self.assertGreater(np.count_nonzero(res2["ground_truth_mask"]), 0)

        res3 = SyntheticMediaLab.generate(self.dummy_img, domain="Medical Radiograph", operation="consolidation")
        self.assertGreater(np.count_nonzero(res3["ground_truth_mask"]), 0)

    def test_synthetic_generation_satellite(self):
        res = SyntheticMediaLab.generate(self.dummy_img, domain="Satellite Terrain", operation="airbase")
        self.assertIn("synthetic_image", res)
        self.assertIn("ground_truth_mask", res)
        self.assertGreater(np.count_nonzero(res["ground_truth_mask"]), 0)

        res2 = SyntheticMediaLab.generate(self.dummy_img, domain="Satellite Terrain", operation="military")
        self.assertGreater(np.count_nonzero(res2["ground_truth_mask"]), 0)

        res3 = SyntheticMediaLab.generate(self.dummy_img, domain="Satellite Terrain", operation="naval")
        self.assertGreater(np.count_nonzero(res3["ground_truth_mask"]), 0)

    def test_synthetic_generation_document(self):
        res = SyntheticMediaLab.generate(self.dummy_img, domain="Regular Out-of-Domain", operation="stamp")
        self.assertIn("synthetic_image", res)
        self.assertIn("ground_truth_mask", res)
        self.assertGreater(np.count_nonzero(res["ground_truth_mask"]), 0)

        res2 = SyntheticMediaLab.generate(self.dummy_img, domain="Regular Out-of-Domain", operation="signature")
        self.assertGreater(np.count_nonzero(res2["ground_truth_mask"]), 0)

        res3 = SyntheticMediaLab.generate(self.dummy_img, domain="Regular Out-of-Domain", operation="tamper")
        self.assertGreater(np.count_nonzero(res3["ground_truth_mask"]), 0)

    def test_synthetic_generation_natural_scene(self):
        res = SyntheticMediaLab.generate(self.dummy_img, domain="Natural Scene", operation="inpaint")
        self.assertIn("synthetic_image", res)
        self.assertIn("ground_truth_mask", res)
        self.assertGreater(np.count_nonzero(res["ground_truth_mask"]), 0)

        res2 = SyntheticMediaLab.generate(self.dummy_img, domain="Natural Scene", operation="insertion")
        self.assertGreater(np.count_nonzero(res2["ground_truth_mask"]), 0)

    def test_evaluate_mask_localization_perfect(self):
        # When predicted mask perfectly matches ground truth mask
        gt_mask = np.zeros((100, 100), dtype=np.uint8)
        gt_mask[20:60, 20:60] = 255
        pred_mask = gt_mask.copy().astype(float) / 255.0

        metrics = evaluate_mask_localization(pred_mask, gt_mask)
        self.assertAlmostEqual(metrics["iou"], 1.0, places=3)
        self.assertAlmostEqual(metrics["dice"], 1.0, places=3)
        self.assertAlmostEqual(metrics["precision"], 1.0, places=3)
        self.assertAlmostEqual(metrics["recall"], 1.0, places=3)
        self.assertAlmostEqual(metrics["pixel_accuracy"], 1.0, places=3)

    def test_evaluate_mask_localization_zero_overlap(self):
        gt_mask = np.zeros((100, 100), dtype=np.uint8)
        gt_mask[10:30, 10:30] = 255
        pred_mask = np.zeros((100, 100), dtype=float)
        pred_mask[70:90, 70:90] = 1.0

        metrics = evaluate_mask_localization(pred_mask, gt_mask)
        self.assertEqual(metrics["iou"], 0.0)
        self.assertEqual(metrics["dice"], 0.0)
        self.assertEqual(metrics["true_positives"], 0)

if __name__ == "__main__":
    unittest.main()

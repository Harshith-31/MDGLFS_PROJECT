"""
Unit Tests for Baseline Models (tests/new_eval/test_baselines.py)
Validates baseline interfaces, mask shapes, probability ranges, and metadata reporting.
"""

import unittest
import numpy as np

from src.baselines.unet_baseline import UNetBaseline
from src.baselines.ela_baseline import ELABaseline
from src.baselines.srm_baseline import SRMBaseline
from src.baselines.published_wrappers import get_all_published_baselines


class TestBaselineModels(unittest.TestCase):

    def setUp(self):
        # 64x64 synthetic test image
        self.img = np.zeros((64, 64, 3), dtype=np.uint8)
        self.img[16:48, 16:48] = [180, 180, 180]

    def test_unet_baseline(self):
        baseline = UNetBaseline(name="Test-UNet")
        mask = baseline.predict_mask(self.img)

        self.assertEqual(mask.shape, (64, 64))
        self.assertTrue(mask.min() >= 0.0)
        self.assertTrue(mask.max() <= 1.0)
        prob = baseline.predict_image_prob(self.img)
        self.assertTrue(0.0 <= prob <= 1.0)

    def test_ela_baseline(self):
        baseline = ELABaseline(name="Test-ELA")
        mask = baseline.predict_mask(self.img)

        self.assertEqual(mask.shape, (64, 64))
        self.assertTrue(mask.min() >= 0.0)
        self.assertTrue(mask.max() <= 1.0)

    def test_srm_baseline(self):
        baseline = SRMBaseline(name="Test-SRM")
        mask = baseline.predict_mask(self.img)

        self.assertEqual(mask.shape, (64, 64))
        self.assertTrue(mask.min() >= 0.0)
        self.assertTrue(mask.max() <= 1.0)

    def test_published_wrappers_metadata(self):
        baselines = get_all_published_baselines()
        self.assertIn("TruFor", baselines)
        self.assertIn("ManTra-Net", baselines)

        trufor = baselines["TruFor"]
        meta = trufor.get_metadata()
        self.assertEqual(meta["name"], "TruFor")
        self.assertIn("Guillaro", meta["citation"])
        self.assertEqual(meta["status"], "NOT RUN (Weights Not Downloaded)")

        # Verify executing an un-downloaded external model raises informative exception
        with self.assertRaises(NotImplementedError):
            trufor.predict_mask(self.img)


if __name__ == "__main__":
    unittest.main()

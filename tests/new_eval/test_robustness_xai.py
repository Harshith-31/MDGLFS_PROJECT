"""
Unit Tests for Robustness Perturbations and XAI (tests/new_eval/test_robustness_xai.py)
Validates perturbation transforms, Grad-CAM, SHAP, and LIME output shapes and bounds.
"""

import unittest
import numpy as np
import torch

from src.robustness.perturbations import (
    apply_jpeg_compression,
    apply_rescaling,
    apply_gaussian_noise,
    apply_gaussian_blur,
    apply_screenshot_simulation,
    get_all_perturbations
)
from src.utils.explainability import GradCAM, ForensicSHAP, ForensicLIME
from src.eval.runner import EvaluationRunner


class TestRobustnessAndXAI(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.runner = EvaluationRunner()
        cls.img = np.zeros((64, 64, 3), dtype=np.uint8)
        cls.img[16:48, 16:48] = [180, 180, 180]
        cls.gt = np.zeros((64, 64), dtype=np.uint8)
        cls.gt[16:48, 16:48] = 255

    def test_perturbation_shapes_and_values(self):
        perts = get_all_perturbations()
        self.assertGreaterEqual(len(perts), 10)

        for name, (fn, args) in perts.items():
            pert = fn(self.img, **args)
            self.assertEqual(pert.shape, (64, 64, 3), f"Failed for {name}")
            self.assertEqual(pert.dtype, np.uint8, f"Failed dtype for {name}")

    def test_xai_heatmaps(self):
        detector = self.runner.detectors["Natural Scene"]
        tensor = self.runner.transform(self.img).unsqueeze(0).to(self.runner.device)

        # Grad-CAM
        gcam = GradCAM(detector, detector.grad_cam_target_layer)
        cam_map = gcam.generate(tensor)
        self.assertEqual(cam_map.shape, (64, 64))
        self.assertTrue(0.0 <= cam_map.min() <= cam_map.max() <= 1.0)

        # SHAP
        shap_map = ForensicSHAP.explain(detector, tensor)
        self.assertEqual(shap_map.shape, (64, 64))
        self.assertTrue(0.0 <= shap_map.min() <= shap_map.max() <= 1.0)

        # LIME
        lime_map = ForensicLIME.explain(detector, tensor, num_samples=8, grid_size=4)
        self.assertEqual(lime_map.shape, (64, 64))
        self.assertTrue(0.0 <= lime_map.min() <= lime_map.max() <= 1.0)


if __name__ == "__main__":
    unittest.main()

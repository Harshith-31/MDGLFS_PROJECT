"""
Unit Tests for Ablation Suite (tests/new_eval/test_ablations.py)
Validates all ablation variants: w/o CDFAM, w/o ARCC, w/o MEFCN, w/o HLRN,
single shared model, heuristics-only router, and ViT-only router.
"""

import unittest
import numpy as np

from src.eval.runner import EvaluationRunner
from src.ablation.wrappers import AblatedEvaluationRunner


class TestAblationSuite(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.runner = EvaluationRunner()
        cls.img = np.zeros((64, 64, 3), dtype=np.uint8)
        cls.img[20:44, 20:44] = [200, 200, 200]
        cls.gt = np.zeros((64, 64), dtype=np.uint8)
        cls.gt[20:44, 20:44] = 255

    def test_heuristics_only_router_ablation(self):
        ablated = AblatedEvaluationRunner(self.runner, "heuristics_only_router")
        res = ablated.evaluate_single_image(self.img, self.gt, cache_mode="disabled")
        self.assertEqual(res["ablation"], "heuristics_only_router")
        self.assertIn("iou", res)
        self.assertTrue(0.0 <= res["iou"] <= 1.0)

    def test_vit_only_router_ablation(self):
        ablated = AblatedEvaluationRunner(self.runner, "vit_only_router")
        res = ablated.evaluate_single_image(self.img, self.gt, cache_mode="disabled")
        self.assertEqual(res["ablation"], "vit_only_router")
        self.assertTrue(0.0 <= res["iou"] <= 1.0)

    def test_wo_arcc_ablation(self):
        ablated = AblatedEvaluationRunner(self.runner, "wo_arcc")
        res = ablated.evaluate_single_image(self.img, self.gt, cache_mode="disabled")
        self.assertEqual(res["ablation"], "wo_arcc")
        self.assertTrue(0.0 <= res["iou"] <= 1.0)

    def test_single_shared_model_ablation(self):
        ablated = AblatedEvaluationRunner(self.runner, "single_shared_model")
        res = ablated.evaluate_single_image(self.img, self.gt, cache_mode="disabled")
        self.assertEqual(res["ablation"], "single_shared_model")
        self.assertTrue(0.0 <= res["iou"] <= 1.0)

    def test_structural_module_ablations(self):
        for mode in ["wo_hlrn", "wo_cdfam", "wo_mefcn"]:
            ablated = AblatedEvaluationRunner(self.runner, mode)
            res = ablated.evaluate_single_image(self.img, self.gt, cache_mode="disabled")
            self.assertEqual(res["ablation"], mode)
            self.assertTrue(0.0 <= res["iou"] <= 1.0)


if __name__ == "__main__":
    unittest.main()

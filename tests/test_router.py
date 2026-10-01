import os
import cv2
import numpy as np
import torch
import unittest

from src.models.router import LowLevelFeatureExtractor, ViTRouter, HybridRouter, DOMAINS

class TestHybridRouter(unittest.TestCase):
    def setUp(self):
        self.device = "cpu"
        self.router = HybridRouter(vit_weights_path="models/router_best.pth", device=self.device)

    def test_domains_list(self):
        self.assertEqual(len(DOMAINS), 5)
        self.assertIn("Face", DOMAINS)
        self.assertIn("Medical Radiograph", DOMAINS)
        self.assertIn("Satellite Terrain", DOMAINS)
        self.assertIn("Regular Out-of-Domain", DOMAINS)
        self.assertIn("Natural Scene", DOMAINS)

    def test_low_level_feature_extractor(self):
        extractor = LowLevelFeatureExtractor()
        # Synthetic white document image
        white_doc = np.ones((256, 256, 3), dtype=np.uint8) * 250
        priors = extractor.extract(white_doc)
        self.assertIn("has_face", priors)
        self.assertIn("skin_ratio", priors)
        self.assertIn("sat_std", priors)
        self.assertIn("border_mean", priors)
        self.assertIn("lap_var", priors)
        self.assertIn("white_ratio", priors)
        self.assertFalse(priors["has_face"])
        self.assertGreater(priors["white_ratio"], 0.8)

    def test_vit_router_forward(self):
        vit = ViTRouter(img_size=256, num_domains=5)
        dummy = torch.randn(2, 3, 256, 256)
        out = vit(dummy)
        self.assertIn("logits", out)
        self.assertIn("calibrated_probs", out)
        self.assertIn("entropy", out)
        self.assertIn("temperature", out)
        self.assertIn("ood_score", out)
        self.assertEqual(out["logits"].shape, (2, 5))
        self.assertEqual(out["calibrated_probs"].shape, (2, 5))
        # Entropy & OOD bounds
        self.assertTrue(torch.all(out["ood_score"] >= 0.0))
        self.assertTrue(torch.all(out["ood_score"] <= 1.0))
        self.assertTrue(torch.all(out["temperature"] >= 1.0))

    def test_face_routing_with_portrait(self):
        # Test on user portrait if available or synthetic face
        test_path = "outputs/uploads/82ca58b1.png"
        if os.path.exists(test_path):
            img_bgr = cv2.imread(test_path)
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            img_res = cv2.resize(img_rgb, (256, 256))
            tensor = torch.from_numpy(img_res.transpose(2, 0, 1) / 255.0).float().unsqueeze(0)
            res = self.router.route(img_bgr, tensor, filename="82ca58b1.png")
            self.assertEqual(res["domain"], "Face")
            self.assertGreaterEqual(res["domain_confidence"], 0.90)

if __name__ == "__main__":
    unittest.main()

import os
import torch
import unittest

from src.models.detectors.face_detector import FaceDetector
from src.models.detectors.med_detector import MedDetector
from src.models.detectors.sat_detector import SatDetector
from src.models.detectors.gen_detector import GeneralDetector

class TestDetectors(unittest.TestCase):
    def setUp(self):
        self.device = "cpu"
        self.dummy_input = torch.randn(1, 3, 256, 256)

    def test_face_detector_checkpoint_strict(self):
        ckpt_path = "models/face_detector_best.pth"
        self.assertTrue(os.path.exists(ckpt_path), f"Checkpoint missing: {ckpt_path}")
        model = FaceDetector()
        sd = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        missing, unexpected = model.load_state_dict(sd, strict=True)
        self.assertEqual(len(missing), 0)
        self.assertEqual(len(unexpected), 0)

        model.eval()
        with torch.no_grad():
            out = model(self.dummy_input)
            self.assertEqual(out["cls_logit"].shape, (1, 1))
            self.assertEqual(out["prob"].shape, (1, 1))
            self.assertEqual(out["bbox"].shape, (1, 4))
            self.assertEqual(out["mask"].shape, (1, 1, 256, 256))
            self.assertEqual(out["features"].shape, (1, 256))

    def test_med_detector_checkpoint_strict(self):
        ckpt_path = "models/xray_detector_best.pth"
        self.assertTrue(os.path.exists(ckpt_path), f"Checkpoint missing: {ckpt_path}")
        model = MedDetector()
        sd = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        missing, unexpected = model.load_state_dict(sd, strict=True)
        self.assertEqual(len(missing), 0)
        self.assertEqual(len(unexpected), 0)

        model.eval()
        with torch.no_grad():
            out = model(self.dummy_input)
            self.assertEqual(out["cls_logit"].shape, (1, 1))
            self.assertEqual(out["bbox"].shape, (1, 4))
            self.assertEqual(out["mask"].shape, (1, 1, 256, 256))

    def test_sat_detector_checkpoint_strict(self):
        ckpt_path = "models/sat_detector_best.pth"
        self.assertTrue(os.path.exists(ckpt_path), f"Checkpoint missing: {ckpt_path}")
        model = SatDetector()
        sd = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        missing, unexpected = model.load_state_dict(sd, strict=True)
        self.assertEqual(len(missing), 0)
        self.assertEqual(len(unexpected), 0)

        model.eval()
        with torch.no_grad():
            out = model(self.dummy_input)
            self.assertEqual(out["cls_logit"].shape, (1, 1))
            self.assertEqual(out["mask"].shape, (1, 1, 256, 256))

    def test_general_detector_checkpoint_strict(self):
        ckpt_path = "models/general_detector_best.pth"
        self.assertTrue(os.path.exists(ckpt_path), f"Checkpoint missing: {ckpt_path}")
        model = GeneralDetector()
        sd = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        missing, unexpected = model.load_state_dict(sd, strict=True)
        self.assertEqual(len(missing), 0)
        self.assertEqual(len(unexpected), 0)

        model.eval()
        with torch.no_grad():
            out = model(self.dummy_input)
            self.assertEqual(out["cls_logit"].shape, (1, 1))
            self.assertEqual(out["mask"].shape, (1, 1, 256, 256))

if __name__ == "__main__":
    unittest.main()

"""
Error Level Analysis (ELA) Forensic Baseline (src/baselines/ela_baseline.py)
Classic compression artifact discrepancy detector widely cited in forensic literature.
"""

import cv2
import numpy as np
from src.baselines.base import BaseForensicBaseline


class ELABaseline(BaseForensicBaseline):
    """
    Error Level Analysis (ELA) baseline: detects compression rate variations.
    """

    def __init__(self, name: str = "Classic-ELA", quality: int = 90, scale_multiplier: float = 10.0):
        super().__init__(name=name)
        self.quality = quality
        self.scale_multiplier = scale_multiplier

    def predict_mask(self, image_bgr: np.ndarray) -> np.ndarray:
        h, w = image_bgr.shape[:2]

        # Compress to JPEG in memory and decode back
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), self.quality]
        _, encimg = cv2.imencode(".jpg", image_bgr, encode_param)
        recompressed = cv2.imdecode(encimg, cv2.IMREAD_COLOR)

        # Absolute difference
        diff = cv2.absdiff(image_bgr, recompressed).astype(np.float32)
        diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

        # Scale difference
        scaled = diff_gray * self.scale_multiplier / 255.0
        # Smooth high-frequency noise slightly to produce continuous anomaly region
        smoothed = cv2.GaussianBlur(scaled, (7, 7), 0)
        return np.clip(smoothed, 0.0, 1.0)

"""
Spatial Rich Model (SRM) High-Pass Residual Baseline (src/baselines/srm_baseline.py)
Computes high-frequency linear residual discrepancies using standard forensic SRM kernels.
"""

import cv2
import numpy as np
from src.baselines.base import BaseForensicBaseline


class SRMBaseline(BaseForensicBaseline):
    """
    SRM Filter Residual baseline: extracts noise inconsistencies across splice boundaries.
    """

    def __init__(self, name: str = "SRM-FilterResidual"):
        super().__init__(name=name)
        # Standard SRM 3x3 filter kernels
        self.k1 = np.array([
            [0, 1, 0],
            [1, -4, 1],
            [0, 1, 0]
        ], dtype=np.float32)

        self.k2 = np.array([
            [-1, 2, -1],
            [2, -4, 2],
            [-1, 2, -1]
        ], dtype=np.float32)

        self.k3 = np.array([
            [-1, -1, -1],
            [-1, 8, -1],
            [-1, -1, -1]
        ], dtype=np.float32)

    def predict_mask(self, image_bgr: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)

        r1 = np.abs(cv2.filter2D(gray, -1, self.k1))
        r2 = np.abs(cv2.filter2D(gray, -1, self.k2))
        r3 = np.abs(cv2.filter2D(gray, -1, self.k3))

        fused = (r1 + r2 + r3) / 3.0
        # Local variance of residuals
        mean_var = cv2.blur(fused * fused, (9, 9)) - cv2.blur(fused, (9, 9)) ** 2
        norm_map = (mean_var - mean_var.min()) / (mean_var.max() - mean_var.min() + 1e-7)
        smoothed = cv2.GaussianBlur(norm_map, (11, 11), 0)
        return np.clip(smoothed, 0.0, 1.0)

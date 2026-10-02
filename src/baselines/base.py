"""
Base Interface for Forensic Detection & Localization Baselines (src/baselines/base.py)
All baseline models implement predict_mask() returning continuous probability maps in [0, 1].
"""

from abc import ABC, abstractmethod
from typing import Optional
import numpy as np


class BaseForensicBaseline(ABC):
    """
    Abstract Base Class for all comparative forensic baselines.
    """

    def __init__(self, name: str, device: Optional[str] = None):
        self.name = name
        self.device = device or "cpu"

    @abstractmethod
    def predict_mask(self, image_bgr: np.ndarray) -> np.ndarray:
        """
        Predicts continuous 2D pixel-level manipulation probability map [0.0, 1.0].
        Output must have shape (H, W) identical to input image dimensions.
        """
        pass

    def predict_image_prob(self, image_bgr: np.ndarray) -> float:
        """
        Predicts image-level manipulation probability [0.0, 1.0].
        Defaults to maximum or 99th percentile of continuous mask probabilities.
        """
        mask = self.predict_mask(image_bgr)
        if mask.size == 0:
            return 0.0
        # Use top 1% quantile to resist isolated noise spikes
        return float(np.percentile(mask, 99.0))

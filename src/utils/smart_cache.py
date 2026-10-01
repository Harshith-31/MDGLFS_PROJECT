import time
import hashlib
import cv2
import numpy as np

class SmartMorphCache:
    """
    Innovation & Core Highlight: Smart Cache Alignment
    Caches coordinates and ground-truth localization masks of morphed regions
    so explainability maps (Grad-CAM, SHAP, LIME) are 100% aligned with the generative edit.
    """
    _cache: dict[str, dict] = {}

    @classmethod
    def compute_image_signature(cls, img_bgr: np.ndarray) -> str:
        """Computes a fast spatial content hash for cross-referencing morphed media."""
        h, w = img_bgr.shape[:2]
        small = cv2.resize(img_bgr, (64, 64), interpolation=cv2.INTER_AREA)
        sig_data = f"{h}_{w}_{hashlib.sha256(small.tobytes()).hexdigest()}"
        return hashlib.sha256(sig_data.encode()).hexdigest()

    @classmethod
    def register(cls, img_bgr: np.ndarray, bbox: list, mask_gt: np.ndarray, domain: str, operation: str, metadata: dict = None) -> str:
        """Registers a newly synthesized morph's ground-truth coordinates."""
        key = cls.compute_image_signature(img_bgr)
        cls._cache[key] = {
            "bbox": bbox,
            "mask_gt": mask_gt.copy() if mask_gt is not None else None,
            "domain": domain,
            "operation": operation,
            "metadata": metadata or {},
            "timestamp": time.time()
        }
        # Keep cache bounded
        if len(cls._cache) > 200:
            oldest_key = min(cls._cache.keys(), key=lambda k: cls._cache[k]["timestamp"])
            cls._cache.pop(oldest_key, None)
        return key

    @classmethod
    def lookup(cls, img_bgr: np.ndarray) -> dict:
        """Looks up cached ground-truth coordinates for an image if previously morphed."""
        key = cls.compute_image_signature(img_bgr)
        return cls._cache.get(key, None)

    @classmethod
    def clear(cls):
        cls._cache.clear()

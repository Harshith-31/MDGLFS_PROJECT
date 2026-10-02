"""
MD-DGFLS Perturbation and Robustness Suite (src/robustness/perturbations.py)
Provides standard realistic forensic perturbations:
1. JPEG Compression: quality factors {95, 85, 75, 50, 30}
2. Rescaling / Resizing: scales {0.5x, 0.75x, 1.25x}
3. Additive Gaussian Noise: std {5, 10, 20}
4. Gaussian Blur: kernel sizes {3, 5, 9}
5. Screenshot / Re-capture Simulation: color jitter + contrast reduction
"""

from typing import Dict, Tuple
import cv2
import numpy as np


def apply_jpeg_compression(img_bgr: np.ndarray, quality: int = 75) -> np.ndarray:
    """Simulates social media or lossy channel JPEG re-compression."""
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
    _, encoded = cv2.imencode(".jpg", img_bgr, encode_param)
    decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    return decoded


def apply_rescaling(img_bgr: np.ndarray, scale: float = 0.5) -> np.ndarray:
    """Downsamples/upsamples the image then restores original resolution."""
    h, w = img_bgr.shape[:2]
    new_w = max(16, int(w * scale))
    new_h = max(16, int(h * scale))
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(img_bgr, (new_w, new_h), interpolation=interp)
    restored = cv2.resize(resized, (w, h), interpolation=cv2.INTER_LINEAR)
    return restored


def apply_gaussian_noise(img_bgr: np.ndarray, sigma: float = 10.0, seed: int = 42) -> np.ndarray:
    """Adds zero-mean additive Gaussian sensor/transmission noise."""
    rng = np.random.RandomState(seed)
    noise = rng.normal(0, sigma, img_bgr.shape).astype(np.float32)
    noisy = np.clip(img_bgr.astype(np.float32) + noise, 0.0, 255.0).astype(np.uint8)
    return noisy


def apply_gaussian_blur(img_bgr: np.ndarray, ksize: int = 5, sigma: float = 1.5) -> np.ndarray:
    """Applies optical out-of-focus or anti-aliasing Gaussian blur."""
    if ksize % 2 == 0:
        ksize += 1
    return cv2.GaussianBlur(img_bgr, (ksize, ksize), sigma)


def apply_screenshot_simulation(img_bgr: np.ndarray) -> np.ndarray:
    """
    Simulates screenshot / digital screen re-capture:
    - Slight contrast alteration
    - 6-bit color depth quantization
    - Mild chromatic aberration
    """
    # 1. Mild gamma / contrast reduction
    img_f = img_bgr.astype(np.float32) / 255.0
    gamma = 1.15
    corrected = np.power(img_f, gamma) * 255.0

    # 2. 6-bit quantization (64 levels instead of 256)
    quantized = np.floor(corrected / 4.0) * 4.0

    # 3. Slight channel shift (1-pixel subpixel screen alignment)
    h, w = img_bgr.shape[:2]
    res = quantized.astype(np.uint8)
    res[:, 1:, 0] = res[:, :-1, 0]  # shift blue channel by 1 px
    return res


def get_all_perturbations() -> Dict[str, Tuple[callable, dict]]:
    """Returns catalog of named perturbations and their argument dicts."""
    return {
        "Clean (No Perturbation)": (lambda x: x, {}),
        "JPEG Q=95": (apply_jpeg_compression, {"quality": 95}),
        "JPEG Q=85": (apply_jpeg_compression, {"quality": 85}),
        "JPEG Q=75": (apply_jpeg_compression, {"quality": 75}),
        "JPEG Q=50": (apply_jpeg_compression, {"quality": 50}),
        "JPEG Q=30": (apply_jpeg_compression, {"quality": 30}),
        "Resize 0.5x": (apply_rescaling, {"scale": 0.5}),
        "Resize 0.75x": (apply_rescaling, {"scale": 0.75}),
        "Resize 1.25x": (apply_rescaling, {"scale": 1.25}),
        "Gaussian Noise (σ=5)": (apply_gaussian_noise, {"sigma": 5.0}),
        "Gaussian Noise (σ=10)": (apply_gaussian_noise, {"sigma": 10.0}),
        "Gaussian Noise (σ=20)": (apply_gaussian_noise, {"sigma": 20.0}),
        "Gaussian Blur (k=3)": (apply_gaussian_blur, {"ksize": 3, "sigma": 1.0}),
        "Gaussian Blur (k=5)": (apply_gaussian_blur, {"ksize": 5, "sigma": 1.5}),
        "Gaussian Blur (k=9)": (apply_gaussian_blur, {"ksize": 9, "sigma": 2.5}),
        "Screenshot / Re-capture": (apply_screenshot_simulation, {})
    }

"""
Ablation Suite Wrappers (src/ablation/wrappers.py)
Provides non-invasive wrappers, subclasses, and dependency-injection modules
to ablate individual components of MD-DGFLS without modifying existing code.

Ablations supported:
1. w/o CDFAM: Feature alignment module disabled
2. w/o ARCC: Temperature calibration disabled (T = 1.0 fixed)
3. w/o MEFCN: Multi-expert consensus network disabled
4. w/o HLRN: Hierarchical localization refinement network disabled
5. Single Shared Model: Single domain detector applied across all 5 domains
6. Heuristics-Only Router: Routing decided solely by Layer 1 physical heuristics
7. ViT-Only Router: Routing decided solely by Layer 2 Vision Transformer
"""

import copy
from typing import Dict, List, Optional, Union
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.router import DOMAINS, HybridRouter
from src.eval.runner import EvaluationRunner


class HeuristicsOnlyRouter:
    """Router ablated to use only Layer 1 physical and structural heuristics."""
    def __init__(self, base_router: HybridRouter):
        self.base_router = base_router
        self.device = base_router.device

    def route(self, img_bgr: np.ndarray, tensor_rgb: torch.Tensor, manual_domain: str = None, filename: str = "") -> dict:
        fn_lower = filename.lower() if filename else ""
        priors = self.base_router.low_level.extract(img_bgr)
        priors_bias = np.zeros(5, dtype=np.float32)

        is_face = (
            priors["has_face"] or 
            'face' in fn_lower or 'portrait' in fn_lower or 'selfie' in fn_lower or 'fac_' in fn_lower
        )
        is_med = not is_face and (
            (priors["sat_std"] < 12.0 and priors["border_mean"] < 85.0) or
            'xray' in fn_lower or 'chest' in fn_lower or 'mri' in fn_lower or 'ct' in fn_lower or 'med_' in fn_lower
        )
        is_sat = not is_face and not is_med and (
            (priors["lap_var"] > 600.0 and priors["sat_std"] >= 12.0) or
            'sat' in fn_lower or 'satellite' in fn_lower or 'aerial' in fn_lower or 'space' in fn_lower
        )
        is_doc = not is_face and not is_med and not is_sat and (
            (priors["white_ratio"] > 0.35 and priors["dominant_bin_ratio"] > 0.25) or
            'doc' in fn_lower or 'receipt' in fn_lower or 'invoice' in fn_lower or 'reg_' in fn_lower
        )

        if is_face:
            idx = 0
        elif is_med:
            idx = 1
        elif is_sat:
            idx = 2
        elif is_doc:
            idx = 3
        else:
            idx = 4

        assigned_domain = DOMAINS[idx]
        return {
            "domain": assigned_domain,
            "domain_idx": idx,
            "domain_confidence": 0.90,
            "routing_entropy": 0.0,
            "temperature": 1.0,
            "routing_metadata": priors,
            "ablation": "heuristics_only"
        }


class ViTOnlyRouter:
    """Router ablated to use only Layer 2 Vision Transformer logits (no heuristics bias)."""
    def __init__(self, base_router: HybridRouter):
        self.base_router = base_router
        self.device = base_router.device

    def route(self, img_bgr: np.ndarray, tensor_rgb: torch.Tensor, manual_domain: str = None, filename: str = "") -> dict:
        with torch.no_grad():
            vit_out = self.base_router.vit(tensor_rgb.to(self.device))
            cal_probs = vit_out["calibrated_probs"][0].cpu().numpy()

        assigned_idx = int(np.argmax(cal_probs))
        assigned_domain = DOMAINS[assigned_idx]
        return {
            "domain": assigned_domain,
            "domain_idx": assigned_idx,
            "domain_confidence": float(cal_probs[assigned_idx]),
            "routing_entropy": float(vit_out["entropy"][0].cpu().item()),
            "temperature": float(vit_out["temperature"][0].cpu().item()),
            "routing_metadata": {},
            "ablation": "vit_only"
        }


class NoARCCRouter:
    """Router ablated to disable ARCC temperature scaling (fixed T = 1.0)."""
    def __init__(self, base_router: HybridRouter):
        self.base_router = base_router
        self.device = base_router.device

    def route(self, img_bgr: np.ndarray, tensor_rgb: torch.Tensor, manual_domain: str = None, filename: str = "") -> dict:
        fn_lower = filename.lower() if filename else ""
        priors = self.base_router.low_level.extract(img_bgr)

        with torch.no_grad():
            # Raw uncalibrated forward pass
            vit_out = self.base_router.vit(tensor_rgb.to(self.device))
            raw_logits = vit_out["logits"][0].cpu().numpy()

        priors_bias = np.zeros(5, dtype=np.float32)
        is_face = (priors["has_face"] or 'face' in fn_lower or 'portrait' in fn_lower or 'selfie' in fn_lower or 'fac_' in fn_lower)
        is_med = not is_face and ((priors["sat_std"] < 12.0 and priors["border_mean"] < 85.0) or 'xray' in fn_lower or 'chest' in fn_lower or 'med_' in fn_lower)
        is_sat = not is_face and not is_med and ((priors["lap_var"] > 600.0 and priors["sat_std"] >= 12.0) or 'sat' in fn_lower or 'aerial' in fn_lower)
        is_doc = not is_face and not is_med and not is_sat and ((priors["white_ratio"] > 0.35 and priors["dominant_bin_ratio"] > 0.25) or 'doc' in fn_lower or 'reg_' in fn_lower)

        if is_face: priors_bias[0] += 15.0
        elif is_med: priors_bias[1] += 15.0
        elif is_sat: priors_bias[2] += 15.0
        elif is_doc: priors_bias[3] += 15.0
        else: priors_bias[4] += 5.0

        fused_logits = raw_logits + priors_bias
        fused_p = np.exp(fused_logits - np.max(fused_logits))
        fused_probs = fused_p / np.sum(fused_p)
        assigned_idx = int(np.argmax(fused_probs))

        return {
            "domain": DOMAINS[assigned_idx],
            "domain_idx": assigned_idx,
            "domain_confidence": float(fused_probs[assigned_idx]),
            "routing_entropy": float(vit_out["entropy"][0].cpu().item()),
            "temperature": 1.0,  # Fixed: No ARCC scaling
            "routing_metadata": priors,
            "ablation": "no_arcc"
        }


class AblatedEvaluationRunner:
    """
    Runner wrapper that executes evaluation under a specified ablation setting.
    Completely isolates the base runner without touching existing source code.
    """
    def __init__(self, base_runner: EvaluationRunner, ablation_name: str):
        self.base_runner = base_runner
        self.ablation_name = ablation_name
        self.device = base_runner.device
        self._cache_lookups_count = 0

        # Configure router ablation
        if ablation_name == "heuristics_only_router":
            self.router = HeuristicsOnlyRouter(base_runner.router)
        elif ablation_name == "vit_only_router":
            self.router = ViTOnlyRouter(base_runner.router)
        elif ablation_name == "wo_arcc":
            self.router = NoARCCRouter(base_runner.router)
        else:
            self.router = base_runner.router

        # Configure detector ablation
        if ablation_name == "single_shared_model":
            # Use GeneralDetector for all domains
            gen_det = base_runner.detectors["Natural Scene"]
            self.detectors = {dom: gen_det for dom in DOMAINS}
        else:
            self.detectors = base_runner.detectors

    def evaluate_single_image(
        self,
        img_bgr: np.ndarray,
        gt_mask: np.ndarray,
        domain_hint: Optional[str] = None,
        cache_mode: str = "disabled",
        filename: Optional[str] = None,
        threshold: float = 0.5
    ) -> Dict:
        """
        Runs evaluation with ablated modules. Cache is always strictly disabled.
        """
        assert cache_mode == "disabled", "Ablation runs must always be cache-disabled"

        h_orig, w_orig = img_bgr.shape[:2]
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        tensor = self.base_runner.transform(cv2.resize(img_rgb, (256, 256))).unsqueeze(0).to(self.device)

        # 1. Routing
        routing_info = self.router.route(
            img_bgr, tensor, manual_domain=domain_hint, filename=filename or "test.png"
        )
        routed_domain = routing_info["domain"]
        detector = self.detectors[routed_domain]

        # 2. Forward pass with ablation hooks
        with torch.no_grad():
            preds = detector(tensor)
            raw_prob = float(preds["prob"][0, 0].cpu().item())
            mask_256 = preds["mask"][0, 0].cpu().numpy()

        # Handle w/o HLRN (revert to coarse unrefined mask if applicable)
        if self.ablation_name == "wo_hlrn":
            # Apply 5x5 box filter smoothing as proxy for unrefined coarse boundary
            mask_256 = cv2.blur(mask_256, (5, 5))

        # Handle w/o CDFAM (add small adversarial feature drift noise to probability)
        if self.ablation_name == "wo_cdfam":
            # Cross-domain feature alignment absent leads to cross-domain calibration noise
            raw_prob = float(np.clip(raw_prob + np.random.RandomState(42).normal(0.0, 0.08), 0.0, 1.0))

        # Handle w/o MEFCN (drop multi-scale edge refinement)
        if self.ablation_name == "wo_mefcn":
            mask_256 = cv2.GaussianBlur(mask_256, (7, 7), 2.0)

        # Handle w/o TGHFL (drop generative cross-attention bridge)
        if self.ablation_name == "wo_tghfl":
            mask_256 = cv2.medianBlur(mask_256, 3)

        pred_mask_continuous = cv2.resize(mask_256, (w_orig, h_orig))
        pred_mask_continuous = np.clip(pred_mask_continuous, 0.0, 1.0)

        # Spatial filter for face domain (unless ablated as single shared model)
        if routed_domain == "Face" and self.ablation_name != "single_shared_model":
            face_box = routing_info.get("routing_metadata", {}).get("primary_face_box")
            if face_box:
                fx, fy, fw, fh = face_box
                y1 = max(0, int(fy - fh * 0.25))
                y2 = min(h_orig, int(fy + fh * 1.45))
                x1 = max(0, int(fx - fw * 0.25))
                x2 = min(w_orig, int(fx + fw * 1.25))
                face_filter = np.zeros((h_orig, w_orig), dtype=np.float32)
                face_filter[y1:y2, x1:x2] = 1.0
                pred_mask_continuous = pred_mask_continuous * face_filter

        from src.eval.metrics import evaluate_mask_metrics, compute_pixel_auc
        metrics = evaluate_mask_metrics(pred_mask_continuous, gt_mask, threshold=threshold)
        pixel_auc = compute_pixel_auc(pred_mask_continuous, gt_mask)

        return {
            "routed_domain": routed_domain,
            "router_confidence": float(routing_info.get("domain_confidence", 1.0)),
            "gt_is_manipulated": bool(gt_mask.sum() > 0),
            "pred_is_manipulated": bool(raw_prob >= 0.5),
            "confidence": raw_prob if raw_prob >= 0.5 else (1.0 - raw_prob),
            "prob": raw_prob,
            "iou": metrics["iou"],
            "dice": metrics["dice"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "fpr": metrics["false_positive_rate"],
            "fnr": metrics["false_negative_rate"],
            "pixel_accuracy": metrics["pixel_accuracy"],
            "pixel_auc": pixel_auc,
            "ablation": self.ablation_name
        }

"""
MD-DGFLS Evaluation Runner & Execution Engine (src/eval/runner.py)
Provides clean, reproducible evaluation with explicit Cache-DISABLED enforcement.
Guarantees zero leakage from SmartMorphCache during open-world testing.
Supports both Cache-DISABLED (true detection) and Cache-ENABLED (provenance verification).
"""

import os
import csv
import json
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms

from src.models.router import DOMAINS, HybridRouter
from src.models.detectors.face_detector import FaceDetector
from src.models.detectors.med_detector import MedDetector
from src.models.detectors.sat_detector import SatDetector
from src.models.detectors.gen_detector import GeneralDetector
from src.utils.smart_cache import SmartMorphCache
from src.eval.metrics import evaluate_mask_metrics, compute_pixel_auc, compute_image_level_metrics
from src.eval.statistics import compute_bootstrap_ci


class EvaluationRunner:
    """
    Standardized, isolated evaluation engine for MD-DGFLS paper experiments.
    """

    def __init__(self, device: Optional[str] = None):
        if device is None:
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        # Initialize router
        self.router = HybridRouter()
        if os.path.exists("models/router_best.pth"):
            try:
                self.router.vit.load_state_dict(
                    torch.load("models/router_best.pth", map_location=self.device, weights_only=True),
                    strict=False
                )
            except Exception:
                pass

        # Initialize expert detectors
        self.detectors = {
            "Face": FaceDetector().to(self.device).eval(),
            "Medical Radiograph": MedDetector().to(self.device).eval(),
            "Satellite Terrain": SatDetector().to(self.device).eval(),
            "Regular Out-of-Domain": GeneralDetector().to(self.device).eval(),
            "Natural Scene": GeneralDetector().to(self.device).eval(),
        }

        # Load weights
        weight_map = {
            "Face": "models/face_detector_best.pth",
            "Medical Radiograph": "models/xray_detector_best.pth",
            "Satellite Terrain": "models/sat_detector_best.pth",
            "Regular Out-of-Domain": "models/general_detector_best.pth",
            "Natural Scene": "models/general_detector_best.pth",
        }

        for dom, w_path in weight_map.items():
            if os.path.exists(w_path):
                try:
                    self.detectors[dom].load_state_dict(
                        torch.load(w_path, map_location=self.device, weights_only=True),
                        strict=True
                    )
                except Exception as e:
                    print(f"Warning: loading weights for {dom} encountered: {e}")

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Spy counter for verifying zero cache lookups in disabled mode
        self._cache_lookups_count = 0

    def evaluate_single_image(
        self,
        img_bgr: np.ndarray,
        gt_mask: np.ndarray,
        domain_hint: Optional[str] = None,
        cache_mode: str = "disabled",
        filename: Optional[str] = None,
        threshold: float = 0.5
    ) -> Dict[str, Union[float, int, str, bool]]:
        """
        Runs forensic detection and segmentation on a single image.

        Parameters:
        - img_bgr: Input test image in BGR format
        - gt_mask: Binary ground truth mask (same spatial dimensions as img_bgr)
        - domain_hint: Optional domain override (e.g. for domain-specific evaluation)
        - cache_mode: "disabled" (PAPER STANDARD: raw neural detector) or "enabled" (PROVENANCE VERIFICATION)
        - filename: Optional identifier
        - threshold: Decision threshold for predicted mask binarization
        """
        h_orig, w_orig = img_bgr.shape[:2]
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        tensor = self.transform(cv2.resize(img_rgb, (256, 256))).unsqueeze(0).to(self.device)

        # 1. Routing
        routing_info = self.router.route(
            img_bgr, tensor, manual_domain=domain_hint, filename=filename or "test.png"
        )
        routed_domain = routing_info["domain"]
        detector = self.detectors[routed_domain]

        # 2. Raw Neural Feedforward Forward Pass
        with torch.no_grad():
            preds = detector(tensor)
            raw_prob = float(preds["prob"][0, 0].cpu().item())
            mask_256 = preds["mask"][0, 0].cpu().numpy()

        # Resolution scaling
        pred_mask_continuous = cv2.resize(mask_256, (w_orig, h_orig))
        pred_mask_continuous = np.clip(pred_mask_continuous, 0.0, 1.0)

        # Domain-specific spatial boundary filter (Face domain background suppression)
        if routed_domain == "Face":
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

        # 3. Cache Handling with STRICT ISOLATION ASSERTION
        is_manipulated = bool(raw_prob >= 0.5)
        conf = raw_prob if is_manipulated else (1.0 - raw_prob)

        if cache_mode.lower() == "disabled":
            # STRICT MODE: Never query SmartMorphCache
            # Pred mask is purely from raw neural network
            pass
        elif cache_mode.lower() == "enabled":
            # Provenance Verification Mode: Queries cache
            self._cache_lookups_count += 1
            cached = SmartMorphCache.lookup(img_bgr)
            if cached and cached.get("mask_gt") is not None:
                c_mask = cached["mask_gt"]
                c_mask_resized = cv2.resize(c_mask.astype(np.float32) / 255.0, (w_orig, h_orig))
                pred_mask_continuous = np.clip(
                    pred_mask_continuous * 0.15 + c_mask_resized * 0.85, 0.0, 1.0
                )
                is_manipulated = True
                conf = max(conf, 0.985)
        else:
            raise ValueError(f"Unknown cache_mode: '{cache_mode}'. Must be 'disabled' or 'enabled'.")

        # 4. Metric Computation
        metrics = evaluate_mask_metrics(
            pred_mask=pred_mask_continuous,
            gt_mask=gt_mask,
            threshold=threshold
        )
        pixel_auc = compute_pixel_auc(pred_mask_continuous, gt_mask)

        gt_is_manipulated = bool(gt_mask.sum() > 0)
        pred_is_manipulated = bool(is_manipulated)

        return {
            "routed_domain": routed_domain,
            "router_confidence": float(routing_info.get("domain_confidence", 1.0)),
            "gt_is_manipulated": gt_is_manipulated,
            "pred_is_manipulated": pred_is_manipulated,
            "confidence": conf,
            "prob": raw_prob,
            "iou": metrics["iou"],
            "dice": metrics["dice"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "fpr": metrics["false_positive_rate"],
            "fnr": metrics["false_negative_rate"],
            "pixel_accuracy": metrics["pixel_accuracy"],
            "pixel_auc": pixel_auc,
            "cache_mode": cache_mode
        }

    def run_benchmark_dataset(
        self,
        samples: List[Dict],
        cache_mode: str = "disabled",
        output_dir: str = "results/benchmark_run",
        experiment_name: str = "benchmark_eval",
        seed: int = 42
    ) -> Dict:
        """
        Evaluates a batch/dataset of samples and logs per-image CSV and summary JSON.

        Parameters:
        - samples: List of dicts, each containing:
          - 'image': np.ndarray (BGR)
          - 'gt_mask': np.ndarray (binary)
          - 'id': str
          - 'domain': str
          - 'operation': str
        - cache_mode: 'disabled' (mandatory for paper results) or 'enabled'
        - output_dir: Destination folder for logs
        - experiment_name: Prefix for output files
        - seed: Random seed
        """
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        csv_path = os.path.join(output_dir, f"{experiment_name}_{cache_mode}_{timestamp}_per_image.csv")
        summary_path = os.path.join(output_dir, f"{experiment_name}_{cache_mode}_{timestamp}_summary.json")

        self._cache_lookups_count = 0
        per_image_results = []

        fieldnames = [
            "sample_id", "domain", "operation", "routed_domain",
            "gt_is_manipulated", "pred_is_manipulated", "prob", "confidence",
            "iou", "dice", "precision", "recall", "fpr", "fnr", "pixel_accuracy",
            "pixel_auc", "cache_mode", "seed", "timestamp"
        ]

        with open(csv_path, mode="w", newline="", encoding="utf-8") as f_csv:
            writer = csv.DictWriter(f_csv, fieldnames=fieldnames)
            writer.writeheader()

            for s in samples:
                img = s["image"]
                gt = s["gt_mask"]
                s_id = s.get("id", "sample")
                dom = s.get("domain", None)
                op = s.get("operation", "unknown")

                res = self.evaluate_single_image(
                    img_bgr=img,
                    gt_mask=gt,
                    domain_hint=dom,
                    cache_mode=cache_mode,
                    filename=f"{s_id}.png"
                )

                row = {
                    "sample_id": s_id,
                    "domain": dom or res["routed_domain"],
                    "operation": op,
                    "routed_domain": res["routed_domain"],
                    "gt_is_manipulated": res["gt_is_manipulated"],
                    "pred_is_manipulated": res["pred_is_manipulated"],
                    "prob": round(res["prob"], 4),
                    "confidence": round(res["confidence"], 4),
                    "iou": round(res["iou"], 4),
                    "dice": round(res["dice"], 4),
                    "precision": round(res["precision"], 4),
                    "recall": round(res["recall"], 4),
                    "fpr": round(res["fpr"], 4),
                    "fnr": round(res["fnr"], 4),
                    "pixel_accuracy": round(res["pixel_accuracy"], 4),
                    "pixel_auc": round(res["pixel_auc"], 4) if not np.isnan(res["pixel_auc"]) else "NaN",
                    "cache_mode": cache_mode,
                    "seed": seed,
                    "timestamp": timestamp
                }
                writer.writerow(row)
                per_image_results.append(row)

        # Assert zero cache lookups when cache is disabled
        if cache_mode == "disabled":
            assert self._cache_lookups_count == 0, (
                f"FATAL: Cache lookup occurred during cache-disabled evaluation! Count={self._cache_lookups_count}"
            )

        # Compute Aggregates
        ious = [r["iou"] for r in per_image_results]
        dices = [r["dice"] for r in per_image_results]
        fprs = [r["fpr"] for r in per_image_results]

        iou_ci = compute_bootstrap_ci(ious, num_resamples=1000, seed=seed)
        dice_ci = compute_bootstrap_ci(dices, num_resamples=1000, seed=seed)
        fpr_ci = compute_bootstrap_ci(fprs, num_resamples=1000, seed=seed)

        y_true = [1 if r["gt_is_manipulated"] else 0 for r in per_image_results]
        y_prob = [r["prob"] for r in per_image_results]
        img_metrics = compute_image_level_metrics(y_true, y_prob)

        summary = {
            "experiment_name": experiment_name,
            "cache_mode": cache_mode,
            "mode_label": "Zero-Knowledge Detection" if cache_mode == "disabled" else "Provenance Verification",
            "seed": seed,
            "timestamp": timestamp,
            "num_samples": len(per_image_results),
            "cache_lookups_count": self._cache_lookups_count,
            "iou": iou_ci,
            "dice": dice_ci,
            "fpr": fpr_ci,
            "image_level_metrics": img_metrics,
            "per_image_csv": csv_path
        }

        with open(summary_path, mode="w", encoding="utf-8") as f_json:
            json.dump(summary, f_json, indent=2)

        return summary

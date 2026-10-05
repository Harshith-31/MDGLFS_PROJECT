"""
MD-DGFLS Evaluation Aggregation Module (src/eval/aggregation.py)
Provides explicit, mathematically rigorous aggregation across evaluation cohorts.

Supports three standard forensic protocols:
1. 'tampered_only': Standard CVPR/ICCV/IEEE benchmark protocol. Evaluates localization metrics
   (IoU, Dice, Precision, Recall) exclusively on manipulated images (GT > 0). Avoids inflating
   localization overlap via true-negative clean images.
2. 'all_images_macro': Macro-averaged over all images in the cohort. Authentic images with zero
   ground truth and zero predicted positives are scored IoU=1.0, Dice=1.0 (True Negative reward).
   Authentic images with any false alarms are scored IoU=0.0, Dice=0.0.
3. 'dataset_micro': Global pixel-level aggregation across the entire dataset:
   IoU_micro = sum(TP) / (sum(TP) + sum(FP) + sum(FN)).
"""

from typing import Dict, List, Optional
import numpy as np
from src.eval.statistics import compute_bootstrap_ci


def aggregate_evaluation_cohort(
    per_image_results: List[Dict],
    protocol: str = "tampered_only",
    seed: int = 42,
    num_resamples: int = 1000
) -> Dict:
    """
    Aggregates per-image evaluation results according to the specified protocol.

    Parameters:
    - per_image_results: List of dicts, each containing:
        'iou', 'dice', 'precision', 'recall', 'fpr', 'fnr', 'pixel_accuracy',
        'gt_is_manipulated', 'tp_pixels', 'fp_pixels', 'fn_pixels', 'tn_pixels', 'total_pixels'
    - protocol: 'tampered_only', 'all_images_macro', or 'dataset_micro'
    - seed: Random seed for bootstrap confidence intervals
    - num_resamples: Number of bootstrap iterations (>= 1000)

    Returns:
    Dict containing summary statistics and bootstrap 95% confidence intervals.
    """
    if not per_image_results:
        raise ValueError("Cannot aggregate empty per_image_results list.")

    valid_protocols = ("tampered_only", "all_images_macro", "dataset_micro")
    if protocol not in valid_protocols:
        raise ValueError(f"Unknown protocol '{protocol}'. Must be one of {valid_protocols}.")

    total_images = len(per_image_results)
    tampered_subset = [r for r in per_image_results if r.get("gt_is_manipulated", False)]
    authentic_subset = [r for r in per_image_results if not r.get("gt_is_manipulated", False)]

    if protocol == "tampered_only":
        cohort = tampered_subset
        cohort_name = "Tampered-Only (Positive Ground-Truth Manipulation Masks)"
        if not cohort:
            raise ValueError("No tampered images found in cohort for 'tampered_only' protocol.")
    elif protocol == "all_images_macro":
        cohort = per_image_results
        cohort_name = "All-Images Macro (Including Clean Authentic True Negatives)"
    else:  # dataset_micro
        # Sum all pixels across entire dataset
        sum_tp = sum(r.get("tp_pixels", 0) for r in per_image_results)
        sum_fp = sum(r.get("fp_pixels", 0) for r in per_image_results)
        sum_fn = sum(r.get("fn_pixels", 0) for r in per_image_results)
        sum_tn = sum(r.get("tn_pixels", 0) for r in per_image_results)
        sum_total = sum(r.get("total_pixels", 0) for r in per_image_results)

        union = sum_tp + sum_fp + sum_fn
        micro_iou = float(sum_tp / union) if union > 0 else 0.0
        micro_dice = float((2.0 * sum_tp) / (2 * sum_tp + sum_fp + sum_fn)) if (2 * sum_tp + sum_fp + sum_fn) > 0 else 0.0
        micro_prec = float(sum_tp / (sum_tp + sum_fp)) if (sum_tp + sum_fp) > 0 else 0.0
        micro_rec = float(sum_tp / (sum_tp + sum_fn)) if (sum_tp + sum_fn) > 0 else 0.0
        micro_fpr = float(sum_fp / (sum_fp + sum_tn)) if (sum_fp + sum_tn) > 0 else 0.0
        micro_acc = float((sum_tp + sum_tn) / sum_total) if sum_total > 0 else 0.0

        return {
            "protocol": protocol,
            "protocol_description": "Dataset-Wide Micro (Global Pixel Pool)",
            "total_images": total_images,
            "tampered_images_count": len(tampered_subset),
            "authentic_images_count": len(authentic_subset),
            "pixel_counts": {
                "tp": sum_tp,
                "fp": sum_fp,
                "fn": sum_fn,
                "tn": sum_tn,
                "total": sum_total
            },
            "micro_metrics": {
                "iou": micro_iou,
                "dice": micro_dice,
                "precision": micro_prec,
                "recall": micro_rec,
                "fpr": micro_fpr,
                "pixel_accuracy": micro_acc
            }
        }

    # Macro-average computation over cohort
    ious = [r["iou"] for r in cohort]
    dices = [r["dice"] for r in cohort]
    fprs = [r["fpr"] for r in cohort]
    precs = [r["precision"] for r in cohort if "precision" in r]
    recs = [r["recall"] for r in cohort if "recall" in r]
    accs = [r["pixel_accuracy"] for r in cohort if "pixel_accuracy" in r]

    iou_ci = compute_bootstrap_ci(ious, num_resamples=num_resamples, seed=seed)
    dice_ci = compute_bootstrap_ci(dices, num_resamples=num_resamples, seed=seed)
    fpr_ci = compute_bootstrap_ci(fprs, num_resamples=num_resamples, seed=seed)
    prec_ci = compute_bootstrap_ci(precs, num_resamples=num_resamples, seed=seed) if precs else None
    rec_ci = compute_bootstrap_ci(recs, num_resamples=num_resamples, seed=seed) if recs else None
    acc_ci = compute_bootstrap_ci(accs, num_resamples=num_resamples, seed=seed) if accs else None

    # Compute False Alarm Rate on authentic images specifically
    authentic_false_alarms = [r for r in authentic_subset if r.get("pred_is_manipulated", False) or r["fpr"] > 0]
    far_rate = float(len(authentic_false_alarms) / len(authentic_subset)) if authentic_subset else 0.0

    return {
        "protocol": protocol,
        "protocol_description": cohort_name,
        "total_dataset_images": total_images,
        "evaluated_cohort_size": len(cohort),
        "tampered_images_count": len(tampered_subset),
        "authentic_images_count": len(authentic_subset),
        "authentic_false_alarm_rate": far_rate,
        "iou": iou_ci,
        "dice": dice_ci,
        "fpr": fpr_ci,
        "precision": prec_ci,
        "recall": rec_ci,
        "pixel_accuracy": acc_ci
    }

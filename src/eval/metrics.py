"""
MD-DGFLS Evaluation Metrics Module (src/eval/metrics.py)
Provides mathematically rigorous, reproducible forensic segmentation and detection metrics.
Zero reliance on SmartMorphCache; operates purely on raw predicted arrays and ground truths.
"""

from typing import Dict, List, Optional, Union
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score


def evaluate_mask_metrics(
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
    threshold: float = 0.5,
    zero_division_value: float = 0.0
) -> Dict[str, float]:
    """
    Computes exact pixel-level forensic localization metrics:
    - IoU (Jaccard Index): |P & G| / |P | G|
    - Dice (Sørensen-Dice / F1): 2*|P & G| / (|P| + |G|)
    - Precision: TP / (TP + FP)
    - Recall (True Positive Rate): TP / (TP + FN)
    - False Positive Rate (FPR): FP / (FP + TN)
    - False Negative Rate (FNR): FN / (FN + TP)
    - Pixel Accuracy: (TP + TN) / Total
    - Specificity (True Negative Rate): TN / (TN + FP)

    Parameters:
    - pred_mask: 2D array of predictions (probabilities [0, 1] or integers [0, 255])
    - gt_mask: 2D binary ground truth mask (values in {0, 1} or {0, 255})
    - threshold: Decision threshold for binarizing continuous predictions (default: 0.5)
    - zero_division_value: Value to return when both masks are all zero (authentic image correctly predicted clean).
    """
    if pred_mask.shape != gt_mask.shape:
        raise ValueError(
            f"Shape mismatch between pred_mask {pred_mask.shape} and gt_mask {gt_mask.shape}"
        )

    # Normalize prediction to [0.0, 1.0]
    p_norm = pred_mask.astype(np.float32)
    if p_norm.max() > 1.0:
        p_norm = p_norm / 255.0
    p_bin = p_norm >= threshold

    # Normalize ground truth to boolean
    g_norm = gt_mask.astype(np.float32)
    if g_norm.max() > 1.0:
        g_norm = g_norm / 255.0
    g_bin = g_norm >= 0.5

    tp = int(np.logical_and(p_bin, g_bin).sum())
    fp = int(np.logical_and(p_bin, ~g_bin).sum())
    fn = int(np.logical_and(~p_bin, g_bin).sum())
    tn = int(np.logical_and(~p_bin, ~g_bin).sum())
    total_pixels = int(p_bin.size)

    gt_positive = g_bin.sum() > 0
    pred_positive = p_bin.sum() > 0

    # Handling empty ground truth (authentic / un-tampered image)
    if not gt_positive:
        if not pred_positive:
            # Correctly predicted completely clean
            iou = 1.0
            dice = 1.0
            prec = 1.0
            rec = 1.0
            fpr = 0.0
            fnr = 0.0
            spec = 1.0
        else:
            # False alarms on clean image
            iou = 0.0
            dice = 0.0
            prec = 0.0
            rec = 0.0
            fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 1.0
            fnr = 0.0
            spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        pixel_acc = float((tp + tn) / total_pixels)
        return {
            "iou": float(iou),
            "dice": float(dice),
            "precision": float(prec),
            "recall": float(rec),
            "false_positive_rate": float(fpr),
            "false_negative_rate": float(fnr),
            "pixel_accuracy": float(pixel_acc),
            "specificity": float(spec),
            "tp_pixels": tp,
            "fp_pixels": fp,
            "fn_pixels": fn,
            "tn_pixels": tn,
            "total_pixels": total_pixels
        }

    # Standard positive ground truth case
    intersection = tp
    union = tp + fp + fn

    iou = float(intersection / union) if union > 0 else zero_division_value
    dice = float((2.0 * intersection) / (2 * tp + fp + fn)) if (2 * tp + fp + fn) > 0 else zero_division_value
    prec = float(tp / (tp + fp)) if (tp + fp) > 0 else zero_division_value
    rec = float(tp / (tp + fn)) if (tp + fn) > 0 else zero_division_value
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else zero_division_value
    fnr = float(fn / (fn + tp)) if (fn + tp) > 0 else zero_division_value
    spec = float(tn / (tn + fp)) if (tn + fp) > 0 else zero_division_value
    pixel_acc = float((tp + tn) / total_pixels)

    return {
        "iou": float(iou),
        "dice": float(dice),
        "precision": float(prec),
        "recall": float(rec),
        "false_positive_rate": float(fpr),
        "false_negative_rate": float(fnr),
        "pixel_accuracy": float(pixel_acc),
        "specificity": float(spec),
        "tp_pixels": tp,
        "fp_pixels": fp,
        "fn_pixels": fn,
        "tn_pixels": tn,
        "total_pixels": total_pixels
    }


def compute_pixel_auc(
    pred_prob_map: np.ndarray,
    gt_mask: np.ndarray,
    downsample_factor: int = 1
) -> float:
    """
    Computes Area Under ROC Curve (AUC-ROC) for continuous pixel-level probabilities.
    Optionally downsamples very large masks (e.g. 2K medical scans) for computational efficiency.
    Returns np.nan if ground truth contains only one class.
    """
    if pred_prob_map.shape != gt_mask.shape:
        raise ValueError("Shape mismatch between probability map and ground truth.")

    p_flat = pred_prob_map.astype(np.float32)
    if p_flat.max() > 1.0:
        p_flat = p_flat / 255.0

    g_flat = gt_mask.astype(np.float32)
    if g_flat.max() > 1.0:
        g_flat = g_flat / 255.0
    g_bin = (g_flat >= 0.5).astype(np.int32)

    if downsample_factor > 1:
        p_flat = p_flat[::downsample_factor, ::downsample_factor]
        g_bin = g_bin[::downsample_factor, ::downsample_factor]

    p_vec = p_flat.ravel()
    g_vec = g_bin.ravel()

    # If only 1 class present, AUC is mathematically undefined
    if len(np.unique(g_vec)) < 2:
        return float(np.nan)

    try:
        return float(roc_auc_score(g_vec, p_vec))
    except Exception:
        return float(np.nan)


def compute_image_level_metrics(
    y_true: Union[List[int], np.ndarray],
    y_prob: Union[List[float], np.ndarray],
    threshold: float = 0.5
) -> Dict[str, float]:
    """
    Computes image-level binary classification forensic metrics:
    - Accuracy: (TP + TN) / Total
    - Balanced Accuracy: (Recall_pos + Recall_neg) / 2
    - Precision, Recall, F1
    - Specificity, False Positive Rate (FPR)
    - ROC-AUC: Area under ROC curve (handled gracefully if single-class)
    - Average Precision (AP / PR-AUC)
    """
    y_t = np.asarray(y_true, dtype=int)
    y_p = np.asarray(y_prob, dtype=float)

    if len(y_t) != len(y_p):
        raise ValueError(f"Lengths differ: y_true={len(y_t)}, y_prob={len(y_p)}")

    if len(y_t) == 0:
        return {
            "accuracy": float(np.nan),
            "balanced_accuracy": float(np.nan),
            "auc": float(np.nan),
            "ap": float(np.nan),
            "precision": float(np.nan),
            "recall": float(np.nan),
            "f1": float(np.nan),
            "fpr": float(np.nan),
            "specificity": float(np.nan)
        }

    y_pred = (y_p >= threshold).astype(int)

    tp = int(np.logical_and(y_pred == 1, y_t == 1).sum())
    fp = int(np.logical_and(y_pred == 1, y_t == 0).sum())
    fn = int(np.logical_and(y_pred == 0, y_t == 1).sum())
    tn = int(np.logical_and(y_pred == 0, y_t == 0).sum())
    total = len(y_t)

    acc = float((tp + tn) / total)
    prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = float(2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
    spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    bal_acc = float((rec + spec) / 2.0)

    # AUC & AP
    if len(np.unique(y_t)) >= 2:
        try:
            auc = float(roc_auc_score(y_t, y_p))
        except Exception:
            auc = float(np.nan)
        try:
            ap = float(average_precision_score(y_t, y_p))
        except Exception:
            ap = float(np.nan)
    else:
        auc = float(np.nan)
        ap = float(np.nan)

    return {
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "auc": auc,
        "ap": ap,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "fpr": fpr,
        "specificity": spec,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "total": total
    }

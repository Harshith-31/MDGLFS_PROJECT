import os
import sys
import json
import time
import argparse
import datetime
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.datasets.dataset import MultiDomainForensicDataset
from src.models.router import HybridRouter, DOMAINS
from src.models.detectors.face_detector import FaceDetector
from src.models.detectors.med_detector import MedDetector
from src.models.detectors.sat_detector import SatDetector
from src.models.detectors.gen_detector import GeneralDetector

def compute_binary_metrics(y_true, y_pred_prob, threshold=0.5):
    y_true = np.array(y_true, dtype=int)
    y_pred_prob = np.array(y_pred_prob, dtype=float)
    y_pred = (y_pred_prob >= threshold).astype(int)

    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))

    acc = (tp + tn) / max(1, len(y_true))
    prec = tp / max(1, (tp + fp))
    rec = tp / max(1, (tp + fn))
    f1 = 2 * (prec * rec) / max(1e-8, (prec + rec))

    # ROC AUC calculation (exact trapezoidal rule)
    try:
        from sklearn.metrics import roc_auc_score
        auc = float(roc_auc_score(y_true, y_pred_prob))
    except Exception:
        # Fallback ranking AUC
        order = np.argsort(y_pred_prob)[::-1]
        y_sorted = y_true[order]
        n_pos = np.sum(y_true == 1)
        n_neg = np.sum(y_true == 0)
        if n_pos == 0 or n_neg == 0:
            auc = 1.0
        else:
            rank = np.arange(len(y_true), 0, -1)
            auc = float((np.sum(rank[y_sorted == 1]) - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))

    return {
        "accuracy": round(float(acc), 4),
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "f1": round(float(f1), 4),
        "roc_auc": round(float(auc), 4),
        "confusion_matrix": {"tp": tp, "fp": fp, "fn": fn, "tn": tn}
    }

def compute_mask_metrics(pred_mask, gt_mask, threshold=0.35):
    p_bin = (pred_mask >= threshold).astype(float)
    g_bin = (gt_mask >= threshold).astype(float)
    inter = np.sum(p_bin * g_bin)
    union = np.sum(np.clip(p_bin + g_bin, 0, 1))
    total = np.sum(p_bin) + np.sum(g_bin)

    if union == 0:
        iou = 1.0 if np.sum(g_bin) == 0 else 0.0
        dice = 1.0 if total == 0 else 0.0
    else:
        iou = float(inter / union)
        dice = float((2.0 * inter) / (total + 1e-8))

    return {"iou": round(iou, 4), "dice": round(dice, 4)}

def evaluate_full_system(test_manifest_path="datasets/manifests/test_manifest.json", device="cpu"):
    print(f"\n================================================================================")
    print(f" MD-DGFLS Comprehensive Forensic Evaluation Suite")
    print(f" Test Set: {test_manifest_path} | Device: {device}")
    print(f"================================================================================")

    if not os.path.exists(test_manifest_path):
        raise FileNotFoundError(f"Test manifest not found at {test_manifest_path}")

    with open(test_manifest_path) as f:
        test_data = json.load(f)

    # Initialize Router and Detectors
    router = HybridRouter(vit_weights_path="models/router_best.pth", device=device)

    detectors = {
        "Face": FaceDetector(),
        "Medical Radiograph": MedDetector(),
        "Satellite Terrain": SatDetector(),
        "Regular Out-of-Domain": GeneralDetector(),
        "Natural Scene": GeneralDetector()
    }
    weight_map = {
        "Face": "models/face_detector_best.pth",
        "Medical Radiograph": "models/xray_detector_best.pth",
        "Satellite Terrain": "models/sat_detector_best.pth",
        "Regular Out-of-Domain": "models/general_detector_best.pth",
        "Natural Scene": "models/general_detector_best.pth"
    }

    for dom, model in detectors.items():
        w_path = weight_map[dom]
        if os.path.exists(w_path):
            try:
                model.load_state_dict(torch.load(w_path, map_location="cpu", weights_only=True), strict=True)
                print(f"  [✓] Loaded {dom} checkpoint with strict=True from {w_path}")
            except Exception as e:
                print(f"  [!] Failed to load {w_path} strictly: {e}")
        else:
            print(f"  [!] Checkpoint {w_path} not found.")
        model.to(device)
        model.eval()

    # Domain mapping
    domain_map = {
        "Face": "Face",
        "Medical Radiograph": "Medical Radiograph",
        "Satellite Terrain": "Satellite Terrain",
        "Regular Out-of-Domain": "Regular Out-of-Domain",
        "Natural Scene": "Natural Scene"
    }

    results_by_domain = {d: {"y_true": [], "y_prob": [], "ious": [], "dices": []} for d in DOMAINS}
    routing_correct = 0
    total_samples = len(test_data)

    overall_y_true = []
    overall_y_prob = []
    overall_ious = []
    overall_dices = []
    latencies = []

    for item in test_data:
        img_path = item["image_path"]
        gt_domain = item["domain"]
        gt_label = int(item["label"])

        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            continue

        h, w = img_bgr.shape[:2]
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (256, 256))
        tensor = torch.from_numpy(img_resized.transpose(2, 0, 1) / 255.0).float()
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        tensor = ((tensor - mean) / std).unsqueeze(0).to(device)

        # Load GT Mask
        mask_path = item.get("mask_path", "")
        if mask_path and os.path.exists(mask_path):
            gt_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) / 255.0
            gt_mask = cv2.resize(gt_mask, (256, 256))
        else:
            gt_mask = np.zeros((256, 256), dtype=np.float32)

        # Time inference
        t0 = time.perf_counter()

        # Step 1: Route
        routing = router.route(img_bgr, tensor, filename=os.path.basename(img_path))
        pred_domain = routing["domain"]
        if pred_domain == gt_domain:
            routing_correct += 1

        # Step 2: Routed Detector Inference
        detector = detectors[pred_domain]
        with torch.no_grad():
            preds = detector(tensor)
            prob = float(preds["prob"][0, 0].cpu().item())
            pred_mask = preds["mask"][0, 0].cpu().numpy()

        t_end = time.perf_counter()
        latencies.append((t_end - t0) * 1000.0)

        # Localization metrics
        m_metrics = compute_mask_metrics(pred_mask, gt_mask)

        # Accumulate
        results_by_domain[gt_domain]["y_true"].append(gt_label)
        results_by_domain[gt_domain]["y_prob"].append(prob)
        results_by_domain[gt_domain]["ious"].append(m_metrics["iou"])
        results_by_domain[gt_domain]["dices"].append(m_metrics["dice"])

        overall_y_true.append(gt_label)
        overall_y_prob.append(prob)
        overall_ious.append(m_metrics["iou"])
        overall_dices.append(m_metrics["dice"])

    # Aggregate Metrics
    routing_acc = routing_correct / max(1, total_samples)
    overall_clf = compute_binary_metrics(overall_y_true, overall_y_prob)

    domain_reports = {}
    for d, d_data in results_by_domain.items():
        if len(d_data["y_true"]) > 0:
            clf_m = compute_binary_metrics(d_data["y_true"], d_data["y_prob"])
            clf_m["mean_iou"] = round(float(np.mean(d_data["ious"])), 4)
            clf_m["mean_dice"] = round(float(np.mean(d_data["dices"])), 4)
            clf_m["sample_count"] = len(d_data["y_true"])
            domain_reports[d] = clf_m

    avg_latency = float(np.mean(latencies)) if latencies else 0.0
    p95_latency = float(np.percentile(latencies, 95)) if latencies else 0.0

    summary = {
        "evaluation_timestamp": datetime.datetime.now().isoformat(),
        "total_test_samples": total_samples,
        "routing_accuracy": round(routing_acc, 4),
        "overall_classification": overall_clf,
        "overall_mean_iou": round(float(np.mean(overall_ious)), 4),
        "overall_mean_dice": round(float(np.mean(overall_dices)), 4),
        "latency_ms": {
            "mean": round(avg_latency, 2),
            "p95": round(p95_latency, 2)
        },
        "domain_breakdown": domain_reports
    }

    # Phase 14: Cross-Domain Generalization Transferability Matrix
    print(f"\nComputing Cross-Domain Generalization Benchmark (Phase 14)...")
    expert_models = {
        "FaceDetector": detectors["Face"],
        "MedDetector": detectors["Medical Radiograph"],
        "SatDetector": detectors["Satellite Terrain"],
        "GeneralDetector": detectors["Natural Scene"]
    }

    cross_domain_matrix = {}
    for exp_name, exp_model in expert_models.items():
        cross_domain_matrix[exp_name] = {}
        for target_dom in DOMAINS:
            dom_items = [it for it in test_data if it["domain"] == target_dom]
            y_t = []
            y_p = []
            for it in dom_items:
                im = cv2.imread(it["image_path"])
                if im is None:
                    continue
                im_rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
                im_res = cv2.resize(im_rgb, (256, 256))
                t = torch.from_numpy(im_res.transpose(2, 0, 1) / 255.0).float()
                t = ((t - mean) / std).unsqueeze(0).to(device)
                with torch.no_grad():
                    out = exp_model(t)
                    y_p.append(float(out["prob"][0, 0].cpu().item()))
                    y_t.append(int(it["label"]))
            if len(y_t) > 0:
                m_res = compute_binary_metrics(y_t, y_p)
                cross_domain_matrix[exp_name][target_dom] = {
                    "accuracy": m_res["accuracy"],
                    "roc_auc": m_res["roc_auc"],
                    "f1": m_res["f1"]
                }

    summary["cross_domain_generalization"] = cross_domain_matrix

    # Print Clean Formatted Report
    print(f"\n==================== EVALUATION REPORT ====================")
    print(f"Total Test Samples: {total_samples}")
    print(f"Router Accuracy:    {routing_acc * 100:.1f}%")
    print(f"Classification Acc: {overall_clf['accuracy'] * 100:.1f}%")
    print(f"ROC-AUC:            {overall_clf['roc_auc']:.4f}")
    print(f"F1 Score:           {overall_clf['f1']:.4f}")
    print(f"Mean IoU:           {summary['overall_mean_iou']:.4f}")
    print(f"Mean Dice:          {summary['overall_mean_dice']:.4f}")
    print(f"Mean Latency:       {avg_latency:.2f} ms (p95: {p95_latency:.2f} ms)")
    print(f"\n--- In-Domain Performance Breakdown ---")
    for d, rep in domain_reports.items():
        print(f"  [{d}] (N={rep['sample_count']}): Acc={rep['accuracy']*100:.1f}%, F1={rep['f1']:.3f}, IoU={rep['mean_iou']:.3f}, Dice={rep['mean_dice']:.3f}")

    print(f"\n--- Cross-Domain Transferability Accuracy Matrix ---")
    header = f"{'Expert Model':<18} | " + " | ".join([f"{d[:8]:<8}" for d in DOMAINS])
    print(header)
    print("-" * len(header))
    for exp_name, row in cross_domain_matrix.items():
        vals = " | ".join([f"{row.get(d, {}).get('accuracy', 0.0)*100:6.1f}%" for d in DOMAINS])
        print(f"{exp_name:<18} | {vals}")
    print(f"===========================================================\n")

    os.makedirs("outputs", exist_ok=True)
    with open("outputs/evaluation_results.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[✓] Evaluation results exported to outputs/evaluation_results.json")
    return summary

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-manifest", default="datasets/manifests/test_manifest.json")
    args = parser.parse_args()

    dev = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    evaluate_full_system(test_manifest_path=args.test_manifest, device=dev)

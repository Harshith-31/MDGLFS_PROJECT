"""
Router Calibration and Routing Sensitivity Analysis (src/analysis/router_eval.py)
Implements:
1. Routing accuracy and confusion matrix across domains
2. Threshold sensitivity sweeps for physical priors:
   - HSV saturation std threshold (medical grayscale boundary)
   - Laplacian edge variance threshold (satellite high-frequency detail)
   - Dominant color bin ratio threshold (document flat graphics)
3. Expected Calibration Error (ECE) and reliability curve computation:
   - Evaluates ARCC dynamic temperature calibration against uncalibrated baseline (T=1.0)
"""

import os
import json
import numpy as np
import torch
import torchvision.transforms as transforms
import cv2
from typing import Dict, List, Tuple

from src.models.router import DOMAINS, HybridRouter
from src.eval.datasets.loaders import load_manifest_split, generate_controlled_protocol_samples, load_real_world_benchmark_samples


def compute_ece(confidences: np.ndarray, accuracies: np.ndarray, num_bins: int = 10) -> Tuple[float, List[Dict]]:
    """
    Computes Expected Calibration Error (ECE) and bin-wise reliability diagram metrics.
    ECE = Sum_m (|B_m| / N) * |acc(B_m) - conf(B_m)|
    """
    bin_boundaries = np.linspace(0.0, 1.0, num_bins + 1)
    ece = 0.0
    n = len(confidences)
    bin_details = []

    for i in range(num_bins):
        b_low = bin_boundaries[i]
        b_high = bin_boundaries[i + 1]
        in_bin = (confidences > b_low) & (confidences <= b_high) if i > 0 else (confidences >= b_low) & (confidences <= b_high)
        bin_count = int(np.sum(in_bin))

        if bin_count > 0:
            bin_acc = float(np.mean(accuracies[in_bin]))
            bin_conf = float(np.mean(confidences[in_bin]))
            bin_error = abs(bin_acc - bin_conf)
            ece += (bin_count / n) * bin_error
            bin_details.append({
                "bin_idx": i,
                "range": [round(b_low, 2), round(b_high, 2)],
                "count": bin_count,
                "accuracy": round(bin_acc, 4),
                "confidence": round(bin_conf, 4),
                "error": round(bin_error, 4)
            })
        else:
            bin_details.append({
                "bin_idx": i,
                "range": [round(b_low, 2), round(b_high, 2)],
                "count": 0,
                "accuracy": 0.0,
                "confidence": 0.0,
                "error": 0.0
            })

    return float(ece), bin_details


def evaluate_router(samples: List[Dict], router: HybridRouter, device: torch.device) -> Dict:
    """
    Evaluates router performance, builds confusion matrix, and computes ARCC calibration ECE.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    conf_matrix = {d_true: {d_pred: 0 for d_pred in DOMAINS} for d_true in DOMAINS}
    total_samples = 0
    correct_samples = 0

    calibrated_confs = []
    uncalibrated_confs = []
    accuracies = []

    for s in samples:
        img_bgr = s["image"]
        true_domain = s.get("domain")
        if true_domain not in DOMAINS:
            continue

        h, w = img_bgr.shape[:2]
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        tensor = transform(cv2.resize(img_rgb, (256, 256))).unsqueeze(0).to(device)

        # 1. Full ARCC routing
        route_out = router.route(img_bgr, tensor, manual_domain=None, filename=f"{s.get('id', 'test')}.png")
        pred_domain = route_out["domain"]
        cal_conf = route_out["domain_confidence"]

        # 2. Raw uncalibrated ViT probabilities
        with torch.no_grad():
            vit_out = router.vit(tensor)
            uncal_probs = vit_out["raw_probs"][0].cpu().numpy()
            uncal_conf = float(np.max(uncal_probs))

        is_correct = int(pred_domain == true_domain)
        conf_matrix[true_domain][pred_domain] += 1
        total_samples += 1
        correct_samples += is_correct

        calibrated_confs.append(cal_conf)
        uncalibrated_confs.append(uncal_conf)
        accuracies.append(is_correct)

    acc = correct_samples / total_samples if total_samples > 0 else 0.0

    cal_confs_arr = np.array(calibrated_confs)
    uncal_confs_arr = np.array(uncalibrated_confs)
    accs_arr = np.array(accuracies)

    ece_arcc, bins_arcc = compute_ece(cal_confs_arr, accs_arr, num_bins=10)
    ece_raw, bins_raw = compute_ece(uncal_confs_arr, accs_arr, num_bins=10)

    # Per-domain recall (precision of domain classification)
    per_domain_stats = {}
    for d in DOMAINS:
        total_d = sum(conf_matrix[d].values())
        correct_d = conf_matrix[d][d]
        per_domain_stats[d] = {
            "total": total_d,
            "correct": correct_d,
            "accuracy": round(correct_d / total_d, 4) if total_d > 0 else 0.0
        }

    return {
        "total_samples": total_samples,
        "overall_accuracy": round(acc, 4),
        "confusion_matrix": conf_matrix,
        "per_domain_accuracy": per_domain_stats,
        "calibration": {
            "arcc_ece": round(ece_arcc, 4),
            "uncalibrated_ece": round(ece_raw, 4),
            "ece_reduction_percent": round(((ece_raw - ece_arcc) / max(ece_raw, 1e-6)) * 100.0, 2),
            "arcc_bins": bins_arcc,
            "raw_bins": bins_raw
        }
    }


def sweep_heuristic_thresholds(samples: List[Dict]) -> Dict:
    """
    Performs parameter sweeps on physical prior thresholds to determine robustness and stability.
    """
    extractor = HybridRouter().low_level

    # Collect ground truth physical metrics
    priors_by_domain = {d: [] for d in DOMAINS}
    for s in samples:
        dom = s.get("domain")
        if dom in priors_by_domain:
            img = s["image"]
            p = extractor.extract(img)
            priors_by_domain[dom].append(p)

    # 1. Sweep Saturation Std for Medical (default 12.0)
    med_sat_values = [p["sat_std"] for p in priors_by_domain["Medical Radiograph"]]
    other_sat_values = [p["sat_std"] for d, plist in priors_by_domain.items() if d != "Medical Radiograph" for p in plist]

    sat_sweep = []
    for thresh in [6.0, 9.0, 12.0, 15.0, 18.0]:
        med_correct = sum(1 for v in med_sat_values if v < thresh)
        med_acc = med_correct / len(med_sat_values) if med_sat_values else 0.0
        sat_sweep.append({"threshold": thresh, "medical_filter_accuracy": round(med_acc, 4)})

    # 2. Sweep Laplacian Edge Variance for Satellite (default 600.0)
    sat_lap_values = [p["lap_var"] for p in priors_by_domain["Satellite Terrain"]]
    lap_sweep = []
    for thresh in [300.0, 450.0, 600.0, 750.0, 900.0]:
        sat_correct = sum(1 for v in sat_lap_values if v > thresh)
        sat_acc = sat_correct / len(sat_lap_values) if sat_lap_values else 0.0
        lap_sweep.append({"threshold": thresh, "satellite_edge_filter_accuracy": round(sat_acc, 4)})

    # 3. Sweep Document Dominant Bin Ratio (default 0.25)
    doc_bin_values = [p["dominant_bin_ratio"] for p in priors_by_domain["Regular Out-of-Domain"]]
    bin_sweep = []
    for thresh in [0.15, 0.20, 0.25, 0.30, 0.35]:
        doc_correct = sum(1 for v in doc_bin_values if v > thresh)
        doc_acc = doc_correct / len(doc_bin_values) if doc_bin_values else 0.0
        bin_sweep.append({"threshold": thresh, "doc_color_filter_accuracy": round(doc_acc, 4)})

    return {
        "saturation_threshold_sweep": sat_sweep,
        "laplacian_threshold_sweep": lap_sweep,
        "dominant_bin_threshold_sweep": bin_sweep
    }


def main():
    print("=" * 70)
    print("MD-DGFLS Router Evaluation & ARCC Calibration Study (IEEE)")
    print("=" * 70)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    router = HybridRouter(device=str(device))

    # Load test samples
    samples = load_manifest_split("test") + generate_controlled_protocol_samples(seed=42) + load_real_world_benchmark_samples()
    print(f"Loaded {len(samples)} samples for router evaluation.")

    print("\n[1/3] Computing Routing Accuracy, Confusion Matrix, and ARCC ECE...")
    eval_results = evaluate_router(samples, router, device)
    print(f"Overall Routing Accuracy: {eval_results['overall_accuracy'] * 100:.2f}%")
    print(f"Raw ViT ECE: {eval_results['calibration']['uncalibrated_ece']:.4f}")
    print(f"ARCC Calibrated ECE: {eval_results['calibration']['arcc_ece']:.4f}")
    print(f"ECE Reduction: {eval_results['calibration']['ece_reduction_percent']:.2f}%")

    print("\n[2/3] Performing Heuristic Physical Prior Threshold Sweeps...")
    sweep_results = sweep_heuristic_thresholds(samples)

    # Export results
    output = {
        "router_evaluation": eval_results,
        "threshold_sweeps": sweep_results
    }

    os.makedirs("results", exist_ok=True)
    with open("results/router_evaluation_summary.json", "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    # Generate LaTeX Confusion Matrix
    cm = eval_results["confusion_matrix"]
    tex_cm = r"""\begin{table}[h]
\centering
\caption{Hybrid Router Domain Classification Confusion Matrix across 5 Forensic Domains.}
\label{tab:router_confusion_matrix}
\resizebox{\columnwidth}{!}{
\begin{tabular}{lccccc}
\toprule
\textbf{Ground Truth} & \textbf{Face} & \textbf{Medical} & \textbf{Satellite} & \textbf{Document} & \textbf{Natural} \\
\midrule
"""
    for d_true in DOMAINS:
        counts = [str(cm[d_true][d_pred]) for d_pred in DOMAINS]
        tex_cm += f"{d_true} & " + " & ".join(counts) + r" \\" + "\n"

    tex_cm += r"""\bottomrule
\end{tabular}
}
\end{table}
"""
    with open("results/router_confusion_matrix.tex", "w", encoding="utf-8") as f:
        f.write(tex_cm)

    print("\n[3/3] Exported:")
    print("  -> results/router_evaluation_summary.json")
    print("  -> results/router_confusion_matrix.tex")
    print("=" * 70)


if __name__ == "__main__":
    main()

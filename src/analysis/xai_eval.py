"""
Explainable AI (XAI) Quantitative Benchmark (src/analysis/xai_eval.py)
Evaluates forensic interpretability methods:
1. Grad-CAM (Gradient-weighted Class Activation Mapping)
2. Forensic SHAP (Gradient-weighted Input Attribution)
3. Forensic LIME (Superpixel Perturbation Surrogate)

Metrics computed against pixel-level Ground Truth Tamper Masks:
- Pointing Game Accuracy: Hits when max attribution point falls inside GT tampered region
- Heatmap-to-GT IoU Overlap (threshold tau = 0.5)
- Heatmap-to-GT Dice Overlap
- Attribution Contrast Ratio (ACR): Mean attribution inside tampered region vs authentic background

Generates:
- results/xai_evaluation_table.csv
- results/xai_evaluation_table.tex
- results/xai_summary.json
"""

import os
import csv
import json
import numpy as np
import torch
import torchvision.transforms as transforms
import cv2
from typing import Dict, List

from src.eval.runner import EvaluationRunner
from src.eval.metrics import evaluate_mask_metrics
from src.eval.statistics import compute_bootstrap_ci
from src.utils.explainability import GradCAM, ForensicSHAP, ForensicLIME
from src.eval.datasets.loaders import (
    load_manifest_split,
    generate_controlled_protocol_samples,
    load_real_world_benchmark_samples
)


def evaluate_xai_method(method_name: str, samples: List[Dict], runner: EvaluationRunner) -> Dict:
    pointing_hits = []
    ious = []
    dices = []
    acrs = []

    transform = runner.transform
    device = runner.device

    for s in samples:
        img_bgr = s["image"]
        gt_mask = s["gt_mask"]
        if gt_mask.sum() == 0:
            continue  # only evaluate localization interpretability on manipulated samples

        h, w = img_bgr.shape[:2]
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        tensor = transform(cv2.resize(img_rgb, (256, 256))).unsqueeze(0).to(device)

        # Route to domain expert
        route_info = runner.router.route(img_bgr, tensor, filename=f"{s.get('id', 'test')}.png")
        dom = route_info["domain"]
        detector = runner.detectors[dom]

        # Generate XAI attribution map
        if method_name == "Grad-CAM":
            target_layer = detector.grad_cam_target_layer
            gcam = GradCAM(detector, target_layer)
            heat_256 = gcam.generate(tensor)
        elif method_name == "Forensic SHAP":
            heat_256 = ForensicSHAP.explain(detector, tensor)
        elif method_name == "Forensic LIME":
            heat_256 = ForensicLIME.explain(detector, tensor, num_samples=16, grid_size=8)
        else:
            raise ValueError(f"Unknown XAI method: {method_name}")

        heat_orig = cv2.resize(heat_256, (w, h))
        heat_norm = np.clip(heat_orig, 0.0, 1.0)

        # 1. Pointing Game: argmax pixel inside GT mask
        max_idx = np.unravel_index(np.argmax(heat_norm), heat_norm.shape)
        hit = int(gt_mask[max_idx[0], max_idx[1]] > 0)
        pointing_hits.append(hit)

        # 2. IoU & Dice overlap at threshold 0.5
        m = evaluate_mask_metrics(heat_norm, gt_mask, threshold=0.5)
        ious.append(m["iou"])
        dices.append(m["dice"])

        # 3. Attribution Contrast Ratio (ACR)
        tamper_mask_bin = (gt_mask > 0)
        bg_mask_bin = (~tamper_mask_bin)
        mean_tamper = float(np.mean(heat_norm[tamper_mask_bin])) if np.any(tamper_mask_bin) else 0.0
        mean_bg = float(np.mean(heat_norm[bg_mask_bin])) if np.any(bg_mask_bin) else 1e-4
        acr = mean_tamper / max(mean_bg, 1e-4)
        acrs.append(acr)

    return {
        "num_samples": len(pointing_hits),
        "pointing_game_accuracy": float(np.mean(pointing_hits)),
        "iou": compute_bootstrap_ci(ious, num_resamples=1000, seed=42),
        "dice": compute_bootstrap_ci(dices, num_resamples=1000, seed=42),
        "attribution_contrast_ratio": compute_bootstrap_ci(acrs, num_resamples=1000, seed=42)
    }


def main():
    print("=" * 70)
    print("MD-DGFLS Phase 4: XAI Interpretability Quantification Benchmark (IEEE)")
    print("=" * 70)

    runner = EvaluationRunner()

    # Load manipulated samples with ground truth masks
    samples = (
        load_manifest_split("test") +
        generate_controlled_protocol_samples(seed=42) +
        load_real_world_benchmark_samples()
    )
    manipulated_samples = [s for s in samples if s["gt_mask"].sum() > 0]
    print(f"Loaded {len(manipulated_samples)} manipulated samples for XAI quantification.")

    methods = ["Grad-CAM", "Forensic SHAP", "Forensic LIME"]
    results_by_method = {}

    for m in methods:
        print(f"\nQuantifying XAI Method: [{m}]...")
        res = evaluate_xai_method(m, manipulated_samples, runner)
        results_by_method[m] = res
        print(f"  Pointing Game Acc: {res['pointing_game_accuracy'] * 100:.1f}%")
        print(f"  mIoU Overlap:      {res['iou']['mean']:.4f} [{res['iou']['ci_lower']:.4f}, {res['iou']['ci_upper']:.4f}]")
        print(f"  mDice Overlap:     {res['dice']['mean']:.4f}")
        print(f"  Attribution Contrast Ratio (ACR): {res['attribution_contrast_ratio']['mean']:.2f}x")

    # Save summary JSON
    os.makedirs("results", exist_ok=True)
    summary_data = {
        "timestamp": "20261002_xai_eval",
        "num_manipulated_samples": len(manipulated_samples),
        "results": results_by_method
    }
    with open("results/xai_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    # Save CSV
    csv_rows = []
    fieldnames = [
        "XAI Method", "Pointing Game Accuracy (%)", "Heatmap-GT mIoU",
        "95% CI (mIoU)", "Heatmap-GT mDice", "Attribution Contrast Ratio (ACR)"
    ]

    for m in methods:
        res = results_by_method[m]
        iou = res["iou"]
        dice = res["dice"]
        acr = res["attribution_contrast_ratio"]
        csv_rows.append({
            "XAI Method": m,
            "Pointing Game Accuracy (%)": f"{res['pointing_game_accuracy'] * 100:.2f}%",
            "Heatmap-GT mIoU": f"{iou['mean']:.4f}",
            "95% CI (mIoU)": f"[{iou['ci_lower']:.4f}, {iou['ci_upper']:.4f}]",
            "Heatmap-GT mDice": f"{dice['mean']:.4f}",
            "Attribution Contrast Ratio (ACR)": f"{acr['mean']:.2f}x"
        })

    with open("results/xai_evaluation_table.csv", "w", newline="", encoding="utf-8") as f_csv:
        writer = csv.DictWriter(f_csv, fieldnames=fieldnames)
        writer.writeheader()
        for r in csv_rows:
            writer.writerow(r)

    # Save LaTeX Table
    tex_content = r"""\begin{table}[t]
\centering
\caption{Quantitative Evaluation of Explainable AI (XAI) Attributions against Ground-Truth Manipulation Regions. Pointing Game accuracy measures whether peak saliency falls inside the tampered boundary. ACR denotes attribution contrast ratio between tampered and authentic pixels.}
\label{tab:xai_quantification}
\resizebox{\columnwidth}{!}{
\begin{tabular}{lcccc}
\toprule
\textbf{XAI Method} & \textbf{Pointing Game} $\uparrow$ & \textbf{Heatmap mIoU} $\uparrow$ & \textbf{95\% CI} & \textbf{ACR Contrast} $\uparrow$ \\
\midrule
"""
    for m in methods:
        res = results_by_method[m]
        iou = res["iou"]
        acr = res["attribution_contrast_ratio"]
        tex_content += (
            f"{m} & {res['pointing_game_accuracy'] * 100:.1f}\\% & "
            f"{iou['mean']:.4f} & [{iou['ci_lower']:.4f}, {iou['ci_upper']:.4f}] & "
            f"{acr['mean']:.2f}$\\times$ \\\\\n"
        )

    tex_content += r"""\bottomrule
\end{tabular}
}
\end{table}
"""
    with open("results/xai_evaluation_table.tex", "w", encoding="utf-8") as f_tex:
        f_tex.write(tex_content)

    print("\nSuccessfully generated:")
    print("  -> results/xai_summary.json")
    print("  -> results/xai_evaluation_table.csv")
    print("  -> results/xai_evaluation_table.tex")
    print("=" * 70)


if __name__ == "__main__":
    main()

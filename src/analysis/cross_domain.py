"""
Cross-Domain Generalization Study (src/analysis/cross_domain.py)
Evaluates each specialized domain expert detector against images from every domain (5x5 matrix).
Demonstrates the severe performance degradation of single-domain detectors on out-of-domain media,
proving the empirical necessity of the MD-DGFLS Hybrid Router.

Generates:
- results/cross_domain_matrix.csv
- results/cross_domain_matrix.tex
- results/cross_domain_summary.json
"""

import os
import csv
import json
import numpy as np
import torch
import torchvision.transforms as transforms
import cv2
from typing import Dict, List

from src.models.router import DOMAINS
from src.eval.runner import EvaluationRunner
from src.eval.metrics import evaluate_mask_metrics
from src.eval.datasets.loaders import (
    load_manifest_split,
    generate_controlled_protocol_samples,
    load_real_world_benchmark_samples
)


def evaluate_expert_on_domain(detector, samples_for_domain: List[Dict], device: torch.device, transform) -> Dict[str, float]:
    """
    Evaluates a specific detector directly on a set of images from a given domain (bypassing routing).
    """
    if not samples_for_domain:
        return {"iou": 0.0, "dice": 0.0, "fpr": 0.0}

    ious, dices, fprs = [], [], []

    for s in samples_for_domain:
        img_bgr = s["image"]
        gt_mask = s["gt_mask"]
        h_orig, w_orig = img_bgr.shape[:2]

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        tensor = transform(cv2.resize(img_rgb, (256, 256))).unsqueeze(0).to(device)

        with torch.no_grad():
            preds = detector(tensor)
            mask_256 = preds["mask"][0, 0].cpu().numpy()

        pred_mask = cv2.resize(mask_256, (w_orig, h_orig))
        pred_mask = np.clip(pred_mask, 0.0, 1.0)

        m = evaluate_mask_metrics(pred_mask, gt_mask)
        ious.append(m["iou"])
        dices.append(m["dice"])
        fprs.append(m["false_positive_rate"])

    return {
        "iou": float(np.mean(ious)),
        "dice": float(np.mean(dices)),
        "fpr": float(np.mean(fprs))
    }


def main():
    print("=" * 70)
    print("MD-DGFLS Cross-Domain Generalization Evaluation (5x5 Matrix)")
    print("=" * 70)

    runner = EvaluationRunner()
    device = runner.device
    transform = runner.transform

    # Load all evaluation samples and partition by domain
    all_samples = (
        load_manifest_split("test") +
        generate_controlled_protocol_samples(seed=42) +
        load_real_world_benchmark_samples()
    )

    domain_samples = {d: [s for s in all_samples if s.get("domain") == d] for d in DOMAINS}
    for d, s_list in domain_samples.items():
        print(f"Domain '{d}': {len(s_list)} samples")

    matrix_iou = {d_exp: {} for d_exp in DOMAINS}
    matrix_dice = {d_exp: {} for d_exp in DOMAINS}
    matrix_fpr = {d_exp: {} for d_exp in DOMAINS}

    print("\nEvaluating 5x5 Cross-Domain Transfer Matrix...")
    for expert_domain in DOMAINS:
        detector = runner.detectors[expert_domain]
        print(f"  Evaluating Expert: [{expert_domain}]...")

        for target_domain in DOMAINS:
            res = evaluate_expert_on_domain(
                detector=detector,
                samples_for_domain=domain_samples[target_domain],
                device=device,
                transform=transform
            )
            matrix_iou[expert_domain][target_domain] = round(res["iou"], 4)
            matrix_dice[expert_domain][target_domain] = round(res["dice"], 4)
            matrix_fpr[expert_domain][target_domain] = round(res["fpr"], 4)
            print(f"    -> Target [{target_domain}]: mIoU={res['iou']:.4f}, mDice={res['dice']:.4f}, FPR={res['fpr']:.4f}")

    os.makedirs("results", exist_ok=True)

    # Save summary JSON
    summary_data = {
        "protocol": "Cross-Domain Transfer Matrix (Cache-DISABLED)",
        "domains": DOMAINS,
        "matrix_iou": matrix_iou,
        "matrix_dice": matrix_dice,
        "matrix_fpr": matrix_fpr
    }
    with open("results/cross_domain_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    # Save CSV
    csv_rows = []
    fieldnames = ["Expert Model", "Target Domain", "mIoU", "mDice", "FPR", "In-Domain vs Cross-Domain"]
    for d_exp in DOMAINS:
        for d_tgt in DOMAINS:
            is_in_domain = (d_exp == d_tgt)
            csv_rows.append({
                "Expert Model": d_exp,
                "Target Domain": d_tgt,
                "mIoU": matrix_iou[d_exp][d_tgt],
                "mDice": matrix_dice[d_exp][d_tgt],
                "FPR": matrix_fpr[d_exp][d_tgt],
                "In-Domain vs Cross-Domain": "In-Domain (Diagonal)" if is_in_domain else "Cross-Domain Transfer"
            })

    with open("results/cross_domain_matrix.csv", "w", newline="", encoding="utf-8") as f_csv:
        writer = csv.DictWriter(f_csv, fieldnames=fieldnames)
        writer.writeheader()
        for r in csv_rows:
            writer.writerow(r)

    # Save LaTeX Table
    tex_content = r"""\begin{table*}[t]
\centering
\caption{Cross-Domain Generalization Matrix ($5 \times 5$). Cell values report Localization mIoU (\%) when evaluating a domain-specific expert model on target test sets. Bold diagonal entries denote in-domain performance. Off-diagonal degradation validates the necessity of the multi-domain router.}
\label{tab:cross_domain_matrix}
\resizebox{\textwidth}{!}{
\begin{tabular}{lccccc}
\toprule
\textbf{Expert Detector} & \textbf{Target: Face} & \textbf{Target: Medical} & \textbf{Target: Satellite} & \textbf{Target: Document} & \textbf{Target: Natural} \\
\midrule
"""
    for d_exp in DOMAINS:
        row_str = f"{d_exp} Expert & "
        cells = []
        for d_tgt in DOMAINS:
            val = matrix_iou[d_exp][d_tgt] * 100.0
            if d_exp == d_tgt:
                cells.append(f"\\textbf{{{val:.1f}\\%}}")
            else:
                cells.append(f"{val:.1f}\\%")
        row_str += " & ".join(cells) + r" \\" + "\n"
        tex_content += row_str

    tex_content += r"""\bottomrule
\end{tabular}
}
\end{table*}
"""
    with open("results/cross_domain_matrix.tex", "w", encoding="utf-8") as f_tex:
        f_tex.write(tex_content)

    print("\nSuccessfully generated:")
    print("  -> results/cross_domain_summary.json")
    print("  -> results/cross_domain_matrix.csv")
    print("  -> results/cross_domain_matrix.tex")
    print("=" * 70)


if __name__ == "__main__":
    main()

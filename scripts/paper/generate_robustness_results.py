"""
Robustness Evaluation Runner (scripts/paper/generate_robustness_results.py)
Evaluates forensic degradation of MD-DGFLS (Cache-DISABLED) and U-Net baseline
under transmission channels, compression artifacts, and adversarial perturbations.

Generates:
- results/robustness_table.csv
- results/robustness_table.tex
- results/robustness_summary.json
"""

import os
import csv
import json
import numpy as np
from datetime import datetime
from typing import Dict, List

from src.eval.runner import EvaluationRunner
from src.eval.metrics import evaluate_mask_metrics
from src.eval.statistics import compute_bootstrap_ci
from src.eval.datasets.loaders import (
    load_manifest_split,
    generate_controlled_protocol_samples,
    load_real_world_benchmark_samples
)
from src.robustness.perturbations import get_all_perturbations
from src.baselines.unet_baseline import UNetBaseline


def evaluate_cohort_under_perturbation(runner, unet, samples: List[Dict], perturb_fn, perturb_args) -> Dict:
    md_ious, md_dices, md_fprs = [], [], []
    un_ious, un_dices, un_fprs = [], [], []

    for s in samples:
        orig_img = s["image"]
        gt_mask = s["gt_mask"]
        dom = s.get("domain")

        # Apply perturbation
        pert_img = perturb_fn(orig_img, **perturb_args)

        # 1. MD-DGFLS evaluation (Cache-Disabled)
        res_md = runner.evaluate_single_image(
            img_bgr=pert_img,
            gt_mask=gt_mask,
            domain_hint=None,
            cache_mode="disabled",
            filename=f"{s.get('id', 'test')}.png"
        )
        md_ious.append(res_md["iou"])
        md_dices.append(res_md["dice"])
        md_fprs.append(res_md["fpr"])

        # 2. U-Net Baseline evaluation
        un_mask = unet.predict_mask(pert_img)
        m_un = evaluate_mask_metrics(un_mask, gt_mask)
        un_ious.append(m_un["iou"])
        un_dices.append(m_un["dice"])
        un_fprs.append(m_un["false_positive_rate"])

    return {
        "md_dgfls": {
            "iou": compute_bootstrap_ci(md_ious, num_resamples=1000, seed=42),
            "dice": compute_bootstrap_ci(md_dices, num_resamples=1000, seed=42),
            "fpr": compute_bootstrap_ci(md_fprs, num_resamples=1000, seed=42)
        },
        "unet": {
            "iou": compute_bootstrap_ci(un_ious, num_resamples=1000, seed=42),
            "dice": compute_bootstrap_ci(un_dices, num_resamples=1000, seed=42),
            "fpr": compute_bootstrap_ci(un_fprs, num_resamples=1000, seed=42)
        }
    }


def main():
    print("=" * 70)
    print("MD-DGFLS Phase 4: Robustness & Transmission Degradation Study (IEEE)")
    print("=" * 70)

    runner = EvaluationRunner()
    unet = UNetBaseline(name="Standard U-Net", device=str(runner.device))

    # Load test cohort
    samples = (
        load_manifest_split("test") +
        generate_controlled_protocol_samples(seed=42) +
        load_real_world_benchmark_samples()
    )
    print(f"Loaded {len(samples)} evaluation samples.")

    perturbations = get_all_perturbations()
    results_by_pert = {}

    clean_md_iou = None
    clean_un_iou = None

    for name, (fn, args) in perturbations.items():
        print(f"\nEvaluating Perturbation: [{name}]...")
        perf = evaluate_cohort_under_perturbation(runner, unet, samples, fn, args)
        results_by_pert[name] = perf

        md_mean = perf["md_dgfls"]["iou"]["mean"]
        un_mean = perf["unet"]["iou"]["mean"]

        if clean_md_iou is None:
            clean_md_iou = md_mean
            clean_un_iou = un_mean

        md_ret = (md_mean / max(clean_md_iou, 1e-6)) * 100.0
        un_ret = (un_mean / max(clean_un_iou, 1e-6)) * 100.0

        print(f"  MD-DGFLS mIoU: {md_mean:.4f} (Retention: {md_ret:.1f}%), FPR: {perf['md_dgfls']['fpr']['mean']:.4f}")
        print(f"  U-Net    mIoU: {un_mean:.4f} (Retention: {un_ret:.1f}%), FPR: {perf['unet']['fpr']['mean']:.4f}")

    # Export Summary JSON
    os.makedirs("results", exist_ok=True)
    summary_data = {
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "protocol": "Forensic Robustness Perturbations (Cache-DISABLED)",
        "num_samples": len(samples),
        "results": results_by_pert
    }
    with open("results/robustness_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    # Export CSV
    csv_rows = []
    fieldnames = [
        "Perturbation", "MD-DGFLS mIoU", "MD-DGFLS 95% CI", "MD-DGFLS FPR", "MD-DGFLS Retention %",
        "U-Net mIoU", "U-Net 95% CI", "U-Net FPR", "U-Net Retention %"
    ]

    for name, perf in results_by_pert.items():
        md_i = perf["md_dgfls"]["iou"]
        un_i = perf["unet"]["iou"]
        md_fpr = perf["md_dgfls"]["fpr"]["mean"]
        un_fpr = perf["unet"]["fpr"]["mean"]

        md_ret = (md_i["mean"] / max(clean_md_iou, 1e-6)) * 100.0
        un_ret = (un_i["mean"] / max(clean_un_iou, 1e-6)) * 100.0

        csv_rows.append({
            "Perturbation": name,
            "MD-DGFLS mIoU": f"{md_i['mean']:.4f}",
            "MD-DGFLS 95% CI": f"[{md_i['ci_lower']:.4f}, {md_i['ci_upper']:.4f}]",
            "MD-DGFLS FPR": f"{md_fpr:.4f}",
            "MD-DGFLS Retention %": f"{md_ret:.1f}%",
            "U-Net mIoU": f"{un_i['mean']:.4f}",
            "U-Net 95% CI": f"[{un_i['ci_lower']:.4f}, {un_i['ci_upper']:.4f}]",
            "U-Net FPR": f"{un_fpr:.4f}",
            "U-Net Retention %": f"{un_ret:.1f}%"
        })

    with open("results/robustness_table.csv", "w", newline="", encoding="utf-8") as f_csv:
        writer = csv.DictWriter(f_csv, fieldnames=fieldnames)
        writer.writeheader()
        for r in csv_rows:
            writer.writerow(r)

    # Export LaTeX Table
    tex_content = r"""\begin{table*}[t]
\centering
\caption{Robustness and Distortion Resilience Analysis. Compares MD-DGFLS against standard U-Net baseline across compression, scaling, noise, blur, and screenshot channels under zero-knowledge evaluation.}
\label{tab:robustness_results}
\resizebox{\textwidth}{!}{
\begin{tabular}{lcccccc}
\toprule
\multirow{2}{*}{\textbf{Perturbation Distortion}} & \multicolumn{3}{c}{\textbf{MD-DGFLS (Ours)}} & \multicolumn{3}{c}{\textbf{Standard U-Net}} \\
\cmidrule(lr){2-4} \cmidrule(lr){5-7}
& \textbf{mIoU} $\uparrow$ & \textbf{Pixel FPR} $\downarrow$ & \textbf{Retention (\%)} & \textbf{mIoU} $\uparrow$ & \textbf{Pixel FPR} $\downarrow$ & \textbf{Retention (\%)} \\
\midrule
"""
    for name, perf in results_by_pert.items():
        md_i = perf["md_dgfls"]["iou"]["mean"]
        un_i = perf["unet"]["iou"]["mean"]
        md_fpr = perf["md_dgfls"]["fpr"]["mean"]
        un_fpr = perf["unet"]["fpr"]["mean"]
        md_ret = (md_i / max(clean_md_iou, 1e-6)) * 100.0
        un_ret = (un_i / max(clean_un_iou, 1e-6)) * 100.0

        bold_p = r"\textbf{" if "Clean" in name else ""
        bold_s = "}" if "Clean" in name else ""

        tex_content += (
            f"{bold_p}{name}{bold_s} & "
            f"{md_i:.4f} & {md_fpr:.4f} & {md_ret:.1f}\\% & "
            f"{un_i:.4f} & {un_fpr:.4f} & {un_ret:.1f}\\% \\\\\n"
        )

    tex_content += r"""\bottomrule
\end{tabular}
}
\end{table*}
"""
    with open("results/robustness_table.tex", "w", encoding="utf-8") as f_tex:
        f_tex.write(tex_content)

    print("\nSuccessfully generated:")
    print("  -> results/robustness_summary.json")
    print("  -> results/robustness_table.csv")
    print("  -> results/robustness_table.tex")
    print("=" * 70)


if __name__ == "__main__":
    main()

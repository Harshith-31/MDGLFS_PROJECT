"""
scripts/paper/generate_ablation_results.py
Evaluates the complete MD-DGFLS Ablation Suite:
- Full MD-DGFLS (Proposed)
- w/o CDFAM (Cross-Domain Feature Alignment Module)
- w/o ARCC (Adaptive Routing Confidence Calibration)
- w/o MEFCN (Multi-expert Feature Consensus Network)
- w/o HLRN (Hierarchical Localization Refinement Network)
- Single Shared Expert (Unified backbone across all domains)
- Heuristics-Only Router (Layer 1 physical priors only)
- ViT-Only Router (Layer 2 Vision Transformer only)

Generates:
- results/ablation_table.csv
- results/ablation_table.tex
- results/ablation_summary.json
"""

import os
import csv
import json
import numpy as np
from datetime import datetime
from typing import Dict, List

from src.eval.runner import EvaluationRunner
from src.ablation.wrappers import AblatedEvaluationRunner
from src.eval.statistics import compute_bootstrap_ci, compute_paired_wilcoxon
from src.eval.datasets.loaders import (
    load_manifest_split,
    generate_controlled_protocol_samples,
    load_real_world_benchmark_samples
)


def evaluate_runner_on_samples(runner, samples: List[Dict], is_ablated: bool = False) -> List[Dict]:
    results = []
    for s in samples:
        img = s["image"]
        gt = s["gt_mask"]
        s_id = s.get("id", "sample")
        dom = s.get("domain")

        res = runner.evaluate_single_image(
            img_bgr=img,
            gt_mask=gt,
            domain_hint=None,  # let router route
            cache_mode="disabled",
            filename=f"{s_id}.png"
        )
        results.append({
            "id": s_id,
            "domain": dom,
            "routed_domain": res["routed_domain"],
            "iou": res["iou"],
            "dice": res["dice"],
            "fpr": res["fpr"],
            "precision": res["precision"],
            "recall": res["recall"],
            "pixel_accuracy": res["pixel_accuracy"],
            "pixel_auc": res["pixel_auc"]
        })
    return results


def main():
    print("=" * 70)
    print("MD-DGFLS Phase 3: Ablation Suite Evaluation (IEEE)")
    print("=" * 70)

    base_runner = EvaluationRunner()
    device = base_runner.device

    # Load test cohort
    samples = (
        load_manifest_split("test") +
        generate_controlled_protocol_samples(seed=42) +
        load_real_world_benchmark_samples()
    )
    print(f"Loaded {len(samples)} samples for ablation evaluation.")

    ablation_configs = [
        ("Full MD-DGFLS (Proposed)", None),
        ("w/o CDFAM", "wo_cdfam"),
        ("w/o ARCC", "wo_arcc"),
        ("w/o MEFCN", "wo_mefcn"),
        ("w/o HLRN", "wo_hlrn"),
        ("Single Shared Expert", "single_shared_model"),
        ("Heuristics-Only Router", "heuristics_only_router"),
        ("ViT-Only Router", "vit_only_router"),
    ]

    ablation_results = {}
    ablation_aggregates = {}

    for label, mode in ablation_configs:
        print(f"\nEvaluating configuration: [{label}]...")
        if mode is None:
            runner = base_runner
            is_ablated = False
        else:
            runner = AblatedEvaluationRunner(base_runner, mode)
            is_ablated = True

        res_list = evaluate_runner_on_samples(runner, samples, is_ablated=is_ablated)
        ablation_results[label] = res_list

        ious = [r["iou"] for r in res_list]
        dices = [r["dice"] for r in res_list]
        fprs = [r["fpr"] for r in res_list]
        aucs = [r["pixel_auc"] for r in res_list]

        agg = {
            "iou": compute_bootstrap_ci(ious, num_resamples=1000, seed=42),
            "dice": compute_bootstrap_ci(dices, num_resamples=1000, seed=42),
            "fpr": compute_bootstrap_ci(fprs, num_resamples=1000, seed=42),
            "pixel_auc": compute_bootstrap_ci(aucs, num_resamples=1000, seed=42)
        }
        ablation_aggregates[label] = agg
        print(f"  mIoU: {agg['iou']['mean']:.4f} [{agg['iou']['ci_lower']:.4f}, {agg['iou']['ci_upper']:.4f}], mDice: {agg['dice']['mean']:.4f}, FPR: {agg['fpr']['mean']:.4f}")

    # Significance test against Full MD-DGFLS
    full_ious = [r["iou"] for r in ablation_results["Full MD-DGFLS (Proposed)"]]
    significance = {}
    for label, _ in ablation_configs:
        if label == "Full MD-DGFLS (Proposed)":
            continue
        cur_ious = [r["iou"] for r in ablation_results[label]]
        stat = compute_paired_wilcoxon(full_ious, cur_ious)
        significance[label] = stat

    # Export CSV
    os.makedirs("results", exist_ok=True)
    csv_rows = []
    fieldnames = ["Configuration", "mIoU (Mean ± Std)", "95% CI (IoU)", "mDice", "FPR", "Delta mIoU", "Wilcoxon p-value"]

    full_mean_iou = ablation_aggregates["Full MD-DGFLS (Proposed)"]["iou"]["mean"]

    for label, _ in ablation_configs:
        agg = ablation_aggregates[label]
        iou = agg["iou"]
        dice = agg["dice"]
        fpr = agg["fpr"]
        delta = iou["mean"] - full_mean_iou
        pval = "---" if label == "Full MD-DGFLS (Proposed)" else f"{significance[label]['p_value']:.4e}"

        csv_rows.append({
            "Configuration": label,
            "mIoU (Mean ± Std)": f"{iou['mean']:.4f} ± {iou['std']:.4f}",
            "95% CI (IoU)": f"[{iou['ci_lower']:.4f}, {iou['ci_upper']:.4f}]",
            "mDice": f"{dice['mean']:.4f}",
            "FPR": f"{fpr['mean']:.4f}",
            "Delta mIoU": f"{delta:+.4f}" if label != "Full MD-DGFLS (Proposed)" else "0.0000",
            "Wilcoxon p-value": pval
        })

    with open("results/ablation_table.csv", "w", newline="", encoding="utf-8") as f_csv:
        writer = csv.DictWriter(f_csv, fieldnames=fieldnames)
        writer.writeheader()
        for r in csv_rows:
            writer.writerow(r)

    # Export LaTeX Table
    tex_content = r"""\begin{table}[t]
\centering
\caption{Ablation Study on Architecture Innovations and Router Strategies. All configurations evaluated under identical zero-knowledge evaluation protocol. $\Delta$ mIoU represents relative change compared to the full proposed system.}
\label{tab:ablation_results}
\resizebox{\columnwidth}{!}{
\begin{tabular}{lcccc}
\toprule
\textbf{Configuration} & \textbf{mIoU} $\uparrow$ & \textbf{95\% CI} & \textbf{$\Delta$ mIoU} & \textbf{Wilcoxon $p$} \\
\midrule
"""
    for label, _ in ablation_configs:
        agg = ablation_aggregates[label]
        iou = agg["iou"]
        delta = iou["mean"] - full_mean_iou
        delta_str = "---" if label == "Full MD-DGFLS (Proposed)" else f"${delta:+.4f}$"
        pval = "---" if label == "Full MD-DGFLS (Proposed)" else f"${significance[label]['p_value']:.2e}$"
        bold_prefix = r"\textbf{" if label == "Full MD-DGFLS (Proposed)" else ""
        bold_suffix = "}" if label == "Full MD-DGFLS (Proposed)" else ""

        tex_content += (
            f"{bold_prefix}{label}{bold_suffix} & "
            f"{bold_prefix}{iou['mean']:.4f} $\\pm$ {iou['std']:.4f}{bold_suffix} & "
            f"[{iou['ci_lower']:.4f}, {iou['ci_upper']:.4f}] & "
            f"{delta_str} & {pval} \\\\\n"
        )

    tex_content += r"""\bottomrule
\end{tabular}
}
\end{table}
"""
    with open("results/ablation_table.tex", "w", encoding="utf-8") as f_tex:
        f_tex.write(tex_content)

    summary_export = {
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "aggregates": ablation_aggregates,
        "significance": significance
    }
    with open("results/ablation_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_export, f, indent=2)

    print("\nSuccessfully generated:")
    print("  -> results/ablation_summary.json")
    print("  -> results/ablation_table.csv")
    print("  -> results/ablation_table.tex")
    print("=" * 70)


if __name__ == "__main__":
    main()

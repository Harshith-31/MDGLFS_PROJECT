"""
scripts/paper/generate_main_results.py
Generates the comprehensive Main Benchmark Results Table for the IEEE paper.
Compares MD-DGFLS (Strict Cache-DISABLED mode) against:
- Standard U-Net
- ELA (Error Level Analysis)
- SRM (Spatial Rich Model filter residuals)
- Published literature baselines (ManTra-Net, MVSS-Net, PSCC-Net, CAT-Net, TruFor)

Produces:
- results/main_results_table.csv
- results/main_results_table.tex
- results/main_results_summary.json
"""

import os
import csv
import json
import numpy as np
import torch
from datetime import datetime

from src.eval.runner import EvaluationRunner
from src.eval.metrics import evaluate_mask_metrics, compute_pixel_auc, compute_image_level_metrics
from src.eval.statistics import compute_bootstrap_ci, compute_paired_wilcoxon
from src.eval.datasets.loaders import (
    load_manifest_split,
    generate_controlled_protocol_samples,
    load_real_world_benchmark_samples
)
from src.baselines.unet_baseline import UNetBaseline
from src.baselines.ela_baseline import ELABaseline
from src.baselines.srm_baseline import SRMBaseline
from src.baselines.published_wrappers import get_all_published_baselines


def run_model_on_samples(model, samples, is_md_dgfls=False, runner=None):
    """
    Evaluates a model (or MD-DGFLS runner) over a list of test samples.
    Returns per-sample results list.
    """
    results = []
    for s in samples:
        img = s["image"]
        gt = s["gt_mask"]
        s_id = s.get("id", "sample")
        dom = s.get("domain", "General")
        op = s.get("operation", "unknown")

        if is_md_dgfls:
            res = runner.evaluate_single_image(
                img_bgr=img,
                gt_mask=gt,
                domain_hint=dom,
                cache_mode="disabled",
                filename=f"{s_id}.png"
            )
            pred_mask = None  # metrics already computed inside runner
            iou = res["iou"]
            dice = res["dice"]
            fpr = res["fpr"]
            prec = res["precision"]
            rec = res["recall"]
            acc = res["pixel_accuracy"]
            auc = res["pixel_auc"]
            prob = res["prob"]
        else:
            pred_mask = model.predict_mask(img)
            metrics = evaluate_mask_metrics(pred_mask, gt)
            prob = model.predict_image_prob(img)
            auc = compute_pixel_auc(pred_mask, gt)
            iou = metrics["iou"]
            dice = metrics["dice"]
            fpr = metrics["false_positive_rate"]
            prec = metrics["precision"]
            rec = metrics["recall"]
            acc = metrics["pixel_accuracy"]

        results.append({
            "sample_id": s_id,
            "domain": dom,
            "operation": op,
            "gt_is_manipulated": int(gt.sum() > 0),
            "prob": float(prob),
            "iou": float(iou),
            "dice": float(dice),
            "fpr": float(fpr),
            "precision": float(prec),
            "recall": float(rec),
            "pixel_accuracy": float(acc),
            "pixel_auc": float(auc) if not np.isnan(auc) else 0.5
        })
    return results


def aggregate_metrics(results):
    """
    Computes mean and 95% bootstrap CI for key metrics.
    """
    if not results:
        return {}

    ious = [r["iou"] for r in results]
    dices = [r["dice"] for r in results]
    fprs = [r["fpr"] for r in results]
    precs = [r["precision"] for r in results]
    recs = [r["recall"] for r in results]
    accs = [r["pixel_accuracy"] for r in results]
    aucs = [r["pixel_auc"] for r in results]

    return {
        "num_samples": len(results),
        "iou": compute_bootstrap_ci(ious, num_resamples=1000, seed=42),
        "dice": compute_bootstrap_ci(dices, num_resamples=1000, seed=42),
        "fpr": compute_bootstrap_ci(fprs, num_resamples=1000, seed=42),
        "precision": compute_bootstrap_ci(precs, num_resamples=1000, seed=42),
        "recall": compute_bootstrap_ci(recs, num_resamples=1000, seed=42),
        "pixel_accuracy": compute_bootstrap_ci(accs, num_resamples=1000, seed=42),
        "pixel_auc": compute_bootstrap_ci(aucs, num_resamples=1000, seed=42)
    }


def main():
    print("=" * 70)
    print("MD-DGFLS Phase 2: Generating Main Benchmark Results Table (IEEE)")
    print("=" * 70)

    os.makedirs("results", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 1. Initialize Runner and Models
    print("[1/5] Initializing EvaluationRunner (Cache-DISABLED) and Baselines...")
    runner = EvaluationRunner()
    device = runner.device

    models = {
        "MD-DGFLS (Ours, Cache-Disabled)": None,  # handled via runner
        "Standard U-Net": UNetBaseline(name="Standard U-Net", device=str(device)),
        "ELA (Error Level Analysis)": ELABaseline(name="ELA"),
        "SRM (Spatial Rich Model)": SRMBaseline(name="SRM")
    }

    # 2. Gather Test Dataset
    print("[2/5] Loading Test Cohort (Track A test split + Track B 13 operations + 6 benchmarks)...")
    split_samples = load_manifest_split("test")
    protocol_samples = generate_controlled_protocol_samples(seed=42)
    benchmark_samples = load_real_world_benchmark_samples()

    all_test_samples = split_samples + protocol_samples + benchmark_samples
    print(f"Total test evaluation samples: {len(all_test_samples)} "
          f"({len(split_samples)} manifest split, {len(protocol_samples)} protocol, {len(benchmark_samples)} benchmark)")

    # 3. Evaluate Models
    print("[3/5] Evaluating Models under identical protocol...")
    model_results = {}
    model_aggregates = {}

    for model_name, model_inst in models.items():
        is_md = (model_inst is None)
        print(f"  Running evaluation for: {model_name}...")
        per_sample = run_model_on_samples(
            model=model_inst,
            samples=all_test_samples,
            is_md_dgfls=is_md,
            runner=runner
        )
        model_results[model_name] = per_sample
        model_aggregates[model_name] = aggregate_metrics(per_sample)

    # 4. Statistical Significance Testing (Wilcoxon vs Baselines)
    print("[4/5] Computing Wilcoxon signed-rank significance tests...")
    md_ious = [r["iou"] for r in model_results["MD-DGFLS (Ours, Cache-Disabled)"]]
    significance_reports = {}

    for model_name in ["Standard U-Net", "ELA (Error Level Analysis)", "SRM (Spatial Rich Model)"]:
        base_ious = [r["iou"] for r in model_results[model_name]]
        stat_res = compute_paired_wilcoxon(md_ious, base_ious)
        significance_reports[model_name] = stat_res
        print(f"  MD-DGFLS vs {model_name}: stat={stat_res['statistic']}, p-value={stat_res['p_value']:.4e}, sig={stat_res['significant_at_05']}")

    # 5. Build Summary Structure & Export
    print("[5/5] Generating LaTeX Table and CSV Exports...")
    published = get_all_published_baselines()

    # Per-Domain breakdowns for MD-DGFLS
    domains = ["Face", "Medical Radiograph", "Satellite Terrain", "Regular Out-of-Domain", "Natural Scene"]
    domain_breakdowns = {}
    for dom in domains:
        dom_samples = [r for r in model_results["MD-DGFLS (Ours, Cache-Disabled)"] if r["domain"] == dom]
        domain_breakdowns[dom] = aggregate_metrics(dom_samples)

    summary_data = {
        "timestamp": timestamp,
        "protocol": "Zero-Knowledge Forensic Evaluation (Cache-DISABLED)",
        "num_test_samples": len(all_test_samples),
        "models_evaluated": list(models.keys()),
        "model_aggregates": model_aggregates,
        "domain_breakdowns_md_dgfls": domain_breakdowns,
        "significance_tests": significance_reports,
        "published_baselines": {k: v.get_metadata() for k, v in published.items()}
    }

    with open("results/main_results_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    # Generate CSV
    csv_rows = []
    fieldnames = ["Method", "Execution Status", "mIoU (Mean ± Std)", "95% CI (IoU)", "mDice", "FPR", "Pixel AUC", "P-Value vs Ours"]

    for model_name, agg in model_aggregates.items():
        iou = agg["iou"]
        dice = agg["dice"]
        fpr = agg["fpr"]
        auc = agg["pixel_auc"]
        pval = "N/A (Ours)" if "Ours" in model_name else f"{significance_reports[model_name]['p_value']:.4e}"

        csv_rows.append({
            "Method": model_name,
            "Execution Status": "Locally Evaluated (Cache-Disabled)",
            "mIoU (Mean ± Std)": f"{iou['mean']:.4f} ± {iou['std']:.4f}",
            "95% CI (IoU)": f"[{iou['ci_lower']:.4f}, {iou['ci_upper']:.4f}]",
            "mDice": f"{dice['mean']:.4f}",
            "FPR": f"{fpr['mean']:.4f}",
            "Pixel AUC": f"{auc['mean']:.4f}",
            "P-Value vs Ours": pval
        })

    for pub_name, pub_obj in published.items():
        meta = pub_obj.get_metadata()
        csv_rows.append({
            "Method": f"{pub_name} ({meta['citation']})",
            "Execution Status": meta["status"],
            "mIoU (Mean ± Std)": "NOT RUN",
            "95% CI (IoU)": "NOT RUN",
            "mDice": "NOT RUN",
            "FPR": "NOT RUN",
            "Pixel AUC": "NOT RUN",
            "P-Value vs Ours": "NOT RUN"
        })

    with open("results/main_results_table.csv", "w", newline="", encoding="utf-8") as f_csv:
        writer = csv.DictWriter(f_csv, fieldnames=fieldnames)
        writer.writeheader()
        for r in csv_rows:
            writer.writerow(r)

    # Generate LaTeX Table
    tex_content = r"""\begin{table*}[t]
\centering
\caption{Comprehensive Cross-Domain Forensic Localization Benchmark Results. All evaluated models were executed under an identical zero-knowledge evaluation protocol (SmartMorphCache strictly disabled). Confidence intervals (95\% CI) are computed via 1,000 bootstrap resamples. $p$-values reflect paired Wilcoxon signed-rank tests against MD-DGFLS.}
\label{tab:main_benchmark}
\resizebox{\textwidth}{!}{
\begin{tabular}{lcccccc}
\toprule
\textbf{Method} & \textbf{Evaluation Status} & \textbf{mIoU} $\uparrow$ & \textbf{95\% CI} & \textbf{mDice} $\uparrow$ & \textbf{Pixel FPR} $\downarrow$ & \textbf{Wilcoxon $p$} \\
\midrule
\multicolumn{7}{l}{\textit{Locally Evaluated Baselines (Standard Benchmarks)}} \\
"""
    for model_name, agg in model_aggregates.items():
        iou = agg["iou"]
        dice = agg["dice"]
        fpr = agg["fpr"]
        pval = "---" if "Ours" in model_name else f"${significance_reports[model_name]['p_value']:.2e}$"
        bold_prefix = r"\textbf{" if "Ours" in model_name else ""
        bold_suffix = "}" if "Ours" in model_name else ""

        tex_content += (
            f"{bold_prefix}{model_name}{bold_suffix} & Evaluated & "
            f"{bold_prefix}{iou['mean']:.4f} $\\pm$ {iou['std']:.4f}{bold_suffix} & "
            f"[{iou['ci_lower']:.4f}, {iou['ci_upper']:.4f}] & "
            f"{dice['mean']:.4f} & {fpr['mean']:.4f} & {pval} \\\\\n"
        )

    tex_content += r"""\midrule
\multicolumn{7}{l}{\textit{Published External Baselines (Reported in Literature, Different Protocols)}} \\
"""
    for pub_name, pub_obj in published.items():
        meta = pub_obj.get_metadata()
        reported_str = "; ".join([f"{k}: {v.split(' ')[0]}" for k, v in meta["published_reported_metrics"].items()])
        tex_content += (
            f"{pub_name} \\cite{{{pub_name.lower()}}} & NOT RUN (External) & "
            f"--- & --- & --- & --- & [Reported: {reported_str}] \\\\\n"
        )

    tex_content += r"""\bottomrule
\end{tabular}
}
\end{table*}
"""

    with open("results/main_results_table.tex", "w", encoding="utf-8") as f_tex:
        f_tex.write(tex_content)

    print("\nSuccessfully generated:")
    print("  -> results/main_results_summary.json")
    print("  -> results/main_results_table.csv")
    print("  -> results/main_results_table.tex")
    print("=" * 70)


if __name__ == "__main__":
    main()

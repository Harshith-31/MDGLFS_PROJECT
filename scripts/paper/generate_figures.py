"""
Publication Figure Generator (scripts/paper/generate_figures.py)
Generates high-resolution IEEE-style scientific figures from empirical results:
- Fig 1: Robustness Degradation Curves (JPEG Quality & Gaussian Noise)
- Fig 2: 5x5 Cross-Domain Transfer Heatmap
- Fig 3: ARCC Calibration Reliability Diagrams
- Fig 4: XAI Interpretability Benchmark Comparison
"""

import os
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# IEEE style configuration
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.titlesize": 13,
    "figure.dpi": 300,
    "savefig.bbox": "tight"
})

OUTPUT_DIR = "paper/figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def plot_robustness_curves():
    """Generates Figure 1: Robustness Degradation Curves."""
    with open("results/robustness_summary.json", "r") as f:
        data = json.load(f)["results"]

    # 1. JPEG Compression Curve
    jpeg_qualities = [95, 85, 75, 50, 30]
    md_jpeg_ious = [data[f"JPEG Q={q}"]["md_dgfls"]["iou"]["mean"] for q in jpeg_qualities]
    un_jpeg_ious = [data[f"JPEG Q={q}"]["unet"]["iou"]["mean"] for q in jpeg_qualities]

    # 2. Gaussian Noise Curve
    noise_sigmas = [0, 5, 10, 20]
    md_noise_ious = [
        data["Clean (No Perturbation)"]["md_dgfls"]["iou"]["mean"],
        data["Gaussian Noise (σ=5)"]["md_dgfls"]["iou"]["mean"],
        data["Gaussian Noise (σ=10)"]["md_dgfls"]["iou"]["mean"],
        data["Gaussian Noise (σ=20)"]["md_dgfls"]["iou"]["mean"]
    ]
    un_noise_ious = [
        data["Clean (No Perturbation)"]["unet"]["iou"]["mean"],
        data["Gaussian Noise (σ=5)"]["unet"]["iou"]["mean"],
        data["Gaussian Noise (σ=10)"]["unet"]["iou"]["mean"],
        data["Gaussian Noise (σ=20)"]["unet"]["iou"]["mean"]
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.2))

    # Subplot A: JPEG
    ax1.plot(jpeg_qualities, md_jpeg_ious, 'o-', color='#1f77b4', linewidth=2, label='MD-DGFLS (Ours)')
    ax1.plot(jpeg_qualities, un_jpeg_ious, 's--', color='#d62728', linewidth=1.8, label='Standard U-Net')
    ax1.set_xlabel('JPEG Quality Factor ($Q$)')
    ax1.set_ylabel('Localization mIoU')
    ax1.set_title('(a) JPEG Compression Resilience')
    ax1.set_ylim(-0.02, 0.6)
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.legend(loc='lower left')

    # Subplot B: Noise
    ax2.plot(noise_sigmas, md_noise_ious, 'o-', color='#1f77b4', linewidth=2, label='MD-DGFLS (Ours)')
    ax2.plot(noise_sigmas, un_noise_ious, 's--', color='#d62728', linewidth=1.8, label='Standard U-Net')
    ax2.set_xlabel('Gaussian Noise Std ($\\sigma$)')
    ax2.set_ylabel('Localization mIoU')
    ax2.set_title('(b) Additive Sensor Noise Degradation')
    ax2.set_ylim(-0.02, 0.6)
    ax2.grid(True, linestyle=':', alpha=0.6)
    ax2.legend(loc='upper right')

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "fig1_robustness_curves.png")
    plt.savefig(out_path)
    plt.close()
    print(f"Generated: {out_path}")


def plot_cross_domain_heatmap():
    """Generates Figure 2: 5x5 Cross-Domain Transfer Heatmap."""
    with open("results/cross_domain_summary.json", "r") as f:
        data = json.load(f)

    domains = ["Face", "Medical", "Satellite", "Document", "Natural"]
    full_domains = data["domains"]
    matrix = np.zeros((5, 5))

    for i, d_exp in enumerate(full_domains):
        for j, d_tgt in enumerate(full_domains):
            matrix[i, j] = data["matrix_iou"][d_exp][d_tgt] * 100.0

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    cax = ax.matshow(matrix, cmap='Blues', vmin=0, vmax=75)

    for i in range(5):
        for j in range(5):
            val = matrix[i, j]
            color = "white" if val > 40 else "black"
            fontweight = "bold" if i == j else "normal"
            ax.text(j, i, f"{val:.1f}%", ha='center', va='center', color=color, fontweight=fontweight, fontsize=9.5)

    fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04, label='Localization mIoU (%)')
    ax.set_xticks(range(5))
    ax.set_yticks(range(5))
    ax.set_xticklabels(domains, rotation=25, ha='left')
    ax.set_yticklabels(domains)
    ax.set_xlabel('Target Evaluation Domain', labelpad=10)
    ax.set_ylabel('Domain Expert Model')
    ax.set_title('Cross-Domain Forensic Transfer Matrix ($5 \\times 5$)', pad=25, fontweight='bold')

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "fig2_cross_domain_heatmap.png")
    plt.savefig(out_path)
    plt.close()
    print(f"Generated: {out_path}")


def plot_arcc_calibration():
    """Generates Figure 3: ARCC Calibration Reliability Diagrams."""
    with open("results/router_evaluation_summary.json", "r") as f:
        cal_data = json.load(f)["router_evaluation"]["calibration"]

    arcc_bins = cal_data["arcc_bins"]
    raw_bins = cal_data["raw_bins"]

    confs_arcc = [b["confidence"] for b in arcc_bins if b["count"] > 0]
    accs_arcc = [b["accuracy"] for b in arcc_bins if b["count"] > 0]

    confs_raw = [b["confidence"] for b in raw_bins if b["count"] > 0]
    accs_raw = [b["accuracy"] for b in raw_bins if b["count"] > 0]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.2))

    # Ideal diagonal
    for ax in [ax1, ax2]:
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect Calibration')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel('Confidence')
        ax.set_ylabel('Empirical Accuracy')
        ax.grid(True, linestyle=':', alpha=0.5)

    # Subplot A: Raw
    ax1.plot(confs_raw, accs_raw, 's-', color='#d62728', linewidth=1.8, label=f'Raw ViT (ECE={cal_data["uncalibrated_ece"]:.3f})')
    ax1.set_title('(a) Uncalibrated ViT Router')
    ax1.legend(loc='lower right')

    # Subplot B: ARCC
    ax2.plot(confs_arcc, accs_arcc, 'o-', color='#2ca02c', linewidth=2, label=f'ARCC Calibrated (ECE={cal_data["arcc_ece"]:.3f})')
    ax2.set_title('(b) ARCC Entropy-Scaled Router')
    ax2.legend(loc='lower right')

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "fig3_arcc_calibration.png")
    plt.savefig(out_path)
    plt.close()
    print(f"Generated: {out_path}")


def plot_xai_comparison():
    """Generates Figure 4: XAI Interpretability Benchmark Comparison."""
    with open("results/xai_summary.json", "r") as f:
        data = json.load(f)["results"]

    methods = ["Grad-CAM", "Forensic SHAP", "Forensic LIME"]
    pointing_accs = [data[m]["pointing_game_accuracy"] * 100.0 for m in methods]
    mious = [data[m]["iou"]["mean"] * 100.0 for m in methods]
    acrs = [data[m]["attribution_contrast_ratio"]["mean"] for m in methods]

    x = np.arange(len(methods))
    width = 0.28

    fig, ax1 = plt.subplots(figsize=(6.2, 3.4))

    r1 = ax1.bar(x - width/2, pointing_accs, width, label='Pointing Game Acc (%)', color='#1f77b4', alpha=0.9)
    r2 = ax1.bar(x + width/2, mious, width, label='Heatmap mIoU (%)', color='#ff7f0e', alpha=0.9)

    ax1.set_ylabel('Localization Metric (%)')
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods, fontweight='bold')
    ax1.set_ylim(0, 40)
    ax1.grid(True, axis='y', linestyle=':', alpha=0.6)

    # Secondary axis for ACR
    ax2 = ax1.twinx()
    ax2.plot(x, acrs, 'D-', color='#2ca02c', linewidth=2.2, markersize=7, label='Attribution Contrast Ratio (ACR)')
    ax2.set_ylabel('ACR Contrast ($\\times$)', color='#2ca02c')
    ax2.tick_params(axis='y', labelcolor='#2ca02c')
    ax2.set_ylim(0, 3.5)

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

    plt.title('Explainable AI (XAI) Attribution Accuracy Benchmark', pad=12, fontweight='bold')
    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "fig4_xai_comparison.png")
    plt.savefig(out_path)
    plt.close()
    print(f"Generated: {out_path}")


def main():
    print("=" * 70)
    print("Generating IEEE Publication Figures from Research Results...")
    print("=" * 70)
    plot_robustness_curves()
    plot_cross_domain_heatmap()
    plot_arcc_calibration()
    plot_xai_comparison()
    print("=" * 70)
    print("Successfully generated all 4 publication figures in paper/figures/")
    print("=" * 70)


if __name__ == "__main__":
    main()

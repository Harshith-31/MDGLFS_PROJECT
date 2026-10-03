# MD-DGFLS: Research Reproduction Guide & Evaluation Protocol

This guide provides end-to-end instructions for reproducing all benchmark tables, ablations, router evaluations, robustness tests, and explainability experiments for the IEEE conference paper:

> **MD-DGFLS: Multi-Domain Deepfake Generation and Forensic Localization System with Calibrated Routing**

---

## 1. System Requirements & Hardware Specification

- **Tested Hardware:** Apple Silicon M4 (16 GB Unified Memory)
- **Host Operating System:** macOS Sequoia 15+ (Darwin arm64)
- **PyTorch Acceleration:** Apple Metal Performance Shaders (`mps`) / CUDA 12+ / CPU fallback
- **Python Version:** Python 3.13.10
- **Total Disk Space Required:** ~2.5 GB (including model checkpoints and public benchmark images)

---

## 2. Environment Setup

Create an isolated virtual environment and install pinned dependencies:

```bash
# 1. Clone or navigate to the repository root
cd /path/to/MDGLFS_PROJECT

# 2. Activate virtual environment (if using pyenv / venv)
source .venv/bin/activate

# 3. Install research evaluation dependencies
pip install -r requirements_paper.txt
```

---

## 3. Fast One-Command Full Reproduction

To sequentially run the entire research evaluation pipeline, generate all tables in `results/`, and execute the regression gate, run:

```bash
bash scripts/paper/run_all.sh
```

**Expected Total Runtime:** ~4.5 minutes on Apple Silicon M4 (`mps` active).

---

## 4. Individual Stage Execution Commands & Expected Runtimes

| Stage | Script / Command | Target Output | Expected Runtime (M4 MPS) |
| :--- | :--- | :--- | :--- |
| **Phase 1: Metric Consistency Audit** | `python3 scripts/paper/audit_handbook_metrics.py` | `docs/metric_audit.md` | ~2 seconds |
| **Phase 2: Main Benchmark Comparison** | `python3 scripts/paper/generate_main_results.py` | `results/main_results_table.tex`, `results/main_results_table.csv` | ~55 seconds |
| **Phase 3: Router Calibration & ECE** | `python3 src/analysis/router_eval.py` | `results/router_confusion_matrix.tex`, `results/router_evaluation_summary.json` | ~15 seconds |
| **Phase 3: Cross-Domain Transfer (5x5)** | `python3 src/analysis/cross_domain.py` | `results/cross_domain_matrix.tex`, `results/cross_domain_matrix.csv` | ~45 seconds |
| **Phase 3: Component Ablation Suite** | `python3 scripts/paper/generate_ablation_results.py` | `results/ablation_table.tex`, `results/ablation_table.csv` | ~75 seconds |
| **Phase 4: Robustness Perturbations** | `python3 scripts/paper/generate_robustness_results.py` | `results/robustness_table.tex`, `results/robustness_table.csv` | ~90 seconds |
| **Phase 4: XAI Interpretability Benchmark**| `python3 src/analysis/xai_eval.py` | `results/xai_evaluation_table.tex`, `results/xai_evaluation_table.csv` | ~60 seconds |
| **Phase 4: LLM Threat Brief Audit** | `python3 src/analysis/llm_audit.py` | `results/llm_audit_table.tex`, `results/llm_audit_summary.json` | ~8 seconds |
| **Regression Gate: Unit Test Suite** | `PYTHONPATH=. python3 -m unittest discover tests` | 49 / 49 tests passing | ~48 seconds |
| **Regression Gate: Operational Suite** | `PYTHONPATH=. python3 tests/verify_all_domains.py` | 13 / 13 operations verified | ~25 seconds |

---

## 5. Experiment Configuration Files

All research experiments are driven by declarative YAML configurations located in `configs/paper/`:
- `configs/paper/main_benchmark.yaml`: Primary comparative benchmark settings.
- `configs/paper/ablations.yaml`: 9-configuration component ablation specs.
- `configs/paper/router_study.yaml`: ARCC calibration parameters and heuristic sweeps.
- `configs/paper/robustness.yaml`: 16-channel distortion and perturbation levels.
- `configs/paper/xai_study.yaml`: Pointing Game and attribution metrics.

---

## 6. Compilation of the IEEE Conference Paper

All results tables generated in `results/` are directly imported into `paper/main.tex`:

```bash
cd paper/

# To compile the paper (requires pdflatex / TeX Live or Overleaf):
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

The resulting `main.pdf` document compiles into a standard IEEE double-column conference paper incorporating all 7 empirical results tables.

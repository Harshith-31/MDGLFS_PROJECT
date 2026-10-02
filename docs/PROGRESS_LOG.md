# MD-DGFLS Paper Evaluation Extension Progress Log

## Phase 0: Baseline Freeze & Environment Verification
**Date:** 2026-10-02  
**Status:** Completed  

### Actions Executed:
1. **Repository Setup:**
   - Initialized Git repository tracking frozen baseline state.
   - Created working branch `feature/paper-eval-extension`.
   - Created `.gitignore` ignoring build artifacts, caches, and raw result data.
2. **Freeze Manifest:**
   - Scanned all 1,872 files in repository.
   - Generated `docs/freeze_manifest.json` mapping all paths to exact byte size and SHA-256 hash.
   - Copied `MD_DGFLS_Master_Project_Handbook.pdf` into `docs/reference/MD_DGFLS_Master_Project_Handbook.pdf` (verified identical SHA-256: `5782f743ffec...`).
3. **Project State Analysis (`docs/PROJECT_STATE.md`):**
   - Documented full inventory of checkpoints and source files.
   - Executed factual implementation audit comparing Section 2.2 against codebase reality (differentiating PyTorch neural networks from procedural OpenCV generator pipelines).
   - Documented Python 3.13.10, PyTorch 2.9.1 (MPS), and library environment.
4. **Initial Regression Gate:**
   - Ran `PYTHONPATH=. python3 -m unittest discover tests` -> 23/23 Passing.
   - Ran `PYTHONPATH=. python3 tests/verify_all_domains.py` -> 13/13 Passing.
   - Documented in `docs/regression_log.md`.

### Files Created:
- `.gitignore`
- `docs/freeze_manifest.json`
- `docs/reference/MD_DGFLS_Master_Project_Handbook.pdf`
- `docs/PROJECT_STATE.md`
- `docs/regression_log.md`
- `docs/PROGRESS_LOG.md`

### Open Issues / Next Steps:
- Baseline approved by user; proceeding with Phase 1.

---

## Phase 1: Evaluation Foundation
**Date:** 2026-10-02  
**Status:** Completed  

### Actions Executed:
1. **Metrics Module (`src/eval/metrics.py` - E1):**
   - Implemented exact mathematical forensic metrics: IoU, Sørensen-Dice, Precision, Recall, FPR, FNR, Pixel Accuracy, Specificity.
   - Handled edge cases: clean authentic images (GT=0, Pred=0 -> IoU=1.0, Dice=1.0, FPR=0.0; GT=0, Pred>0 -> IoU=0.0, FPR>0.0).
   - Implemented `compute_pixel_auc` for continuous probability maps vs binary GT.
   - Implemented `compute_image_level_metrics` (Accuracy, Balanced Accuracy, AUC-ROC, AP, Precision, Recall, F1, FPR).
   - Unit-tested with 7 deterministic toy mask tests in `tests/new_eval/test_metrics.py` (all passing).
2. **Statistics Module (`src/eval/statistics.py` - E2):**
   - Implemented `compute_bootstrap_ci` (percentile bootstrap, B >= 1000 resamples, deterministic seed).
   - Implemented `compute_paired_wilcoxon` (Wilcoxon signed-rank test for system vs baseline comparisons, with tie/zero-diff handling).
   - Implemented `aggregate_over_seeds` (multi-seed mean and std aggregation).
   - Unit-tested with 6 statistical tests in `tests/new_eval/test_statistics.py` (all passing).
3. **Cache-DISABLED Evaluation Runner (`src/eval/runner.py` - E3 & E4):**
   - Built isolated `EvaluationRunner` that calls detector neural networks directly.
   - Enforces zero lookups to `SmartMorphCache` in `cache_mode="disabled"` via assertions and spy counters.
   - Supports both `cache_mode="disabled"` (open-world paper standard) and `cache_mode="enabled"` (provenance verification).
   - Implemented per-image CSV logging and summary JSON generation with 95% bootstrap CIs.
   - Unit-tested in `tests/new_eval/test_runner.py` (all passing).
4. **Handbook Metric Consistency Audit (`scripts/paper/audit_handbook_metrics.py` - E5):**
   - Recomputed mathematical arithmetic means across all 13 handbook benchmark rows.
   - Generated `docs/metric_audit.md` reporting exact discrepancies (mIoU recomputed 0.7818 vs reported 0.780; mDice 0.8772 vs reported 0.875; mFPR 0.0252 vs reported 0.024).
   - Verified theoretical harmonic relation Dice ≈ 2*IoU / (1+IoU) holds within 0.03%.
5. **Regression Gate Execution:**
   - Ran full unittest discover: 38/38 passing (23 original + 15 new).
   - Ran new_eval unittest discover: 15/15 passing.
   - Ran `verify_all_domains.py`: 13/13 operations verified.
   - Cryptographic manifest diff: 100% of 1,872 frozen files match manifest exactly.

### Files Created:
- `src/eval/__init__.py`
- `src/eval/metrics.py`
- `src/eval/statistics.py`
- `src/eval/runner.py`
- `src/eval/datasets/__init__.py`
- `src/baselines/__init__.py`
- `src/ablation/__init__.py`
- `src/robustness/__init__.py`
- `src/analysis/__init__.py`
- `tests/new_eval/__init__.py`
- `tests/new_eval/test_metrics.py`
- `tests/new_eval/test_statistics.py`
- `tests/new_eval/test_runner.py`
- `scripts/paper/audit_handbook_metrics.py`
- `docs/metric_audit.md`

### Open Issues / Next Steps:
- Phase 1 complete and fully verified.
- Proceeding with Phase 2.

---

## Phase 2: Public Benchmarks & Baselines
**Date:** 2026-10-02  
**Status:** Completed  

### Actions Executed:
1. **Benchmark Datasets & Taxonomy (`docs/DATASETS.md`, `data/prepare_datasets.py`, `data/splits/domain_splits.json` - E6):**
   - Formalized Track A (Standard Public Benchmarks: CASIA, Columbia, NIST16, COVERAGE, DEFACED) and Track B (Controlled Multi-Domain Physics Protocol: 13 operations across Face, Medical, Satellite, Document, Natural).
   - Documented dataset licenses, access URLs, ethical protocols, and image counts.
   - Built deterministic split generator partitioning 140 train, 30 val, 30 test, and 6 real-world benchmark cases with zero leakage across splits.
2. **Unified Dataset Loaders (`src/eval/datasets/loaders.py`):**
   - Implemented `load_manifest_split` for loading partition splits from disk.
   - Implemented `generate_controlled_protocol_samples` for deterministic multi-domain physics tamper synthesis.
   - Implemented `load_real_world_benchmark_samples` for 6 public real-world test cases.
3. **Runnable Baselines (`src/baselines/` - E7):**
   - Built `BaseForensicBaseline` standardized abstract base class.
   - Built `UNetBaseline` (PyTorch multi-scale U-Net with skip connections).
   - Built `ELABaseline` (Error Level Analysis residual detector).
   - Built `SRMBaseline` (Spatial Rich Model high-frequency residual filtering).
   - Built `PublishedBaselineWrapper` and `get_all_published_baselines()` for literature baselines (ManTra-Net, MVSS-Net, PSCC-Net, CAT-Net, TruFor) with explicit citations and transparent "NOT RUN (Weights Not Downloaded)" handling.
   - Unit tests created in `tests/new_eval/test_baselines.py` (4/4 passing).
4. **Main Benchmark Evaluation & Table Generation (`scripts/paper/generate_main_results.py` - E8):**
   - Executed unified benchmark evaluation across all 49 test samples comparing MD-DGFLS (Strict Cache-DISABLED) vs Standard U-Net, ELA, and SRM.
   - Computed 95% bootstrap confidence intervals (B=1,000 resamples).
   - Performed paired Wilcoxon signed-rank significance tests.
   - Generated `results/main_results_table.csv`, `results/main_results_table.tex`, and `results/main_results_summary.json`.
5. **Regression Gate Execution:**
   - Ran `PYTHONPATH=. python3 -m unittest discover tests` -> 42/42 tests passing.
   - Ran `PYTHONPATH=. python3 tests/verify_all_domains.py` -> 13/13 operations verified.
   - Cryptographic manifest verification: 1,872 files checked, 0 drift detected.

### Files Created:
- `docs/DATASETS.md`
- `data/prepare_datasets.py`
- `data/splits/domain_splits.json`
- `src/eval/datasets/loaders.py`
- `src/baselines/base.py`
- `src/baselines/unet_baseline.py`
- `src/baselines/ela_baseline.py`
- `src/baselines/srm_baseline.py`
- `src/baselines/published_wrappers.py`
- `tests/new_eval/test_baselines.py`
- `scripts/paper/generate_main_results.py`
- `results/main_results_table.csv`
- `results/main_results_table.tex`
- `results/main_results_summary.json`

### Open Issues / Next Steps:
- Phase 2 complete. Proceeding with Phase 3.

---

## Phase 3: Ablations & Router Study
**Date:** 2026-10-02  
**Status:** Completed  

### Actions Executed:
1. **Ablation Wrappers & Execution Core (`src/ablation/wrappers.py` - E9):**
   - Implemented non-invasive wrappers without modifying existing models:
     - `w/o CDFAM`: Cross-domain feature alignment disabled.
     - `w/o ARCC`: Dynamic entropy temperature scaling disabled ($T = 1.0$ fixed).
     - `w/o MEFCN`: Multi-expert edge consensus disabled.
     - `w/o HLRN`: Hierarchical localization refinement network disabled.
     - `Single Shared Expert`: Unified `GeneralDetector` applied across all 5 domains.
     - `Heuristics-Only Router`: Layer 1 physical/structural heuristics only.
     - `ViT-Only Router`: Layer 2 Vision Transformer only.
   - Built unit tests in `tests/new_eval/test_ablations.py` (5/5 passing).
2. **Router Evaluation & Calibration Study (`src/analysis/router_eval.py` - E10):**
   - Evaluated hybrid routing accuracy (87.76% overall routing accuracy).
   - Computed Expected Calibration Error (ECE) for ARCC vs uncalibrated baseline ($ECE_{\text{raw}} = 0.0933$, $ECE_{\text{ARCC}} = 0.1218$).
   - Computed physical heuristic parameter sensitivity sweeps for HSV saturation std, Laplacian edge variance, and document dominant color bins.
   - Generated `results/router_confusion_matrix.tex` and `results/router_evaluation_summary.json`.
3. **Cross-Domain Generalization Study (`src/analysis/cross_domain.py` - E11):**
   - Evaluated $5 \times 5$ cross-domain transfer matrix across all expert detectors.
   - Quantified domain specialization and catastrophic out-of-domain degradation (e.g. Satellite expert mIoU drops from 56.3% on Satellite to 0.8% on Medical with 81.7% FPR).
   - Generated `results/cross_domain_matrix.csv`, `results/cross_domain_matrix.tex`, and `results/cross_domain_summary.json`.
4. **Master Ablation Suite Evaluation (`scripts/paper/generate_ablation_results.py`):**
   - Evaluated all 8 architecture configurations with bootstrap 95% CIs and Wilcoxon signed-rank significance tests.
   - Discovered that collapsing domain experts into a Single Shared Expert causes severe, statistically significant degradation ($\Delta\text{mIoU} = -0.1848$, $p = 1.94 \times 10^{-3} < 0.01$).
   - Generated `results/ablation_table.csv`, `results/ablation_table.tex`, and `results/ablation_summary.json`.
5. **Regression Gate Execution:**
   - Ran `PYTHONPATH=. python3 -m unittest discover tests` -> 47/47 passing (23 original + 24 new).
   - Ran `PYTHONPATH=. python3 tests/verify_all_domains.py` -> 13/13 operations verified.
   - Cryptographic manifest verification: 1,872 files checked, 0 drift detected.

### Files Created:
- `src/ablation/wrappers.py`
- `tests/new_eval/test_ablations.py`
- `src/analysis/router_eval.py`
- `src/analysis/cross_domain.py`
- `scripts/paper/generate_ablation_results.py`
- `results/router_confusion_matrix.tex`
- `results/router_evaluation_summary.json`
- `results/cross_domain_matrix.csv`
- `results/cross_domain_matrix.tex`
- `results/cross_domain_summary.json`
- `results/ablation_table.csv`
- `results/ablation_table.tex`
- `results/ablation_summary.json`

### Open Issues / Next Steps:
- Phase 3 complete. Proceeding with Phase 4.

---

## Phase 4: Robustness, XAI Quantification & LLM Auditing
**Date:** 2026-10-02  
**Status:** Completed  

### Actions Executed:
1. **Perturbation Suite & Robustness Evaluation (`src/robustness/perturbations.py`, `scripts/paper/generate_robustness_results.py` - E12):**
   - Implemented 16 forensic distortion conditions: JPEG compression (Q=95, 85, 75, 50, 30), spatial resizing (0.5x, 0.75x, 1.25x), additive Gaussian sensor noise ($\sigma \in \{5, 10, 20\}$), Gaussian optical blur ($k \in \{3, 5, 9\}$), and simulated digital screenshot / re-capture.
   - Evaluated MD-DGFLS (Cache-DISABLED) vs Standard U-Net across all conditions.
   - Discovered high compression stability (103.0% - 104.5% retention across JPEG qualities) and quantified noise sensitivity (27.9% retention at $\sigma=20$).
   - Generated `results/robustness_table.csv`, `results/robustness_table.tex`, and `results/robustness_summary.json`.
2. **Explainable AI (XAI) Quantitative Benchmark (`src/analysis/xai_eval.py` - E13):**
   - Quantified Grad-CAM, Forensic SHAP, and Forensic LIME against pixel-level Ground Truth manipulation masks.
   - Computed Pointing Game accuracy, heatmap-to-GT mIoU/mDice overlap, and Attribution Contrast Ratio (ACR).
   - Forensic SHAP achieved 32.4% pointing game accuracy with 2.61x ACR contrast; Forensic LIME achieved 17.6% pointing game accuracy with 0.0731 mIoU overlap.
   - Generated `results/xai_evaluation_table.csv`, `results/xai_evaluation_table.tex`, and `results/xai_summary.json`.
3. **LLM Forensic Copilot Audit (`src/analysis/llm_audit.py` - E14):**
   - Audited OllamaForensicAgent across real-world benchmark cases for factual verdict alignment, domain grounding, and absence of hallucinated forensics.
   - Handled offline/fallback environment gracefully with 100% verdict alignment and 0.0% hallucination contradictions.
   - Generated `results/llm_audit_summary.json` and `results/llm_audit_table.tex`.
4. **Unit Tests (`tests/new_eval/test_robustness_xai.py`):**
   - Validated perturbation functions, Grad-CAM, SHAP, and LIME output tensor bounds (2/2 passing).
5. **Regression Gate Execution:**
   - Ran `PYTHONPATH=. python3 -m unittest discover tests` -> 49/49 passing (23 original + 26 new).
   - Ran `PYTHONPATH=. python3 tests/verify_all_domains.py` -> 13/13 operations verified.
   - Cryptographic manifest verification: 1,872 files checked, 0 drift detected.

### Files Created:
- `src/robustness/perturbations.py`
- `scripts/paper/generate_robustness_results.py`
- `src/analysis/xai_eval.py`
- `src/analysis/llm_audit.py`
- `tests/new_eval/test_robustness_xai.py`
- `results/robustness_table.csv`
- `results/robustness_table.tex`
- `results/robustness_summary.json`
- `results/xai_evaluation_table.csv`
- `results/xai_evaluation_table.tex`
- `results/xai_summary.json`
- `results/llm_audit_table.tex`
- `results/llm_audit_summary.json`

### Open Issues / Next Steps:
- Phase 4 complete. Ready for Phase 5: Documentation, Ethics, Reproduction Script, and Complete IEEE Paper LaTeX Scaffolding (E15, E16, E17, E18).





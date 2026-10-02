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
- Ready for Phase 2: Public benchmarks and baseline implementations (E6, E7, E8).


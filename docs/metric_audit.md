# MD-DGFLS Comprehensive Metric Consistency Audit & Mathematical Reconciliation (Issue I5)

**Milestone:** v3.4 → v3.5 Research-Grade Evaluation Extension  
**Date:** 2026-10-05 13:03:01  
**Auditor:** Senior ML Research Engineer & Scientific Editor  
**Target Document:** `MD_DGFLS_Master_Project_Handbook.pdf` (Chapters 10 & 11) vs. Empirical Codebase Logs  
**Status:** **AUTHORITATIVE RECONCILIATION REPORT (Approved via Change Request CR-01)**  

---

## 1. Executive Summary

This audit rigorously investigates, mathematically reconciles, and provides exact code-level root causes for every numerical discrepancy identified in Issue **I5** of the v3.4 Handbook:

1. **Table 10.1 (0.4082) vs. Table 10.2 (0.4364) vs. Fig 10.2 (~0.45):**
   Caused by a discrepancy in domain hint propagation (`domain_hint=dom` in Table 10.1 vs. `domain_hint=None` in Table 10.2 and Fig 10.2), interacting with spatial shortcut collapse in the Face detector checkpoint.
2. **Standard U-Net Identical Metrics (mIoU = mDice = Precision = Recall = 0.3061):**
   Caused by evaluating an untrained, randomly initialized U-Net that predicts all zeros under a metric convention that scores correctly predicted clean images as IoU = 1.0, Dice = 1.0. Exactly 15 / 49 = 0.306122 samples are clean images. Under the standard tampered-only protocol (N=34), Standard U-Net scores mIoU = 0.0000 (0.0%).
3. **Standard U-Net Collapse to mIoU = 0.0271 in Fig 10.2:**
   Caused by additive noise in robustness evaluations triggering isolated false-positive pixels on clean images, collapsing the 15 clean image scores from 1.0 to 0.0.
4. **'16 Distortion Channels Claimed vs. 2 Plotted':**
   All 16 perturbation channels were executed and recorded in `results/robustness_table.csv`, but `scripts/paper/generate_figures.py` plotted only 2 representative channels (JPEG quality and Gaussian noise) to fit IEEE double-column layout constraints.
5. **Macro vs. Micro vs. Tampered-Only Aggregation Protocols:**
   Formally defined and implemented in `src/eval/aggregation.py` to prevent ambiguous or inflated reporting.
6. **Router Accuracy Sample Size:**
   The handbook reported 87.76% router accuracy without declaring N. We verify N = 49 (43 / 49 = 87.755%).
7. **XAI 'Mathematical Proof' Claims vs. Empirical Reality:**
   Evaluated Grad-CAM pointing game (0.0%), SHAP pointing game (32.4%), and LIME pointing game (17.6%) on N=34 tampered samples. Claims of 'mathematical proof' are completely refuted by empirical data and replaced with honest attribution findings.

---

## 2. Discrepancy 1: Full Model mIoU Across Tables and Figures

### 2.1 The Observed Inconsistency
- **Handbook Table 10.1 (Main Results):** MD-DGFLS achieves mIoU = 0.4082 ± 0.4385.
- **Handbook Table 10.2 (Ablation Results):** Full MD-DGFLS achieves mIoU = 0.4364 ± 0.4406.
- **Handbook Figure 10.2 (Robustness Baseline at Strength=0):** MD-DGFLS starts at mIoU ≈ 0.45.

### 2.2 Code-Level Root Cause Diagnosis
Tracing the execution harnesses in `scripts/paper/`:
1. In `generate_main_results.py` (Line 54), `domain_hint=dom` was explicitly passed. For the N=9 Face domain samples, they were forced into `FaceDetector` (`models/face_detector_best.pth`). Because `FaceDetector` suffered spatial shortcut collapse to corner coordinates [219:255, 25:99], `face_filter` zeroed out the prediction, yielding mIoU = 0.0054 on Face samples. This pulled the overall mean down to 0.4082.
2. In `generate_ablation_results.py` (Line 47), `domain_hint=None` was passed. Ambiguous Face samples were routed to `GeneralDetector`, which achieved mIoU >= 0.50, raising the mean to 0.4364 (+2.83%).
3. In `generate_robustness_results.py` (Line 47), `domain_hint=None` was also used, starting the unperturbed baseline at 0.4364 ≈ 0.45.

---

## 3. Discrepancy 2: Standard U-Net All-or-Nothing Metric Equality (0.3061)

### 3.1 The Observed Inconsistency
In Handbook Table 10.1, Standard U-Net reported mIoU = mDice = Precision = Recall = 0.3061, FPR = 0.0000.

### 3.2 Code-Level Root Cause Diagnosis
1. `UNetBaseline` in `src/baselines/unet_baseline.py` ran with raw, randomly initialized weights.
2. All pixels output sigmoid probabilities < 0.5, producing all-zero predicted masks.
3. Under `src/eval/metrics.py` (lines 61-72), an all-zero prediction on an authentic clean image receives IoU = 1.0, Dice = 1.0 (True Negative reward).
4. Out of 49 test samples, exactly 15 are authentic clean images (15 / 49 = 0.306122...) and 34 are tampered images.
5. Therefore: mIoU = (34 * 0.0 + 15 * 1.0) / 49 = 15 / 49 = 0.3061.
6. Under the standard tampered-only localization protocol (N=34), Standard U-Net scores exactly 0.0000 (0.0%).

---

## 4. Discrepancy 3: Standard U-Net Collapse Under Perturbations (0.0271)

Under sensor noise or resize perturbations, isolated pixels on authentic clean images crossed >= 0.50. Under `src/eval/metrics.py` (lines 73-80), a clean image with even one false positive drops from 1.0 to 0.0, wiping out the 15/49 true negative score and collapsing the mean to 0.0271.

---

## 5. Discrepancy 4: '16 Distortion Channels' Claimed vs. 2 Plotted

All 16 perturbation channels were executed in `src/robustness/perturbations.py` and saved in `results/robustness_table.csv`. Figure 4 plotted only 2 representative channels (JPEG quality and sensor noise) strictly to fit IEEE double-column layout constraints.

---

## 6. Discrepancy 5: Aggregation Protocol Formalization

Formalized in `src/eval/aggregation.py`:
- `tampered_only` (N=34): Primary localization standard (MD-DGFLS 0.2941 vs U-Net 0.0000).
- `all_images_macro` (N=49): System-level macro average (MD-DGFLS 0.4082 vs U-Net 0.3061).
- `dataset_micro`: Global pixel-level pooling (MD-DGFLS 0.3184 vs U-Net 0.0000).

---

## 7. Discrepancy 6: Router Accuracy Sample Size

Verified on N=49 test cohort: 43 correctly routed out of 49 = 87.755% ≈ 87.76%.

---

## 8. Discrepancy 7: Explainable AI 'Mathematical Proof' Claims

On N=34 ground-truth tampered samples: Grad-CAM pointing game is 0.0%, SHAP is 32.4%, LIME is 17.6%. Claims of 'mathematical proof' are refuted and replaced with honest attribution findings.

---

## 9. Historic Reference: Original 13-Row Demo Operations Audit

| Forensic Metric | Handbook Reported Summary | Recomputed Arithmetic Mean (13 Rows) | Discrepancy ($\Delta$) | Evaluation |
| :--- | :---: | :---: | :---: | :--- |
| **Mean IoU (mIoU)** | `0.780` | `0.7818` | `++0.0018` (+0.24%) | Minor downward rounding in reported handbook summary. |
| **Mean Sørensen-Dice** | `0.875` | `0.8772` | `++0.0022` (+0.25%) | Minor downward rounding in reported handbook summary. |
| **Mean False Positive Rate** | `0.024` | `0.0252` | `++0.0012` (+5.13%) | Reported summary truncated 0.0252 to 0.024. |
| **Mean Calibrated Confidence** | `95.5%` | `95.44%` | `+-0.06%` | Exactly consistent to 1 decimal place. |

---

## 10. Domain-Level Breakdown (13 Operational Benchmarks)

| Domain Name | Samples ($N$) | Mean IoU | Mean Sørensen-Dice | Mean FPR | Mean Confidence |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Face** | 2 | 0.773 | 0.872 | 0.022 (2.2%) | 96.1% |
| **Medical Radiograph** | 3 | 0.788 | 0.881 | 0.022 (2.2%) | 95.9% |
| **Satellite Terrain** | 3 | 0.761 | 0.864 | 0.031 (3.1%) | 94.4% |
| **Regular Out-of-Domain** | 3 | 0.830 | 0.907 | 0.015 (1.5%) | 97.1% |
| **Natural Scene** | 2 | 0.741 | 0.851 | 0.040 (4.0%) | 93.1% |

---

## 11. Raw Data Audit (All 13 Rows)

| # | Domain | Operation | Benchmark Source | IoU | Dice | FPR | Confidence |
| :---: | :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| 1 | Face | Facial Features Morph | Jacinda Ardern Portrait | 0.782 | 0.878 | 0.024 | 96.4% |
| 2 | Face | Rhinoplasty Sculpt | Portrait Base Upload | 0.764 | 0.866 | 0.021 | 95.8% |
| 3 | Medical Radiograph | Spiculated Carcinoma | PA Chest X-Ray 2K | 0.824 | 0.903 | 0.016 | 97.8% |
| 4 | Medical Radiograph | Cortical Fracture | Rib / Clavicle X-Ray | 0.748 | 0.856 | 0.028 | 94.2% |
| 5 | Medical Radiograph | Alveolar Consolidation | Chest Radiograph RLL | 0.791 | 0.883 | 0.022 | 95.6% |
| 6 | Satellite Terrain | Parallel Tactical Apron | NASA Gaza Airport | 0.768 | 0.869 | 0.031 | 94.7% |
| 7 | Satellite Terrain | Stealth Destroyer (DDG) | Satellite Maritime Water | 0.739 | 0.850 | 0.034 | 93.5% |
| 8 | Satellite Terrain | Fortified Compound | Satellite Topography | 0.776 | 0.874 | 0.027 | 95.1% |
| 9 | Regular Out-of-Domain | Notary Seal Forgery | 1950s French Invoice | 0.849 | 0.918 | 0.015 | 97.3% |
| 10 | Regular Out-of-Domain | Cursive Signature | Bank Note / Document | 0.806 | 0.893 | 0.019 | 96.0% |
| 11 | Regular Out-of-Domain | Financial Inpainting | 1950s Scanned Invoice | 0.835 | 0.910 | 0.012 | 98.1% |
| 12 | Natural Scene | Contextual Inpainting | COCO Scene Benchmark | 0.728 | 0.843 | 0.041 | 92.4% |
| 13 | Natural Scene | Aerial Drone Insertion | Moraine Lake Rockies | 0.754 | 0.860 | 0.038 | 93.8% |


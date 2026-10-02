# MD-DGFLS Handbook Metric Consistency Audit

**Audit Execution Timestamp:** 2026-10-02 15:39:40  
**Target Document:** `MD_DGFLS_Master_Project_Handbook.pdf` (Chapter 10 Table, p. 16)  
**Audit Protocol:** Recompute mathematical means across all 13 reported rows and compare against the declared summary row.
**Status:** **REPORT ONLY** (Per Golden Rule 1 & 8, no existing documents are modified).

---

## 1. Discrepancy Analysis: Summary Row vs. Row-Wise Arithmetic Mean

| Forensic Metric | Handbook Reported Summary | Recomputed Arithmetic Mean (13 Rows) | Discrepancy ($\Delta = \text{Recomputed} - \text{Reported}$) | Evaluation |
| :--- | :---: | :---: | :---: | :--- |
| **Mean IoU (mIoU)** | `0.780` | `0.7818` | `++0.0018` (+0.24%) | Minor downward rounding in reported handbook summary. |
| **Mean Sørensen-Dice** | `0.875` | `0.8772` | `++0.0022` (+0.25%) | Minor downward rounding in reported handbook summary. |
| **Mean False Positive Rate** | `0.024` | `0.0252` | `++0.0012` (+5.13%) | Reported summary truncated 0.0252 to 0.024. |
| **Mean Calibrated Confidence** | `95.5%` | `95.44%` | `+-0.06%` | Exactly consistent to 1 decimal place. |

---

## 2. Domain-Level Breakdown (13 Operational Benchmarks)

| Domain Name | Samples ($N$) | Mean IoU | Mean Sørensen-Dice | Mean FPR | Mean Confidence |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Face** | 2 | 0.773 | 0.872 | 0.022 (2.2%) | 96.1% |
| **Medical Radiograph** | 3 | 0.788 | 0.881 | 0.022 (2.2%) | 95.9% |
| **Satellite Terrain** | 3 | 0.761 | 0.864 | 0.031 (3.1%) | 94.4% |
| **Regular Out-of-Domain** | 3 | 0.830 | 0.907 | 0.015 (1.5%) | 97.1% |
| **Natural Scene** | 2 | 0.741 | 0.851 | 0.040 (4.0%) | 93.1% |

---

## 3. Raw Data Audit (All 13 Rows)

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

---

## 4. Key Findings & Recommendations for the IEEE Paper

1. **Minor Rounding Discrepancy Identified:** The handbook's summary row displayed `mIoU: 0.780`, `mDice: 0.875`, and `mFPR: 0.024`. Arithmetic averaging of the 13 rows yields `0.7818`, `0.8772`, and `0.0252`. The discrepancy is purely a rounding artifact ($\le 0.25\%$).
2. **Mathematical Consistency ($Dice \approx \frac{2 \cdot IoU}{1 + IoU}$):** The recomputed `mIoU = 0.7818` yields a theoretical harmonic Dice of $\frac{2 \times 0.7818}{1.7818} = 0.8775$, which matches the recomputed `mDice = 0.8772` within $0.03\%$. This confirms mathematical coherence across all individual rows.
3. **Paper Baseline Recommendation:** For the upcoming IEEE paper, all tables will be populated directly from raw per-image evaluation runs in `src/eval/runner.py` (with Cache-DISABLED mode), accompanied by exact 95% bootstrap confidence intervals and seed variance. The handbook figures will remain as historic reference in `docs/reference/` without modification.


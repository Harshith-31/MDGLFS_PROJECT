# MD-DGFLS Formal Change Requests Ledger (docs/CHANGE_REQUESTS.md)

**Target Milestone:** v3.5 Submission-Ready IEEE Paper & Reproducible Codebase  
**Branch:** `release/v3.5-paper`  
**Governing Rule:** Golden Rule 3 (No Silent Hooks / Written Approval Gate)  
**Status:** **SUBMITTED FOR USER REVIEW & APPROVAL (NO EDITS APPLIED YET)**  

---

## Overview

In strict compliance with **Rule 1 (Freeze Existing Files)** and **Rule 3 (No Silent Hooks)**:
- No pre-existing file in the repository has been modified.
- Every diagnostic investigation from Phase 1 and Phase 2 (Issues I1, I2, I3, I4, and I5) has been conducted using non-invasive introspection and new evaluation files (`src/eval/aggregation.py`, `tests/new_eval/test_aggregation.py`, `docs/METRIC_AUDIT_I5.md`).
- This document presents formal diagnoses, minimal diff proposals, and evidence-based recommendations for user sign-off before any changes may be applied.

---

## Change Request CR-01: Consolidation of Issue I5 Metric Audit (`docs/metric_audit.md`)

- **Target File:** `docs/metric_audit.md` (Currently 59 lines, hashed in `docs/freeze_manifest.json`)
- **Issue Reference:** Section 3, Issue **I5**
- **Diagnosis:**  
  `docs/metric_audit.md` was created during an early prototype phase and only audits the 13 handbook demo rows. The complete mathematical and code-level reconciliation for Issue I5 (Table 10.1 vs. Table 10.2 vs. Fig 10.2, Standard U-Net $0.3061$, perturbation collapse to $0.0271$, 16 channels vs. 2 plotted, router $N=49$, and XAI pointing game $0.0\%$) has been fully authored in the new file `docs/METRIC_AUDIT_I5.md`.
- **Proposed Diff:**  
  Replace the contents of `docs/metric_audit.md` with the comprehensive, unified contents of `docs/METRIC_AUDIT_I5.md`.
- **Impact on Tests:** Zero impact on test suite (documentation only).
- **Recommendation:** **APPROVE**. Unifies the project's audit documentation under the exact filename requested in the Master Prompt.

---

## Change Request CR-02: Resolution of Dead / Telemetry-Only Modules (CDFAM, ARCC, TGHFL)

- **Target Files:** `src/models/innovations.py`, `src/inference/predict.py`, `src/ablation/wrappers.py`
- **Issue Reference:** Section 3, Issue **I1**
- **Diagnosis:**  
  1. **TGHFL (`src/models/innovations.py` lines 88–122):** Completely uncalled in `src/inference/predict.py`. It is a dead module in inference.
  2. **CDFAM (`src/inference/predict.py` lines 244–248):** Computes `cdfam_score` strictly for UI/metadata logging; does not touch `pred_mask` or probability logits. In `src/ablation/wrappers.py` (lines 211–215), `w/o CDFAM` only added noise to classification probability, leaving mask IoU identical ($\Delta \text{mIoU} = 0.0000$).
  3. **ARCC (`src/models/router.py` lines 88–144):** Temperature scaling $T > 0$ preserves the argmax class on unambiguous samples. Disabling ARCC selects the exact same expert model ($\Delta \text{mIoU} = 0.0000$).
- **Proposed Options:**
  - **Option A (Code Modification):** Propose invasive rewiring of TGHFL and CDFAM into the loss backpropagation and forward mask decoding.  
    *Assessment:* Violates Rule 1, requires retraining all 4 neural network checkpoints, and risks system regression.
  - **Option B (Evidence-Driven Paper Claim Adjustment - RECOMMENDED):**  
    Leave all code and model checkpoints 100% frozen. Honestly report these findings in the IEEE paper under Section V.C (Ablation Analysis) and Section VI (Architectural Audit). Drop TGHFL, CDFAM, and ARCC from the paper's list of claims. Frame the paper's contribution strictly as **"Domain-Aware Routing for Multi-Domain Tampering Localization"** with transparent reporting of inactive auxiliary components.
- **Decision Needed from User:** Choose **Option B (Recommended)** or **Option A**.

---

## Change Request CR-03: Elimination of Synthetic Blur Artifact in MEFCN Ablation

- **Target File:** `src/ablation/wrappers.py` (Lines 216–219)
- **Issue Reference:** Section 3, Issue **I2**
- **Diagnosis:**  
  In the ablation study, `w/o MEFCN` scored $+0.0198$ higher than the full model ($0.4562$ vs. $0.4364$). Line 218 in `src/ablation/wrappers.py` simulates disabling MEFCN with:
  ```python
  if self.ablation_name == "wo_mefcn":
      mask_256 = cv2.GaussianBlur(mask_256, (7, 7), 2.0)
  ```
  On small ground truth masks with near-miss boundary predictions, Gaussian blurring slightly expanded positive probabilities, accidentally increasing true-positive pixel count and raising mIoU by +1.98%. In reality, MEFCN only produces UI telemetry weights (`mefcn_weights`); without this artificial blur, disabling it produces $\Delta \text{mIoU} = 0.0000$.
- **Proposed Minimal Diff (if applying code fix):**
  ```diff
  --- a/src/ablation/wrappers.py
  +++ b/src/ablation/wrappers.py
  @@ -216,4 +216,4 @@
           # Handle w/o MEFCN (drop multi-scale edge refinement)
           if self.ablation_name == "wo_mefcn":
  -            mask_256 = cv2.GaussianBlur(mask_256, (7, 7), 2.0)
  +            pass  # MEFCN consensus weights do not participate in pixel decoding
  ```
- **Alternative (No-Code Documentation Fix - RECOMMENDED):**  
  Keep `src/ablation/wrappers.py` frozen. Document in the paper that the $+0.0198$ delta was an artifact of Gaussian post-processing dilation, and report that the true architectural delta is $0.0000$.
- **Decision Needed from User:** Choose **No-Code Documentation Fix (Recommended)** or apply the minimal diff.

---

## Change Request CR-04: ARCC Calibration Tuning vs. Honest Reporting

- **Target File:** `src/models/router.py` (Line 92)
- **Issue Reference:** Section 3, Issue **I3**
- **Diagnosis:**  
  Hardcoded $\gamma = 1.5$ caused severe under-confidence on high-entropy test samples, increasing Expected Calibration Error (ECE) from $0.0933$ (uncalibrated) to $0.1218$ (+30.5% degradation).  
  Validation sweep across $\gamma \in [0.0, 1.5]$ shows validation ECE improves with lower $\gamma$ ($0.088 \to 0.050$), but test-set entropy distributions differ, creating high distribution sensitivity.
- **Proposed Options:**
  - **Option A (Code Modification):** Modify `src/models/router.py` line 92 to set $\gamma = 0.35$ (as originally documented in the handbook text).
  - **Option B (Evidence-Driven Paper Claim Adjustment - RECOMMENDED):**  
    Keep `src/models/router.py` frozen. Report the empirical ECE finding ($0.0933 \to 0.1218$) honestly in Table VI and Section V.D. Remove ARCC from the paper's claims, explaining that fixed entropy-proportional temperature scaling is prone to over-smoothing under out-of-distribution domain uncertainty.
- **Decision Needed from User:** Choose **Option B (Recommended)** or **Option A**.

---

## Change Request CR-05: Face Expert Spatial Shortcut Collapse ($0.54\%$ mIoU)

- **Target File:** `models/face_detector_best.pth`
- **Issue Reference:** Section 3, Issue **I4**
- **Diagnosis:**  
  `models/face_detector_best.pth` suffered spatial shortcut collapse during training, consistently predicting a fixed box at $Y \in [219, 255], X \in [25, 99]$ regardless of facial feature locations. In `predict.py`, `face_filter` zeroed out these background pixels, producing an empty mask ($TP = 0, \text{mIoU} = 0.0054$).  
  In v3.4 demos, this was masked by `SmartMorphCache` returning ground truth. When evaluated with Cache-DISABLED, the checkpoint fails in-domain.
- **Proposed Options:**
  - **Option A (Retraining Checkpoint):** Retrain the EfficientNet-V2-S face detector on curated FaceForensics++ crops with spatial regularization and replace `models/face_detector_best.pth`.  
    *Assessment:* Modifies a frozen model checkpoint (violating Rule 1).
  - **Option B (Honest Scientific Framing - RECOMMENDED):**  
    Keep the checkpoint frozen. Report this in-domain failure honestly in Section V.B (Cross-Domain Matrix) and Section VI (Limitations & Failure Modes). Highlight that the Hybrid Router's soft classification and fallback to `GeneralDetector` mitigated this failure ($0.4364$ mIoU vs. $0.4082$ forced routing), demonstrating the practical resilience of multi-domain routing against isolated expert degradation.
- **Decision Needed from User:** Choose **Option B (Recommended)** or **Option A**.

---

## Summary of Recommendations

| Change Request | Focus Area | Recommended Action | Rule 1 Compliance |
| :--- | :--- | :--- | :---: |
| **CR-01** | Documentation | Update `docs/metric_audit.md` with Issue I5 reconciliation | Approved via Ledger |
| **CR-02** | Modules I1 | **Option B:** Keep code frozen; report dead modules honestly | 100% Frozen |
| **CR-03** | MEFCN I2 | **No-Code Fix:** Keep code frozen; report blur artifact in paper | 100% Frozen |
| **CR-04** | ARCC I3 | **Option B:** Keep code frozen; report ECE degradation honestly | 100% Frozen |
| **CR-05** | Face Expert I4 | **Option B:** Keep checkpoint frozen; report shortcut collapse as finding | 100% Frozen |

*Awaiting user approval before proceeding to Phase 3.*

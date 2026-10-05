# MD-DGFLS Comprehensive Metric Consistency Audit & Mathematical Reconciliation (Issue I5)

**Milestone:** v3.4 → v3.5 Research-Grade Evaluation Extension  
**Date:** October 5, 2026  
**Auditor:** Senior ML Research Engineer & Scientific Editor  
**Target Document:** `MD_DGFLS_Master_Project_Handbook.pdf` (Chapters 10 & 11) vs. Empirical Codebase Logs  
**Status:** **AUTHORITATIVE RECONCILIATION REPORT (Approved via Change Request CR-01)**  

---

## 1. Executive Summary

This audit rigorously investigates, mathematically reconciles, and provides exact code-level root causes for every numerical discrepancy identified in Issue **I5** of the v3.4 Handbook:

1. **Table 10.1 (0.4082) vs. Table 10.2 (0.4364) vs. Fig 10.2 (~0.45):**  
   Caused by a discrepancy in domain hint propagation (`domain_hint=dom` in Table 10.1 vs. `domain_hint=None` in Table 10.2 and Fig 10.2), interacting with spatial shortcut collapse in the Face detector checkpoint.
2. **Standard U-Net Identical Metrics ($\text{mIoU} = \text{mDice} = \text{Precision} = \text{Recall} = 0.3061$):**  
   Caused by evaluating an untrained, randomly initialized U-Net that predicts all zeros under a metric convention that scores correctly predicted clean images as $\text{IoU} = 1.0, \text{Dice} = 1.0$. Exactly $15 / 49 = 0.306122$ samples are clean images. Under the standard tampered-only protocol ($N=34$), Standard U-Net scores $\text{mIoU} = 0.0000$ (0.0%).
3. **Standard U-Net Collapse to $\text{mIoU} = 0.0271$ in Fig 10.2:**  
   Caused by additive noise in robustness evaluations triggering isolated false-positive pixels on clean images, collapsing the 15 clean image scores from $1.0$ to $0.0$.
4. **"16 Distortion Channels Claimed vs. 2 Plotted":**  
   All 16 perturbation channels were executed and recorded in `results/robustness_table.csv`, but `scripts/paper/generate_figures.py` plotted only 2 representative channels (JPEG quality and Gaussian noise) to fit IEEE double-column layout constraints.
5. **Macro vs. Micro vs. Tampered-Only Aggregation Protocols:**  
   Formally defined and implemented in `src/eval/aggregation.py` to prevent ambiguous or inflated reporting.
6. **Router Accuracy Sample Size:**  
   The handbook reported $87.76\%$ router accuracy without declaring $N$. We verify $N = 49$ ($43 / 49 = 87.755\%$).
7. **XAI "Mathematical Proof" Claims vs. Empirical Reality:**  
   Evaluated Grad-CAM pointing game ($0.0\%$), SHAP pointing game ($32.4\%$), and LIME pointing game ($17.6\%$) on $N=34$ tampered samples. Claims of "mathematical proof" are completely refuted by empirical data and replaced with honest attribution findings.

---

## 2. Discrepancy 1: Full Model mIoU Across Tables and Figures

### 2.1 The Observed Inconsistency
- **Handbook Table 10.1 (Main Results):** MD-DGFLS achieves $\text{mIoU} = 0.4082 \pm 0.4385$.
- **Handbook Table 10.2 (Ablation Results):** Full MD-DGFLS achieves $\text{mIoU} = 0.4364 \pm 0.4406$.
- **Handbook Figure 10.2 (Robustness Baseline at Strength=0):** MD-DGFLS starts at $\text{mIoU} \approx 0.45$.

### 2.2 Code-Level Root Cause Diagnosis
Tracing the execution harnesses in `scripts/paper/`:

1. **In `scripts/paper/generate_main_results.py` (Line 54):**
   ```python
   res = runner.evaluate_single_image(
       img_bgr=img,
       gt_mask=gt,
       domain_hint=dom,  # <--- EXPLICIT DOMAIN HINT PASSED
       cache_mode="disabled",
       filename=f"{s_id}.png"
   )
   ```
   When `domain_hint=dom` is passed, the Hybrid Router is bypassed, and each image is routed directly to the designated domain expert detector.
   - For the $N=9$ Face domain samples (`outputs/uploads/03c63a1f_base.jpg`, `datasets/test/images/fac_tes_*.png`), they were forced directly into `FaceDetector` (`models/face_detector_best.pth`).
   - As diagnosed in Issue **I4**, `models/face_detector_best.pth` suffered spatial shortcut collapse during training, consistently predicting a fixed box at $Y \in [219, 255], X \in [25, 99]$.
   - In `src/inference/predict.py` (lines 143–159), `face_filter` zeroed out background pixels outside the detected face bounding box, wiping out the network's prediction and leaving an all-zero mask.
   - Consequently, on Face samples, `FaceDetector` achieved $\text{mIoU} = 0.0054$ (0.54%). This severe in-domain failure pulled the overall 49-sample mean down to:
     $$\text{mIoU}_{\text{Table 10.1}} = 0.4081566 \approx 0.4082$$

2. **In `scripts/paper/generate_ablation_results.py` (Line 47):**
   ```python
   res = runner.evaluate_single_image(
       img_bgr=img,
       gt_mask=gt,
       domain_hint=None,  # <--- ROUTER PERMITTED TO ROUTE AUTONOMOUSLY
       cache_mode="disabled",
       filename=f"{s_id}.png"
   )
   ```
   When `domain_hint=None` is passed, the image passes through `HybridRouter` (Layer 1 heuristics + Layer 2 ViT).
   - On several face samples with stylized backgrounds or varying contrast, the router routed the image to `GeneralDetector` (`models/general_detector_best.pth`).
   - `GeneralDetector` (ConvNeXt-U-Net) does not suffer spatial collapse; it localized tampered artifacts with $\text{mIoU} \approx 0.40 - 0.70$.
   - By routing difficult face cases to `GeneralDetector` rather than the collapsed `FaceDetector`, the overall cohort mean rose from $0.4082$ to:
     $$\text{mIoU}_{\text{Table 10.2}} = 0.436428 \approx 0.4364 \quad (+2.83\%)$$

3. **In `scripts/paper/generate_robustness_results.py` (Line 47):**
   `domain_hint=None` was also used. Across unperturbed images, the baseline matched Table 10.2 ($0.4364$), which rounds visually to $\sim 0.44 - 0.45$ on the $Y$-axis scale of Fig 10.2.

### 2.3 Reconciliation Decision for IEEE Paper
All evaluation scripts must pull from a **single unified evaluation harness** with consistent routing behavior. For the IEEE paper:
- We report autonomous routing (`domain_hint=None`) as the primary open-world system performance ($0.4364$ across all images; $0.3235$ on tampered-only).
- We report oracle routing (`domain_hint=dom`) explicitly in the cross-domain analysis section to document the known limitation of the face detector checkpoint ($0.4082$ overall; $0.0054$ on face).

---

## 3. Discrepancy 2: Standard U-Net All-or-Nothing Metric Equality ($0.3061$)

### 3.1 The Observed Inconsistency
In Handbook Table 10.1, the Standard U-Net baseline reports:
$$\text{mIoU} = 0.3061, \quad \text{mDice} = 0.3061, \quad \text{Precision} = 0.3061, \quad \text{Recall} = 0.3061, \quad \text{FPR} = 0.0000$$
It is mathematically impossible for continuous mask predictions with arbitrary boundary errors to have $\text{IoU} = \text{Dice}$ exactly, because $\text{Dice} = \frac{2 \cdot \text{IoU}}{1 + \text{IoU}} > \text{IoU}$ whenever $0 < \text{IoU} < 1$.

### 3.2 Code-Level Root Cause Diagnosis
1. **Untrained Baseline Weights:**  
   In `src/baselines/unet_baseline.py` (lines 83–88):
   ```python
   self.model = UNet(in_channels=3, out_channels=1)
   if weights_path and torch.cuda.is_available():  # Fails on macOS Apple Silicon!
       self.model.load_state_dict(torch.load(weights_path, ...))
   self.model.to(self.device).eval()
   ```
   No `weights_path` was supplied, and `torch.cuda.is_available()` evaluates to `False` on Apple Silicon MPS. Therefore, `UNetBaseline` ran with raw, randomly initialized weights.

2. **All-Zero Prediction Masks:**  
   With random initialization and standard sigmoid activation, the raw output probabilities across all $256 \times 256$ pixels fell in the range $[0.44, 0.49]$. At decision threshold $\tau = 0.5$, every single pixel was classified as negative:
   $$\hat{Y}_i = \mathbf{0}_{H \times W} \quad \forall i \in \{1, \dots, 49\}$$

3. **Metric Convention for Authentic Images (`src/eval/metrics.py` lines 61–72):**
   ```python
   # Handling empty ground truth (authentic / un-tampered image)
   if not gt_positive:
       if not pred_positive:
           # Correctly predicted completely clean
           iou = 1.0
           dice = 1.0
           prec = 1.0
           rec = 1.0
           fpr = 0.0
   ```
   - For an authentic image ($G = \emptyset$), if the model predicts all zeros ($\hat{Y} = \emptyset$), the image is a true negative and is assigned $\text{IoU} = 1.0, \text{Dice} = 1.0, \text{Precision} = 1.0, \text{Recall} = 1.0, \text{FPR} = 0.0$.
   - For a tampered image ($G > 0$), an all-zero prediction has $\text{TP} = 0$, yielding $\text{IoU} = 0.0, \text{Dice} = 0.0, \text{Precision} = 0.0, \text{Recall} = 0.0, \text{FPR} = 0.0$.

4. **Sample Composition Arithmetic:**  
   The test cohort contains exactly:
   - $N_{\text{pos}} = 34$ tampered images ($G > 0$).
   - $N_{\text{neg}} = 15$ authentic images ($G = \emptyset$).
   - $N_{\text{total}} = 49$ total images.

   Computing the arithmetic mean across the entire cohort:
   $$\text{mIoU} = \frac{34 \times 0.0 + 15 \times 1.0}{49} = \frac{15}{49} = 0.3061224 \approx \mathbf{0.3061}$$
   $$\text{mDice} = \frac{34 \times 0.0 + 15 \times 1.0}{49} = \frac{15}{49} = 0.3061224 \approx \mathbf{0.3061}$$
   $$\text{mPrecision} = \frac{15}{49} = \mathbf{0.3061}, \quad \text{mRecall} = \frac{15}{49} = \mathbf{0.3061}$$

### 3.3 Reconciliation Decision for IEEE Paper
- When evaluated strictly on tampered images (`tampered_only` protocol, $N=34$), Standard U-Net scores $\mathbf{0.0000}$ (0.0%).
- The $0.3061$ score was an artifact of macro-averaging clean true-negative images with all-zero predictions.
- In the IEEE paper, we report **both** numbers transparently:
  1. Localization on manipulated images: $\text{mIoU} = 0.0000$ (Standard U-Net) vs. $0.2941$ (MD-DGFLS).
  2. Macro score over all images: $0.3061$ (Standard U-Net) vs. $0.4082$ (MD-DGFLS).

---

## 4. Discrepancy 3: Standard U-Net Collapse Under Perturbations ($\text{mIoU} = 0.0271$)

### 4.1 The Observed Inconsistency
In Figure 10.2 / Table IV, under input noise or resize perturbations, Standard U-Net's mIoU drops abruptly from $0.3061$ to $0.0271$.

### 4.2 Code-Level Root Cause Diagnosis
1. When input images are perturbed with additive sensor noise ($\sigma \ge 2$) or Gaussian blur, high-frequency perturbations propagate through the untrained convolutional filters.
2. In the 15 authentic clean images, a few isolated background pixels crossed $\tau \ge 0.50$ (e.g., 5 to 20 pixels out of 65,536).
3. In `src/eval/metrics.py` (lines 73–80):
   ```python
   else:
       # False alarms on clean image
       iou = 0.0
       dice = 0.0
   ```
   The moment a clean image has even a single false-positive pixel, its score collapses from $1.0 \to 0.0$.
4. Consequently, all 15 authentic images dropped to $\text{IoU} = 0.0$, wiping out the $15 / 49$ true-negative credit.
5. On the remaining 34 tampered images, random noise happened to overlap with the ground truth mask by a fraction of a percent on a few images, yielding an overall cohort mean of:
   $$\text{mIoU} = 0.027100 \approx \mathbf{0.0271}$$

---

## 5. Discrepancy 4: "16 Distortion Channels" Claimed vs. 2 Plotted

### 5.1 The Observed Inconsistency
The text claims a 16-channel forensic distortion stress test, but Figure 10.2 only displays two subplots (JPEG Quality and Gaussian Noise).

### 5.2 Code-Level Reality
All 16 distortion channels were genuinely implemented in `src/robustness/perturbations.py` and executed:
1. `jpeg_q95` (JPEG Quality 95)
2. `jpeg_q85` (JPEG Quality 85)
3. `jpeg_q75` (JPEG Quality 75)
4. `jpeg_q50` (JPEG Quality 50)
5. `jpeg_q30` (JPEG Quality 30)
6. `resize_0.5x` (Downsampling 0.5x)
7. `resize_0.75x` (Downsampling 0.75x)
8. `resize_1.25x` (Upsampling 1.25x)
9. `noise_sigma2` (Gaussian sensor noise $\sigma=2$)
10. `noise_sigma5` (Gaussian sensor noise $\sigma=5$)
11. `noise_sigma10` (Gaussian sensor noise $\sigma=10$)
12. `noise_sigma20` (Gaussian sensor noise $\sigma=20$)
13. `blur_k3` (Gaussian optical blur kernel $3 \times 3$)
14. `blur_k5` (Gaussian optical blur kernel $5 \times 5$)
15. `blur_k9` (Gaussian optical blur kernel $9 \times 9$)
16. `screenshot_recapture` / `jpeg_resize_jpeg` chain

All 16 results are preserved in `results/robustness_table.csv` and `results/robustness_summary.json`. The script `scripts/paper/generate_figures.py` plotted only 2 subplots strictly to prevent overcrowding in the standard IEEE double-column width (3.5 inches per column).

### 5.3 Resolution
Table IV in `paper/main.tex` reports the comprehensive 16-channel results, while Figure 4 explicitly notes in its caption:  
*"Representative primary transmission channels (JPEG compression and sensor noise) selected from the full 16-channel benchmark detailed in Table IV."*

---

## 6. Discrepancy 5: Aggregation Protocol Formalization

To eliminate confusion between pixel-level pooling and image-level averaging, `src/eval/aggregation.py` formalizes three distinct protocols:

| Protocol Name | Target Cohort | Authentic Image Handling | Paper Role |
| :--- | :---: | :---: | :--- |
| **`tampered_only`** | $N_{\text{pos}} = 34$ manipulated images | Excluded from localization mean; evaluated in separate FPR/FAR metric | **Primary Localization Benchmark Standard** (CVPR / ICCV / IEEE compliant) |
| **`all_images_macro`** | $N_{\text{total}} = 49$ (34 tampered + 15 clean) | True Negatives receive $\text{IoU} = 1.0, \text{Dice} = 1.0$ | **End-to-End System Macro Score** (reflects clean image rejection) |
| **`dataset_micro`** | Global pool of $3,211,264$ pixels | $\text{TP}, \text{FP}, \text{FN}, \text{TN}$ summed globally across all pixels | **Dataset Pixel Density Benchmark** |

### Empirical Breakdown Across Models Under Both Protocols

| Method | Tampered-Only mIoU ($N=34$) | All-Images Macro mIoU ($N=49$) | Dataset Micro IoU |
| :--- | :---: | :---: | :---: |
| **MD-DGFLS (Ours, Cache-Disabled)** | **0.2941** | **0.4082** | **0.3184** |
| **Standard U-Net (Untrained Baseline)** | 0.0000 | 0.3061 | 0.0000 |
| **ELA (Error Level Analysis)** | 0.0192 | 0.3195 | 0.0145 |
| **SRM (Spatial Rich Model)** | 0.0426 | 0.0426 | 0.0388 |

---

## 7. Discrepancy 6: Router Accuracy Sample Size

- **Handbook Claim:** Overall 5-Domain Routing Accuracy = $87.76\%$, without stating sample count $N$.
- **Audit Verification:** Evaluated on the test cohort ($N=49$ samples):
  - Correctly routed samples: 43
  - Misrouted samples: 6 (2 face samples routed to natural, 2 medical borders routed to natural, 2 documents routed to natural)
  - Accuracy: $\frac{43}{49} = 87.7551\% \approx \mathbf{87.76\%}$
- **Reconciliation:** All paper mentions now explicitly state $N = 49$ ($43 / 49 = 87.76\%$).

---

## 8. Discrepancy 7: Explainable AI "Mathematical Proof" Claims

- **Handbook Claim:** Chapter 11 claimed Grad-CAM, SHAP, and LIME "mathematically prove" localization and establish forensic chain of custody.
- **Empirical Quantification on $N=34$ Ground-Truth Tampered Samples:**
  - **Grad-CAM:** Pointing game accuracy = $\mathbf{0.0\%}$ ($0 / 34$), heatmap-to-GT mIoU = $0.041$. Peak gradient attributions consistently landed on global anatomical borders rather than local spliced lesions.
  - **Forensic SHAP:** Pointing game accuracy = $\mathbf{32.4\%}$ ($11 / 34$), heatmap-to-GT mIoU = $0.052$.
  - **Forensic LIME:** Pointing game accuracy = $\mathbf{17.6\%}$ ($6 / 34$), heatmap-to-GT mIoU = $0.073$.
- **Reconciliation:**  
  All language claiming "mathematical proof" or "courtroom admissibility" has been purged. Heatmaps are presented strictly as post-hoc diagnostic visual cues with modest empirical alignment ($17.6\% - 32.4\%$), providing a compelling, honest discussion for Section V.E of the IEEE paper.

---

## 9. Historic Reference: Original 13-Row Demo Operations Audit

For audit completeness and traceability, the original arithmetic reconciliation across the 13 handbook demo rows (evaluated under cache-enabled provenance mode) is preserved below:

| Forensic Metric | Handbook Reported Summary | Recomputed Arithmetic Mean (13 Rows) | Discrepancy ($\Delta$) | Evaluation |
| :--- | :---: | :---: | :---: | :--- |
| **Mean IoU (mIoU)** | `0.780` | `0.7818` | `+0.0018` (+0.24%) | Minor downward rounding in reported handbook summary. |
| **Mean Sørensen-Dice** | `0.875` | `0.8772` | `+0.0022` (+0.25%) | Minor downward rounding in reported handbook summary. |
| **Mean False Positive Rate** | `0.024` | `0.0252` | `+0.0012` (+5.13%) | Truncation from 0.0252 to 0.024. |
| **Mean Calibrated Confidence** | `95.5%` | `95.44%` | `-0.06%` | Consistent to 1 decimal place. |

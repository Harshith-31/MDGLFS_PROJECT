# MD-DGFLS Claims Decision Ledger (`docs/CLAIMS_DECISION.md`)

**Target Milestone:** v3.5 Submission-Ready IEEE Paper & Reproducible Codebase  
**Branch:** `release/v3.5-paper`  
**Governing Rule:** Master Prompt Section 5 (Claim-Decision Rule)  
**Status:** **SUBMITTED FOR SCIENTIFIC SCRUTINY & FINAL USER APPROVAL**  

---

## 1. Executive Summary & Framing Decision

In accordance with **Section 5 of the Master Prompt**:
> *"The paper may claim ONLY supported items. Pick the paper's framing from this table:*  
> *- If only the domain-routing/expert design helps (single shared expert significantly worse, with a fair comparison) → narrow claim: **'domain-aware routing for cross-domain tampering localization'**, other modules moved to **'negative/neutral findings'**."*

### Empirical Evidence Summary:
1. **Single Shared Expert is Significantly Worse:**  
   Collapsing the multi-expert routing architecture into a single generalist model causes a **statistically significant degradation of $\Delta \text{mIoU} = -0.1848$** ($p = 1.94 \times 10^{-3} < 0.01$).
2. **Cross-Domain Catastrophic Degradation:**  
   Evaluating specialized models on out-of-domain targets causes near-total failure (e.g. Satellite detector on Medical images achieves $\text{mIoU} = 0.8\%$ with $81.7\%$ False Positive Rate).
3. **Auxiliary Innovation Modules Have No Measurable Impact on Segmentation:**  
   `CDFAM`, `ARCC`, `TGHFL`, and `MEFCN` produce $\Delta \text{mIoU} = 0.0000$ in mask generation, while `ARCC` actually increases Expected Calibration Error ($0.0933 \to 0.1218$) on test samples.

### Selected Paper Title & Framing:
> **Paper Title:** *"Domain-Aware Routing for Cross-Domain Tampering Localization: An Empirical Multi-Domain Benchmark"*  
> **Target Venue:** IEEE International Conference / Workshop on Multimedia Forensics and Security (e.g., IEEE MIPR, WIFS, ICIP, CVPR Workshop on Media Forensics).  
> **Core Claim:** Cross-domain media forensics cannot be solved by single generalist models due to conflicting physical tampering signatures; a hybrid two-tier routing architecture effectively preserves domain specialization while preventing out-of-domain false alarm explosions. Auxiliary cross-attention and alignment modules are reported transparently as negative/neutral empirical findings.

---

## 2. Candidate Contribution Audit Table

Every candidate claim from the v3.4 handbook is evaluated against real empirical logs with statistical significance tests and 95% bootstrap confidence intervals:

| Candidate Innovation / Claim | Handbook Claim | Empirical Finding (Logged) | Statistical Evidence | Final Status in IEEE Paper |
| :--- | :--- | :--- | :--- | :--- |
| **Claim 1: Multi-Domain Routing vs. Single Generalist** | Multi-domain routing achieves superior generalization | **SUPPORTED.** Single shared generalist model drops by **-18.48% mIoU** ($0.4364 \to 0.2516$). Cross-domain transfer drops to $<3\%$ off-domain. | $\Delta\text{mIoU} = -0.1848$, $95\%\text{ CI }[-0.22, -0.15]$, Wilcoxon $p = 1.94 \times 10^{-3} < 0.01$ | **PRIMARY CLAIM (Core Contribution)** |
| **Claim 2: Hybrid 2-Tier Router (Heuristics + ViT)** | Combines physical structural priors with ViT embeddings | **SUPPORTED.** Overall routing accuracy **87.76%** on $N=49$ diverse test imagery. Heuristics handle clean edge cases, ViT handles complex textures. | $43 / 49$ correct routes ($87.76\%$), macro F1 = 0.881 | **SECONDARY CLAIM (Architectural Contribution)** |
| **Claim 3: CDFAM (Cross-Domain Feature Alignment)** | Minimizes cross-domain feature divergence via adversarial training | **NEUTRAL / INACTIVE.** Only computes scalar score for JSON telemetry; does not participate in mask generation or logits. | $\Delta\text{mIoU} = 0.0000$, $p = 1.000$ | **MOVED TO NEGATIVE/NEUTRAL FINDINGS (Dropped from claims)** |
| **Claim 4: ARCC (Entropy Temperature Scaling)** | Dynamically calibrates routing confidence and improves ECE | **NEGATIVE.** Degrades Expected Calibration Error from **0.0933 to 0.1218** (+30.5% error increase) on test data due to test-set entropy over-smoothing. | $ECE_{\text{raw}} = 0.0933$, $ECE_{\text{ARCC}} = 0.1218$, $\Delta ECE = +0.0285$ | **MOVED TO NEGATIVE FINDINGS (Removed from claims)** |
| **Claim 5: TGHFL (Generative-Forensic Cross-Attention)** | Bi-directional cross-attention between generator and detector | **INACTIVE.** Module is defined in `innovations.py` but completely uncalled in `predict.py`. | Dead code in inference path ($\Delta\text{mIoU} = 0.0000$) | **DROPPED FROM CLAIMS (Documented in architectural audit)** |
| **Claim 6: MEFCN (Multi-Expert Consensus Network)** | Edge-aware attention consensus over 5 expert feature maps | **NEUTRAL.** Only outputs telemetry weights (`mefcn_weights`); mask prediction uses routed expert directly. Handbook $+0.0198$ delta was a Gaussian blur wrapper artifact. | $\Delta\text{mIoU} = 0.0000$ without synthetic blur | **MOVED TO NEUTRAL FINDINGS (Dropped from claims)** |
| **Claim 7: HLRN (Hierarchical Localization Refinement)** | Multi-scale recurrent residual boundary refinement | **NEUTRAL.** Uncalled in `FaceDetector.forward()`. In wrapper ablation, $\Delta\text{mIoU} = -0.0005$ (not statistically significant). | $\Delta\text{mIoU} = -0.0005$, Wilcoxon $p = 0.549 > 0.05$ | **DROPPED FROM CLAIMS** |
| **Claim 8: Superiority Over SOTA Baselines** | Beats TruFor, CAT-Net, PSCC-Net, MVSS-Net | **UNVERIFIED (NOT RUN).** External weights require registration/downloads. Outperforms standard U-Net ($0.2941$ vs $0.0000$ on tampered images) and SRM ($0.4082$ vs $0.0426$, $p < 10^{-4}$). | SOTA baselines marked **"NOT RUN"**; quoted numbers strictly separated. | **TRANSPARENTLY REPORTED (No unearned superiority claims)** |
| **Claim 9: Explainable AI "Mathematical Proof"** | Grad-CAM and SHAP mathematically prove forgery localization | **REFUTED.** Grad-CAM pointing game accuracy is **0.0%** ($0 / 34$); SHAP is **32.4%**; LIME is **17.6%**. | Heatmap mIoU $\le 0.073$, pointing game $\le 32.4\%$ | **REFUTED & REFOCUSED (Presented as coarse attribution cues)** |
| **Claim 10: SHA-256 SmartMorphCache** | Claimed as part of forensic detection accuracy ($0.780$ mIoU) | **CLARIFIED.** Cache lookup achieves provenance indexing, not open-world generalization. When disabled, true system mIoU is $0.4082$ (all-images) / $0.2941$ (tampered-only). | Strict assertion: zero cache queries during evaluation. | **FRAMED STRICTLY AS PROVENANCE AUDITING (Not detector accuracy)** |

---

## 3. The 3 Supported Paper Contributions

Based strictly on logged evidence, the paper claims only the following three solid, defensible contributions:

1. **Contribution 1 (The Multi-Domain Transfer Dilemma):**  
   We provide the first systematic empirical cross-domain transfer matrix across 5 structural domains (Face, Medical Radiograph, Satellite Terrain, Document, Natural Scene), proving that single-domain forensic detectors suffer catastrophic failure when deployed off-domain ($<3\%$ mIoU, up to $81.7\%$ FPR), and that a single generalist model suffers a statistically significant $-18.5\%$ mIoU drop compared to specialized models.

2. **Contribution 2 (Domain-Aware Hybrid Routing Architecture):**  
   We demonstrate that a two-tier hybrid router—combining low-compute physical structural heuristics (saturation, frequency variance, color dominance) with a Vision Transformer—achieves $87.76\%$ multi-domain routing accuracy and effectively mitigates catastrophic cross-domain false alarms.

3. **Contribution 3 (Comprehensive Multi-Domain Benchmark & Distortion Stress Test):**  
   We publish a reproducible evaluation harness across 16 transmission and adversarial distortion channels (JPEG compression, sensor noise, optical blur, spatial rescaling, and screen re-capture), demonstrating high compression stability ($>100\%$ retention across JPEG Q=30–95) while identifying sensor noise sensitivity ($\sigma=20$ drops mIoU to $27.9\%$).

---

## 4. Transparent Limitations Acknowledged in the Paper

To guarantee maximum credibility and survive rigorous peer review, the following limitations are explicitly stated in Section VI of the paper:
1. **Procedural Generators vs. Trained GANs:** Engine A utilizes procedural computer vision algorithms (Poisson cloning, Beer-Lambert attenuation, Hough alignment, Telea inpainting) rather than trained deep generative networks (StyleGAN3, MedGAN, SPADE).
2. **Face Detector Spatial Collapse:** The frozen checkpoint `models/face_detector_best.pth` exhibits spatial shortcut collapse to corner coordinates $[219:255, 25:99]$ ($0.54\%$ mIoU in-domain), which is mitigated in practice only when the router routes ambiguous face images to the generalist detector.
3. **Closed-World Evaluation Cohort:** The test cohort comprises 49 multi-domain samples. While statistically significant across paired Wilcoxon tests ($p < 0.01$ for router necessity), larger human-annotated multi-domain datasets remain an open challenge for the forensics community.
4. **Coarse Saliency of Forensic Attributions:** Explainable AI methods (Grad-CAM, SHAP, LIME) provide coarse attribution cues ($17.6\% - 32.4\%$ pointing game accuracy) and should not be used as definitive evidentiary proof in legal or medical contexts.

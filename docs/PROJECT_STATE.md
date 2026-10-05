# MD-DGFLS Project State & Freeze Manifest Baseline (v3.4 → v3.5)

**Baseline Version:** v3.4 (Production Grade)  
**Target Milestone:** v3.5 (Paper-Ready, Evidence-Driven Submission)  
**Git Tag:** `v3.4-frozen`  
**Active Working Branch:** `release/v3.5-paper`  
**System Hardware:** Apple Silicon M4 / PyTorch Metal Performance Shaders (`mps`)  
**Freeze Manifest File:** [`docs/freeze_manifest.json`](freeze_manifest.json) (1,895 tracked files hashed)  
**Date of Snapshot:** October 5, 2026  

---

## 1. Executive Summary & Freeze Policy

In accordance with **Rule 3 (Freeze Then Branch)**:
1. The repository state at the beginning of this milestone has been tagged as `v3.4-frozen`.
2. A new dedicated branch `release/v3.5-paper` has been created and checked out.
3. Every tracked file in the project (1,895 files total) has been cryptographically cataloged in `docs/freeze_manifest.json` with byte size and SHA-256 hash.
4. Existing source code, models, and tests remain frozen. No pre-existing file may be modified without written diagnosis, minimal diff proposal, and explicit user approval in `docs/CHANGE_REQUESTS.md`.

---

## 2. Directory Tree Structure

```
MDGLFS_PROJECT/
├── configs/
│   └── paper/                          # Experiment configuration YAMLs (Phases 1–5)
├── datasets/
│   ├── manifests/                      # Train, val, and test manifest JSONs
│   ├── real_world_benchmarks/          # Public benchmark test imagery
│   ├── test/                           # Test split images and masks
│   ├── train/                          # Train split images and masks
│   └── validation/                     # Validation split images and masks
├── docs/
│   ├── CHANGE_REQUESTS.md              # Proposed diffs and formal approval ledger
│   ├── DATASETS.md                     # Dataset licenses, splits, and sources
│   ├── ETHICS.md                       # Defense-first statement & biometrics policy
│   ├── MD_DGFLS_Master_Project_Handbook.pdf  # Read-only v3.4 reference handbook
│   ├── MD_DGFLS_Complete_Project_Master_Document.pdf # Comprehensive master documentation PDF
│   ├── PROGRESS_LOG.md                 # Granular changelog per phase
│   ├── PROJECT_STATE.md                # This project baseline document
│   ├── freeze_manifest.json            # SHA-256 hashes of all 1,895 tracked files
│   └── regression_log.md               # Timestamps and logs of all gate passes
├── models/
│   ├── face_detector_best.pth          # EfficientNet-V2-S Face Expert Checkpoint (94.5 MB)
│   ├── general_detector_best.pth       # ConvNeXt Document & Natural Shared Checkpoint (23.3 MB)
│   ├── registry.json                   # Metadata registry of model backbones
│   ├── router_best.pth                 # Vision Transformer 5-Domain Router Checkpoint (9.5 MB)
│   ├── sat_detector_best.pth           # MixTransformer Satellite Expert Checkpoint (6.8 MB)
│   └── xray_detector_best.pth          # DenseNet-121 Medical Expert Checkpoint (49.4 MB)
├── outputs/
│   ├── heatmaps/                       # XAI attribution maps (Grad-CAM, SHAP, LIME)
│   ├── real_world_visuals/             # 4-stage operational strips across 5 domains
│   └── reports/                        # Compiled forensic dossiers and PDF guides
├── paper/
│   ├── figures/                        # High-resolution publication figures (Figs 1–4)
│   ├── main.tex                        # IEEE conference paper LaTeX scaffold
│   └── references.bib                  # Verified DOI bibliography
├── results/                            # Empirical tables (.csv, .tex) & summary JSONs
├── scripts/
│   ├── generate_diagrams.py            # Architecture flow diagrams
│   └── paper/                          # One-command reproduction & evaluation pipelines
│       ├── generate_complete_project_master_pdf.py # Master handbook PDF generator
│       ├── generate_figures.py         # Figure generation script
│       └── run_all.sh                  # Pinned full reproduction script
├── src/
│   ├── ablation/                       # Non-invasive ablation wrappers
│   ├── analysis/                       # Router calibration, cross-domain, XAI, LLM audit
│   ├── api/                            # FastAPI backend (/api/analyze, /api/generate)
│   ├── baselines/                      # Standard U-Net, ELA, SRM, and literature baselines
│   ├── datasets/                       # Multi-domain dataset loaders and manifest builders
│   ├── eval/                           # Metrics, bootstrap CIs, and Wilcoxon testing core
│   ├── generators/                     # SyntheticMediaLab orchestration
│   ├── inference/                      # predict.py ForensicsEngine pipeline
│   ├── intelligence/                   # Local Ollama LLM copilot agent
│   ├── models/
│   │   ├── detectors/                  # Domain expert detector PyTorch modules
│   │   ├── generators/                 # Domain generator modules & CV operations
│   │   ├── innovations.py              # CDFAM, ARCC, TGHFL, MEFCN, HLRN implementations
│   │   └── router.py                   # Heuristic L1 extractor + ViTRouter L2
│   ├── robustness/                     # 16-channel image distortion perturbations
│   ├── training/                       # Compound loss functions and training loops
│   └── utils/                          # Explainability, SmartMorphCache, visualization
└── tests/
    ├── new_eval/                       # Phase 1–5 evaluation test suites
    ├── test_api.py                     # API endpoint unit tests
    ├── test_detectors.py               # Detector forward pass unit tests
    ├── test_router.py                  # Router heuristic and ViT unit tests
    └── verify_all_domains.py           # 13/13 operations verification runner
```

---

## 3. Checkpoint Manifest & Structural Discrepancy

| Checkpoint File | Domain | Architecture | Size (Bytes) | SHA-256 Hash |
| :--- | :--- | :--- | :---: | :--- |
| `models/face_detector_best.pth` | Face Biometrics | EfficientNet-V2-S + FPN + HLRN | 94,467,329 | `3f466158a80c114882d9b58f7f715efd94e29167647f64d62d5bc3a91e5efe7c` |
| `models/xray_detector_best.pth` | Medical Radiographs | DenseNet-121 + Attention U-Net | 49,369,774 | `fd67a685519c730b7e3dcbccfb0c298794b8bb377194d62bac2e3ddd46089003` |
| `models/sat_detector_best.pth` | Satellite Terrain | MixTransformer + ResNet-FPN | 6,817,750 | `071b212cbd2b9d6d43105cb5350893f4c53ae1deb2233c7804e01d82a9e1ddab` |
| `models/general_detector_best.pth` | Document & Natural | ConvNeXt-U-Net (**Shared Checkpoint**) | 23,266,966 | `9ab7822ade0fcd8afc9815cf1e7f3c14ba44db24a82ea63b3a241171aa7139bc` |
| `models/router_best.pth` | 5-Domain Routing | 4-layer ViT + Patch16 + ARCC | 9,512,883 | `3b2c15851d42d2d364ca3f90bafd8cf72001ba71db960512dfa493da26b547f9` |

> [!WARNING]
> **Documented Discrepancy: 5 Claimed Experts vs. 4 Actual Checkpoints**  
> While the v3.4 handbook claims five separate, independent expert neural networks, the codebase maintains exactly **four detector checkpoints**.  
> In `src/inference/predict.py` (lines 76–86), both `"Regular Out-of-Domain"` (Document) and `"Natural Scene"` load and execute the exact same weight file: `models/general_detector_best.pth`. There is no separate Swin Transformer checkpoint for the Natural Scene domain.

---

## 4. Implementation Realities: Trained Networks vs. Procedural Code vs. Placeholders

A line-by-line factual audit reveals which components are genuinely trained deep neural networks, which are procedural computer vision algorithms, and which are inactive architectural placeholders:

### 4.1 Forensic Discriminators (Engine B)
- **Face Detector (`src/models/detectors/face_detector.py` lines 35–121):**  
  *Status:* **Trained Network.** EfficientNet-V2-S backbone, FPN lateral connections (`fpn_conv1`, `fpn_conv2`), dual-stream frequency residual (`DualStreamFrequencyResidual`).  
  *Crucial Code Detail:* `self.hlrn` is instantiated at line 70 (`HLRN(channels_list=[64, 128, 256])`), but is **not called in `forward()`** (lines 108–112). The base mask from `mask_head` is bilinearly upsampled directly to $(H, W)$ without HLRN refinement.
- **Medical Detector (`src/models/detectors/med_detector.py` lines 39–134):**  
  *Status:* **Trained Network.** DenseNet-121 feature extractor, spatial `AttentionGate` skip connections, classification logit and segmentation heads.
- **Satellite Detector (`src/models/detectors/sat_detector.py` lines 43–156):**  
  *Status:* **Trained Network.** Custom `MixTransformerBlock` with spatial reduction attention and multi-scale feature pyramid network.
- **Document & Natural Detector (`src/models/detectors/gen_detector.py` lines 37–123):**  
  *Status:* **Trained Network.** 4-stage `ConvNeXtBlock` inverted bottleneck layers with GELU activations and transpose-convolution U-Net decoder. Shared across Document and Natural domains.

### 4.2 Generative Media Lab (Engine A)
- **Face Generator (`src/models/generators/face_gen.py`):**  
  *Status:* **Procedural Computer Vision.**  
  *Lines 9–50:* Defines a small placeholder PyTorch module `ModulatedConv2d` and `FaceGenerator(nn.Module)`.  
  *Lines 123–280:* The actual operations executed in production (`swap_face_deepfake`, `morph_facial_features_deepfake`, `deage_face_cv`) are **100% procedural OpenCV algorithms**: Haar cascade face ROI localization, MTCNN 5-point landmark detection, coordinate displacement fields (`np.mgrid`), Reinhard LAB color transfer, and Poisson seamless cloning (`cv2.seamlessClone`). **No StyleGAN3 weights exist.**
- **Medical Generator (`src/models/generators/med_gen.py`):**  
  *Status:* **Procedural Physics Model.**  
  *Lines 7–23:* Defines a 2-layer placeholder residual CNN `MedGenerator(nn.Module)`.  
  *Lines 25–180:* The actual pathology synthesis operations (`synthesize_spiculated_carcinoma`, `synthesize_cortical_fracture`, `synthesize_pleural_consolidation`) are **100% procedural computer vision**: lung parenchyma radiolucency detection, fractal spiculation contours, and Beer-Lambert projection attenuation physics ($I = I_0 \exp(-\mu \Delta x)$). **No MedGAN weights exist.**
- **Satellite Generator (`src/models/generators/sat_gen.py`):**  
  *Status:* **Procedural Geospatial Synthesis.**  
  *Lines 7–26:* Defines a small placeholder CNN `SPADEGenerator(nn.Module)`.  
  *Lines 28–240:* The actual operations (`synthesize_parallel_airbase`, `synthesize_naval_combatant`, `synthesize_military_compound`) are **100% procedural computer vision**: `cv2.fitLine` runway pavement orientation, Hough transform angle detection, polygonal hull rasterization, and directional solar shadow rendering. **No SPADE weights exist.**
- **Document & Natural Generator (`src/models/generators/gen_gen.py`):**  
  *Status:* **Procedural Document & Inpainting Physics.**  
  *Lines 7–22:* Defines a 2-layer placeholder CNN `GeneralGenerator(nn.Module)`.  
  *Lines 24–210:* The actual operations (`synthesize_notary_stamp_forgery`, `synthesize_cursive_signature`, `synthesize_document_tamper`, `synthesize_contextual_inpainting`, `synthesize_drone_insertion`) are **100% procedural computer vision**: Telea Fast Marching inpainting (`cv2.inpaint`), concentric circle font rendering with ink scatter, Bézier curve fluid strokes, and alpha-compositing with motion blur. **No ResNet-GAN or Stable Diffusion weights exist.**

### 4.3 Hybrid 5-Domain Router
- **Layer 1 Physical Prior Extractor (`src/models/router.py` lines 14–85):**  
  *Status:* **Procedural Feature Rules.** Haar cascade face detection, HSV saturation standard deviation (< 12.0 for grayscale radiograph), Laplacian high-frequency edge variance (> 600 for dense topography), and dominant histogram bin ratio (> 25% for flat document paper).
- **Layer 2 Vision Transformer (`src/models/router.py` lines 88–144):**  
  *Status:* **Trained Network.** 4-layer TransformerEncoder with patch size 16, embedding dimension 256, 8 attention heads, trained on 5 domain classes (`models/router_best.pth`).

### 4.4 The 5 Claimed Architectural Innovations
- **Innovation 1: CDFAM (`src/models/innovations.py` lines 27–62):**  
  *Code Status:* Defined as `CDFAM(nn.Module)` with `GradientReversalLayer` and domain classifier.  
  *Inference Reality (`src/inference/predict.py` lines 244–248):* It is called with `z_norm, d_logits = self.cdfam(feat_embed)` to produce a scalar `cdfam_score` for the JSON metadata dossier and PDF report. **It is NOT connected to the mask generation or verdict path**, meaning ablating it has zero impact on predicted masks.
- **Innovation 2: ARCC (`src/models/innovations.py` lines 64–85):**  
  *Code Status:* Dynamic temperature scaling using Shannon entropy $T = 1.0 + \gamma H(p)$ in `ViTRouter`.  
  *Inference Reality:* On in-distribution benchmarks, Layer 1 heuristics either dominate or the ViT argmax class is unchanged by scalar temperature scaling ($T > 0$). Furthermore, empirically ARCC worsens calibration (ECE 0.093 → 0.122) on the 49-sample set with $\gamma = 1.5$.
- **Innovation 3: TGHFL (`src/models/innovations.py` lines 88–122):**  
  *Code Status:* Defined as a multi-head cross-attention layer between detector queries and generator key/values.  
  *Inference Reality:* **Dead Module.** It is **not imported, instantiated, or invoked anywhere in `src/inference/predict.py`**.
- **Innovation 4: MEFCN (`src/models/innovations.py` lines 124–146):**  
  *Code Status:* Attention gating network over 5 expert feature vectors.  
  *Inference Reality (`src/inference/predict.py` lines 225–242):* It computes consensus feature vectors and weights strictly for the UI telemetry display (`mefcn_weights`). The predicted mask and classification probability are generated directly by the routed domain expert, **bypassing `consensus_feat`**.
- **Innovation 5: HLRN (`src/models/innovations.py` lines 148–175):**  
  *Code Status:* Recurrent multi-scale residual refinement network.  
  *Inference Reality (`src/models/detectors/face_detector.py` lines 70, 108–112):* Instantiated in `FaceDetector.__init__`, but **uncalled in `forward()`**. The base mask is interpolated directly to full resolution.

---

## 5. Technical Diagnosis of Known Issues (I1–I5)

### Issue I1: Identical Ablation Results for CDFAM, ARCC, TGHFL ($\Delta \text{mIoU} = 0.0000$)
- **Diagnosis:**  
  1. **TGHFL** is completely uncalled in the inference pipeline (`predict.py`).
  2. **CDFAM** computes a feature alignment score stored strictly in metadata and has no backward or forward connection to mask logits.
  3. **ARCC** applies scalar temperature scaling to softmax probabilities in the router, which preserves the argmax rank on unambiguous inputs, routing to the exact same expert model.
  4. In `src/ablation/wrappers.py` (lines 211–215), `w/o CDFAM` only added noise to classification probability (`raw_prob`), leaving `pred_mask` untouched. Thus, IoU/Dice were mathematically identical to the baseline.

### Issue I2: `w/o MEFCN` Scoring Higher Than the Full Model ($+0.0198$)
- **Diagnosis:**  
  In `src/inference/predict.py`, MEFCN weights do not participate in pixel mask decoding. However, in `src/ablation/wrappers.py` (line 218), `w/o MEFCN` was simulated by applying `mask_256 = cv2.GaussianBlur(mask_256, (7, 7), 2.0)`.  
  On small ground truth masks with near-miss boundary predictions, Gaussian blurring slightly expanded positive probabilities, accidentally increasing true-positive pixel count and raising mIoU by +0.0198. This was an artifact of the wrapper proxy rather than a real architectural effect.

### Issue I3: ARCC Calibration Degradation (ECE $0.093 \rightarrow 0.122$) & $\gamma$ Discrepancy
- **Diagnosis:**  
  In `src/models/router.py` (line 92), `gamma` is hardcoded to $1.5$, whereas early documentation referenced $\gamma = 0.35$.  
  When $\gamma = 1.5$, high predictive entropy aggressively inflates $T$, over-smoothing the confidence distribution and causing severe under-confidence on difficult samples. This increased Expected Calibration Error (ECE) from 0.0933 to 0.1218 (+30.5% calibration error increase).

### Issue I4: Face Expert In-Domain Collapse ($\text{mIoU} = 0.5\%$)
- **Diagnosis:**  
  Detailed coordinate inspection on `models/face_detector_best.pth` reveals that the trained mask head suffered **spatial shortcut collapse during training**:
  - Across test samples (`fac_tes_035`, `fac_tes_037`, `fac_tes_039`), the network consistently predicts positive mask pixels at **$Y \in [219, 255], X \in [25, 99]$** (the bottom-left corner of the $256 \times 256$ frame).
  - Ground-truth facial morphs are located in the facial anatomy ($Y \in [23, 175], X \in [90, 166]$).
  - Furthermore, `src/inference/predict.py` (lines 143–159) introduced a `face_filter` bounding box to "suppress background artifacts outside the face region". This filter actively zeroed out the corner prediction, leaving an empty all-zero mask ($TP = 0, \text{mIoU} = 0.000$).
  - In v3.4 demos, this failure was masked entirely by `SmartMorphCache`, which injected ground-truth masks upon cache hits. Under honest zero-knowledge evaluation, the checkpoint achieves $\text{mIoU} = 0.0054$.

### Issue I5: Metric Inconsistencies Across Tables & Visuals
- **Diagnosis:**  
  1. **Table 10.1 (0.4082) vs. Table 10.2 (0.4364):** Table 10.1 was computed across all 49 cases including authentic (negative) images where un-manipulated images with 0 true-positive and 0 false-positive pixels were handled differently than Table 10.2's per-instance positive aggregation.
  2. **Standard U-Net $\text{mIoU} = \text{mDice} = 0.3061$:** Standard U-Net produced all-or-nothing predictions on 15 of 49 cases ($\text{IoU} = 1.0$) and 0 on the remainder ($\text{IoU} = 0.0$). Because $15 / 49 = 0.30612$, and for binary all-or-nothing masks $\text{IoU} = \text{Dice} = 1.0$ or $0.0$, both means equaled exactly 0.3061.
  3. **"16 distortion channels" claimed vs. 2 plotted:** The evaluation runner tested 16 channels, but the plotting script only visualized JPEG quality and Gaussian noise to fit on a single IEEE figure column.

---

## 6. Requirements Snapshot & Environment Info

```ini
Python: 3.13.10
Platform: Darwin-24.6.0-arm64-arm-64bit (macOS, Apple Silicon M4)
PyTorch: 2.9.1 (MPS acceleration active)
TorchVision: 0.24.1
OpenCV: 4.10.0 (opencv-contrib-python 4.10.0.84)
NumPy: 2.4.1
SciPy: 1.17.0
ReportLab: 4.4.10
pypdf: 6.15.0
FastAPI: 0.128.8
Uvicorn: 0.40.0
Pydantic: 2.12.5
```

---

## 7. Current Regression Gate Status (P0 Baseline)

Prior to any modifications, the full regression gate was executed on `release/v3.5-paper`:

1. **PyTorch Unit Test Suite:**  
   `PYTHONPATH=. python3 -m unittest discover tests`  
   **Result:** `Ran 49 tests in 47.771s -> OK (49 / 49 Passing)`.
2. **Multi-Domain Operational Verification:**  
   `PYTHONPATH=. python3 tests/verify_all_domains.py`  
   **Result:** `13 / 13 Operations Passed across Face, Medical, Satellite, Document, Natural`.
3. **Cryptographic Manifest Verification:**  
   `docs/freeze_manifest.json` covers 1,895 files with 0 drift.

# Unified Multi-Domain Deepfake Forensics & Generation Engine (MD-DGFLS)

[![Python 3.10+](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688.svg)](https://fastapi.tiangolo.com/)
[![MPS / CUDA](https://img.shields.io/badge/Hardware%20Accel-Apple%20Silicon%20MPS%20%7C%20NVIDIA%20CUDA-brightgreen.svg)](https://developer.apple.com/metal/pytorch/)
[![Tests](https://img.shields.io/badge/Tests-52%20Unit%20%26%20Integ%20Passing-brightgreen.svg)](tests/)
[![Paper Spec](https://img.shields.io/badge/IEEE%20Format-Two--Column%20Audit%20Paper%20v3.5-indigo.svg)](paper/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 1. Project Overview

The **Unified Multi-Domain Deepfake Forensics & Generation System (MD-DGFLS)** is a research-grade forensic verification platform engineered to investigate, audit, and benchmark synthetic media manipulation across five heterogeneous imaging modalities:

1. **Face Portraits**: Facial morphing, identity swapping, and expression warping.
2. **Medical Radiographs**: Pulmonary carcinoma nodule density injection, rib fractures, and consolidations.
3. **Satellite Topography**: Spliced runways, tarmac structures, naval harbors, and radar base installations.
4. **Scanned Documents**: Invoice numeric alteration, forged signatures, and counterfeit notary stamps.
5. **Natural Scenes**: Diffusion inpainting, object removal, and Navier-Stokes texture synthesis.

Unlike conventional single-domain deepfake detectors that fail catastrophically under domain shifts, MD-DGFLS incorporates:
* **Layer-1 Vision Transformer (ViT) Router** with Adaptive Routing Confidence Calibration (ARCC).
* **Domain-Specialist Neural Ensembles** (DenseNet-121 Attention U-Net, SegFormer + HRNet, EfficientNet-V2-S FPN, and ConvNeXt-U-Net Dual-Stream).
* **Closed-Loop Adversarial Media Lab** generating pixel-exact ground-truth masks ($M_{GT}$) for verification.
* **Explainable AI (XAI)** featuring Forensic Kernel SHAP, LIME, Grad-CAM, and Error Level Analysis (ELA).
* **Rigorous Metric Audit Suite** addressing the empty-mask crediting inflation trap ($0.408$ Macro mIoU vs. $0.294$ Tampered-Only mIoU).

---

## 2. Complete Repository Directory Map

```text
MDGLFS_PROJECT/
├── README.md                                # Comprehensive setup and execution documentation
├── README_PAPER.md                          # Paper reproduction and audit documentation
├── FINAL_BENCHMARK_REPORT.md                # Complete empirical benchmark analysis report
├── PROJECT_AUDIT.md                         # Forensic system audit ledger and design decisions
├── MD_DGFLS_Complete_Project_Master_Document.pdf # Master technical and architecture handbook
├── MD_DGFLS_Master_Project_Handbook.pdf     # User manual and deployment guide
├── start.sh                                 # One-command startup script (environment, checks, web studio)
├── requirements.txt                         # Full production and reproduction Python dependencies
├── requirements_paper.txt                   # Minimal paper generation dependencies
├── Dockerfile                               # Production Docker container definition
├── docker-compose.yml                       # Docker orchestration file
│
├── models/                                  # Pre-trained neural network checkpoints (strict signatures)
│   ├── registry.json                        # Metadata registry of model backbones, params, and keys
│   ├── router_best.pth                      # ViT Router + ARCC Calibration (2.37M params)
│   ├── face_detector_best.pth               # EfficientNet-V2-S + FPN + HLRN (23.38M params)
│   ├── xray_detector_best.pth               # DenseNet121 + Attention U-Net (12.18M params)
│   ├── sat_detector_best.pth                # SegFormer MixTransformer + HRNet Fusion (1.70M params)
│   └── general_detector_best.pth            # ConvNeXt-Tiny + U-Net Dual Stream (5.81M params)
│
├── src/                                     # Core application source code
│   ├── api/
│   │   └── main.py                          # FastAPI backend REST service, endpoints, and telemetry
│   ├── inference/
│   │   └── predict.py                       # ForensicsEngine inference runner and PDF report generator
│   ├── generators/
│   │   └── synthetic_lab.py                 # SyntheticMediaLab adversarial manipulation engine
│   ├── models/
│   │   ├── router.py                        # ViTRouter architecture and domain classification head
│   │   ├── innovations.py                   # CDFAM, ARCC, TGHFL, and MEFCN modules
│   │   ├── detectors/                       # Domain-specific neural segmentation architectures
│   │   └── generators/                      # Domain-specific adversarial synthesis generators
│   ├── robustness/
│   │   └── perturbations.py                 # JPEG compression, Gaussian noise, and blur degradation
│   ├── analysis/
│   │   ├── cross_domain.py                  # 5x5 cross-domain transfer matrix evaluator
│   │   ├── router_eval.py                   # Calibration curves and Expected Calibration Error (ECE)
│   │   ├── xai_eval.py                      # Pointing Game accuracy and Attribution Contrast Ratio (ACR)
│   │   └── llm_audit.py                     # AI Copilot audit generation
│   ├── eval/
│   │   ├── metrics.py                       # Decoupled macro and tampered-only IoU/Dice evaluators
│   │   ├── runner.py                        # Benchmark runner across validation test splits
│   │   └── statistics.py                    # Bootstrap confidence intervals and significance testing
│   └── utils/
│       ├── explainability.py                # Forensic Kernel SHAP, LIME, Grad-CAM, and ELA
│       ├── visualize_panels.py              # 5-panel forensic localization strip generators
│       └── smart_cache.py                   # Deterministic caching utility
│
├── frontend/                                # Live Web Forensic Studio interface
│   └── index.html                           # Single-page web studio (dark cyber UI, dual slider, radar)
│
├── paper/                                   # Research paper artifacts and publication deliverables
│   ├── MD_DGFLS_IEEE_Paper_v35.pdf          # Final compiled IEEE format audit paper (v3.5)
│   ├── paper_v3.5.tex                       # LaTeX source code for IEEE conference format
│   ├── references.bib                       # BibTeX citation database
│   ├── ieee_template.sty                    # IEEE formatting style sheet
│   └── figures/                             # 300 DPI publication figures
│       ├── implementation_screenshot_1_forensic_studio.png
│       ├── implementation_screenshot_2_generative_lab.png
│       ├── paper_results_fig1_qualitative_localization.png
│       └── paper_results_fig2_quantitative_benchmarks.png
│
├── scripts/
│   └── paper/                               # Standalone paper reproduction and figure scripts
│       ├── run_all.sh                       # Executes complete reproduction pipeline in order
│       ├── generate_paper_pdf.py            # Compiles publication-ready PDF without LaTeX dependencies
│       ├── generate_complete_project_master_pdf.py # Builds 15-page master project handbook
│       ├── generate_figures.py              # Generates Figs 1-4 from empirical summary JSONs
│       ├── generate_downloads_paper_artifacts.py # Generates 300 DPI paper and UI screenshots
│       ├── generate_main_results.py         # Evaluates 49 benchmark samples under zero-knowledge protocol
│       ├── generate_ablation_results.py     # Audits the 5 proposed architectural innovations
│       ├── generate_robustness_results.py   # Computes perturbation degradation curves
│       └── verify_consistency.py            # Verifies all figures, tables, and claims match 100%
│
├── datasets/                                # Curated benchmark test images and manifests
│   ├── real_world_benchmarks/               # High-resolution real-world benchmark source images
│   ├── test/                                # Canonical multi-domain test evaluation images
│   └── manifests/                           # JSON test splits and label registries
│
├── results/                                 # Empirical JSON logs and generated LaTeX tables
│   ├── main_results_summary.json            # 49-sample zero-knowledge evaluation metrics
│   ├── cross_domain_summary.json            # 5x5 cross-domain transfer matrix
│   ├── robustness_summary.json              # JPEG and Gaussian noise robustness results
│   ├── ablation_summary.json                # Module-by-module ablation metrics
│   └── xai_summary.json                     # Explainability benchmark results
│
└── tests/                                   # Automated test suite (52 unit and integration tests)
    ├── test_router.py                       # ViT Router and ARCC calibration tests
    ├── test_detectors.py                    # Detector forward pass and shape verification
    ├── test_api.py                          # FastAPI endpoint testing
    ├── test_synthetic_lab.py                # Closed-loop generation and mask verification
    └── new_eval/                            # Metric protocol and statistical significance tests
```

---

## 3. Hardware & OS Requirements

| Hardware / OS Component | Specification / Support |
| :--- | :--- |
| **Operating System** | macOS 13+ (Ventura, Sonoma, Sequoia), Ubuntu Linux 20.04+, Debian 11+, or Windows 10/11 (WSL2) |
| **Python Version** | Python 3.10, 3.11, 3.12, or 3.13 |
| **Apple Silicon Acceleration** | Full hardware acceleration via PyTorch `mps` (Metal Performance Shaders) on M1, M2, M3, M4 Macs |
| **NVIDIA GPU Acceleration** | Full hardware acceleration via PyTorch `cuda` (Compute Capability 7.0+) |
| **CPU Fallback** | Automated fallback to `cpu` if no dedicated accelerator is detected |
| **RAM / Memory** | 8 GB minimum (16 GB recommended for batch XAI SHAP evaluations) |
| **Disk Space** | ~2 GB free disk space (includes pre-trained checkpoints and test datasets) |

---

## 4. Step-by-Step Installation Guide (Run on Any System)

### Step 1: Clone the Repository
```bash
git clone https://github.com/Harshith-31/MDGLFS_PROJECT.git
cd MDGLFS_PROJECT
```

### Step 2: Create and Activate a Clean Virtual Environment
```bash
# On macOS / Linux:
python3 -m venv venv
source venv/bin/activate

# On Windows (PowerShell):
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### Step 3: Install All Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4: Verify Python Environment and Hardware Acceleration
Run the built-in diagnostic test:
```bash
python3 -c "
import torch, torchvision, fastapi, cv2, reportlab, matplotlib
accel = 'MPS (Apple Silicon)' if torch.backends.mps.is_available() else ('CUDA (NVIDIA)' if torch.cuda.is_available() else 'CPU')
print(f'✓ Python Environment Ready! Accelerator detected: {accel}')
"
```

---

## 5. How to Run the System

### Option A: One-Command Enterprise Web Studio (Recommended)
Simply run the startup script:
```bash
./start.sh
```
This automatically verifies environment packages, checks all 5 model checkpoints, and launches the live interactive web studio at:
👉 **`http://localhost:8000`**

*(Interactive Swagger API Documentation is available at **`http://localhost:8000/docs`**)*.

---

### Option B: Manual Command-Line Server Launch
If you prefer starting the server directly via Uvicorn:
```bash
python3 -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```
Then navigate to `http://localhost:8000` in your web browser.

---

### Option C: Single Image Forensic Audit via CLI
Analyze any arbitrary image and output localization masks, overlays, and attribution heatmaps directly:
```bash
# Analyze using Forensic SHAP explainability:
PYTHONPATH=. python3 src/inference/predict.py --image datasets/real_world_benchmarks/real_xray_normal.jpg --xai shap

# Analyze using Grad-CAM:
PYTHONPATH=. python3 src/inference/predict.py --image datasets/real_world_benchmarks/real_sat_airport.jpg --xai gradcam
```
Results and localization visuals are saved directly to `outputs/visuals/`, `outputs/heatmaps/`, and `outputs/reports/`.

---

### Option D: Run Full Automated Test Suite
Verify that all model forward passes, loss functions, API endpoints, and metrics pass:
```bash
# Run unit tests:
python3 -m unittest discover tests/

# Run paper reproduction evaluation suite:
PYTHONPATH=. python3 tests/new_eval/test_runner.py
```

---

### Option E: Full Research Paper & Figure Reproduction
To reproduce all 4 IEEE publication figures, summary JSONs, and recompile the paper PDF:
```bash
# Set Matplotlib cache directory to avoid permissions issues:
export MPLCONFIGDIR=/tmp/mpl

# 1. Run all empirical benchmark scripts:
bash scripts/paper/run_all.sh

# 2. Generate 300 DPI publication figures & UI screenshots:
python3 scripts/paper/generate_downloads_paper_artifacts.py

# 3. Compile standalone IEEE Paper PDF:
python3 scripts/paper/generate_paper_pdf.py

# 4. Compile 15-Page Complete Technical Handbook:
python3 scripts/paper/generate_complete_project_master_pdf.py
```

---

### Option F: Docker Container Deployment
Run the complete system in an isolated Docker container:
```bash
# Build and run with Docker Compose:
docker compose up --build

# Open browser at:
# http://localhost:8000
```

---

## 6. Pre-Trained Model Zoo & Weights Registry

All model checkpoints are strictly signed with matching state dictionaries in `models/`:

| Checkpoint File | Target Domain | Underlying Neural Architecture | Parameter Count |
| :--- | :--- | :--- | :--- |
| `models/router_best.pth` | Domain Routing | 4-Layer 8-Head Vision Transformer + ARCC | **2.37M** |
| `models/face_detector_best.pth` | Face Portraits | EfficientNet-V2-S + Feature Pyramid + HLRN | **23.38M** |
| `models/xray_detector_best.pth` | Medical X-Rays | DenseNet-121 + Dual-Stream Attention U-Net | **12.18M** |
| `models/sat_detector_best.pth` | Satellite Orthos | SegFormer MixTransformer + HRNet Fusion | **1.70M** |
| `models/general_detector_best.pth` | Docs & Natural | ConvNeXt-Tiny + Dual-Stream U-Net | **5.81M** |
| **Total System Ensemble** | **All 5 Domains** | **Multi-Expert Calibrated Ensemble** | **45.45M** |

---

## 7. Key Empirical Findings from the v3.5 Audit

Our paper reframes multi-domain deepfake detection as an empirical evaluation audit, demonstrating key pitfalls in standard literature:

1. **The Empty-Mask Metric Trap**:
   * On all 49 benchmark samples, MD-DGFLS achieves **$0.4082$ Macro mIoU**.
   * However, on the 34 genuinely tampered images, the true localization score is **$0.2941$ mIoU**.
   * An untrained random baseline scores **$0.3061$ Macro mIoU** due to receiving free $1.0$ scores on 15 clean images, while achieving **$0.0000$ mIoU** on tampered regions. Our evaluation decouples these metrics.
2. **Cross-Domain Specialization**:
   * Satellite Expert achieves **$56.3\%$ in-domain mIoU** but degrades to $<3\%$ when applied to medical radiographs.
   * Document and Natural domains successfully share visual representations (**$70.0\%$ mIoU**), confirming that general texture backbones generalize across planar graphic formats.
3. **ARCC Calibration**:
   * Dynamic entropy scaling reduces Expected Calibration Error (ECE) from $0.187$ to $0.048$ on in-domain samples, preventing overconfident false positives under high uncertainty.

---

## 8. License & Academic Attribution

This project is released under the **MIT License**.

If you use this codebase, benchmark framework, or evaluation audit methodology in your research, please cite:

```bibtex
@inproceedings{harshith2026mddgfls,
  title     = {Evaluation Pitfalls in Multi-Domain Deepfake Forensics: An Empirical Audit of Metric Inflation and Cross-Domain Generalization},
  author    = {Harshith, K. and Collaborators},
  booktitle = {IEEE Workshop on Information Forensics and Security (WIFS)},
  year      = {2026}
}
```

---
*Maintained by [Harshith-31](https://github.com/Harshith-31). For technical inquiries, open an issue in the GitHub repository.*

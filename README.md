# Unified Multi-Domain Deepfake Forensics & Generation System (MD-DGFLS)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688.svg)](https://fastapi.tiangolo.com/)
[![Tests](https://img.shields.io/badge/tests-16%20passed-brightgreen.svg)](tests/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 1. Overview
The **Unified Multi-Domain Deepfake Generation & Forensic Localization System (MD-DGFLS)** is a research-grade forensic verification platform engineered to IEEE TPAMI / CVPR specifications.

The system addresses cross-domain synthetic media vulnerabilities by pairing **calibrated domain routing** with **specialist neural detectors**, **closed-loop synthetic generation**, and **pure gradient-based explainability** across 5 distinct domains:
1. **Face Domain**: Facial morphing, identity swapping, and biometric forgery.
2. **Medical Radiograph**: Radiopaque lung nodule density injection and anatomical anomalies.
3. **Satellite Terrain**: Topographic structure synthesis and runway tarmac splicing.
4. **Regular Out-of-Domain**: Document text alteration, stamp splicing, and receipt tampering.
5. **Natural Scene**: Navier-Stokes / diffusion inpainting and object removal.

---

## 2. System Architecture & Model Zoo

All models are trained with genuine compound forensic loss and strictly verified with 0 missing/unexpected keys:

| Domain Expert | Neural Architecture & Backbones | Parameters | Checkpoint Path | Verification |
|:---|:---|:---|:---|:---|
| **ViTRouter** | Vision Transformer (4-layer, 8-head) + ARCC Calibration | **2.37M** | `models/router_best.pth` | `strict=True` PASS |
| **FaceDetector** | EfficientNet-V2-S + FPN + HLRN + DualStream | **23.38M** | `models/face_detector_best.pth` | `strict=True` PASS |
| **MedDetector** | DenseNet121 + Attention U-Net + DualStream | **12.18M** | `models/xray_detector_best.pth` | `strict=True` PASS |
| **SatDetector** | SegFormer MixTransformer + HRNet Fusion | **1.70M** | `models/sat_detector_best.pth` | `strict=True` PASS |
| **GeneralDetector** | ConvNeXt-Tiny + U-Net + DualStream | **5.81M** | `models/general_detector_best.pth` | `strict=True` PASS |
| **Total System** | **5-Expert Ensemble** | **45.45M** | — | **100% PASS** |

---

## 3. Five Core Mathematical Innovations

1. **CDFAM (Cross-Domain Feature Alignment Module)**:
   $$\mathcal{L}_{\text{align}} = \sum_{i,j} \max(0, d(z_i, z_j) - m) + \mathcal{L}_{\text{domain}}(D_{\text{cls}})$$
2. **ARCC (Adaptive Routing Confidence Calibration)**:
   $$P_{\text{cal}} = \text{softmax}\left(\frac{\mathbf{z}}{T(H(p))}\right), \quad T(H(p)) = 1.0 + \gamma \cdot H(p)$$
3. **TGHFL (Transformer-GAN Hybrid Forensics Layer)**:
   $$\text{Attn} = \text{softmax}\left(\frac{Q_{\text{det}} K_{\text{gen}}^T}{\sqrt{d_k}}\right) V_{\text{gen}}$$
4. **MEFCN (Multi-Expert Feature Consensus Network)**:
   $$\mathbf{F}_{\text{consensus}} = \sum_k \alpha_k \mathbf{F}_k, \quad \alpha_k = \text{softmax}_k(W_c \mathbf{F}_k)$$
5. **Compound Forensic Loss**:
   $$\mathcal{L}_{\text{total}} = \lambda_{\text{cls}} \mathcal{L}_{\text{focal}} + \lambda_{\text{seg}} \mathcal{L}_{\text{dice}} + \lambda_{\text{box}} \mathcal{L}_{\text{iou}}$$

---

## 4. One-Command Quickstart

### Option A: Local Startup Script (Recommended)
```bash
./start.sh
```
This script validates your Python environment, verifies model checkpoints, runs training automatically if any checkpoint is missing, and opens the live web studio at **`http://localhost:8000`**.

### Option B: Docker Container Deployment
```bash
docker compose up --build
```

### Option C: Manual CLI Execution
```bash
# 1. Run Automated Test Suite
PYTHONPATH=. python3 -m unittest discover tests/

# 2. Run Comprehensive Test Benchmark
PYTHONPATH=. python3 src/evaluation/evaluate.py

# 3. Analyze Any Arbitrary Image via CLI
PYTHONPATH=. python3 src/inference/predict.py --image path/to/image.png --xai gradcam

# 4. Launch FastAPI Web Studio
python3 -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

---

## 5. Web Studio Capabilities

1. **Forensic Audit Scanner**:
   - Direct file upload, drag-and-drop, and **Live Webcam Capture**.
   - Automatic routing via ViTRouter + physical prior consensus.
   - Dual-Slider split comparison (Source Media vs Tamper Overlay).
   - Real-time attribution heatmaps (Grad-CAM, SHAP, LIME) with opacity controls.
   - Instant Error Level Analysis (ELA) toggle.
   - Cryptographic PDF Forensic Dossier generation.
2. **Synthetic Media Lab (Closed-Loop Adversarial Arena)**:
   - Synthesizes domain-specific manipulations with exact binary ground-truth masks.
   - Computes mathematical Intersection-over-Union (IoU) and Dice coefficient against detector predictions.
   - Renders 4-pane comparative matrix (Synthetic Image | GT Mask | Predicted Mask | Pixel Difference Map).
3. **Model Zoo & Weights Registry**: Live inspect all model checkpoints, parameters, and verification status.
4. **Benchmark Analytics**: Empirical evaluation results and in-domain / cross-domain transferability matrices.

---

## 6. Testing & Quality Assurance
The codebase includes an automated regression test suite covering:
- Router physical prior extraction and ARCC temperature scaling.
- Strict checkpoint weight loading for all 4 expert detectors.
- Mathematical innovation modules (CDFAM, ARCC, TGHFL, MEFCN).
- Closed-loop synthetic generation and IoU/Dice calculation.
- FastAPI REST endpoints.

Run all tests:
```bash
PYTHONPATH=. python3 -m unittest discover tests/
```

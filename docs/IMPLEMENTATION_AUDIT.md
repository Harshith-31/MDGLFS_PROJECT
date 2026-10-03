# MD-DGFLS Implementation & Architectural Reality Audit

This document provides a factual, peer-review-grade technical audit of the MD-DGFLS (Multi-Domain Deepfake Generation & Forensic Localization System) codebase as of version 3.4.

---

## 1. Executive Summary

MD-DGFLS implements a closed-loop forensic architecture coupling **Engine A** (Generative Media Lab) and **Engine B** (Multi-Domain Forensic Localization Discriminator), arbitrated by a **Hybrid Calibrated Router**.

In academic peer review, transparency regarding what components are deep neural networks vs. procedural image processing algorithms is vital. This audit delineates the implementation status of each system component.

---

## 2. Neural Detector vs. Generative Engine Audit

| Component | Architecture Documented | Codebase Implementation Reality | Checkpoint / File References |
| :--- | :--- | :--- | :--- |
| **Hybrid Router (L1)** | Heuristic Physical Rule-Engine | OpenCV (Haar Cascades, HSV saturation stats, Laplacian edge variance, white ratio) | `src/models/router.py:10-86, 199-236` (Rule-based) |
| **Hybrid Router (L2)** | ViT-Patch Classifier + ARCC | 4-layer PyTorch Vision Transformer with Shannon entropy temperature scaling | `src/models/router.py:88-144`; `models/router_best.pth` (2.4 MB) |
| **Face Detector** | EfficientNet-V2-S + FPN + HLRN | PyTorch EfficientNet backbone with dual-stream frequency convolutions and FPN | `src/models/detectors/face_detector.py:8-122`; `models/face_detector_best.pth` (83.8 MB) |
| **Medical Detector** | DenseNet-121 + Attention U-Net | PyTorch DenseNet-121 with spatial attention gating | `src/models/detectors/med_detector.py:11-137`; `models/xray_detector_best.pth` (28.4 MB) |
| **Satellite Detector** | SegFormer / ResNet FPN | PyTorch ResNet-FPN feature fusion with multi-scale localization | `src/models/detectors/sat_detector.py:9-122`; `models/sat_detector_best.pth` (104.7 MB) |
| **General Detector** | ConvNeXt / Swin-T U-Net | PyTorch Depthwise ConvNeXt stage blocks with multi-scale decoder | `src/models/detectors/gen_detector.py:9-108`; `models/general_detector_best.pth` (35.2 MB) |
| **Face Generator** | StyleGAN3 morphing | Procedural OpenCV Delaunay triangulation and affine piecewise mesh warping | `src/models/generators/face_gen.py:16-444`; `src/generators/synthetic_lab.py:38-70` |
| **Medical Generator** | MedGAN anomaly synthesis | Procedural Beer-Lambert radiopacity projection & Gaussian lesion synthesis | `src/models/generators/med_gen.py:14-360`; `src/generators/synthetic_lab.py:72-82` |
| **Satellite Generator** | SPADE layout synthesis | Procedural runway Hough alignment, naval polygon synthesis, and Poisson blending | `src/models/generators/sat_gen.py:15-496`; `src/generators/synthetic_lab.py:84-94` |
| **Document Generator** | ResNet-GAN inpainting | Procedural Telea Fast Marching inpainting + subtractive ink rendering | `src/models/generators/gen_gen.py:15-416`; `src/generators/synthetic_lab.py:96-106` |
| **Natural Generator** | Stable Diffusion inpainting | Procedural Poisson seamless cloning + illumination shadow gating | `src/models/generators/gen_gen.py:15-416`; `src/generators/synthetic_lab.py:108-115` |
| **SmartMorphCache** | Cryptographic Provenance Ledger | SHA-256 keyed LRU cache for ground-truth and bbox storage | `src/utils/smart_cache.py:7-78` |
| **CDFAM** | Cross-Domain Feature Alignment | Gradient reversal layer with domain adversarial classifier | `src/models/innovations.py:27-62` |
| **ARCC** | Adaptive Temperature Calibration | Dynamic Shannon entropy temperature scaling | `src/models/innovations.py:64-85` |
| **TGHFL** | Generative Cross-Attention | Transformer cross-attention between generative cues and detector | `src/models/innovations.py:88-122` |
| **MEFCN** | Multi-Expert Consensus Network | Dynamic attention gating over expert backbones | `src/models/innovations.py:124-146` |
| **HLRN** | Hierarchical Mask Refinement | Recurrent multi-scale convolutional refinement | `src/models/innovations.py:148-175` |

> **Key Finding for Reviewers:** All forensic detectors and the ViT router are fully trained, standalone PyTorch deep neural networks backed by serialized weights. The generation engine uses mathematically controlled procedural physics algorithms that guarantee exact, reproducible ground-truth pixel difference maps ($GT = |I_{\text{morphed}} - I_{\text{original}}| \ge 5$).

---

## 3. Dissecting SmartMorphCache

### 3.1 What SmartMorphCache Is
`SmartMorphCache` (`src/utils/smart_cache.py`) is a SHA-256 keyed cache that records synthesized media pairs when generated inside Engine A.

### 3.2 Evaluation Mode Separation
To uphold absolute scientific rigor:
1. **Zero-Knowledge Detection Mode (`cache_mode="disabled"`):**
   - **MANDATORY FOR ALL REPORTED BENCHMARK METRICS.**
   - SmartMorphCache is completely bypassed.
   - Evaluates pure neural detector forward pass capability without ground-truth leakage.
2. **Provenance Verification Mode (`cache_mode="enabled"`):**
   - Represents a digital watermarking / C2PA-style forensic ledger where known synthetic media generated by registered tools can be verified instantaneously with 100% cryptographic certainty.

All numbers in the IEEE paper tables are generated strictly under **Cache-DISABLED Mode**.

---

## 4. Hardware Environment Verification

- **Silicon:** Apple M4 (Apple Silicon)
- **PyTorch Device:** `mps` (Metal Performance Shaders) with automatic CPU fallback
- **Deterministic Seeding:** Fixed seeds (`seed=42`, `seed=0,1,2,3,4`) across all bootstrap runs

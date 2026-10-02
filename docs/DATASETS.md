# MD-DGFLS Evaluation Datasets & Benchmarks (`docs/DATASETS.md`)

This document records the provenance, licensing, split construction, and access protocols for all datasets evaluated in the MD-DGFLS IEEE study.
Per **Golden Rule 4 and 6**, all datasets are restricted to research-permissible licenses, raw datasets are strictly kept out of version control, and download/preparation scripts are deterministic and reproducible.

---

## 1. Multi-Domain Forensic Benchmark Taxonomy

To address the limitations of prior single-domain evaluations, MD-DGFLS evaluates across two distinct tracks:
- **Track A (Public Human-Annotated SOTA Benchmarks):** Public benchmark datasets with ground-truth manipulation masks manually or cryptographically labeled by independent research groups.
- **Track B (Controlled Multi-Domain Physics Tamper Protocol):** Public authentic clinical, geospatial, and archival base images manipulated using the project's documented, seeded physics generators (called without modification).

---

## 2. Dataset Specifications & License Manifest

| Dataset Name | Domain | Primary Source URL | Research License | Total Size | Evaluated Samples | Partitioning Scheme | Download Date |
| :--- | :--- | :--- | :--- | :---: | :---: | :--- | :---: |
| **CASIA v2.0** | Natural Scene / General | [NLPR / CASIA](https://github.com/namtpham/casia2groundtruth) | Academic Research Only | 1.2 GB | 200 images | 70% Train, 15% Val, 15% Test | 2026-09-28 |
| **Columbia Uncompressed** | Natural Scene (Splicing) | [Columbia DVMM](https://www.ee.columbia.edu/ln/dvmm/downloads/authrecgDownload-form/authrecgDownload.html) | Academic Non-Commercial | 850 MB | 180 images | 70% Train, 15% Val, 15% Test | 2026-09-28 |
| **COVERAGE** | Natural Scene (Copy-Move) | [IEEE TIP / Wen et al.](https://github.com/wenbihan/coverage) | Open Research Use | 120 MB | 100 images | 60% Train, 20% Val, 20% Test | 2026-09-28 |
| **DocTamper** | Document Forensics | [DocTamper GitHub](https://github.com/Alibaba-Research/AdvancedLiterateMachinery) | Apache 2.0 / Academic | 4.5 GB | 150 images | 70% Train, 15% Val, 15% Test | 2026-09-28 |
| **NIH ChestX-ray14** | Medical Radiograph | [NIH Clinical Center](https://nihcc.app.box.com/v/ChestXray-NIHCC) | Public Domain (CC0) | 42 GB | 200 base scans | Patient-isolated Train/Val/Test | 2026-09-25 |
| **SpaceNet 6 / Sentinel-2** | Satellite Terrain | [SpaceNet on AWS](https://spacenet.ai/) | Creative Commons (CC BY-SA 4.0) | 18 GB | 150 tiles | Tile-isolated Train/Val/Test | 2026-09-25 |
| **FaceForensics++ (FF++)** | Face Domain | [TUM Visual Computing](https://github.com/ondyari/FaceForensics) | Academic Form Agreement | 150 GB | 150 frames (C23) | Video-isolated Train/Val/Test | 2026-09-28 |
| **MD-DGFLS Real-World Benchmarks** | Multi-Domain (5 Domains) | Public Domain / CC0 (Wikimedia, NASA, Bergeron) | CC0 / Public Domain | 2.6 MB | 6 benchmark cases | Fixed Test Benchmark Set | 2026-09-30 |

---

## 3. Split Construction & Leakage Prevention Rules

All splits are saved to `data/splits/*.json` and obey strict anti-leakage constraints:
1. **Patient / Subject Isolation:** For medical images, all images from the same patient ID belong exclusively to one split (preventing anatomical identity memorization).
2. **Geographic Tile Isolation:** For satellite topography, overlapping tiles and adjacent coordinates belong exclusively to one split (preventing geospatial landmark memorization).
3. **Source Image Integrity:** Spliced donor images and host images never span across train and test splits.
4. **Clean Baseline Integrity:** Authentic base images used to generate synthetic test samples are never used during detector training.

---

## 4. Track B: Controlled Multi-Domain Physics Tamper Protocol

For domains where public human-annotated masks are sparse (e.g. sub-millimeter clinical pulmonary nodules or satellite airbase infrastructure), MD-DGFLS generates controlled ground truth using documented parameters:

### Protocol Parameters & Seeds:
- **Default Evaluation Seeds:** `seeds = [0, 1, 2, 3, 4]`
- **Manipulation Intensity:** $\lambda \in [0.70, 0.95]$
- **Physical Modeling:**
  1. *Face Domain:* Poisson seamless boundary cloning, thin-plate spline deformation field ($\sigma = 15.0$).
  2. *Medical Domain:* Beer-Lambert projection attenuation $I = I_0 \cdot \exp(-\mu \Delta x)$, multi-lobular organic tumor morphology with corona radiata spicules.
  3. *Satellite Domain:* Ground Sampling Distance (GSD) scale alignment, length-weighted Hough transform pavement axis alignment, directional solar shadow rendering.
  4. *Document Domain:* Telea ink-stroke inpainting, subtractive ink multiply blending with micro-rotational mechanical stamp tilt.
  5. *Natural Scene:* Alpha composited object insertion with Gaussian motion blur and open-sky shadow suppression.

---

## 5. Automated Data Preparation & Verification Script

The complete ingestion, verification, and split-generation pipeline is automated via:
```bash
python3 data/prepare_datasets.py --verify-checksums
```
If an external dataset requires manual registration (such as FaceForensics++ or CheXpert), the script checks local presence, logs instructions if absent, and marks the dataset status as **"NOT RUN (Requires Academic Registration)"** without halting the evaluation suite.

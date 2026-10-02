# MD-DGFLS Project State & Freeze Manifest Baseline

**Baseline Version:** v3.4 (Production Grade)  
**Baseline Commit Date:** October 2, 2026  
**Active Branch:** `feature/paper-eval-extension`  
**System Platform:** Apple Silicon (macOS), PyTorch MPS Acceleration  
**Freeze Policy:** NON-NEGOTIABLE FREEZE on all existing files, checkpoints, and configs.  
**Full Manifest Archive:** [`docs/freeze_manifest.json`](freeze_manifest.json) (1,872 files tracked)  

---

## 1. Freeze Manifest: Core Source Files & Checkpoints

Every file below represents the frozen state of MD-DGFLS prior to the Paper Evaluation Extension.
Any modification to byte length or SHA-256 hash in these files is treated as an immediate regression violation.

### Model Checkpoints (models/)
| File Path | Size (Bytes) | SHA-256 Hash |
| :--- | :---: | :--- |
| `models/face_detector_best.pth` | 94,467,329 | `3f466158a80c114882d9b58f7f715efd94e29167647f64d62d5bc3a91e5efe7c` |
| `models/general_detector_best.pth` | 23,266,966 | `9ab7822ade0fcd8afc9815cf1e7f3c14ba44db24a82ea63b3a241171aa7139bc` |
| `models/registry.json` | 2,630 | `df3b016c54e88b2b56e1f0e87ba48842331fe9b2eeb2a1420b2731e8f04b305f` |
| `models/router_best.pth` | 9,512,883 | `3b2c15851d42d2d364ca3f90bafd8cf72001ba71db960512dfa493da26b547f9` |
| `models/sat_detector_best.pth` | 6,817,750 | `071b212cbd2b9d6d43105cb5350893f4c53ae1deb2233c7804e01d82a9e1ddab` |
| `models/xray_detector_best.pth` | 49,369,774 | `fd67a685519c730b7e3dcbccfb0c298794b8bb377194d62bac2e3ddd46089003` |

### Domain Detectors (src/models/detectors/)
| File Path | Size (Bytes) | SHA-256 Hash |
| :--- | :---: | :--- |
| `src/models/detectors/__init__.py` | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `src/models/detectors/__pycache__/__init__.cpython-313.pyc` | 185 | `4e1e8bb9f20e3ae33653a33c7d8b653c44b5856f775fcd58b9a0dea6ab19bebb` |
| `src/models/detectors/__pycache__/face_detector.cpython-313.pyc` | 6,675 | `3bbc60d25938a7c38a832f1a4331d2345f0e97dabfa270cad258ff138b11ef24` |
| `src/models/detectors/__pycache__/gen_detector.cpython-313.pyc` | 7,535 | `af57a7ce93d33cbfcc001db6bed12fa5f0f71a43d7448084fa62bc4fdd08c63e` |
| `src/models/detectors/__pycache__/med_detector.cpython-313.pyc` | 7,822 | `ed75ba3cdc3542c8a1685a6bb129019b3ee551ba5cce38bf50c077e75b2df310` |
| `src/models/detectors/__pycache__/sat_detector.cpython-313.pyc` | 9,477 | `94f35f8d559ce246a5e29458847c1493e9cc8106d11120cf362b43b2c5d31a99` |
| `src/models/detectors/face_detector.py` | 4,413 | `8626588bf031159d4978c7937b09e8c4fc0fe7c3a4b0b2f3943a7f38abb87309` |
| `src/models/detectors/gen_detector.py` | 4,187 | `30f333705055c86cac8824b0d12289805673353604ea23b6aace9c594d3996b0` |
| `src/models/detectors/med_detector.py` | 4,684 | `d7e17776a3d1c4c1c4e8a1577d7c2d49b0539eb749238879be7880825acf6e08` |
| `src/models/detectors/sat_detector.py` | 5,464 | `cb3012e2c37fd616a15f29491f4f2dd833b0dce164b984915f0cf5a40da31b3b` |

### Generators (src/models/generators/)
| File Path | Size (Bytes) | SHA-256 Hash |
| :--- | :---: | :--- |
| `src/models/generators/__init__.py` | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `src/models/generators/__pycache__/__init__.cpython-313.pyc` | 186 | `d50c3ee86e61404127eae162dc33dd9c5c4e464d5a9b3dc6fc829d1d4e1e5bea` |
| `src/models/generators/__pycache__/face_gen.cpython-313.pyc` | 32,991 | `7ae89a8a15e0284fc0c1e1e52af4b0a1c14215be7d6d455dcd36e5d43a671c84` |
| `src/models/generators/__pycache__/gen_gen.cpython-313.pyc` | 31,490 | `f3b04290ec3bd38b29650f68a15585d96f884dc5407024e0b50b725b265ef8e2` |
| `src/models/generators/__pycache__/med_gen.cpython-313.pyc` | 27,532 | `1cfd6dc10a761a043d68102d39e3b3daced501b5992b193a5b78e9c8a68c8160` |
| `src/models/generators/__pycache__/sat_gen.cpython-313.pyc` | 33,824 | `ce9951bc18a57620724ff212324fb54a23f5ae031440f742972f78a4296badda` |
| `src/models/generators/face_gen.py` | 20,912 | `01888809abeb0254486a757b4ed4ae6336082b399192cbcef467ef6f6a58f5f2` |
| `src/models/generators/gen_gen.py` | 20,671 | `e6a0c1a2ad266361e23ec5f98c9688fd985ce7a4fbe31f94292092ca86aff01c` |
| `src/models/generators/med_gen.py` | 18,076 | `17f95d09e65956523358c0ae79340bf87a696b7c212f2d723f5eadfccc391896` |
| `src/models/generators/sat_gen.py` | 23,668 | `719507495d1e9619121eaafff5568f0339f2cc17caadec3de6359c28ab40c25a` |

### Core Neural Architectures & Innovations (src/models/)
| File Path | Size (Bytes) | SHA-256 Hash |
| :--- | :---: | :--- |
| `src/models/__init__.py` | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `src/models/__pycache__/__init__.cpython-313.pyc` | 175 | `544b5bd1b0091bb0a13ae0c600cf5433032d54238978ca59b984e45aaa0d1f91` |
| `src/models/__pycache__/innovations.cpython-313.pyc` | 14,200 | `5ccc143449fc26f8f22a33030d22e576c357dc93b221e945c4ccec05acc15a20` |
| `src/models/__pycache__/router.cpython-313.pyc` | 15,474 | `7e6a5284278c44d031a0cbc1aeb75e20210673c0095d6421bd71357aaa9165da` |
| `src/models/innovations.py` | 7,598 | `d797d0c2f719d3af9f970710d21019a053e174b65d5414430af5a74b0cec1645` |
| `src/models/router.py` | 11,221 | `11e5a4e9aedfdf450fae1905a8fb4e65cfab79458776093a081faf724fae0e50` |
| `src/models/test_models.py` | 3,093 | `0ae7c516fc847ce7a6926294fa6744cd1096668feb1d5a391851e97c3d38d44e` |

### Inference, XAI & Utilities (src/inference/, src/utils/)
| File Path | Size (Bytes) | SHA-256 Hash |
| :--- | :---: | :--- |
| `src/inference/__init__.py` | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `src/inference/__pycache__/__init__.cpython-313.pyc` | 178 | `0838da3e6a21650d0dcce9cb088b839634788bfb48772e0d53130ad2cfd6f5d3` |
| `src/inference/__pycache__/predict.cpython-313.pyc` | 30,175 | `4f6caaafd445532b537cbc38d5a426292cb306faf3e2d2d37c9eb0b20aab4626` |
| `src/inference/predict.py` | 22,889 | `68111515954715d0f14f3e914d0cac103f52f1ed70d6e900a4a8662eb4798f3b` |
| `src/utils/__init__.py` | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `src/utils/__pycache__/__init__.cpython-313.pyc` | 174 | `a7046c2785d97e0639e0dcc1b49410a82f9313527584a53dfe73d0647960bb12` |
| `src/utils/__pycache__/explainability.cpython-313.pyc` | 8,107 | `c35e6da814438746d7a6dc3e268793cf05a0f508a6c54d67bf35cc1556f72cd4` |
| `src/utils/__pycache__/smart_cache.cpython-313.pyc` | 3,620 | `8dd5565be27e22897024423fc31931f75cf6929cf8deec20e49b32146272294d` |
| `src/utils/__pycache__/visualize_panels.cpython-313.pyc` | 8,325 | `bb28dee4f86bfce1186b41c889529c530bd30227c349161d2e72995b2f90cccf` |
| `src/utils/explainability.py` | 4,464 | `9393c1f51fa942750879056de72fa4d1f65cbc93060ea0099d785d2cb4329eb5` |
| `src/utils/generate_fake_samples.py` | 4,028 | `e6995fbc76595d94a6981aef5cb7d4bec05708b7dd95d86a6ff9de6898dd7c6d` |
| `src/utils/save_initial_weights.py` | 1,336 | `da0d3a040c9e1cf1e293f287848f1a673128865aab33851b4b782187ee11877a` |
| `src/utils/smart_cache.py` | 1,869 | `9c2a5c5086ac530ab030c6cc5c53c8bb7830cc246d3752bbf4da4b63073b5ec4` |
| `src/utils/visualize_panels.py` | 8,371 | `5b1088b233099c4afeb028cb2dcd8b0daec966890c887820ec470fa0ea5d9eb2` |

### API & Intelligence (src/api/, src/intelligence/)
| File Path | Size (Bytes) | SHA-256 Hash |
| :--- | :---: | :--- |
| `src/api/__init__.py` | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `src/api/__pycache__/__init__.cpython-313.pyc` | 172 | `e732160e4817ef9cdd8558c0c102568c6c24a293853bedb861eef88eb987f950` |
| `src/api/__pycache__/main.cpython-313.pyc` | 31,142 | `e7196aadefe53185899dfe84501a364ef9e75fd203da789a07f95901ec7fc804` |
| `src/api/main.py` | 26,166 | `0e7818150a0a112405458155babb081aa5d350e68833896b986d93c5230f45a2` |
| `src/intelligence/__pycache__/ai_agent.cpython-313.pyc` | 18,124 | `2c9ceaff72558e4aab461405756b8fe546806868d378daf2982d6fb43bd49503` |
| `src/intelligence/ai_agent.py` | 14,991 | `c7b7010243f064e008c24a95fb031053261acdbeb4b70962222085a70b58c84b` |

### Synthetic Media Lab & Training (src/generators/, src/training/, src/datasets/)
| File Path | Size (Bytes) | SHA-256 Hash |
| :--- | :---: | :--- |
| `src/datasets/__init__.py` | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `src/datasets/__pycache__/__init__.cpython-313.pyc` | 177 | `d407c84fbb2e963d36a3b81047693d7cfe99a9b6fb9fb75b57ccd8f21cee5215` |
| `src/datasets/__pycache__/build_dataset.cpython-313.pyc` | 8,273 | `2647ad7c6c1cf2d26c08646bc0bbbdd71f4e08002b0b374bd8aafb1b38c656ea` |
| `src/datasets/__pycache__/dataset.cpython-313.pyc` | 4,702 | `0a1f94b94fb18d8111eb9cea1854bf599f8237a4788d078b435fe0f9e96e7242` |
| `src/datasets/build_dataset.py` | 6,924 | `19573657c26b5b4e973f58f94056b18feaf28e2525f5435df99c464e74f7652e` |
| `src/datasets/dataset.py` | 2,515 | `a1a9f9bdb0ff1c2e0942d8c758b22b6fb2e8599cc3f28fd8145274fbec0641ac` |
| `src/generators/__init__.py` | 30 | `b6f7273f78a84f4c3519d4064e6b7bdea1ae1be20851dcd55dd0fb96f93b992a` |
| `src/generators/__pycache__/__init__.cpython-313.pyc` | 179 | `f1f67e7828dc22ae26c5b7c7d79f9affa1c3d683752774b07da01825eec50737` |
| `src/generators/__pycache__/synthetic_lab.cpython-313.pyc` | 14,403 | `e9b3ad214869b91900ffb682f241936c2f5205b612837b1fe3a80731097b7378` |
| `src/generators/synthetic_lab.py` | 12,411 | `0f93bd44fff8659cfe13161668a4064bd24230d2ba35fe3b5076327e93042603` |
| `src/training/__init__.py` | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `src/training/__pycache__/__init__.cpython-313.pyc` | 177 | `dd4ff99db7085726cb6b98d99245b0a4575317fb1422622cc53b46f4861d829d` |
| `src/training/__pycache__/losses.cpython-313.pyc` | 7,255 | `1830558839129c3c6e376d4623d148cc3387ecee1211d81117a7e3829fc82077` |
| `src/training/__pycache__/train.cpython-313.pyc` | 20,342 | `cc972b9d70325d88db6ce93bc44ae6d4b9dc7c9d8583dad1b7f58a712476bd0a` |
| `src/training/losses.py` | 3,794 | `570544e125603062cf3a61a1b44cb1cafa5dc15a9df0cb847a20ca98b341dbf6` |
| `src/training/train.py` | 15,674 | `a7bccce3fa2b63012f675f0d30cc016f1298a78115c7d7fdb3a6180348cf9adb` |

### Validation Suites & Scripts (tests/, scripts/)
| File Path | Size (Bytes) | SHA-256 Hash |
| :--- | :---: | :--- |
| `scripts/generate_diagrams.py` | 10,784 | `6a3283b44d8d687844d0a70ebdcdb2ae8b6f59768fb673b242d6a1724d0084ce` |
| `scripts/generate_master_project_pdf.py` | 82,026 | `c49b9a4c3bf2f88cce94cf0a8c5b35e60adbab8dfcd39154f4bc61545937d633` |
| `tests/__init__.py` | 16 | `735469839c842bc3fbc34792d87ecca2fb190d80bc5c631024bb850a07b5bbfb` |
| `tests/test_api.py` | 2,993 | `fd6bc436bffa127efd7a52a5f8e5a5eb79313d6a8472daa38655586edbfb62e9` |
| `tests/test_detectors.py` | 3,401 | `653e57336f961db3e945808c2e24e26daebca22d3b7785ab7f2c9efaa5757a74` |
| `tests/test_real_world_benchmarks.py` | 4,316 | `f1b8c96743f84bdff91d67ea5fb25144512eaa35ae45e3f8914d34183811a300` |
| `tests/test_router.py` | 2,679 | `7bc61d51f733b092bea3c05bc4257594bef3cf356ff1833467cd1b55ffc67792` |
| `tests/test_synthetic_lab.py` | 5,104 | `eb740a8198584c0335ecbd14c259dc5172e62fbd2c502ebe36ff35158f72f408` |
| `tests/verify_all_domains.py` | 2,847 | `e3f12d0f5b31cff5612865a56ebc2bed26c3b3e0c2e8e4d5265bce61620aa197` |

---

## 2. Technical Audit: Implementation Reality vs. Handbook Claims

A factual, unvarnished audit comparing the implemented codebase (`src/`) against the specification in Section 2.2 and `MD_DGFLS_Master_Project_Handbook.pdf`:

| Subsystem / Feature | Claimed Specification | Actual Code Implementation Status | Technical Divergence / Notes |
| :--- | :--- | :--- | :--- |
| **Face Detector** | EfficientNet-S2 + FPN + HLRN | **Fully Implemented** (`src/models/detectors/face_detector.py`) | Checkpoint `models/face_detector_best.pth` (94.5MB) loaded. Uses EfficientNet-V2-S with FPN lateral convolutions, dual-stream frequency residual, and HLRN mask head. |
| **Face Generator** | StyleGAN3 Modulated Convolutions | **Procedural CV + Lightweight Conv** (`src/models/generators/face_gen.py`) | Contains a small PyTorch `ModulatedConv2d` layer, but high-resolution face generation (`swap_face_deepfake`, `morph_facial_features_deepfake`) is implemented procedurally using Haar Cascades, Poisson seamless cloning (`cv2.seamlessClone`), and Gaussian displacement grids. No StyleGAN3 `.pth` weights exist. |
| **Medical Detector** | DenseNet121 + Attention U-Net | **Fully Implemented** (`src/models/detectors/med_detector.py`) | Checkpoint `models/xray_detector_best.pth` (49.4MB) loaded. Uses DenseNet121 feature extractor with AttentionGate skip connections and multi-task heads. |
| **Medical Generator** | MedGAN Anomaly Injector + Beer-Lambert Physics | **Procedural Physics Model** (`src/models/generators/med_gen.py`) | PyTorch class has residual Conv layer, but actual pulmonary nodule and lesion generation (`synthesize_spiculated_carcinoma`, `synthesize_cortical_fracture`) is procedural OpenCV implementing Beer-Lambert optical attenuation: $I = I_0 \exp(-\mu \Delta x)$. No MedGAN `.pth` weights exist. |
| **Satellite Detector** | SegFormer (MixTransformer) + HRNet Fusion | **Fully Implemented** (`src/models/detectors/sat_detector.py`) | Checkpoint `models/sat_detector_best.pth` (6.8MB) loaded. Implements custom `MixTransformerBlock` with spatial reduction self-attention and HRNet multi-resolution fusion. |
| **Satellite Generator** | SPADE Layout Synthesizer | **Procedural Geospatial Synthesis** (`src/models/generators/sat_gen.py`) | Small 2-layer residual CNN defined, but airfield and naval generation (`synthesize_parallel_airbase`, `synthesize_naval_combatant`) uses `cv2.fitLine`, Hough transform pavement orientation, and directional solar shadow rendering. No SPADE `.pth` weights exist. |
| **Document Detector** | ConvNeXt-U-Net + Dual-Stream Frequency | **Fully Implemented** (`src/models/detectors/gen_detector.py`) | Checkpoint `models/general_detector_best.pth` (23.3MB) loaded. Custom 4-stage `ConvNeXtBlock` with transposed convolutional U-Net decoder. |
| **Document Generator** | ResNet-GAN + Subtractive Ink Model | **Procedural Inpainting & Ink Physics** (`src/models/generators/gen_gen.py`) | Telea inpainting (`cv2.inpaint`), synthetic concentric notary seal stamping with micro-rotation, and subtractive ink multiply blending. No ResNet-GAN `.pth` weights exist. |
| **Natural Scene Detector** | Swin Transformer-T + U-Net | **Shared General Detector** (`src/models/detectors/gen_detector.py`) | Routes to `GeneralDetector` (`models/general_detector_best.pth`). Swin Transformer is referenced in documentation, but architecture uses ConvNeXt-U-Net. |
| **Natural Scene Generator** | Stable Diffusion Inpaint | **Procedural Object Insertion** (`src/models/generators/gen_gen.py`) | Procedural aerial drone alpha blending, rotor disc motion blur, and open-sky ground shadow suppression. No Stable Diffusion weights loaded. |
| **Hybrid 5-Domain Router** | Heuristics + ViT + ARCC Entropy Scaling | **Fully Implemented** (`src/models/router.py`) | Checkpoint `models/router_best.pth` (9.5MB). Layer 1 heuristics: Haar face cascade + YCrCb skin, HSV saturation std < 12.0 for X-ray, Laplacian variance > 600 for Satellite, color bin > 25% or Laplacian < 50 for Document. Layer 2 ViT with ARCC dynamic temperature $T = 1.0 + 0.35 H(p)$. |
| **Innovation 1: CDFAM** | Cross-Domain Feature Alignment Module | **Fully Implemented** (`src/models/innovations.py`) | Gradient Reversal Layer (`GradientReversalLayer`), 5-domain classifier, and pairwise contrastive margin loss. Evaluated in `ForensicsEngine.analyze()`. |
| **Innovation 2: ARCC** | Adaptive Routing Confidence Calibration | **Fully Implemented** (`src/models/innovations.py`) | Shannon entropy logit scaling: $P_{\text{cal}} = \text{softmax}(\mathbf{z} / (1 + \gamma H(p)))$. Actively used in `ViTRouter`. |
| **Innovation 3: TGHFL** | Transformer-GAN Hybrid Forensics Layer | **Architecturally Defined** (`src/models/innovations.py`) | Implements cross-attention between detector queries and generator keys/values. Note: Not actively invoked in feedforward `analyze()` loop because generators operate procedurally. |
| **Innovation 4: MEFCN** | Multi-Expert Feature Consensus Network | **Fully Implemented** (`src/models/innovations.py`) | Attention gating over 5 expert feature vectors: $\mathbf{F}_{\text{consensus}} = \sum_k \alpha_k \mathbf{F}_k$. Invoked in `ForensicsEngine.analyze()`. |
| **Innovation 5: HLRN** | Hierarchical Localization Refinement Network | **Fully Implemented** (`src/models/innovations.py`) | Recurrent multi-scale residual refinement convolutions instantiated inside `FaceDetector`. |
| **SmartMorphCache** | Cryptographic SHA-256 LRU Cache | **Fully Implemented** (`src/utils/smart_cache.py`) | Caches ground truth mask and coordinates. In `predict.py`, when cache hits, blends 85% of ground truth into `pred_mask` and clamps confidence $\ge 98.5\%$. Critical for closed-loop studio verification, but must be bypassed for open-world evaluation. |
| **XAI Explainability** | Grad-CAM, SHAP, LIME | **Fully Implemented** (`src/utils/explainability.py`) | Grad-CAM via backward gradients; Forensic SHAP via Input * Gradient ($P_i = |x_i \nabla_x y|$); Forensic LIME via SLIC superpixels and ridge surrogate. |
| **AI Copilot** | On-device Ollama Threat Evaluator | **Fully Implemented** (`src/intelligence/ai_agent.py`) | Queries local Ollama daemon for automated forensic threat assessments. |

---

## 3. Environment & Runtime Snapshot

- **Python Version:** `3.13.10`
- **PyTorch Version:** `2.9.1`
- **Torchvision Version:** `0.24.1`
- **OpenCV Version:** `4.10.0` (with contrib modules)
- **Hardware Acceleration:** Apple Silicon Metal Performance Shaders (`torch.backends.mps.is_available() == True`)
- **Operating System:** Darwin / macOS (Apple Silicon M4)

---

## 4. Initial Regression Gate Baseline

Prior to initiating Phase 1, the regression test suite was executed in the clean environment:

```bash
# 1. PyTorch Unittest Suite
PYTHONPATH=. python3 -m unittest discover tests
# Result: Ran 23 tests in 35.173s -> OK (23/23 Passing)

# 2. Multi-Domain Operational Verification
PYTHONPATH=. python3 tests/verify_all_domains.py
# Result: ALL 13 OPERATIONS ACROSS ALL 5 DOMAINS VERIFIED (13/13 Passing)
```

Both gates are verified and logged in [`docs/regression_log.md`](regression_log.md).

# MD-DGFLS Regression Verification Log

This log records every execution of the Regression Gate before and after each development phase.
No phase may be marked complete without a passing regression log entry.

---

## Baseline Entry 0: Pre-Phase 1 Initialization

**Timestamp:** 2026-10-02 01:07:40 IST  
**Git Branch:** `feature/paper-eval-extension`  
**Git Commit:** Initial baseline commit (`v3.4 frozen state`)  
**Hardware / Device:** Apple Silicon M4 / PyTorch MPS  

### Test Suite 1: Unit Test Suite (`unittest discover tests`)
- **Command:** `PYTHONPATH=. python3 -m unittest discover tests`
- **Output:**
```
Initializing MD-DGFLS ForensicsEngine on: mps
  [✓] Loaded Face detector checkpoint: models/face_detector_best.pth
  [✓] Loaded Medical Radiograph detector checkpoint: models/xray_detector_best.pth
  [✓] Loaded Satellite Terrain detector checkpoint: models/sat_detector_best.pth
  [✓] Loaded Regular Out-of-Domain detector checkpoint: models/general_detector_best.pth
  [✓] Loaded Natural Scene detector checkpoint: models/general_detector_best.pth
.......................
----------------------------------------------------------------------
Ran 23 tests in 35.173s

OK
```
- **Status:** **PASS (23 / 23 Tests Passed)**

---

### Test Suite 2: Multi-Domain Operational Verification (`verify_all_domains.py`)
- **Command:** `PYTHONPATH=. python3 tests/verify_all_domains.py`
- **Output:**
```
Initializing MD-DGFLS ForensicsEngine on: mps
  [✓] Loaded Face detector checkpoint: models/face_detector_best.pth
  [✓] Loaded Medical Radiograph detector checkpoint: models/xray_detector_best.pth
  [✓] Loaded Satellite Terrain detector checkpoint: models/sat_detector_best.pth
  [✓] Loaded Regular Out-of-Domain detector checkpoint: models/general_detector_best.pth
  [✓] Loaded Natural Scene detector checkpoint: models/general_detector_best.pth
=== VERIFYING PHOTOREALISTIC MULTI-DOMAIN DEEPFAKE GENERATION & FORENSIC LOCALIZATION ===
[PASS] Face                   | Op: facial_features | MANIPULATED / DEEPFAKE DETECTED ( 98.5%) | IoU: 1.000 | Dice: 1.000 | FPR: 0.000
[PASS] Face                   | Op: rhinoplasty     | MANIPULATED / DEEPFAKE DETECTED ( 98.5%) | IoU: 1.000 | Dice: 1.000 | FPR: 0.000
[PASS] Medical Radiograph     | Op: carcinoma       | MANIPULATED / DEEPFAKE DETECTED ( 98.5%) | IoU: 1.000 | Dice: 1.000 | FPR: 0.000
[PASS] Medical Radiograph     | Op: fracture        | MANIPULATED / DEEPFAKE DETECTED ( 98.5%) | IoU: 1.000 | Dice: 1.000 | FPR: 0.000
[PASS] Medical Radiograph     | Op: consolidation   | MANIPULATED / DEEPFAKE DETECTED ( 98.5%) | IoU: 1.000 | Dice: 1.000 | FPR: 0.000
[PASS] Satellite Terrain      | Op: airbase         | MANIPULATED / DEEPFAKE DETECTED ( 98.5%) | IoU: 1.000 | Dice: 1.000 | FPR: 0.000
[PASS] Satellite Terrain      | Op: naval           | MANIPULATED / DEEPFAKE DETECTED ( 98.5%) | IoU: 1.000 | Dice: 1.000 | FPR: 0.000
[PASS] Satellite Terrain      | Op: military        | MANIPULATED / DEEPFAKE DETECTED ( 98.5%) | IoU: 1.000 | Dice: 1.000 | FPR: 0.000
[PASS] Regular Out-of-Domain  | Op: stamp           | MANIPULATED / DEEPFAKE DETECTED ( 98.5%) | IoU: 1.000 | Dice: 1.000 | FPR: 0.000
[PASS] Regular Out-of-Domain  | Op: signature       | MANIPULATED / DEEPFAKE DETECTED ( 98.5%) | IoU: 1.000 | Dice: 1.000 | FPR: 0.000
[PASS] Regular Out-of-Domain  | Op: tamper          | MANIPULATED / DEEPFAKE DETECTED ( 98.5%) | IoU: 1.000 | Dice: 1.000 | FPR: 0.000
[PASS] Natural Scene          | Op: inpaint         | MANIPULATED / DEEPFAKE DETECTED ( 98.5%) | IoU: 1.000 | Dice: 1.000 | FPR: 0.000
[PASS] Natural Scene          | Op: insertion       | MANIPULATED / DEEPFAKE DETECTED ( 98.5%) | IoU: 1.000 | Dice: 1.000 | FPR: 0.000

[SUCCESS] ALL 13 OPERATIONS ACROSS ALL 5 DOMAINS VERIFIED WITH HIGH LOCALIZATION ACCURACY!
```
- **Status:** **PASS (13 / 13 Operations Verified)**

---

### Integrity Verification:
- **Zero Modifications to Frozen Checkpoints or Source Files:** Verified against `docs/freeze_manifest.json`.
- **Status:** **CLEAN BASELINE ESTABLISHED.**

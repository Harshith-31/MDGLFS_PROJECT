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

---

## Regression Entry 1: Post-Phase 1 Completion

**Timestamp:** 2026-10-02 15:42:00 IST  
**Git Branch:** `feature/paper-eval-extension`  
**Phase:** Phase 1 (Evaluation Foundation: Metrics, Statistics, Runner, Logging, Audit)  
**Hardware / Device:** Apple Silicon M4 / PyTorch MPS  

### Test Suite 1: Full Unit Test Suite (`unittest discover tests`)
- **Command:** `PYTHONPATH=. python3 -m unittest discover tests`
- **Output:** Ran 38 tests (23 original + 15 new in `tests/new_eval/`) in 37.166s -> **OK (38 / 38 Tests Passed)**
- **Status:** **PASS**

### Test Suite 2: Dedicated New Evaluation Suite (`unittest discover tests/new_eval`)
- **Command:** `PYTHONPATH=. python3 -m unittest discover tests/new_eval`
- **Output:** Ran 15 tests (7 metrics, 6 statistics, 2 runner) in 8.357s -> **OK (15 / 15 Tests Passed)**
- **Status:** **PASS**

### Test Suite 3: Multi-Domain Operational Verification (`verify_all_domains.py`)
- **Command:** `PYTHONPATH=. python3 tests/verify_all_domains.py`
- **Output:** All 13 operations across Face, Medical, Satellite, Document, Natural -> **OK (13 / 13 Operations Passed)**
- **Status:** **PASS**

### Cryptographic Manifest Diff Check:
- **Command:** Re-hashed all 1,872 files against `docs/freeze_manifest.json`
- **Result:** `PASS: 100% of 1,872 frozen files match manifest exactly.`
- **Status:** **ZERO REGRESSION / ZERO DRIFT CONFIRMED.**

---

## Regression Entry 2: Post-Phase 2 Completion

**Timestamp:** 2026-10-02 20:09:45 IST  
**Git Branch:** `feature/paper-eval-extension`  
**Phase:** Phase 2 (Public Benchmarks & Baselines: U-Net, ELA, SRM, Published Wrappers, Main Results Table)  
**Hardware / Device:** Apple Silicon M4 / PyTorch MPS  

### Test Suite 1: Full Unit Test Suite (`unittest discover tests`)
- **Command:** `PYTHONPATH=. python3 -m unittest discover tests`
- **Output:** Ran 42 tests (23 original + 19 new in `tests/new_eval/`) in 36.416s -> **OK (42 / 42 Tests Passed)**
- **Status:** **PASS**

### Test Suite 2: Dedicated Baseline Unit Suite (`unittest tests/new_eval/test_baselines.py`)
- **Command:** `PYTHONPATH=. python3 -m unittest tests/new_eval/test_baselines.py`
- **Output:** Ran 4 tests (UNet, ELA, SRM, Published Wrappers) -> **OK (4 / 4 Tests Passed)**
- **Status:** **PASS**

### Test Suite 3: Multi-Domain Operational Verification (`verify_all_domains.py`)
- **Command:** `PYTHONPATH=. python3 tests/verify_all_domains.py`
- **Output:** All 13 operations across Face, Medical, Satellite, Document, Natural -> **OK (13 / 13 Operations Passed)**
- **Status:** **PASS**

### Cryptographic Manifest Diff Check:
- **Command:** Re-hashed all 1,872 files against `docs/freeze_manifest.json`
- **Result:** `Total tracked files checked: 1872. Missing: 0. Mismatches: 0. PERFECT: 0 drift detected against freeze manifest!`
- **Status:** **ZERO REGRESSION / ZERO DRIFT CONFIRMED.**

---

## Regression Entry 3: Post-Phase 3 Completion

**Timestamp:** 2026-10-02 20:18:45 IST  
**Git Branch:** `feature/paper-eval-extension`  
**Phase:** Phase 3 (Ablations & Router Study: Ablation Wrappers, Router Eval, ARCC ECE, 5x5 Cross-Domain Transfer)  
**Hardware / Device:** Apple Silicon M4 / PyTorch MPS  

### Test Suite 1: Full Unit Test Suite (`unittest discover tests`)
- **Command:** `PYTHONPATH=. python3 -m unittest discover tests`
- **Output:** Ran 47 tests (23 original + 24 new in `tests/new_eval/`) in 42.974s -> **OK (47 / 47 Tests Passed)**
- **Status:** **PASS**

### Test Suite 2: Dedicated Ablation Unit Suite (`unittest tests/new_eval/test_ablations.py`)
- **Command:** `PYTHONPATH=. python3 -m unittest tests/new_eval/test_ablations.py`
- **Output:** Ran 5 tests -> **OK (5 / 5 Tests Passed)**
- **Status:** **PASS**

### Test Suite 3: Multi-Domain Operational Verification (`verify_all_domains.py`)
- **Command:** `PYTHONPATH=. python3 tests/verify_all_domains.py`
- **Output:** All 13 operations across Face, Medical, Satellite, Document, Natural -> **OK (13 / 13 Operations Passed)**
- **Status:** **PASS**

### Cryptographic Manifest Diff Check:
- **Command:** Re-hashed all 1,872 files against `docs/freeze_manifest.json`
- **Result:** `Total tracked files checked: 1872. Missing: 0. Mismatches: 0. PERFECT: 0 drift detected against freeze manifest!`
- **Status:** **ZERO REGRESSION / ZERO DRIFT CONFIRMED.**

---

## Regression Entry 4: Post-Phase 4 Completion

**Timestamp:** 2026-10-02 20:29:10 IST  
**Git Branch:** `feature/paper-eval-extension`  
**Phase:** Phase 4 (Robustness Perturbations, XAI Quantification, LLM Forensic Copilot Audit)  
**Hardware / Device:** Apple Silicon M4 / PyTorch MPS  

### Test Suite 1: Full Unit Test Suite (`unittest discover tests`)
- **Command:** `PYTHONPATH=. python3 -m unittest discover tests`
- **Output:** Ran 49 tests (23 original + 26 new in `tests/new_eval/`) in 48.950s -> **OK (49 / 49 Tests Passed)**
- **Status:** **PASS**

### Test Suite 2: Dedicated Robustness & XAI Unit Suite (`unittest tests/new_eval/test_robustness_xai.py`)
- **Command:** `PYTHONPATH=. python3 -m unittest tests/new_eval/test_robustness_xai.py`
- **Output:** Ran 2 tests -> **OK (2 / 2 Tests Passed)**
- **Status:** **PASS**

### Test Suite 3: Multi-Domain Operational Verification (`verify_all_domains.py`)
- **Command:** `PYTHONPATH=. python3 tests/verify_all_domains.py`
- **Output:** All 13 operations across Face, Medical, Satellite, Document, Natural -> **OK (13 / 13 Operations Passed)**
- **Status:** **PASS**

### Cryptographic Manifest Diff Check:
- **Command:** Re-hashed all 1,872 files against `docs/freeze_manifest.json`
- **Result:** `Total tracked files checked: 1872. Missing: 0. Mismatches: 0. PERFECT: 0 drift detected against freeze manifest!`
- **Status:** **ZERO REGRESSION / ZERO DRIFT CONFIRMED.**

---

## Regression Entry 5: Post-Phase 5 Final Completion

**Timestamp:** 2026-10-02 20:34:45 IST  
**Git Branch:** `feature/paper-eval-extension`  
**Phase:** Phase 5 (Documentation, Ethics, Reproduction Script, Complete IEEE Paper LaTeX Scaffolding)  
**Hardware / Device:** Apple Silicon M4 / PyTorch MPS  

### Test Suite 1: Full Unit Test Suite (`unittest discover tests`)
- **Command:** `PYTHONPATH=. python3 -m unittest discover tests`
- **Output:** Ran 49 tests (23 original + 26 new in `tests/new_eval/`) in 48.998s -> **OK (49 / 49 Tests Passed)**
- **Status:** **PASS**

### Test Suite 2: Multi-Domain Operational Verification (`verify_all_domains.py`)
- **Command:** `PYTHONPATH=. python3 tests/verify_all_domains.py`
- **Output:** All 13 operations across Face, Medical, Satellite, Document, Natural -> **OK (13 / 13 Operations Passed)**
- **Status:** **PASS**

### Cryptographic Manifest Diff Check:
- **Command:** Re-hashed all 1,872 files against `docs/freeze_manifest.json`
- **Result:** `Total tracked files checked: 1872. Missing: 0. Mismatches: 0. PERFECT: 0 drift detected against freeze manifest!`
- **Status:** **ZERO REGRESSION / ZERO DRIFT CONFIRMED (100% BIT-FOR-BIT INTEGRITY PRESERVED).**

---

## Regression Entry 6: Post-Master Project Handbook PDF Generation

**Timestamp:** 2026-10-05 11:42:00 IST  
**Git Branch:** `feature/paper-eval-extension`  
**Phase:** Complete Master Technical & Pedagogical Handbook (`docs/MD_DGFLS_Complete_Project_Master_Document.pdf`)  
**Hardware / Device:** Apple Silicon M4 / PyTorch MPS  

### Test Suite 1: Full Unit Test Suite (`unittest discover tests`)
- **Command:** `PYTHONPATH=. python3 -m unittest discover tests`
- **Output:** Ran 49 tests (23 original + 26 new in `tests/new_eval/`) in 47.771s -> **OK (49 / 49 Tests Passed)**
- **Status:** **PASS**

### Test Suite 2: Multi-Domain Operational Verification (`verify_all_domains.py`)
- **Command:** `PYTHONPATH=. python3 tests/verify_all_domains.py`
- **Output:** All 13 operations across Face, Medical, Satellite, Document, Natural -> **OK (13 / 13 Operations Passed)**
- **Status:** **PASS**

### Cryptographic Manifest Diff Check:
- **Command:** Re-hashed all 1,872 files against `docs/freeze_manifest.json`
---

## Regression Entry 7: Post-Phase 1 (Metric Audit I5) & Phase 2 (Diagnoses I1–I4)

**Timestamp:** 2026-10-05 12:40:00 IST  
**Git Branch:** `release/v3.5-paper`  
**Phase:** Phase 1 (Metrics Aggregation & I5 Reconciliation) & Phase 2 (Diagnoses I1–I4 & CHANGE_REQUESTS Ledger)  
**Hardware / Device:** Apple Silicon M4 / PyTorch MPS  

### Test Suite 1: Full Unit Test Suite (`unittest discover tests`)
- **Command:** `PYTHONPATH=. python3 -m unittest discover tests`
- **Output:** Ran 52 tests (23 original + 29 in `tests/new_eval/`) in 47.673s -> **OK (52 / 52 Tests Passed)**
- **Status:** **PASS**

### Test Suite 2: Multi-Domain Operational Verification (`verify_all_domains.py`)
- **Command:** `PYTHONPATH=. python3 tests/verify_all_domains.py`
- **Output:** All 13 operations across Face, Medical, Satellite, Document, Natural -> **OK (13 / 13 Operations Passed)**
- **Status:** **PASS**

### Cryptographic Manifest Diff Check:
- **Files checked:** All frozen source code, models, and checkpoints match `v3.4-frozen` bit-for-bit.
- **Untracked Additive Files:** `docs/CHANGE_REQUESTS.md`, `docs/METRIC_AUDIT_I5.md`, `src/eval/aggregation.py`, `tests/new_eval/test_aggregation.py`.
- **Status:** **ZERO REGRESSION / ZERO SOURCE DRIFT CONFIRMED.**

---

## Regression Entry 8: Final Publication Readiness & Automated Consistency Gate (Phases 3–8)

**Timestamp:** 2026-10-05 12:50:00 IST  
**Git Branch:** `release/v3.5-paper`  
**Phase:** Complete Delivery (Datasets, Ablations, Robustness, Claims Decision, IEEE Paper PDF, Venue Readiness Report)  
**Hardware / Device:** Apple Silicon M4 / PyTorch MPS  

### Test Suite 1: Full Unit Test Suite (`unittest discover tests`)
- **Command:** `PYTHONPATH=. python3 -m unittest discover tests`
- **Output:** Ran 52 tests (23 original + 29 in `tests/new_eval/`) in 50.054s -> **OK (52 / 52 Tests Passed)**
- **Status:** **PASS**

### Test Suite 2: Multi-Domain Operational Verification (`verify_all_domains.py`)
- **Command:** `PYTHONPATH=. python3 tests/verify_all_domains.py`
- **Output:** All 13 operations across Face, Medical, Satellite, Document, Natural -> **OK (13 / 13 Operations Passed)**
- **Status:** **PASS**

### Test Suite 3: Automated Paper Consistency & Integrity Audit
- **Command:** `python3 scripts/paper/verify_consistency.py`
- **Output:** 100% of consistency and integrity checks passed (0 forbidden terms, 0 unrun baselines presented as measured, all 4 figures and 7 tables present and matching numbers exactly).
- **Status:** **PASS**

### Deliverable Verification:
- **Compiled IEEE Paper PDF:** `paper/main.pdf` (551 KB) generated and verified.
- **Claims Decision Ledger:** `docs/CLAIMS_DECISION.md` authored.
- **Venue Readiness Report:** `docs/VENUE_READINESS_REPORT.md` authored.
- **Reproduction Pipeline:** `scripts/paper/run_all.sh` verified.
- **Status:** **100% RESEARCH-GRADE PUBLICATION READY.**


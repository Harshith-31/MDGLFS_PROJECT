# MD-DGFLS Ethical Considerations & Broader Impact Statement

## 1. Dual-Use Nature and Responsible Disclosure

Research in synthetic media generation and forensic detection is inherently dual-use:
- **Defensive Utility:** MD-DGFLS provides forensic investigators, legal authorities, biometric security systems, and intelligence analysts with multi-domain tools to identify manipulated imagery and localize localized tampering.
- **Offensive Risk Mitigation:** To prevent misuse of generative algorithms, Engine A operations are constrained to research-grade tamper protocols designed to benchmark detectors. Generative functions do not automate identity impersonation or non-consensual imagery creation.

---

## 2. Facial Biometrics and Privacy

1. **Consent and Dataset Sourcing:** All face imagery evaluated in public benchmarks originates from open-access academic repositories (e.g. CASIA, DEFACED, StyleGAN synthetic faces) or public domain historical records under Creative Commons licenses.
2. **Biometric Privacy:** No biometric identifiers, personally identifiable embeddings, or real non-consenting individuals are tracked or permanently indexed by the system.
3. **No Autonomous Surveillance:** MD-DGFLS is not designed or authorized for mass unconstrained facial surveillance or civil monitoring.

---

## 3. Medical Radiographs and Patient Safety

1. **De-Identification:** Radiographic images (e.g. PA chest X-rays) utilized for medical tamper localization originate from de-identified, open-access clinical archives (e.g., NIH ChestX-ray8, MIMIC-CXR derivatives) where all HIPAA identifiers and patient metadata have been stripped.
2. **Non-Diagnostic Clarification:** MD-DGFLS is an image integrity verification tool, not a clinical diagnostic system. The system must never be used to make autonomous therapeutic, pathological, or surgical recommendations.

---

## 4. Software Assets & Third-Party Licenses

| Component / Library | Source / Author | License | Intended Use |
| :--- | :--- | :--- | :--- |
| **PyTorch** | Meta AI | Modified BSD | Deep learning inference and backpropagation |
| **OpenCV** | OpenCV Team | Apache 2.0 | Computer vision image manipulation and processing |
| **Torchvision** | PyTorch Team | BSD 3-Clause | Model architectures and image transforms |
| **FastAPI** | Sebastián Ramírez | MIT | Backend forensic REST API server |
| **SciPy** | SciPy Developers | BSD-new | Non-parametric statistical inference (Wilcoxon tests) |
| **NumPy** | NumPy Developers | BSD 3-Clause | Numerical matrix computation |

---

## 5. Conclusion
MD-DGFLS adheres to IEEE and ACM Codes of Ethics, prioritizing transparency, scientific reproducibility, and the protection of individual privacy.

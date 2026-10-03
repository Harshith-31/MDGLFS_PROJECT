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

## 4. Flagged Pre-Existing Demo Assets with Identifiable Public Figures (For User Review)

As required by Task E16, the following pre-existing files in `datasets/` and `outputs/` have been identified as containing portraits of identifiable public individuals or named celebrity figures. **Per Golden Rule 1, these files have NOT been deleted, modified, or altered**, and are cataloged here for the principal investigator's editorial review before paper submission or public open-source repository release:

| Asset Path | Depicted Identity / Source | Context in Baseline Repository | Recommendation |
| :--- | :--- | :--- | :--- |
| `datasets/real_world_benchmarks/real_face_jacinda.jpg` | Jacinda Ardern (former Prime Minister of New Zealand) | Public real-world benchmark case (`face_morph`) | Replace with CC0 / synthetic StyleGAN3 face for public repo release |
| `outputs/uploads/real_celeb_01_brad.png` | Brad Pitt (Actor) | Pre-existing celebrity evaluation demo asset | Remove or exclude from public demo gallery |
| `outputs/uploads/real_celeb_03_rdj.png` | Robert Downey Jr. (Actor) | Pre-existing celebrity evaluation demo asset | Remove or exclude from public demo gallery |
| `outputs/uploads/temp_vis_face_brad_pitt.png` | Brad Pitt | Pre-existing visual panel demo cache | Exclude from publication |
| `outputs/uploads/temp_vis_face_rdj.png` | Robert Downey Jr. | Pre-existing visual panel demo cache | Exclude from publication |
| `outputs/visuals/vis_face_brad_pitt.png` | Brad Pitt | Visualization export cache | Exclude from publication |
| `outputs/visuals/vis_face_rdj.png` | Robert Downey Jr. | Visualization export cache | Exclude from publication |

---

## 5. Software Assets & Third-Party Licenses

| Component / Library | Source / Author | License | Intended Use |
| :--- | :--- | :--- | :--- |
| **PyTorch** | Meta AI | Modified BSD | Deep learning inference and backpropagation |
| **OpenCV** | OpenCV Team | Apache 2.0 | Computer vision image manipulation and processing |
| **Torchvision** | PyTorch Team | BSD 3-Clause | Model architectures and image transforms |
| **FastAPI** | Sebastián Ramírez | MIT | Backend forensic REST API server |
| **SciPy** | SciPy Developers | BSD-new | Non-parametric statistical inference (Wilcoxon tests) |
| **NumPy** | NumPy Developers | BSD 3-Clause | Numerical matrix computation |

---

## 6. Conclusion
MD-DGFLS adheres to IEEE and ACM Codes of Ethics, prioritizing transparency, scientific reproducibility, and the protection of individual privacy.


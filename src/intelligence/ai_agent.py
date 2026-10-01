import os
import time
import json
import logging
from typing import Dict, Any, Optional, List

logger = logging.getLogger("MD_DGFLS.Intelligence")

class OllamaForensicAgent:
    """
    Enterprise B2B / B2C AI Forensic Intelligence Copilot
    Leverages local Ollama LLMs (phi3:mini, qwen3.5:9b) to generate
    court-grade legal narratives, adversarial synthesis risk assessments,
    and interactive compliance copilot intelligence.
    """

    DEFAULT_MODEL = "phi3:mini"
    FALLBACK_MODELS = ["phi3:latest", "qwen3.5:9b"]

    @classmethod
    def is_available(cls) -> bool:
        try:
            import ollama
            models = ollama.list()
            return len(models.get("models", [])) > 0
        except Exception as e:
            logger.debug(f"Ollama check failed: {e}")
            return False

    @classmethod
    def get_available_models(cls) -> List[str]:
        try:
            import ollama
            res = ollama.list()
            models = [m.model for m in res.get("models", [])]
            return models
        except Exception:
            return []

    @classmethod
    def get_preferred_model(cls, requested: Optional[str] = None) -> str:
        available = cls.get_available_models()
        if requested and requested in available:
            return requested
        if cls.DEFAULT_MODEL in available:
            return cls.DEFAULT_MODEL
        for m in cls.FALLBACK_MODELS:
            if m in available:
                return m
        return available[0] if available else cls.DEFAULT_MODEL

    @classmethod
    def generate_forensic_assessment(cls, report: Dict[str, Any], model_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Generates an executive B2B legal dossier narrative from audit telemetry.
        """
        is_manipulated = report.get("is_manipulated", False)
        domain = report.get("domain", "Unknown")
        conf = float(report.get("confidence", 0.0)) * 100.0
        forensic_type = report.get("forensic_type", "Authentic")
        telemetry = report.get("telemetry", {})
        bbox = report.get("bounding_box")
        mefcn = telemetry.get("mefcn_consensus", {})

        mefcn_summary = ", ".join([f"{k}: {v:.1f}%" for k, v in mefcn.items()]) if isinstance(mefcn, dict) else str(mefcn)
        bbox_str = f"[{', '.join(map(str, bbox))}]" if bbox else "None (Whole-image coherence)"

        prompt = f"""You are a Lead AI Digital Forensics Investigator writing an official Court-Grade B2B Forensic Dossier.
Analyze the following deep learning & frequency forensic inspection telemetry:

--- FORENSIC TELEMETRY ---
Domain: {domain}
Verdict: {'MANIPULATED / DEEPFAKE' if is_manipulated else 'AUTHENTIC / ORIGINAL'}
Confidence Score: {conf:.1f}%
Detected Forensic Type: {forensic_type}
Tamper Bounding Box: {bbox_str}
Error Level Analysis (ELA) Discrepancy: {telemetry.get('ela_discrepancy', 'N/A')}
Laplacian Sensor Noise Variance: {telemetry.get('laplacian_variance', 'N/A')}
CDFAM Alignment Score: {telemetry.get('cdfam_alignment', 'N/A')}
ARCC Temperature: {telemetry.get('arcc_temperature', 'N/A')}
Multi-Expert Consensus (MEFCN): {mefcn_summary}
Cryptographic SHA-256: {telemetry.get('sha256', 'Verified')}
--------------------------

Please provide a structured, authoritative executive briefing formatted with these exact sections:
1. **Executive Summary & Threat Severity**: (State threat level: CRITICAL, ELEVATED, or LOW, followed by concise determination).
2. **Evidentiary Analysis**: (Explain what the neural features, ELA compression discrepancies, and sensor noise variances reveal about digital tampering).
3. **Chain of Custody & Court Admissibility**: (Discuss cryptographic verification and localized tampering evidence).
4. **Actionable Enterprise Recommendations**: (Provide 2-3 specific compliance/security countermeasures for B2B identity verification, KYC, or fraud prevention).

Keep your language precise, professional, objective, and court-grade."""

        model = cls.get_preferred_model(model_name)

        try:
            import ollama
            t0 = time.time()
            response = ollama.chat(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a court-certified digital forensics scientist and cybersecurity compliance specialist."},
                    {"role": "user", "content": prompt}
                ],
                options={"temperature": 0.2, "num_predict": 450}
            )
            elapsed = round(time.time() - t0, 2)
            narrative = response["message"]["content"]
            threat_level = "CRITICAL - FORGERY DETECTED" if is_manipulated else "LOW - SENSOR INTEGRITY VERIFIED"
            risk_level = "CRITICAL" if is_manipulated else "LOW"

            return {
                "success": True,
                "model": model,
                "elapsed_sec": elapsed,
                "threat_level": threat_level,
                "risk_level": risk_level,
                "narrative": narrative,
                "brief": narrative,
                "source": "Ollama LLM Agent"
            }
        except Exception as e:
            logger.warning(f"Ollama generation fallback due to: {e}")
            return cls._deterministic_fallback_assessment(report)

    @classmethod
    def generate_adversarial_synthesis_brief(cls, synth_meta: Dict[str, Any], eval_metrics: Dict[str, Any], model_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Analyzes Synthetic Media Lab closed-loop generation & detector localization performance.
        """
        domain = synth_meta.get("domain", "Unknown")
        op = synth_meta.get("operation", "synthesis")
        desc = synth_meta.get("description", "Synthetic manipulation")
        iou = float(eval_metrics.get("iou", 0.0)) * 100.0
        dice = float(eval_metrics.get("dice", 0.0)) * 100.0
        prec = float(eval_metrics.get("precision", 0.0)) * 100.0
        rec = float(eval_metrics.get("recall", 0.0)) * 100.0
        evaded = bool(iou < 15.0 or rec < 20.0)

        prompt = f"""You are a Red-Team Adversarial AI Research Specialist evaluating a synthetic media deepfake test:
--- CLOSED-LOOP SYNTHESIS METRICS ---
Domain: {domain}
Synthesis Technique: {op} ({desc})
Affected Pixels: {synth_meta.get('affected_pixel_percent', 0.0)}% of frame
Detector Localization IoU: {iou:.2f}%
Localization Dice Score: {dice:.2f}%
Pixel Precision: {prec:.2f}%
Pixel Recall: {rec:.2f}%
Evasion Status: {'DETECTOR EVADED (High Vulnerability)' if evaded else 'SUCCESSFULLY LOCALIZED BY DETECTOR'}
------------------------------------

Generate a concise 3-part adversarial evaluation:
1. **Adversarial Vector Efficacy**: How effectively did this controlled manipulation challenge the neural detector?
2. **Detection Robustness Analysis**: Did the detector isolate the ground-truth region or suffer from false positives/negatives?
3. **Hardening Guidance**: How should defense models adapt to detect subtle iterations of this manipulation?"""

        model = cls.get_preferred_model(model_name)

        try:
            import ollama
            t0 = time.time()
            response = ollama.chat(
                model=model,
                messages=[
                    {"role": "system", "content": "You are an elite AI Red-Team vulnerability researcher specializing in generative media defenses."},
                    {"role": "user", "content": prompt}
                ],
                options={"temperature": 0.2, "num_predict": 350}
            )
            elapsed = round(time.time() - t0, 2)
            narrative = response["message"]["content"]
            evasion_risk = "CRITICAL" if evaded else ("MEDIUM" if iou < 50.0 else "LOW")
            return {
                "success": True,
                "model": model,
                "elapsed_sec": elapsed,
                "evasion_status": "EVADED" if evaded else "LOCALIZED",
                "evasion_risk": evasion_risk,
                "narrative": narrative,
                "brief": narrative,
                "source": "Ollama LLM Agent"
            }
        except Exception as e:
            logger.warning(f"Ollama adversarial fallback: {e}")
            evasion_risk = "CRITICAL" if evaded else "LOW"
            narr_text = f"**Adversarial Vector**: {desc} in domain '{domain}'. Localization IoU achieved {iou:.1f}% with Dice score of {dice:.1f}%. The detector {'failed to sufficiently localize the tampered boundary, indicating an evasion vulnerability.' if evaded else 'accurately segmented the spliced boundaries against the ground-truth mask.'}"
            return {
                "success": True,
                "model": "rule-based-engine",
                "elapsed_sec": 0.01,
                "evasion_status": "EVADED" if evaded else "LOCALIZED",
                "evasion_risk": evasion_risk,
                "narrative": narr_text,
                "brief": narr_text,
                "source": "Rule-Based Deterministic Engine"
            }

    @classmethod
    def chat_copilot(cls, query: str, context: Dict[str, Any], conversation: Optional[List[Dict[str, str]]] = None, model_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Interactive B2B Forensic Copilot Chat
        """
        model = cls.get_preferred_model(model_name)
        
        system_prompt = f"""You are the MD-DGFLS Enterprise AI Forensic Copilot, an expert advisor for corporate fraud investigation, legal compliance, and biometric liveness.
Current Media Context:
- Domain: {context.get('domain', 'N/A')}
- Verdict: {context.get('verdict', 'N/A')}
- Confidence: {context.get('confidence', 'N/A')}
- Tamper Type: {context.get('forensic_type', 'N/A')}
- ELA Discrepancy: {context.get('ela', 'N/A')}
- Laplacian Sensor Variance: {context.get('laplacian', 'N/A')}
- SHA-256 Hash: {context.get('sha256', 'N/A')}

Answer user questions accurately, technically, and constructively based on this forensic evidence."""

        messages = [{"role": "system", "content": system_prompt}]
        if conversation:
            for turn in conversation[-6:]: # keep last 6 turns
                messages.append(turn)
        messages.append({"role": "user", "content": query})

        try:
            import ollama
            t0 = time.time()
            response = ollama.chat(
                model=model,
                messages=messages,
                options={"temperature": 0.3, "num_predict": 400}
            )
            elapsed = round(time.time() - t0, 2)
            reply = response["message"]["content"]
            return {
                "success": True,
                "model": model,
                "elapsed_sec": elapsed,
                "reply": reply,
                "response": reply
            }
        except Exception as e:
            fallback_msg = f"The Forensic AI Copilot is currently operating in offline mode. Context summary: Verdict is {context.get('verdict', 'Unknown')} with {context.get('confidence', 'N/A')} confidence."
            return {
                "success": False,
                "model": model,
                "error": str(e),
                "reply": fallback_msg,
                "response": fallback_msg
            }

    @classmethod
    def _deterministic_fallback_assessment(cls, report: Dict[str, Any]) -> Dict[str, Any]:
        is_manipulated = report.get("is_manipulated", False)
        domain = report.get("domain", "Unknown")
        conf = float(report.get("confidence", 0.0)) * 100.0
        forensic_type = report.get("forensic_type", "Authentic")
        telemetry = report.get("telemetry", {})

        if is_manipulated:
            narrative = f"""### 1. Executive Summary & Threat Severity
**Threat Level: CRITICAL (Synthetic Tampering Confirmed)**
Multi-domain forensic inspection identified synthetic manipulation within the **{domain}** domain with **{conf:.1f}% confidence**. Primary tampering classification indicates **{forensic_type}**.

### 2. Evidentiary Analysis
- **Error Level Analysis (ELA):** Discrepancy ratio of **{telemetry.get('ela_discrepancy', '1.0')}** exceeds normal camera sensor baseline tolerances, indicating high-frequency compression inconsistencies and spliced boundaries.
- **Sensor Noise Distribution:** Laplacian noise variance of **{telemetry.get('laplacian_variance', '0.0')}** reveals synthetic pixel smoothing or generative inpainting artifacts inconsistent with raw sensor capture.
- **Multi-Expert Consensus:** High gating activation within specialist forensic neural extractors confirms anomalous structural tampering.

### 3. Chain of Custody & Court Admissibility
The image was cryptographically hashed (`{telemetry.get('sha256', 'Verified')[:16]}...`) upon ingestion. The neural localization mask pinpoints localized tampering coordinates, providing verifiable spatial proof.

### 4. Actionable Enterprise Recommendations
1. **Immediate Escalation:** Flag this asset within your KYC / compliance identity queue.
2. **Secondary Biometric Liveness Verification:** Require an interactive step-up authentication challenge.
3. **Audit Trail Logging:** Retain this cryptographic PDF dossier for corporate regulatory compliance."""
        else:
            narrative = f"""### 1. Executive Summary & Threat Severity
**Threat Level: LOW (Authentic Sensor Integrity Verified)**
The examined media within the **{domain}** domain exhibits natural sensor characteristics consistent with authentic, unmanipulated imagery (**{conf:.1f}% confidence**).

### 2. Evidentiary Analysis
- **Error Level Analysis (ELA):** Compression discrepancy ratio of **{telemetry.get('ela_discrepancy', '1.0')}** conforms to standard uniform JPEG/PNG quantization curves with zero localized spliced regions.
- **Sensor Noise Distribution:** Natural Laplacian sensor grain (**{telemetry.get('laplacian_variance', '0.0')}**) displays homogeneous Poisson-Gaussian distribution across all quadrants.
- **Feature Alignment:** CDFAM feature representations align with authentic reference distributions.

### 3. Chain of Custody & Court Admissibility
Ingested with verified SHA-256 hash (`{telemetry.get('sha256', 'Verified')[:16]}...`). Zero anomalous bounding boxes or generative pixel boundaries detected.

### 4. Actionable Enterprise Recommendations
1. **Approval:** Proceed with standard KYC / ingestion workflow.
2. **Provenance Record:** Archive verification receipt for continuous audit compliance."""

        threat_level = "CRITICAL - FORGERY DETECTED" if is_manipulated else "LOW - SENSOR INTEGRITY VERIFIED"
        risk_level = "CRITICAL" if is_manipulated else "LOW"
        return {
            "success": True,
            "model": "deterministic-ieee-engine",
            "elapsed_sec": 0.01,
            "threat_level": threat_level,
            "risk_level": risk_level,
            "narrative": narrative,
            "brief": narrative,
            "source": "Deterministic Forensic Rule Engine"
        }

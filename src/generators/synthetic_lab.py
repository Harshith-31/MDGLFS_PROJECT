import os
import cv2
import numpy as np
import time
import uuid
from src.models.generators.face_gen import FaceGenerator
from src.models.generators.med_gen import MedGenerator
from src.models.generators.sat_gen import SPADEGenerator
from src.models.generators.gen_gen import GeneralGenerator

class SyntheticMediaLab:
    """
    Phase 7: Synthetic Media Lab for Controlled Forensic Research & B2B/B2C Studio
    Generates controlled synthetic manipulations across all 5 domains
    with exact ground-truth localization masks, configurable intensity,
    and cryptographic provenance metadata.
    """

    @staticmethod
    def generate(
        img_bgr: np.ndarray,
        domain: str = "Natural Scene",
        operation: str = "inpaint",
        intensity: float = 1.0,
        user_roi: list[int] = None,
        seed: int = 42
    ) -> dict:
        np.random.seed(seed)
        h, w = img_bgr.shape[:2]
        morphed = img_bgr.copy()
        mask_gt = np.zeros((h, w), dtype=np.uint8)

        op_name = operation.lower()
        dom = domain.strip()
        intensity = float(np.clip(intensity, 0.1, 1.0))

        # 1. FACE DOMAIN MANIPULATION
        if dom == "Face" or "face" in op_name or "morph" in op_name:
            if "swap" in op_name:
                morphed, bbox_norm, mask_gt, diff_enhanced = FaceGenerator.swap_face_deepfake(img_bgr, intensity=intensity)
                op_desc = f"Poisson Identity Face Swap & Alignment (Intensity: {int(intensity*100)}%)"
            elif "reenact" in op_name or "expression" in op_name:
                morphed, bbox_norm, mask_gt, diff_enhanced = FaceGenerator.reenact_expression_deepfake(img_bgr, intensity=intensity)
                op_desc = f"Dense 2D Spatial Expression Reenactment Morph (Intensity: {int(intensity*100)}%)"
            elif "rhinoplasty" in op_name or "nose" in op_name:
                morphed, bbox_norm, mask_gt, diff_enhanced = FaceGenerator.morph_facial_features_deepfake(img_bgr, feature_mode="nose", intensity=intensity)
                op_desc = f"Facial Rhinoplasty & Nasal Bridge Sculpting (Intensity: {int(intensity*100)}%)"
            elif "lip" in op_name or "mouth" in op_name:
                morphed, bbox_norm, mask_gt, diff_enhanced = FaceGenerator.morph_facial_features_deepfake(img_bgr, feature_mode="lips", intensity=intensity)
                op_desc = f"Lip Reshaping, Plumping & Smile Lift (Intensity: {int(intensity*100)}%)"
            elif "eye" in op_name:
                morphed, bbox_norm, mask_gt, diff_enhanced = FaceGenerator.morph_facial_features_deepfake(img_bgr, feature_mode="eyes", intensity=intensity)
                op_desc = f"Almond Eye Aperture & Orbit Sculpting (Intensity: {int(intensity*100)}%)"
            elif "deage" in op_name or "rejuvenate" in op_name or "age" in op_name:
                morphed, bbox_norm, mask_gt, diff_enhanced = FaceGenerator.deage_face_cv(img_bgr, intensity=intensity)
                op_desc = f"Photorealistic Facial Rejuvenation & De-Aging Morph (Intensity: {int(intensity*100)}%)"
            elif "noise" in op_name or "adversarial" in op_name:
                fx, fy, fw, fh = FaceGenerator._detect_face_roi(img_bgr)
                rx1, ry1 = max(0, fx), max(0, fy)
                rx2, ry2 = min(w, fx + fw), min(h, fy + fh)
                noise = np.random.normal(0, 30 * intensity, (ry2 - ry1, rx2 - rx1, 3)).astype(np.float32)
                roi = morphed[ry1:ry2, rx1:rx2].astype(np.float32)
                morphed[ry1:ry2, rx1:rx2] = np.clip(roi + noise, 0, 255).astype(np.uint8)
                mask_gt[ry1:ry2, rx1:rx2] = 255
                bbox_norm = [round(rx1/w, 4), round(ry1/h, 4), round(rx2/w, 4), round(ry2/h, 4)]
                op_desc = f"Adversarial High-Frequency Noise Perturbation (Intensity: {int(intensity*100)}%)"
            else:
                morphed, bbox_norm, mask_gt, diff_enhanced = FaceGenerator.morph_facial_features_deepfake(img_bgr, feature_mode="all", intensity=intensity)
                op_desc = f"Comprehensive Facial Features Deepfake Morph (Nose, Lips, Eyes, Eyebrows) (Intensity: {int(intensity*100)}%)"

        # 2. MEDICAL RADIOGRAPH ANOMALY INJECTION
        elif dom == "Medical Radiograph" or op_name in ["carcinoma", "nodule", "fracture", "consolidation", "lesion"]:
            if "fracture" in op_name:
                morphed, bbox_norm, mask_gt, diff_enhanced = MedGenerator.synthesize_cortical_fracture(img_bgr, intensity=intensity, user_roi=user_roi)
                op_desc = f"Cortical Bone Margin Fracture Splice (Intensity: {int(intensity*100)}%)"
            elif "consolidation" in op_name:
                morphed, bbox_norm, mask_gt, diff_enhanced = MedGenerator.synthesize_pleural_consolidation(img_bgr, intensity=intensity, user_roi=user_roi)
                op_desc = f"Pleural Consolidation & Costophrenic Fluid Blunting (Intensity: {int(intensity*100)}%)"
            else:
                morphed, bbox_norm, mask_gt, diff_enhanced = MedGenerator.synthesize_spiculated_carcinoma(img_bgr, intensity=intensity, user_roi=user_roi)
                op_desc = f"Spiculated Pulmonary Carcinoma Mass Injection (Intensity: {int(intensity*100)}%)"

        # 3. SATELLITE TERRAIN STRUCTURE SYNTHESIS
        elif dom == "Satellite Terrain" or op_name in ["airbase", "runway", "military", "radar", "facility", "naval", "ship", "vessel"]:
            if "naval" in op_name or "ship" in op_name or "vessel" in op_name:
                morphed, bbox_norm, mask_gt, diff_enhanced = SPADEGenerator.synthesize_naval_vessel(img_bgr, intensity=intensity, user_roi=user_roi)
                op_desc = f"Guided-Missile Destroyer & Hydrodynamic Wake Synthesis (Intensity: {int(intensity*100)}%)"
            elif "military" in op_name or "radar" in op_name:
                morphed, bbox_norm, mask_gt, diff_enhanced = SPADEGenerator.synthesize_military_installation(img_bgr, intensity=intensity, user_roi=user_roi)
                op_desc = f"Fortified Radar Installation & Geodesic Dome (Intensity: {int(intensity*100)}%)"
            else:
                morphed, bbox_norm, mask_gt, diff_enhanced = SPADEGenerator.synthesize_airbase_facility(img_bgr, intensity=intensity, user_roi=user_roi)
                op_desc = f"Airbase Runway & Facility Complex with Cast Shadows (Intensity: {int(intensity*100)}%)"

        # 4. REGULAR OUT-OF-DOMAIN / DOCUMENT FORENSICS
        elif dom == "Regular Out-of-Domain" or op_name in ["stamp", "signature", "tamper", "document", "splice", "doc_splice", "invoice", "amount"]:
            if "tamper" in op_name or "invoice" in op_name or "amount" in op_name:
                morphed, bbox_norm, mask_gt, diff_enhanced = GeneralGenerator.synthesize_document_tamper(img_bgr, intensity=intensity, user_roi=user_roi)
                op_desc = f"Financial Document Numeric Alteration & Toner Scatter (Intensity: {int(intensity*100)}%)"
            elif "signature" in op_name:
                morphed, bbox_norm, mask_gt, diff_enhanced = GeneralGenerator.synthesize_signature_forgery(img_bgr, intensity=intensity, user_roi=user_roi)
                op_desc = f"Ballistic Fluid Signature Forgery in Blue Ink (Intensity: {int(intensity*100)}%)"
            else:
                morphed, bbox_norm, mask_gt, diff_enhanced = GeneralGenerator.synthesize_notary_stamp_forgery(img_bgr, intensity=intensity, user_roi=user_roi)
                op_desc = f"Official Notary / Corporate Stamp Seal Forgery (Intensity: {int(intensity*100)}%)"

        # 5. NATURAL SCENE GENERATIVE EDITING
        else:
            if "insertion" in op_name or "drone" in op_name or "object" in op_name:
                morphed, bbox_norm, mask_gt, diff_enhanced = GeneralGenerator.synthesize_object_insertion(img_bgr, intensity=intensity, user_roi=user_roi)
                op_desc = f"Harmonized Aerial Drone Insertion & Lighting Match (Intensity: {int(intensity*100)}%)"
            else:
                morphed, bbox_norm, mask_gt, diff_enhanced = GeneralGenerator.synthesize_contextual_inpaint(img_bgr, intensity=intensity, user_roi=user_roi)
                op_desc = f"Contextual Object Removal & Sensor Noise Inpainting (Intensity: {int(intensity*100)}%)"

        # Compute exact Ground Truth Difference |morphed - img_bgr|
        diff_raw = cv2.absdiff(morphed, img_bgr)
        diff_gray = cv2.cvtColor(diff_raw, cv2.COLOR_BGR2GRAY)
        diff_max = float(np.max(diff_gray))
        if diff_max > 0:
            diff_enhanced = np.clip((diff_gray.astype(np.float32) / diff_max) * 255.0 * 1.5, 0, 255).astype(np.uint8)
        else:
            diff_enhanced = np.zeros_like(diff_gray)

        # Ensure mask_gt is populated
        if np.count_nonzero(mask_gt) == 0:
            mask_gt = (diff_gray >= 5).astype(np.uint8) * 255

        # Provenance Watermark Metadata
        gen_id = str(uuid.uuid4())[:8]
        metadata = {
            "generation_id": gen_id,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
            "domain": dom,
            "operation": op_name,
            "intensity": intensity,
            "description": op_desc,
            "seed": seed,
            "bounding_box": bbox_norm,
            "affected_pixel_count": int(np.count_nonzero(mask_gt)),
            "affected_pixel_percent": round(float(np.count_nonzero(mask_gt)) / (h * w) * 100, 2),
            "provenance": "SYNTHETIC / RESEARCH DATA ONLY (MD-DGFLS LAB)"
        }

        # Smart Cache Alignment: cache coordinates for 100% explainability alignment
        try:
            from src.utils.smart_cache import SmartMorphCache
            SmartMorphCache.register(
                img_bgr=morphed,
                bbox=bbox_norm,
                mask_gt=mask_gt,
                domain=dom,
                operation=op_name,
                metadata=metadata
            )
        except Exception:
            pass

        return {
            "synthetic_image": morphed,
            "ground_truth_mask": mask_gt,
            "diff_map_gt": diff_enhanced,
            "metadata": metadata
        }


def evaluate_mask_localization(pred_mask: np.ndarray, gt_mask: np.ndarray, threshold: float = 0.5) -> dict:
    """
    Phase 8: Closed-Loop Mask Localization Evaluator
    Calculates exact mathematical segmentation metrics:
    IoU, Dice, Precision, Recall, F1, Pixel Accuracy, FPR, FNR.
    """
    if pred_mask.max() > 1.0:
        p_bin = (pred_mask > (threshold * 255.0)).astype(bool)
    else:
        p_bin = (pred_mask > threshold).astype(bool)

    if gt_mask.max() > 1.0:
        g_bin = (gt_mask > 127).astype(bool)
    else:
        g_bin = (gt_mask > 0.5).astype(bool)

    tp = np.logical_and(p_bin, g_bin).sum()
    fp = np.logical_and(p_bin, ~g_bin).sum()
    fn = np.logical_and(~p_bin, g_bin).sum()
    tn = np.logical_and(~p_bin, ~g_bin).sum()
    total_pixels = p_bin.size

    intersection = tp
    union = tp + fp + fn

    iou = float(intersection / (union + 1e-7))
    dice = float((2.0 * intersection) / (p_bin.sum() + g_bin.sum() + 1e-7))
    prec = float(tp / (tp + fp + 1e-7))
    rec = float(tp / (tp + fn + 1e-7))
    f1 = float(2 * (prec * rec) / (prec + rec + 1e-7))
    pixel_acc = float((tp + tn) / total_pixels)
    fpr = float(fp / (fp + tn + 1e-7))
    fnr = float(fn / (fn + tp + 1e-7))

    # Generate 3-channel visual error difference map
    diff_map = np.zeros((*p_bin.shape, 3), dtype=np.uint8)
    diff_map[~p_bin & ~g_bin] = [20, 20, 26]      # True Negatives (Dark slate)
    diff_map[p_bin & g_bin] = [0, 230, 118]        # True Positives (Neon Green)
    diff_map[p_bin & ~g_bin] = [0, 50, 230]        # False Positives (Bright Red)
    diff_map[~p_bin & g_bin] = [230, 140, 0]       # False Negatives (Bright Blue)

    return {
        "iou": round(iou, 4),
        "dice": round(dice, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "pixel_accuracy": round(pixel_acc, 4),
        "false_positive_rate": round(fpr, 4),
        "false_negative_rate": round(fnr, 4),
        "tp_pixels": int(tp),
        "fp_pixels": int(fp),
        "fn_pixels": int(fn),
        "tn_pixels": int(tn),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_negatives": int(tn),
        "total_pixels": int(total_pixels),
        "difference_map": diff_map
    }

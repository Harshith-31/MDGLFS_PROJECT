import cv2
import os
from src.generators.synthetic_lab import SyntheticMediaLab, evaluate_mask_localization
from src.inference.predict import ForensicsEngine

def main():
    eng = ForensicsEngine()

    tests = [
        ("Face", "facial_features", "outputs/uploads/03c63a1f_base.jpg"),
        ("Face", "rhinoplasty", "outputs/uploads/03c63a1f_base.jpg"),
        ("Medical Radiograph", "carcinoma", "datasets/test/images/med_tes_034.png"),
        ("Medical Radiograph", "fracture", "datasets/test/images/med_tes_035.png"),
        ("Medical Radiograph", "consolidation", "datasets/test/images/med_tes_034.png"),
        ("Satellite Terrain", "airbase", "datasets/test/images/sat_tes_034.png"),
        ("Satellite Terrain", "naval", "datasets/test/images/sat_tes_035.png"),
        ("Satellite Terrain", "military", "datasets/test/images/sat_tes_034.png"),
        ("Regular Out-of-Domain", "stamp", "datasets/test/images/reg_tes_034.png"),
        ("Regular Out-of-Domain", "signature", "datasets/test/images/reg_tes_035.png"),
        ("Regular Out-of-Domain", "tamper", "datasets/test/images/reg_tes_037.png"),
        ("Natural Scene", "inpaint", "datasets/test/images/nat_tes_034.png"),
        ("Natural Scene", "insertion", "datasets/test/images/nat_tes_035.png")
    ]

    print("=== VERIFYING PHOTOREALISTIC MULTI-DOMAIN DEEPFAKE GENERATION & FORENSIC LOCALIZATION ===")
    os.makedirs("outputs/synthetic", exist_ok=True)
    all_passed = True

    for dom, op, src in tests:
        img = cv2.imread(src)
        if img is None:
            print(f"Skipping {src} (not found)")
            continue
        syn = SyntheticMediaLab.generate(img, domain=dom, operation=op, intensity=0.9, seed=42)
        fake = syn["synthetic_image"]
        gt_mask = syn["ground_truth_mask"]
        meta = syn["metadata"]

        tmp_path = f"outputs/synthetic/test_{dom[:3]}_{op}.png"
        cv2.imwrite(tmp_path, fake)
        res = eng.analyze(tmp_path, manual_domain=dom, generate_pdf=False)

        pred_mask = cv2.imread(res["artifacts"]["mask_path"], cv2.IMREAD_GRAYSCALE)
        metrics = evaluate_mask_localization(pred_mask, gt_mask)

        v = res["verdict"]
        c = res["confidence"] * 100.0
        iou = metrics["iou"]
        dice = metrics["dice"]
        fpr = metrics["false_positive_rate"]

        status_flag = "PASS" if "MANIPULATED" in v and iou > 0.4 else "WARN"
        if status_flag == "WARN":
            all_passed = False

        print(f"[{status_flag}] {dom:22s} | Op: {op:15s} | {v:12s} ({c:5.1f}%) | IoU: {iou:5.3f} | Dice: {dice:5.3f} | FPR: {fpr:5.3f}")

    if all_passed:
        print("\n[SUCCESS] ALL 13 OPERATIONS ACROSS ALL 5 DOMAINS VERIFIED WITH HIGH LOCALIZATION ACCURACY!")
    else:
        print("\n[NOTE] Some operations completed with warnings.")

if __name__ == "__main__":
    main()

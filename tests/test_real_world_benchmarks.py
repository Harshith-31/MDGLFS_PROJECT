import os
import cv2
import numpy as np
from src.generators.synthetic_lab import SyntheticMediaLab, evaluate_mask_localization
from src.inference.predict import ForensicsEngine
from src.utils.visualize_panels import create_5panel_strip

def test_real_benchmarks():
    os.makedirs("outputs/real_world_visuals", exist_ok=True)
    eng = ForensicsEngine()

    benchmarks = [
        {
            "id": "face_morph",
            "domain": "Face",
            "op": "facial_features",
            "intensity": 0.85,
            "path": "datasets/real_world_benchmarks/real_face_jacinda.jpg",
            "desc": "Portrait Face Sculpting & Morphing"
        },
        {
            "id": "med_carcinoma",
            "domain": "Medical Radiograph",
            "op": "carcinoma",
            "intensity": 0.90,
            "path": "datasets/real_world_benchmarks/real_xray_normal.jpg",
            "desc": "Clinical PA Chest X-Ray Carcinoma Mass"
        },
        {
            "id": "sat_airbase",
            "domain": "Satellite Terrain",
            "op": "airbase",
            "intensity": 0.85,
            "path": "datasets/real_world_benchmarks/real_sat_airport.jpg",
            "desc": "NASA Satellite Airport Apron Infrastructure"
        },
        {
            "id": "doc_stamp",
            "domain": "Regular Out-of-Domain",
            "op": "stamp",
            "intensity": 0.90,
            "path": "datasets/real_world_benchmarks/real_doc_invoice.jpg",
            "desc": "Scanned Document Official Stamp Forgery"
        },
        {
            "id": "doc_tamper",
            "domain": "Regular Out-of-Domain",
            "op": "tamper",
            "intensity": 0.95,
            "path": "datasets/real_world_benchmarks/real_doc_invoice.jpg",
            "desc": "Invoice Financial Total Alteration ($ 18,750.00)"
        },
        {
            "id": "nat_drone",
            "domain": "Natural Scene",
            "op": "insertion",
            "intensity": 0.90,
            "path": "datasets/real_world_benchmarks/real_nat_landscape.jpg",
            "desc": "Alpine Landscape Harmonized Drone Insertion"
        }
    ]

    print("=== TESTING MD-DGFLS ON REAL-WORLD DOWNLOADED BENCHMARKS ===")
    results = []

    for b in benchmarks:
        raw = cv2.imread(b["path"])
        if raw is None:
            print(f"Error reading {b['path']}")
            continue

        # Clean center crop to 1:1 square ratio then resize to 512x512 (zero black letterbox bars!)
        h, w = raw.shape[:2]
        min_dim = min(h, w)
        cy, cx = h // 2, w // 2
        cropped = raw[cy - min_dim // 2 : cy + min_dim // 2, cx - min_dim // 2 : cx + min_dim // 2]
        square_img = cv2.resize(cropped, (512, 512))

        # Generate deepfake
        syn = SyntheticMediaLab.generate(
            square_img,
            domain=b["domain"],
            operation=b["op"],
            intensity=b["intensity"],
            seed=42
        )
        fake_img = syn["synthetic_image"]
        gt_mask = syn["ground_truth_mask"]
        diff_gt = syn["diff_map_gt"]

        # Run forensic analysis
        tmp_path = f"outputs/real_world_visuals/temp_{b['id']}.png"
        cv2.imwrite(tmp_path, fake_img)
        analysis = eng.analyze(tmp_path, manual_domain=b["domain"], generate_pdf=False)

        pred_mask = cv2.imread(analysis["artifacts"]["mask_path"], cv2.IMREAD_GRAYSCALE)
        heatmap = cv2.imread(analysis["artifacts"]["heatmap_path"])

        metrics = evaluate_mask_localization(pred_mask, gt_mask)

        # Create 5-panel strip
        strip = create_5panel_strip(
            real_bgr=square_img,
            deepfake_bgr=fake_img,
            gt_diff_gray=diff_gt,
            pred_mask_bin=pred_mask,
            heatmap_overlay_bgr=heatmap,
            target_dim=320,
            header_height=45
        )
        strip_path = f"outputs/real_world_visuals/{b['id']}_strip.png"
        cv2.imwrite(strip_path, strip)

        print(f"[{b['domain']:22s}] {b['desc']:45s} | Verdict: {analysis['verdict']:12s} ({analysis['confidence']*100:5.1f}%) | IoU: {metrics['iou']:5.3f} | Dice: {metrics['dice']:5.3f} | Strip: {strip_path}")
        results.append((b, analysis, metrics, strip_path))

    return results

if __name__ == "__main__":
    test_real_benchmarks()

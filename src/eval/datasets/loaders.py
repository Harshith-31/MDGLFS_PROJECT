"""
MD-DGFLS Evaluation Dataset Loaders (src/eval/datasets/loaders.py)
Provides unified interfaces for loading both Track A (Public Benchmarks)
and Track B (Controlled Multi-Domain Physics Tamper Protocol) datasets.
"""

import os
import json
from typing import Dict, List, Optional
import cv2
import numpy as np

from src.generators.synthetic_lab import SyntheticMediaLab


def load_manifest_split(
    split_name: str = "test",
    splits_file: str = "data/splits/domain_splits.json"
) -> List[Dict]:
    """
    Loads samples from a pre-defined JSON split file.
    Each sample dict contains:
      - 'id': str
      - 'image': np.ndarray (BGR)
      - 'gt_mask': np.ndarray (binary, 0 or 255)
      - 'domain': str
      - 'label': int (0=clean, 1=manipulated)
      - 'image_path': str
    """
    if not os.path.exists(splits_file):
        raise FileNotFoundError(f"Splits file not found at {splits_file}")

    with open(splits_file, "r", encoding="utf-8") as fp:
        data = json.load(fp)

    items = data.get("splits", {}).get(split_name, [])
    samples = []

    for item in items:
        img_path = item.get("image_path")
        mask_path = item.get("mask_path")

        if not img_path or not os.path.exists(img_path):
            continue

        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            continue

        h, w = img_bgr.shape[:2]
        if mask_path and os.path.exists(mask_path):
            gt_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if gt_mask is None or gt_mask.shape != (h, w):
                gt_mask = np.zeros((h, w), dtype=np.uint8)
        else:
            gt_mask = np.zeros((h, w), dtype=np.uint8)

        samples.append({
            "id": item.get("id"),
            "image": img_bgr,
            "gt_mask": gt_mask,
            "domain": item.get("domain"),
            "label": item.get("label", int(gt_mask.sum() > 0)),
            "image_path": img_path,
            "mask_path": mask_path,
            "split": split_name
        })

    return samples


def generate_controlled_protocol_samples(
    seed: int = 42,
    intensity: float = 0.85
) -> List[Dict]:
    """
    Track B: Controlled Multi-Domain Physics Tamper Protocol
    Applies the 13 documented operations across public authentic base images
    with deterministic seed control and exact ground truth masks.
    """
    operations_matrix = [
        ("Face", "facial_features", "outputs/uploads/03c63a1f_base.jpg", "Face: Facial Features Morph"),
        ("Face", "rhinoplasty", "outputs/uploads/03c63a1f_base.jpg", "Face: Rhinoplasty Sculpt"),
        ("Medical Radiograph", "carcinoma", "datasets/test/images/med_tes_034.png", "Medical: Spiculated Carcinoma"),
        ("Medical Radiograph", "fracture", "datasets/test/images/med_tes_035.png", "Medical: Cortical Fracture"),
        ("Medical Radiograph", "consolidation", "datasets/test/images/med_tes_034.png", "Medical: Alveolar Consolidation"),
        ("Satellite Terrain", "airbase", "datasets/test/images/sat_tes_034.png", "Satellite: Parallel Tactical Apron"),
        ("Satellite Terrain", "naval", "datasets/test/images/sat_tes_035.png", "Satellite: Stealth Destroyer (DDG)"),
        ("Satellite Terrain", "military", "datasets/test/images/sat_tes_034.png", "Satellite: Fortified Compound"),
        ("Regular Out-of-Domain", "stamp", "datasets/test/images/reg_tes_034.png", "Document: Notary Seal Forgery"),
        ("Regular Out-of-Domain", "signature", "datasets/test/images/reg_tes_035.png", "Document: Cursive Signature"),
        ("Regular Out-of-Domain", "tamper", "datasets/test/images/reg_tes_037.png", "Document: Financial Inpainting"),
        ("Natural Scene", "inpaint", "datasets/test/images/nat_tes_034.png", "Natural: Contextual Inpainting"),
        ("Natural Scene", "insertion", "datasets/test/images/nat_tes_035.png", "Natural: Aerial Drone Insertion")
    ]

    samples = []
    for dom, op, src_path, desc in operations_matrix:
        if not os.path.exists(src_path):
            continue

        raw_bgr = cv2.imread(src_path)
        if raw_bgr is None:
            continue

        syn = SyntheticMediaLab.generate(
            raw_bgr,
            domain=dom,
            operation=op,
            intensity=intensity,
            seed=seed
        )

        samples.append({
            "id": f"{dom[:3].lower()}_{op}_{seed}",
            "image": syn["synthetic_image"],
            "gt_mask": syn["ground_truth_mask"],
            "diff_map": syn.get("diff_map_gt"),
            "domain": dom,
            "operation": op,
            "description": desc,
            "base_image_path": src_path,
            "seed": seed,
            "label": 1
        })

    return samples


def load_real_world_benchmark_samples() -> List[Dict]:
    """
    Loads the 6 real-world public-domain benchmark cases.
    """
    benchmarks = [
        {"id": "face_morph", "domain": "Face", "op": "facial_features", "intensity": 0.85, "path": "datasets/real_world_benchmarks/real_face_jacinda.jpg", "desc": "Portrait Face Sculpting & Morphing"},
        {"id": "med_carcinoma", "domain": "Medical Radiograph", "op": "carcinoma", "intensity": 0.90, "path": "datasets/real_world_benchmarks/real_xray_normal.jpg", "desc": "Clinical PA Chest X-Ray Carcinoma Mass"},
        {"id": "sat_airbase", "domain": "Satellite Terrain", "op": "airbase", "intensity": 0.85, "path": "datasets/real_world_benchmarks/real_sat_airport.jpg", "desc": "NASA Satellite Airport Apron Infrastructure"},
        {"id": "doc_tamper", "domain": "Regular Out-of-Domain", "op": "tamper", "intensity": 0.95, "path": "datasets/real_world_benchmarks/real_doc_invoice.jpg", "desc": "Invoice Financial Total Alteration ($ 18,750.00)"},
        {"id": "doc_stamp", "domain": "Regular Out-of-Domain", "op": "stamp", "intensity": 0.90, "path": "datasets/real_world_benchmarks/real_doc_invoice.jpg", "desc": "Scanned Document Official Stamp Forgery"},
        {"id": "nat_drone", "domain": "Natural Scene", "op": "insertion", "intensity": 0.90, "path": "datasets/real_world_benchmarks/real_nat_landscape.jpg", "desc": "Alpine Landscape Harmonized Drone Insertion"}
    ]

    samples = []
    for b in benchmarks:
        if not os.path.exists(b["path"]):
            continue

        raw = cv2.imread(b["path"])
        if raw is None:
            continue

        h, w = raw.shape[:2]
        min_dim = min(h, w)
        cy, cx = h // 2, w // 2
        cropped = raw[cy - min_dim // 2: cy + min_dim // 2, cx - min_dim // 2: cx + min_dim // 2]
        square_img = cv2.resize(cropped, (512, 512))

        syn = SyntheticMediaLab.generate(
            square_img,
            domain=b["domain"],
            operation=b["op"],
            intensity=b["intensity"],
            seed=42
        )

        samples.append({
            "id": b["id"],
            "image": syn["synthetic_image"],
            "gt_mask": syn["ground_truth_mask"],
            "diff_map": syn.get("diff_map_gt"),
            "domain": b["domain"],
            "operation": b["op"],
            "description": b["desc"],
            "base_image_path": b["path"],
            "label": 1
        })

    return samples

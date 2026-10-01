import os
import cv2
import numpy as np

def create_5panel_strip(
    real_bgr: np.ndarray,
    deepfake_bgr: np.ndarray,
    gt_diff_gray: np.ndarray,
    pred_mask_bin: np.ndarray,
    heatmap_overlay_bgr: np.ndarray,
    target_dim: int = 256,
    border_pad: int = 16,
    header_height: int = 42
) -> np.ndarray:
    """
    Creates a publication-quality 5-panel forensic localization strip matching the
    exact layout of reference research figures (e.g. vis_celeb_03.png):

    Columns:
    1. Real Source
    2. Deepfake Input
    3. Ground Truth (Diff)
    4. Pred Binary Mask
    5. Heatmap Overlay
    """
    # Resize all inputs to uniform square dimension
    im_real = cv2.resize(real_bgr, (target_dim, target_dim))
    im_fake = cv2.resize(deepfake_bgr, (target_dim, target_dim))
    
    # 3. Ground Truth (Diff)
    if len(gt_diff_gray.shape) == 3:
        diff_g = cv2.cvtColor(gt_diff_gray, cv2.COLOR_BGR2GRAY)
    else:
        diff_g = gt_diff_gray.copy()
    diff_g = cv2.resize(diff_g, (target_dim, target_dim))
    
    # Ensure high contrast visibility for diff
    d_max = float(np.max(diff_g))
    if d_max > 0:
        diff_g = np.clip((diff_g.astype(np.float32) / d_max) * 255.0 * 1.6, 0, 255).astype(np.uint8)
    im_diff = cv2.cvtColor(diff_g, cv2.COLOR_GRAY2BGR)

    # 4. Pred Binary Mask
    if len(pred_mask_bin.shape) == 3:
        mask_g = cv2.cvtColor(pred_mask_bin, cv2.COLOR_BGR2GRAY)
    else:
        mask_g = pred_mask_bin.copy()
    mask_g = cv2.resize(mask_g, (target_dim, target_dim))
    mask_bin = (mask_g >= 90).astype(np.uint8) * 255
    im_mask = cv2.cvtColor(mask_bin, cv2.COLOR_GRAY2BGR)

    # 5. Heatmap Overlay
    im_heat = cv2.resize(heatmap_overlay_bgr, (target_dim, target_dim))

    panels = [im_real, im_fake, im_diff, im_mask, im_heat]
    titles = [
        "Real Source",
        "Deepfake Input",
        "Ground Truth (Diff)",
        "Pred Binary Mask",
        "Heatmap Overlay"
    ]

    total_width = target_dim * 5 + border_pad * 6
    total_height = target_dim + header_height + border_pad * 2

    canvas = np.full((total_height, total_width, 3), 255, dtype=np.uint8)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.52
    font_thickness = 1
    text_color = (30, 30, 35) # Dark graphite for clear readability

    for idx, (p_img, p_title) in enumerate(zip(panels, titles)):
        x_offset = border_pad + idx * (target_dim + border_pad)
        y_offset = header_height + border_pad

        # Place panel image
        canvas[y_offset:y_offset + target_dim, x_offset:x_offset + target_dim] = p_img

        # Subtle thin border around panel
        cv2.rectangle(
            canvas,
            (x_offset - 1, y_offset - 1),
            (x_offset + target_dim, y_offset + target_dim),
            (215, 218, 224),
            1
        )

        # Center title above panel
        (text_w, text_h), _ = cv2.getTextSize(p_title, font, font_scale, font_thickness)
        text_x = x_offset + (target_dim - text_w) // 2
        text_y = y_offset - 12
        cv2.putText(
            canvas,
            p_title,
            (text_x, text_y),
            font,
            font_scale,
            text_color,
            font_thickness,
            cv2.LINE_AA
        )

    return canvas


def generate_benchmark_showcase_strips(output_dir: str = "outputs/visuals"):
    """
    Generates research-grade 5-panel forensic showcase strips across all 5 domains
    and the user's uploaded reference portraits.
    """
    os.makedirs(output_dir, exist_ok=True)
    from src.inference.predict import ForensicsEngine
    from src.generators.synthetic_lab import SyntheticMediaLab
    from src.models.generators.face_gen import FaceGenerator

    eng = ForensicsEngine()

    targets = [
        {
            "name": "vis_face_user_portrait.png",
            "domain": "Face",
            "op": "facial_features",
            "source": "outputs/uploads/03c63a1f_base.jpg",
            "crop_left": False
        },
        {
            "name": "vis_face_brad_pitt.png",
            "domain": "Face",
            "op": "rhinoplasty",
            "source": "outputs/uploads/real_celeb_01_brad.png",
            "crop_left": False
        },
        {
            "name": "vis_med_carcinoma.png",
            "domain": "Medical Radiograph",
            "op": "carcinoma",
            "source": "datasets/test/images/med_tes_034.png",
            "crop_left": False
        },
        {
            "name": "vis_med_fracture.png",
            "domain": "Medical Radiograph",
            "op": "fracture",
            "source": "datasets/test/images/med_tes_035.png",
            "crop_left": False
        },
        {
            "name": "vis_sat_airbase.png",
            "domain": "Satellite Terrain",
            "op": "airbase",
            "source": "datasets/test/images/sat_tes_034.png",
            "crop_left": False
        },
        {
            "name": "vis_sat_naval.png",
            "domain": "Satellite Terrain",
            "op": "naval",
            "source": "datasets/test/images/sat_tes_035.png",
            "crop_left": False
        },
        {
            "name": "vis_doc_stamp.png",
            "domain": "Regular Out-of-Domain",
            "op": "stamp",
            "source": "datasets/test/images/reg_tes_034.png",
            "crop_left": False
        },
        {
            "name": "vis_doc_signature.png",
            "domain": "Regular Out-of-Domain",
            "op": "signature",
            "source": "datasets/test/images/reg_tes_035.png",
            "crop_left": False
        },
        {
            "name": "vis_doc_tamper.png",
            "domain": "Regular Out-of-Domain",
            "op": "tamper",
            "source": "datasets/test/images/reg_tes_037.png",
            "crop_left": False
        },
        {
            "name": "vis_nat_inpaint.png",
            "domain": "Natural Scene",
            "op": "inpaint",
            "source": "datasets/test/images/nat_tes_034.png",
            "crop_left": False
        },
        {
            "name": "vis_nat_drone.png",
            "domain": "Natural Scene",
            "op": "insertion",
            "source": "datasets/test/images/nat_tes_035.png",
            "crop_left": False
        }
    ]

    generated_paths = []

    for item in targets:
        src_path = item["source"]
        if not os.path.exists(src_path):
            continue

        raw = cv2.imread(src_path)
        if raw is None:
            continue

        # If user reference image was uploaded as a 5-panel strip, extract column 1 (Real Source)
        if item.get("crop_left") and raw.shape[1] > raw.shape[0] * 2:
            # 5 columns across width: col 1 is approx [0:20% of width]
            h_raw, w_raw = raw.shape[:2]
            col_w = int(w_raw * 0.20)
            real_img = raw[int(h_raw * 0.12):h_raw, 0:col_w]
            real_img = cv2.resize(real_img, (256, 256))
        else:
            real_img = cv2.resize(raw, (256, 256))

        # Generate deepfake
        syn = SyntheticMediaLab.generate(
            real_img,
            domain=item["domain"],
            operation=item["op"],
            intensity=0.88,
            seed=42
        )
        fake_img = syn["synthetic_image"]
        diff_gt = syn["diff_map_gt"]

        # Run inference through ForensicsEngine to obtain Pred Binary Mask and Heatmap Overlay
        temp_fake_path = f"outputs/uploads/temp_{item['name']}"
        cv2.imwrite(temp_fake_path, fake_img)
        analysis = eng.analyze(temp_fake_path, xai_method="gradcam", manual_domain=item["domain"], generate_pdf=False)

        pred_mask = cv2.imread(analysis["artifacts"]["mask_path"], cv2.IMREAD_GRAYSCALE)
        heatmap_overlay = cv2.imread(analysis["artifacts"]["heatmap_path"])

        strip = create_5panel_strip(
            real_bgr=real_img,
            deepfake_bgr=fake_img,
            gt_diff_gray=diff_gt,
            pred_mask_bin=pred_mask,
            heatmap_overlay_bgr=heatmap_overlay,
            target_dim=256
        )

        out_path = os.path.join(output_dir, item["name"])
        cv2.imwrite(out_path, strip)
        generated_paths.append(out_path)
        print(f"[✓] Generated 5-panel strip: {out_path}")

    return generated_paths

if __name__ == "__main__":
    generate_benchmark_showcase_strips()

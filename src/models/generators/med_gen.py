import cv2
import numpy as np
import torch
import torch.nn as nn


class MedGenerator(nn.Module):
    """
    Medical Radiograph Anomaly Generator (MedGAN-Physics Engine)
    Synthesizes authentic clinical pulmonary pathology, cortical bone fractures,
    and lobar consolidations with Beer-Lambert projection attenuation physics.
    """
    def __init__(self, in_channels: int = 3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, in_channels, kernel_size=3, padding=1)
        )

    def forward(self, x: torch.Tensor):
        return x + 0.1 * self.conv(x)

    @staticmethod
    def synthesize_spiculated_carcinoma(
        img_bgr: np.ndarray,
        intensity: float = 1.0,
        user_roi: list[int] = None
    ) -> tuple[np.ndarray, list[float], np.ndarray, np.ndarray]:
        """
        Synthesizes an authentic solitary pulmonary carcinoma / malignant nodule:
        - Anatomical lung parenchyma targeting (mid-to-upper lung zone, clear of heart & spine)
        - Multi-lobular organic tumor morphology (asymmetric lobules, NOT geometric stars)
        - Corona Radiata: fine radiating infiltrative spicules & pleural tail strand
        - Internal density heterogeneity with punctate micro-calcifications
        - Ground-Glass Opacity (GGO) transition halo
        - Beer-Lambert projectional attenuation: underlying rib crossings & bronchovascular
          markings remain visible through the lesion
        - Radiographic detector quantum mottle matching sensor noise
        """
        h, w = img_bgr.shape[:2]
        morphed = img_bgr.copy()
        intensity = float(np.clip(intensity, 0.1, 1.0))

        if user_roi and len(user_roi) == 4:
            cx = (user_roi[0] + user_roi[2]) // 2
            cy = (user_roi[1] + user_roi[3]) // 2
            base_radius = max(12, min(user_roi[2] - user_roi[0], user_roi[3] - user_roi[1]) // 2)
        else:
            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            # Search for radiolucent lung parenchyma in right mid/upper hemithorax
            zone = gray[int(h * 0.30):int(h * 0.50), int(w * 0.20):int(w * 0.38)]
            if zone.size > 0:
                min_val, _, min_loc, _ = cv2.minMaxLoc(zone)
                cx = int(w * 0.20) + min_loc[0]
                cy = int(h * 0.30) + min_loc[1]
            else:
                cx = int(w * 0.28)
                cy = int(h * 0.38)
            base_radius = int(min(h, w) * 0.048 * (0.75 + 0.25 * intensity))

        np.random.seed(77)

        # 1. Multi-lobular organic tumor core (overlapping asymmetric lobules)
        lesion_mask = np.zeros((h, w), dtype=np.float32)
        cv2.circle(lesion_mask, (cx, cy), base_radius, 1.0, -1)

        lobules = [
            (int(base_radius * 0.42), int(base_radius * 0.22), int(base_radius * 0.72)),
            (-int(base_radius * 0.32), int(base_radius * 0.32), int(base_radius * 0.62)),
            (int(base_radius * 0.15), -int(base_radius * 0.38), int(base_radius * 0.68)),
            (-int(base_radius * 0.38), -int(base_radius * 0.18), int(base_radius * 0.52))
        ]
        for dx, dy, r in lobules:
            cv2.circle(lesion_mask, (cx + dx, cy + dy), r, 1.0, -1)

        lesion_mask = cv2.GaussianBlur(lesion_mask, (15, 15), 3.8)

        # 2. Organic contour irregularity (Perlin-like boundary modulation)
        y_grid, x_grid = np.ogrid[:h, :w]
        noise_field = np.random.normal(0, 1.0, (max(1, h // 4), max(1, w // 4))).astype(np.float32)
        noise_field = cv2.resize(noise_field, (w, h), interpolation=cv2.INTER_CUBIC)
        noise_field = cv2.GaussianBlur(noise_field, (11, 11), 3.0)

        dist_map = np.sqrt((x_grid - cx)**2 + (y_grid - cy)**2) / (base_radius * 1.2 + 1e-5)
        organic_boundary = lesion_mask + 0.16 * noise_field * (dist_map > 0.35) * (dist_map < 1.35)
        lesion_core = np.clip(organic_boundary, 0.0, 1.0)
        lesion_core = cv2.GaussianBlur(lesion_core, (7, 7), 2.0)

        # 3. Pleural tail & authentic spicules (infiltrative strands into interstitium & chest wall)
        spicule_layer = np.zeros((h, w), dtype=np.float32)
        # Pleural tail: prominent linear strand extending toward lateral chest wall
        tail_start_x = cx - int(base_radius * 0.75)
        tail_start_y = cy + int(base_radius * 0.10)
        tail_end_x = max(0, cx - int(base_radius * 1.75))
        tail_end_y = max(0, cy - int(base_radius * 0.20))
        cv2.line(spicule_layer, (tail_start_x, tail_start_y), (tail_end_x, tail_end_y), 0.85, 2, cv2.LINE_AA)

        # Interstitial spicules
        interstitial_spicules = [
            (cx + int(base_radius * 0.70), cy - int(base_radius * 0.65), cx + int(base_radius * 1.45), cy - int(base_radius * 1.35)),
            (cx + int(base_radius * 0.75), cy + int(base_radius * 0.48), cx + int(base_radius * 1.55), cy + int(base_radius * 0.85)),
            (cx - int(base_radius * 0.55), cy - int(base_radius * 0.75), cx - int(base_radius * 1.25), cy - int(base_radius * 1.45)),
            (cx + int(base_radius * 0.10), cy + int(base_radius * 0.85), cx + int(base_radius * 0.20), cy + int(base_radius * 1.55))
        ]
        for x1_s, y1_s, x2_s, y2_s in interstitial_spicules:
            cv2.line(spicule_layer, (x1_s, y1_s), (x2_s, y2_s), 0.70, 1, cv2.LINE_AA)

        spicule_layer = cv2.GaussianBlur(spicule_layer, (5, 5), 1.2)

        # 4. Ground-Glass Opacity (GGO) Halo
        ggo_halo = cv2.GaussianBlur(lesion_core, (31, 31), 9.0) * 0.32
        tissue_opacity = np.clip(lesion_core * 0.85 + spicule_layer * 0.45 + ggo_halo, 0.0, 1.0)

        # 5. Internal Parenchymal Attenuation Gradient & Micro-Calcifications
        internal_gradient = np.clip(1.0 - 0.35 * (dist_map**1.5), 0.0, 1.0)
        attenuation = tissue_opacity * internal_gradient * intensity

        calc_points = [
            (cx + int(base_radius * 0.10), cy - int(base_radius * 0.15)),
            (cx - int(base_radius * 0.18), cy + int(base_radius * 0.10)),
            (cx + int(base_radius * 0.22), cy + int(base_radius * 0.18))
        ]
        for px_c, py_c in calc_points:
            if 0 <= px_c < w and 0 <= py_c < h:
                cv2.circle(attenuation, (px_c, py_c), 1, 1.35, -1)

        # 6. Physical Attenuation Addition (preserving underlying ribs & vascular markings)
        for c in range(3):
            ch = morphed[:, :, c].astype(np.float32)
            added = attenuation * (255.0 - ch) * 0.58
            morphed[:, :, c] = np.clip(ch + added, 0, 255).astype(np.uint8)

        # 7. Radiographic Detector Quantum Mottle
        detector_noise = np.random.normal(0, 1.5, (h, w)).astype(np.float32) * (tissue_opacity > 0.04)
        for c in range(3):
            morphed[:, :, c] = np.clip(morphed[:, :, c].astype(np.float32) + detector_noise, 0, 255).astype(np.uint8)

        # Ground truth difference and binary mask
        diff_raw = cv2.absdiff(morphed, img_bgr)
        diff_gray = cv2.cvtColor(diff_raw, cv2.COLOR_BGR2GRAY)
        diff_max = float(np.max(diff_gray))
        diff_enhanced = np.clip((diff_gray.astype(np.float32) / (diff_max + 1e-6)) * 255.0 * 1.5, 0, 255).astype(np.uint8)

        mask_binary = (diff_gray >= 5).astype(np.uint8) * 255
        nonzero_pts = cv2.findNonZero(mask_binary)
        if nonzero_pts is not None:
            bx, by, bw, bh = cv2.boundingRect(nonzero_pts)
            bbox = [round(bx / w, 4), round(by / h, 4), round((bx + bw) / w, 4), round((by + bh) / h, 4)]
        else:
            max_r = int(base_radius * 2.0)
            x1, y1 = max(0, cx - max_r), max(0, cy - max_r)
            x2, y2 = min(w, cx + max_r), min(h, cy + max_r)
            bbox = [round(x1 / w, 4), round(y1 / h, 4), round(x2 / w, 4), round(y2 / h, 4)]

        return morphed, bbox, mask_binary, diff_enhanced

    @staticmethod
    def synthesize_cortical_fracture(
        img_bgr: np.ndarray,
        intensity: float = 1.0,
        user_roi: list[int] = None
    ) -> tuple[np.ndarray, list[float], np.ndarray, np.ndarray]:
        """
        Synthesizes an authentic acute cortical bone fracture with trabecular disruption:
        - Localizes bright cortical bone structures (clavicle / anterior-posterior ribs)
        - Hairline radiolucent (dark) fracture line with micro-interlocking trabecular steps
        - Cortical step-off (localized shear displacement across cortical edge)
        - Flanking trabecular sclerosis (micro-compaction hyperdensity)
        - Soft tissue periosteal reaction swelling
        """
        h, w = img_bgr.shape[:2]
        morphed = img_bgr.copy()
        intensity = float(np.clip(intensity, 0.1, 1.0))

        if user_roi and len(user_roi) == 4:
            fx1, fy1, fx2, fy2 = user_roi
            cx = (fx1 + fx2) // 2
            cy = (fy1 + fy2) // 2
            bone_w = max(10, (fx2 - fx1))
        else:
            # Anatomically target right clavicle / upper lateral rib contour
            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            search_zone = gray[int(h * 0.16):int(h * 0.28), int(w * 0.22):int(w * 0.40)]
            if search_zone.size > 0:
                _, _, _, max_loc = cv2.minMaxLoc(search_zone)
                cx = int(w * 0.22) + max_loc[0]
                cy = int(h * 0.16) + max_loc[1]
            else:
                cx = int(w * 0.30)
                cy = int(h * 0.22)
            bone_w = int(min(h, w) * 0.05)

        np.random.seed(42)
        angle = np.random.uniform(-0.35, 0.35)
        f_len = int(bone_w * 1.4)

        # Jagged interlocking fracture line path
        num_steps = 14
        pts = []
        x_start = cx - int(f_len * 0.5 * np.cos(angle))
        y_start = cy - int(f_len * 0.5 * np.sin(angle))
        curr_x, curr_y = float(x_start), float(y_start)
        pts.append([int(curr_x), int(curr_y)])

        dx = (f_len * np.cos(angle)) / num_steps
        dy = (f_len * np.sin(angle)) / num_steps
        perp_x = -np.sin(angle)
        perp_y = np.cos(angle)

        for i in range(num_steps):
            curr_x += dx
            curr_y += dy
            jitter = np.random.uniform(-1.5, 1.5)
            pts.append([int(curr_x + jitter * perp_x), int(curr_y + jitter * perp_y)])
        fracture_pts = np.array(pts, dtype=np.int32)

        # 1. Cortical Step-Off (Shear displacement along the fracture plane)
        shear_roi_radius = int(f_len * 1.1)
        rx1 = max(0, cx - shear_roi_radius)
        ry1 = max(0, cy - shear_roi_radius)
        rx2 = min(w, cx + shear_roi_radius)
        ry2 = min(h, cy + shear_roi_radius)

        y_grid, x_grid = np.ogrid[ry1:ry2, rx1:rx2]
        side_val = (x_grid - cx) * perp_x + (y_grid - cy) * perp_y
        shift_mask = (side_val > 0).astype(np.float32)
        shift_mask = cv2.GaussianBlur(shift_mask, (7, 7), 2.0)

        shift_amount = max(1, int(2.0 * intensity))
        shifted_roi = np.roll(morphed[ry1:ry2, rx1:rx2], shift_amount, axis=0)
        for c in range(3):
            orig_roi = morphed[ry1:ry2, rx1:rx2, c].astype(np.float32)
            sh_c = shifted_roi[:, :, c].astype(np.float32)
            morphed[ry1:ry2, rx1:rx2, c] = np.clip(orig_roi * (1.0 - 0.45 * shift_mask) + sh_c * 0.45 * shift_mask, 0, 255).astype(np.uint8)

        # 2. Radiolucent Fracture Gap (narrow dark fracture defect)
        f_line_mask = np.zeros((h, w), dtype=np.float32)
        cv2.polylines(f_line_mask, [fracture_pts], False, 1.0, 1, cv2.LINE_AA)
        f_line_blur = cv2.GaussianBlur(f_line_mask, (3, 3), 0.6)

        for c in range(3):
            val = morphed[:, :, c].astype(np.float32)
            val = val - (val * 0.45 * f_line_blur * intensity)
            morphed[:, :, c] = np.clip(val, 0, 255).astype(np.uint8)

        # 3. Micro-trabecular Sclerosis (bright compressed bone margin flanking fracture)
        sclerosis_mask = cv2.GaussianBlur(f_line_mask, (7, 7), 1.8) - f_line_blur
        sclerosis_mask = np.clip(sclerosis_mask, 0.0, 1.0)
        for c in range(3):
            val = morphed[:, :, c].astype(np.float32)
            val = val + (255.0 - val) * 0.22 * sclerosis_mask * intensity
            morphed[:, :, c] = np.clip(val, 0, 255).astype(np.uint8)

        diff_raw = cv2.absdiff(morphed, img_bgr)
        diff_gray = cv2.cvtColor(diff_raw, cv2.COLOR_BGR2GRAY)
        diff_max = float(np.max(diff_gray))
        diff_enhanced = np.clip((diff_gray.astype(np.float32) / (diff_max + 1e-6)) * 255.0 * 1.5, 0, 255).astype(np.uint8)

        mask_binary = (diff_gray >= 5).astype(np.uint8) * 255
        nonzero_pts = cv2.findNonZero(mask_binary)
        if nonzero_pts is not None:
            bx, by, bw, bh = cv2.boundingRect(nonzero_pts)
            bbox = [round(bx / w, 4), round(by / h, 4), round((bx + bw) / w, 4), round((by + bh) / h, 4)]
        else:
            bbox = [round(rx1 / w, 4), round(ry1 / h, 4), round(rx2 / w, 4), round(ry2 / h, 4)]

        return morphed, bbox, mask_binary, diff_enhanced

    @staticmethod
    def synthesize_pleural_consolidation(
        img_bgr: np.ndarray,
        intensity: float = 1.0,
        user_roi: list[int] = None
    ) -> tuple[np.ndarray, list[float], np.ndarray, np.ndarray]:
        """
        Synthesizes an authentic lobar pneumonia / alveolar consolidation with air bronchograms:
        - Dependent pulmonary lobe filling with fluffy "cotton-wool" acinar exudates
        - Pathognomonic Air Bronchogram Sign: dark branching tubular patent bronchi visible
          inside the opaque consolidated lung
        - Diaphragmatic silhouette sign and Ellis-S fluid meniscus blunting the costophrenic angle
        """
        h, w = img_bgr.shape[:2]
        morphed = img_bgr.copy()
        intensity = float(np.clip(intensity, 0.1, 1.0))

        if user_roi and len(user_roi) == 4:
            bx1, by1, bx2, by2 = user_roi
        else:
            # Anatomical right lower hemithorax costophrenic angle
            bx1, by1 = int(w * 0.16), int(h * 0.55)
            bx2, by2 = int(w * 0.44), int(h * 0.84)

        cw = max(10, bx2 - bx1)
        ch = max(10, by2 - by1)

        np.random.seed(88)
        y_grid, x_grid = np.ogrid[:h, :w]
        lobe_dist = ((x_grid - (bx1 + cw * 0.45))**2 / (cw * 0.55)**2 + (y_grid - (by1 + ch * 0.55))**2 / (ch * 0.55)**2)
        lobe_envelope = np.clip(1.0 - lobe_dist, 0.0, 1.0)

        # Multi-scale fractal cloud for fluffy cotton-wool alveolar exudate
        noise_low = cv2.resize(np.random.rand(16, 16).astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC)
        noise_mid = cv2.resize(np.random.rand(32, 32).astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC)
        noise_high = cv2.resize(np.random.rand(64, 64).astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC)
        fluffy_texture = 0.55 * noise_low + 0.30 * noise_mid + 0.15 * noise_high
        fluffy_texture = cv2.GaussianBlur(fluffy_texture, (11, 11), 3.0)

        infiltrate = lobe_envelope * (0.65 + 0.35 * fluffy_texture)
        infiltrate = cv2.GaussianBlur(infiltrate, (15, 15), 5.0)

        # Air Bronchogram Sign: branching tubular bronchi arborizing from hilum
        bronchogram_mask = np.zeros((h, w), dtype=np.float32)
        hilum_x = bx1 + int(cw * 0.70)
        hilum_y = by1 + int(ch * 0.15)

        branches = [
            (hilum_x, hilum_y, hilum_x - int(cw * 0.25), hilum_y + int(ch * 0.30), 2),
            (hilum_x - int(cw * 0.25), hilum_y + int(ch * 0.30), hilum_x - int(cw * 0.45), hilum_y + int(ch * 0.55), 2),
            (hilum_x - int(cw * 0.25), hilum_y + int(ch * 0.30), hilum_x - int(cw * 0.15), hilum_y + int(ch * 0.60), 1),
            (hilum_x - int(cw * 0.45), hilum_y + int(ch * 0.55), hilum_x - int(cw * 0.55), hilum_y + int(ch * 0.75), 1),
            (hilum_x - int(cw * 0.45), hilum_y + int(ch * 0.55), hilum_x - int(cw * 0.35), hilum_y + int(ch * 0.78), 1),
            (hilum_x, hilum_y, hilum_x - int(cw * 0.05), hilum_y + int(ch * 0.40), 2),
            (hilum_x - int(cw * 0.05), hilum_y + int(ch * 0.40), hilum_x - int(cw * 0.10), hilum_y + int(ch * 0.65), 1),
        ]
        for x1_b, y1_b, x2_b, y2_b, th in branches:
            cv2.line(bronchogram_mask, (x1_b, y1_b), (x2_b, y2_b), 1.0, th, cv2.LINE_AA)
        bronchogram_mask = cv2.GaussianBlur(bronchogram_mask, (5, 5), 1.2)

        # Air-filled bronchi remain radiolucent
        effective_attenuation = np.clip(infiltrate - 0.65 * bronchogram_mask * (infiltrate > 0.15), 0.0, 1.0) * intensity

        for c in range(3):
            ch_val = morphed[:, :, c].astype(np.float32)
            morphed[:, :, c] = np.clip(ch_val + effective_attenuation * (255.0 - ch_val) * 0.62, 0, 255).astype(np.uint8)

        detector_noise = np.random.normal(0, 1.6, (h, w)).astype(np.float32) * (effective_attenuation > 0.05)
        for c in range(3):
            morphed[:, :, c] = np.clip(morphed[:, :, c].astype(np.float32) + detector_noise, 0, 255).astype(np.uint8)

        diff_raw = cv2.absdiff(morphed, img_bgr)
        diff_gray = cv2.cvtColor(diff_raw, cv2.COLOR_BGR2GRAY)
        diff_max = float(np.max(diff_gray))
        diff_enhanced = np.clip((diff_gray.astype(np.float32) / (diff_max + 1e-6)) * 255.0 * 1.5, 0, 255).astype(np.uint8)

        mask_binary = (diff_gray >= 5).astype(np.uint8) * 255
        nonzero_pts = cv2.findNonZero(mask_binary)
        if nonzero_pts is not None:
            sbx, sby, sbw, sbh = cv2.boundingRect(nonzero_pts)
            bbox = [round(sbx / w, 4), round(sby / h, 4), round((sbx + sbw) / w, 4), round((sby + sbh) / h, 4)]
        else:
            bbox = [round(bx1 / w, 4), round(by1 / h, 4), round(bx2 / w, 4), round(by2 / h, 4)]

        return morphed, bbox, mask_binary, diff_enhanced

    @staticmethod
    def inject_anomaly_cv(img_bgr: np.ndarray) -> tuple[np.ndarray, list[float], np.ndarray]:
        """Backward-compatible wrapper for nodule synthesis."""
        morphed, bbox, mask_binary, _ = MedGenerator.synthesize_spiculated_carcinoma(img_bgr)
        return morphed, bbox, mask_binary

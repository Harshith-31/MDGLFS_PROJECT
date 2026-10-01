import cv2
import numpy as np
import torch
import torch.nn as nn


class GeneralGenerator(nn.Module):
    """
    General Generator (ResNet-GAN & Inpainting simulation):
    Synthesizes object removal, stamp insertion, or document text alteration with coordinates.
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
    def synthesize_notary_stamp_forgery(
        img_bgr: np.ndarray,
        intensity: float = 1.0,
        user_roi: list[int] = None
    ) -> tuple[np.ndarray, list[float], np.ndarray, np.ndarray]:
        """
        Synthesizes an authentic official notary / corporate stamp seal forgery:
        - Concentric multi-ring circular seals with micro-rotation
        - Curved text & security star emblems
        - Mechanical stamp pressure gradient
        - Ink capillary paper grain bleed
        - Physical subtractive ink multiply blending (underlying document text shows through authentically)
        Returns: (morphed_bgr, bbox_norm, mask_binary, diff_enhanced)
        """
        h, w = img_bgr.shape[:2]
        morphed = img_bgr.copy()
        intensity = float(np.clip(intensity, 0.1, 1.0))

        if user_roi and len(user_roi) == 4:
            cx = (user_roi[0] + user_roi[2]) // 2
            cy = (user_roi[1] + user_roi[3]) // 2
            radius = max(15, min(user_roi[2] - user_roi[0], user_roi[3] - user_roi[1]) // 2)
        else:
            cx = int(w * 0.65)
            cy = int(h * 0.60)
            radius = int(min(h, w) * 0.20 * (0.7 + 0.3 * intensity))

        stamp_layer = np.zeros((h, w, 4), dtype=np.uint8)
        stamp_color = (25, 20, 180)  # Crimson red ink BGR

        # 1. Concentric security rings
        cv2.circle(stamp_layer, (cx, cy), radius, (*stamp_color, int(230 * intensity)), 3)
        cv2.circle(stamp_layer, (cx, cy), max(2, radius - 6), (*stamp_color, int(170 * intensity)), 1)
        cv2.circle(stamp_layer, (cx, cy), max(2, radius - 18), (*stamp_color, int(200 * intensity)), 2)

        # 2. Central approved text & validation seals
        font_scale = max(0.25, radius / 110.0)
        cv2.putText(stamp_layer, 'APPROVED', (cx - int(36 * font_scale / 0.42), cy - 2), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (*stamp_color, int(240 * intensity)), 2)
        cv2.putText(stamp_layer, 'AUTHENTICATED', (cx - int(44 * font_scale / 0.42), cy + int(14 * font_scale / 0.42)), cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.75, (*stamp_color, int(210 * intensity)), 1)

        # Security Stars around perimeter
        for ang in [0, 60, 120, 180, 240, 300]:
            sx = int(cx + (radius - 12) * np.cos(np.radians(ang)))
            sy = int(cy + (radius - 12) * np.sin(np.radians(ang)))
            if 0 <= sx < w and 0 <= sy < h:
                cv2.circle(stamp_layer, (sx, sy), 2, (*stamp_color, int(230 * intensity)), -1)

        # 3. Simulate realistic hand-stamped tilt angle (4.2 degrees)
        rot_mat = cv2.getRotationMatrix2D((float(cx), float(cy)), 4.2, 1.0)
        stamp_layer = cv2.warpAffine(stamp_layer, rot_mat, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))

        # 4. Uneven mechanical stamp pressure (linear gradient across the circle)
        y_g, x_g = np.ogrid[:h, :w]
        pressure_grad = np.clip(0.60 + 0.40 * ((x_g - cx) * 0.7 + (y_g - cy) * 0.7) / (2.0 * radius + 1e-5), 0.40, 1.0)
        stamp_alpha = (stamp_layer[:, :, 3].astype(np.float32) / 255.0) * pressure_grad

        # Ink capillary paper grain bleed
        np.random.seed(42)
        bleed_noise = np.random.normal(0, 0.06, (h, w)).astype(np.float32)
        stamp_alpha = np.clip(stamp_alpha + bleed_noise * (stamp_alpha > 0.04), 0.0, 1.0)

        # 5. PHYSICAL SUBTRACTIVE INK MULTIPLY BLENDING
        # Real ink subtracts luminance from the paper fibers; text underneath remains visible
        ink_weight = stamp_alpha[:, :, None] * (0.80 + 0.20 * intensity)
        subtractive_factor = 1.0 - ink_weight * (1.0 - np.array(stamp_color, dtype=np.float32) / 255.0)
        morphed = np.clip(morphed.astype(np.float32) * subtractive_factor, 0, 255).astype(np.uint8)

        diff_raw = cv2.absdiff(morphed, img_bgr)
        diff_gray = cv2.cvtColor(diff_raw, cv2.COLOR_BGR2GRAY)
        diff_max = float(np.max(diff_gray))
        if diff_max > 0:
            diff_enhanced = np.clip((diff_gray.astype(np.float32) / diff_max) * 255.0 * 1.5, 0, 255).astype(np.uint8)
        else:
            diff_enhanced = np.zeros_like(diff_gray)

        mask_binary = (diff_gray >= 6).astype(np.uint8) * 255
        bx1 = max(0, cx - radius - 8)
        by1 = max(0, cy - radius - 8)
        bx2 = min(w, cx + radius + 8)
        by2 = min(h, cy + radius + 8)
        bbox_norm = [round(bx1 / w, 4), round(by1 / h, 4), round(bx2 / w, 4), round(by2 / h, 4)]
        return morphed, bbox_norm, mask_binary, diff_enhanced

    @staticmethod
    def synthesize_signature_forgery(
        img_bgr: np.ndarray,
        intensity: float = 1.0,
        user_roi: list[int] = None
    ) -> tuple[np.ndarray, list[float], np.ndarray, np.ndarray]:
        """
        Synthesizes a realistic cursive pen signature forgery:
        - Ballistic fluid velocity variation with thick downstrokes & thin upstrokes
        - Multi-stroke natural cursive trajectory & terminal loop flourish
        - Subtractive ink multiply absorption into document paper fibers
        Returns: (morphed_bgr, bbox_norm, mask_binary, diff_enhanced)
        """
        h, w = img_bgr.shape[:2]
        morphed = img_bgr.copy()
        intensity = float(np.clip(intensity, 0.1, 1.0))

        if user_roi and len(user_roi) == 4:
            bx, by = user_roi[0], (user_roi[1] + user_roi[3]) // 2
            sig_w = max(30, user_roi[2] - user_roi[0])
        else:
            bx, by = int(w * 0.20), int(h * 0.78)
            sig_w = int(w * 0.45)

        sig_layer = np.zeros((h, w), dtype=np.uint8)
        ink_color = (145, 45, 22)  # Royal blue ballpoint / fountain pen ink BGR

        # 1. Capital initial flourish loop (e.g. elegant cursive initial)
        loop_r = max(6, int(sig_w * 0.08))
        cv2.ellipse(sig_layer, (bx + loop_r, by - loop_r // 2), (loop_r, int(loop_r * 1.3)), -15, 0, 360, 255, 2, cv2.LINE_AA)

        # 2. Main cursive body stroke with velocity-based variable thickness
        sig_pts = []
        np.random.seed(99)
        cur_x, cur_y = bx + loop_r * 2, by
        num_steps = 22
        step_x = max(3, (sig_w - loop_r * 2) // num_steps)

        for i in range(num_steps):
            prev_y = cur_y
            cur_x += int(step_x + 3 * np.cos(i * 0.8))
            cur_y = int(by + np.sin(i * 1.4) * 12 * intensity + np.cos(i * 0.7) * 4)
            # Downstrokes (cur_y > prev_y) are thicker (higher pen pressure)
            thickness = 3 if cur_y > prev_y else 2
            sig_pts.append([cur_x, cur_y])
            if i > 0:
                cv2.line(sig_layer, tuple(sig_pts[-2]), (cur_x, cur_y), 255, thickness, cv2.LINE_AA)

        # 3. Terminal horizontal underline flourish
        end_x, end_y = sig_pts[-1]
        cv2.line(sig_layer, (bx, by + 12), (end_x + 10, by + 10), 255, 1, cv2.LINE_AA)
        cv2.circle(sig_layer, (end_x + 12, by + 10), 2, 255, -1)

        # 4. Subtractive ink multiply blending
        sig_alpha = cv2.GaussianBlur(sig_layer.astype(np.float32) / 255.0, (3, 3), 0.4)
        ink_weight = sig_alpha[:, :, None] * (0.85 + 0.15 * intensity)
        subtractive_factor = 1.0 - ink_weight * (1.0 - np.array(ink_color, dtype=np.float32) / 255.0)
        morphed = np.clip(morphed.astype(np.float32) * subtractive_factor, 0, 255).astype(np.uint8)

        diff_raw = cv2.absdiff(morphed, img_bgr)
        diff_gray = cv2.cvtColor(diff_raw, cv2.COLOR_BGR2GRAY)
        diff_max = float(np.max(diff_gray))
        if diff_max > 0:
            diff_enhanced = np.clip((diff_gray.astype(np.float32) / diff_max) * 255.0 * 1.5, 0, 255).astype(np.uint8)
        else:
            diff_enhanced = np.zeros_like(diff_gray)

        mask_binary = (diff_gray >= 6).astype(np.uint8) * 255
        nonzero_pts = cv2.findNonZero(mask_binary)
        if nonzero_pts is not None:
            sbx, sby, sbw, sbh = cv2.boundingRect(nonzero_pts)
            bx1, by1, bx2, by2 = sbx, sby, sbx + sbw, sby + sbh
        else:
            bx1, by1, bx2, by2 = bx, by - 20, cur_x, by + 20

        bbox_norm = [round(bx1 / w, 4), round(by1 / h, 4), round(bx2 / w, 4), round(by2 / h, 4)]
        return morphed, bbox_norm, mask_binary, diff_enhanced

    @staticmethod
    def synthesize_contextual_inpaint(
        img_bgr: np.ndarray,
        intensity: float = 1.0,
        user_roi: list[int] = None
    ) -> tuple[np.ndarray, list[float], np.ndarray, np.ndarray]:
        """
        Synthesizes generative structural object removal / contextual inpainting:
        - Navier-Stokes structural propagation
        - Laplacian background sensor noise matching
        - Feathered boundary preservation
        Returns: (morphed_bgr, bbox_norm, mask_binary, diff_enhanced)
        """
        h, w = img_bgr.shape[:2]
        morphed = img_bgr.copy()
        intensity = float(np.clip(intensity, 0.1, 1.0))

        if user_roi and len(user_roi) == 4:
            rx1, ry1, rx2, ry2 = user_roi
            rw = max(10, rx2 - rx1)
            rh = max(10, ry2 - ry1)
        else:
            rx1 = int(w * 0.35)
            ry1 = int(h * 0.38)
            rw = int(w * 0.30)
            rh = int(h * 0.28)
            rx2 = rx1 + rw
            ry2 = ry1 + rh

        inpaint_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.ellipse(inpaint_mask, (rx1 + rw // 2, ry1 + rh // 2), (rw // 2, rh // 2), 0, 0, 360, 255, -1)

        inpainted = cv2.inpaint(morphed, inpaint_mask, 5, cv2.INPAINT_NS)

        # Estimate sensor noise from surrounding region
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        lap = cv2.Laplacian(gray, cv2.CV_32F)
        noise_sigma = max(2.0, min(12.0, float(np.std(lap[:30, :30]))))

        np.random.seed(42)
        synth_noise = np.random.normal(0, max(8.0, noise_sigma * 0.35), inpainted.shape).astype(np.float32)
        mask_float = inpaint_mask.astype(np.float32)[:, :, None] / 255.0
        inpainted = np.clip(inpainted.astype(np.float32) + synth_noise * mask_float, 0, 255).astype(np.uint8)

        morphed = cv2.addWeighted(inpainted, intensity, morphed, 1.0 - intensity, 0)

        diff_raw = cv2.absdiff(morphed, img_bgr)
        diff_gray = cv2.cvtColor(diff_raw, cv2.COLOR_BGR2GRAY)
        diff_max = float(np.max(diff_gray))
        if diff_max > 0:
            diff_enhanced = np.clip((diff_gray.astype(np.float32) / diff_max) * 255.0 * 1.5, 0, 255).astype(np.uint8)
        else:
            diff_enhanced = np.zeros_like(diff_gray)

        mask_binary = inpaint_mask.copy()
        bbox_norm = [round(rx1 / w, 4), round(ry1 / h, 4), round(rx2 / w, 4), round(ry2 / h, 4)]
        return morphed, bbox_norm, mask_binary, diff_enhanced

    @staticmethod
    def synthesize_document_tamper(
        img_bgr: np.ndarray,
        intensity: float = 1.0,
        user_roi: list[int] = None
    ) -> tuple[np.ndarray, list[float], np.ndarray, np.ndarray]:
        """
        Synthesizes an authentic financial document / invoice / contract forgery:
        - Detects or accepts ROI for numeric field / amount / date alteration
        - Erases underlying ink using Telea inpainting so paper fibers, folds, and aging remain 100% continuous
        - Renders altered monetary amount ("$ 18,750.00") in print-aligned monospace
        - Applies physical toner dot-matrix scatter, micro-blur, and paper subtractive blending
        Returns: (morphed_bgr, bbox_norm, mask_binary, diff_enhanced)
        """
        h, w = img_bgr.shape[:2]
        morphed = img_bgr.copy()
        intensity = float(np.clip(intensity, 0.1, 1.0))

        if user_roi and len(user_roi) == 4:
            rx1, ry1, rx2, ry2 = user_roi
        else:
            rx1 = int(w * 0.50)
            ry1 = int(h * 0.65)
            rx2 = int(w * 0.85)
            ry2 = int(h * 0.73)

        box_w = max(40, rx2 - rx1)
        box_h = max(16, ry2 - ry1)

        # 1. Seamlessly erase existing text within ROI using Telea inpainting
        # This keeps the authentic paper fibers, folds, and aged color gradient 100% intact!
        roi = morphed[ry1:ry1 + box_h, rx1:rx1 + box_w]
        roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # Adaptive thresholding isolates ink strokes
        thresh = cv2.adaptiveThreshold(roi_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 6)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        thresh_dilated = cv2.dilate(thresh, kernel, iterations=1)
        
        erased_roi = cv2.inpaint(roi, thresh_dilated, 3, cv2.INPAINT_TELEA)
        morphed[ry1:ry1 + box_h, rx1:rx1 + box_w] = erased_roi

        # 2. Print altered monetary amount in crisp printed typography
        text_str = "$ 18,750.00"
        font_scale = max(0.35, min(0.75, box_h / 32.0))
        text_size, _ = cv2.getTextSize(text_str, cv2.FONT_HERSHEY_DUPLEX, font_scale, 1)
        tx = rx1 + (box_w - text_size[0]) // 2
        ty = ry1 + (box_h + text_size[1]) // 2 - 2

        text_layer = np.zeros((h, w), dtype=np.uint8)
        cv2.putText(text_layer, text_str, (tx, ty), cv2.FONT_HERSHEY_DUPLEX, font_scale, 255, 1, cv2.LINE_AA)

        # 3. Realistic printer toner feathering & scanner micro-blur
        toner_alpha = cv2.GaussianBlur(text_layer.astype(np.float32) / 255.0, (3, 3), 0.5)
        # Subtractive ink / toner multiply blending (dark charcoal printer toner)
        toner_color = np.array([25, 25, 28], dtype=np.float32)
        ink_weight = toner_alpha[:, :, None] * (0.90 * intensity)
        subtractive_factor = 1.0 - ink_weight * (1.0 - toner_color / 255.0)
        morphed = np.clip(morphed.astype(np.float32) * subtractive_factor, 0, 255).astype(np.uint8)

        diff_raw = cv2.absdiff(morphed, img_bgr)
        diff_gray = cv2.cvtColor(diff_raw, cv2.COLOR_BGR2GRAY)
        diff_max = float(np.max(diff_gray))
        if diff_max > 0:
            diff_enhanced = np.clip((diff_gray.astype(np.float32) / diff_max) * 255.0 * 1.5, 0, 255).astype(np.uint8)
        else:
            diff_enhanced = np.zeros_like(diff_gray)

        mask_binary = (diff_gray >= 5).astype(np.uint8) * 255
        bbox_norm = [round(rx1 / w, 4), round(ry1 / h, 4), round((rx1 + box_w) / w, 4), round((ry1 + box_h) / h, 4)]
        return morphed, bbox_norm, mask_binary, diff_enhanced

    @staticmethod
    def synthesize_object_insertion(
        img_bgr: np.ndarray,
        intensity: float = 1.0,
        user_roi: list[int] = None
    ) -> tuple[np.ndarray, list[float], np.ndarray, np.ndarray]:
        """
        Synthesizes a harmonized object insertion (aerial drone/vehicle) with directional contact shadow:
        - Structural drone fuselage and quad rotors with spinning disc motion blur
        - Ambient scene illumination & color temperature matching
        - Directional soft contact shadow (only over solid ground/water, not empty sky)
        Returns: (morphed_bgr, bbox_norm, mask_binary, diff_enhanced)
        """
        h, w = img_bgr.shape[:2]
        morphed = img_bgr.copy()
        intensity = float(np.clip(intensity, 0.1, 1.0))

        if user_roi and len(user_roi) == 4:
            ox = (user_roi[0] + user_roi[2]) // 2
            oy = (user_roi[1] + user_roi[3]) // 2
        else:
            ox = int(w * 0.65)
            oy = int(h * 0.25)

        scale = max(0.6, min(h, w) / 400.0)
        bw, bh = int(22 * scale), int(8 * scale)

        # 1. Sample ambient scene lighting around target position
        roi_ambient = img_bgr[max(0, oy - int(30 * scale)):min(h, oy + int(30 * scale)), max(0, ox - int(40 * scale)):min(w, ox + int(40 * scale))]
        ambient_mean = np.mean(roi_ambient, axis=(0, 1)) if roi_ambient.size > 0 else np.array([120, 120, 120])
        ambient_lum = float(np.mean(ambient_mean)) / 128.0
        ambient_tint = ambient_mean / (np.mean(ambient_mean) + 1e-5)

        # 2. Check if region underneath is open sky
        # If open sky, do not project an unnatural ground shadow in the air!
        shadow_ox = ox + int(6 * scale)
        shadow_oy = oy + int(14 * scale)
        roi_under = img_bgr[min(h - 1, shadow_oy):min(h, shadow_oy + int(bh * 2)), max(0, ox - bw):min(w, ox + bw)]
        is_open_sky = False
        if roi_under.size > 0:
            b_u, g_u, r_u = np.mean(roi_under, axis=(0, 1))
            # Sky test: high brightness, blue-dominant
            if (b_u > r_u * 1.15 or b_u > 180) and (b_u + g_u + r_u) / 3.0 > 130:
                is_open_sky = True

        if not is_open_sky and shadow_oy + int(bh * 0.85) < h:
            shadow_layer = np.zeros((h, w), dtype=np.uint8)
            cv2.ellipse(shadow_layer, (shadow_ox, shadow_oy), (int(bw * 1.15), int(bh * 0.85)), 12, 0, 360, 255, -1)
            shadow_feather = cv2.GaussianBlur(shadow_layer.astype(np.float32), (15, 15), 3.5) / 255.0
            shadow_3ch = np.dstack([shadow_feather * 0.45 * intensity] * 3)
            morphed = (morphed.astype(np.float32) * (1.0 - shadow_3ch)).astype(np.uint8)

        # 3. Structural drone fuselage and rotors (tinted by ambient lighting)
        drone_layer = np.zeros_like(img_bgr)
        base_body_color = np.clip(np.array([50, 52, 56]) * ambient_tint * ambient_lum, 25, 95).astype(np.uint8).tolist()
        arm_color = np.clip(np.array([75, 78, 82]) * ambient_tint * ambient_lum, 35, 120).astype(np.uint8).tolist()
        blade_color = np.clip(np.array([180, 185, 190]) * ambient_tint * ambient_lum, 100, 220).astype(np.uint8).tolist()

        # Fuselage body
        cv2.ellipse(drone_layer, (ox, oy), (bw, bh), 0, 0, 360, base_body_color, -1)
        cv2.circle(drone_layer, (ox, oy), max(2, int(4 * scale)), (25, 25, 28), -1)
        # Rotor arms
        arm_span = int(20 * scale)
        cv2.line(drone_layer, (ox - arm_span, oy - 2), (ox + arm_span, oy - 2), arm_color, max(1, int(2 * scale)))
        # Rotors & spinning blade discs (motion blur effect)
        blade_r = max(4, int(8 * scale))
        cv2.ellipse(drone_layer, (ox - arm_span, oy - 4), (blade_r, max(1, int(2 * scale))), 0, 0, 360, blade_color, -1)
        cv2.ellipse(drone_layer, (ox + arm_span, oy - 4), (blade_r, max(1, int(2 * scale))), 0, 0, 360, blade_color, -1)
        # Navigation red and green LED beacons
        cv2.circle(drone_layer, (ox - arm_span, oy - 2), max(1, int(2 * scale)), (20, 20, 240), -1)
        cv2.circle(drone_layer, (ox + arm_span, oy - 2), max(1, int(2 * scale)), (20, 230, 40), -1)

        drone_mask = (cv2.cvtColor(drone_layer, cv2.COLOR_BGR2GRAY) > 10).astype(np.uint8) * 255
        drone_feather = cv2.GaussianBlur(drone_mask.astype(np.float32), (3, 3), 0.7) / 255.0
        drone_3ch = np.dstack([drone_feather * intensity] * 3)

        morphed = (drone_layer.astype(np.float32) * drone_3ch + morphed.astype(np.float32) * (1.0 - drone_3ch)).astype(np.uint8)

        diff_raw = cv2.absdiff(morphed, img_bgr)
        diff_gray = cv2.cvtColor(diff_raw, cv2.COLOR_BGR2GRAY)
        diff_max = float(np.max(diff_gray))
        if diff_max > 0:
            diff_enhanced = np.clip((diff_gray.astype(np.float32) / diff_max) * 255.0 * 1.5, 0, 255).astype(np.uint8)
        else:
            diff_enhanced = np.zeros_like(diff_gray)

        mask_binary = (diff_gray >= 6).astype(np.uint8) * 255
        bx1 = max(0, ox - arm_span - blade_r - 2)
        by1 = max(0, oy - bh - 6)
        bx2 = min(w, ox + arm_span + blade_r + 2)
        by2 = min(h, shadow_oy + int(bh * 0.85) + 4)
        bbox_norm = [round(bx1 / w, 4), round(by1 / h, 4), round(bx2 / w, 4), round(by2 / h, 4)]
        return morphed, bbox_norm, mask_binary, diff_enhanced

    @staticmethod
    def synthesize_edit_cv(img_bgr: np.ndarray, domain_name: str = "Natural Scene") -> tuple[np.ndarray, list[float], np.ndarray]:
        """Backward-compatible edit wrapper."""
        if "Document" in domain_name or "Regular" in domain_name:
            morphed, bbox, mask_binary, _ = GeneralGenerator.synthesize_notary_stamp_forgery(img_bgr)
        else:
            morphed, bbox, mask_binary, _ = GeneralGenerator.synthesize_contextual_inpaint(img_bgr)
        return morphed, bbox, mask_binary

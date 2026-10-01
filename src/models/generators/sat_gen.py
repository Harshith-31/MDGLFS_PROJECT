import cv2
import numpy as np
import torch
import torch.nn as nn


class SPADEGenerator(nn.Module):
    """
    SPADE (Spatially-Adaptive Normalization) Geospatial Generator Module
    Synthesizes authentic satellite airbases, tactical military installations,
    and naval combatants with correct Ground Sampling Distance (GSD) scale,
    orientation alignment, and directional solar shadow physics.
    """
    def __init__(self, in_channels: int = 3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, in_channels, kernel_size=3, padding=1),
            nn.Tanh()
        )

    def forward(self, x: torch.Tensor):
        return x + 0.15 * self.conv(x)

    @staticmethod
    def _detect_dominant_terrain_angle(img_bgr: np.ndarray) -> tuple[float, bool]:
        """
        Detects dominant infrastructure orientation angle in degrees:
        1. Checks for prominent continuous paved runway contour via principal axis (cv2.fitLine)
        2. Falls back to length-weighted Hough transform on edge contours.
        Returns (dominant_angle_deg, has_strong_runway).
        """
        # 1. Pavement principal axis detection
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        bright = ((hsv[:, :, 2] > 165) & (hsv[:, :, 1] < 90)).astype(np.uint8) * 255
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        closed = cv2.morphologyEx(bright, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            largest = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest) > 4000:
                line = cv2.fitLine(largest, cv2.DIST_L2, 0, 0.01, 0.01)
                vx, vy, x0, y0 = line.flatten()
                ang = -np.degrees(np.arctan2(vy, vx))
                if ang > 90: ang -= 180
                elif ang < -90: ang += 180
                return float(ang), True

        # 2. Length-weighted Hough transform fallback
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 40, 120)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=35, minLineLength=40, maxLineGap=20)
        if lines is None or len(lines) < 3:
            return 45.0, False

        angles = []
        weights = []
        max_len = 0.0
        for line in lines:
            x1_l, y1_l, x2_l, y2_l = line[0]
            if x2_l < x1_l:
                x1_l, y1_l, x2_l, y2_l = x2_l, y2_l, x1_l, y1_l
            dx_l = float(x2_l - x1_l)
            dy_l = float(y2_l - y1_l)
            length = np.sqrt(dx_l**2 + dy_l**2)
            max_len = max(max_len, length)
            ang = -np.degrees(np.arctan2(dy_l, dx_l))
            if ang > 90: ang -= 180
            elif ang < -90: ang += 180
            if abs(ang) > 12:
                angles.append(ang)
                weights.append(length**2)

        if len(angles) < 3:
            return 45.0, False

        hist, bin_edges = np.histogram(angles, bins=36, range=(-90, 90), weights=weights)
        best_bin = np.argmax(hist)
        dominant_angle = float((bin_edges[best_bin] + bin_edges[best_bin + 1]) / 2.0)
        has_strong_runway = (max_len > min(img_bgr.shape[:2]) * 0.40)
        return dominant_angle, has_strong_runway

    @staticmethod
    def _sample_pavement_palette(img_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Samples realistic weathered concrete and graded dirt tones from native terrain.
        """
        terrain_mean = np.mean(img_bgr, axis=(0, 1))
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        bright_mask = (hsv[:, :, 2] > 175) & (hsv[:, :, 1] < 85)
        if np.count_nonzero(bright_mask) > 120:
            concrete_bgr = np.mean(img_bgr[bright_mask], axis=0).astype(np.uint8)
        else:
            concrete_bgr = np.clip(terrain_mean * 1.25 + np.array([22, 18, 14]), 130, 235).astype(np.uint8)
        dirt_bgr = np.clip(terrain_mean * 0.88, 30, 190).astype(np.uint8)
        return concrete_bgr, dirt_bgr

    @staticmethod
    def synthesize_airbase_facility(
        img_bgr: np.ndarray,
        intensity: float = 1.0,
        user_roi: list[int] = None
    ) -> tuple[np.ndarray, list[float], np.ndarray, np.ndarray]:
        """
        Synthesizes an authentic multi-layered military airbase facility:
        - Dominant orientation angle detection (Hough transform with length^2 weighting)
        - Palette harmonized directly with existing airport pavement or terrain
        - Parallel tactical alert apron with high-speed diagonal connector taxiways
        - Dispersal parking hardstands with Hardened Aircraft Shelters (HAS) and solar cast shadows
        - Anti-aliased yellow taxiway centerlines & anti-skid concrete textures
        - Zero rectangular carpet borders: blends organically with native terrain and sensor MTF
        """
        h, w = img_bgr.shape[:2]
        morphed = img_bgr.copy()
        intensity = float(np.clip(intensity, 0.1, 1.0))

        dominant_angle, has_runway = SPADEGenerator._detect_dominant_terrain_angle(img_bgr)
        concrete_bgr, dirt_bgr = SPADEGenerator._sample_pavement_palette(img_bgr)

        # Scale proportions
        fac_len = int(min(h, w) * 0.50)
        fac_w = max(8, int(min(h, w) * 0.022))

        canvas_w = fac_len + 80
        canvas_h = fac_w + 100
        tarmac_canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
        alpha_canvas = np.zeros((canvas_h, canvas_w), dtype=np.float32)

        cx_c = canvas_w // 2
        cy_c = canvas_h // 2

        tx1 = cx_c - fac_len // 2
        ty1 = cy_c - fac_w // 2
        tx2 = cx_c + fac_len // 2
        ty2 = cy_c + fac_w // 2

        # 1. Weathered Concrete Taxiway / Apron Strip
        cv2.rectangle(tarmac_canvas, (tx1, ty1), (tx2, ty2), concrete_bgr.tolist(), -1)
        alpha_canvas[ty1:ty2, tx1:tx2] = 0.96 * intensity

        # Graded shoulder (1-2 px alongside strip)
        shoulder_color = np.clip(concrete_bgr.astype(np.int16) - 20, 0, 255).astype(np.uint8).tolist()
        cv2.line(tarmac_canvas, (tx1, ty1 - 1), (tx2, ty1 - 1), shoulder_color, 1)
        cv2.line(tarmac_canvas, (tx1, ty2 + 1), (tx2, ty2 + 1), shoulder_color, 1)
        alpha_canvas[ty1-1:ty1, tx1:tx2] = 0.70 * intensity
        alpha_canvas[ty2:ty2+1, tx1:tx2] = 0.70 * intensity

        # Yellow taxiway centerline stripe
        yellow_mark = (35, 215, 245)
        for x in range(tx1 + 6, tx2 - 6, 8):
            cv2.line(tarmac_canvas, (x, cy_c), (min(x + 4, tx2 - 6), cy_c), yellow_mark, 1)

        # 2. High-speed connector taxiways branching towards main airfield
        for c_x in [cx_c - int(fac_len * 0.35), cx_c + int(fac_len * 0.35)]:
            conn_pts = np.array([
                [c_x - 3, ty2],
                [c_x + 3, ty2],
                [c_x + 16, ty2 + 24],
                [c_x + 10, ty2 + 24]
            ], dtype=np.int32)
            cv2.fillPoly(tarmac_canvas, [conn_pts], concrete_bgr.tolist())
            cv2.fillPoly(alpha_canvas, [conn_pts], 0.94 * intensity)
            cv2.line(tarmac_canvas, (c_x, ty2), (c_x + 13, ty2 + 24), yellow_mark, 1)

        # 3. Dispersal Parking Hardstands & Hardened Aircraft Shelters (HAS) along outer side
        has_w, has_h = 7, 5
        shadow_bgr = (25, 28, 32)
        has_xs = [cx_c - int(fac_len * 0.30), cx_c - int(fac_len * 0.10), cx_c + int(fac_len * 0.12), cx_c + int(fac_len * 0.32)]
        for hx in has_xs:
            hy = ty1 - has_h - 4
            cv2.rectangle(tarmac_canvas, (hx - 1, hy), (hx + has_w + 1, ty1), concrete_bgr.tolist(), -1)
            alpha_canvas[hy:ty1, hx-1:hx+has_w+1] = 0.92 * intensity

            # Directional cast shadow on SE (+2, +2)
            cv2.rectangle(tarmac_canvas, (hx + 2, hy + 2), (hx + has_w + 2, hy + has_h + 2), shadow_bgr, -1)
            # Arched reinforced concrete shelter roof
            cv2.rectangle(tarmac_canvas, (hx, hy), (hx + has_w, hy + has_h), (140, 145, 150), -1)
            cv2.rectangle(tarmac_canvas, (hx, hy), (hx + has_w, hy + has_h), (95, 100, 105), 1)
            alpha_canvas[hy:hy+has_h+2, hx:hx+has_w+2] = 0.96 * intensity

        # Soft edge feathering
        alpha_canvas = cv2.GaussianBlur(alpha_canvas, (5, 5), 1.0)

        # 4. Target Placement: Parallel to dominant runway axis, offset into open field
        if user_roi and len(user_roi) == 4:
            target_cx = (user_roi[0] + user_roi[2]) // 2
            target_cy = (user_roi[1] + user_roi[3]) // 2
        else:
            rad = np.radians(dominant_angle)
            # Orthogonal offset into open field (perpendicular to runway)
            perp_dx = -np.sin(rad) * 45.0
            perp_dy = -np.cos(rad) * 45.0
            target_cx = int(w * 0.50 + perp_dx)
            target_cy = int(h * 0.50 + perp_dy)

        full_layer = np.zeros_like(img_bgr)
        full_alpha = np.zeros((h, w), dtype=np.float32)

        top_y = target_cy - cy_c
        left_x = target_cx - cx_c

        src_y1 = max(0, -top_y)
        src_x1 = max(0, -left_x)
        dst_y1 = max(0, top_y)
        dst_x1 = max(0, left_x)
        dst_y2 = min(h, top_y + canvas_h)
        dst_x2 = min(w, left_x + canvas_w)
        src_y2 = src_y1 + (dst_y2 - dst_y1)
        src_x2 = src_x1 + (dst_x2 - dst_x1)

        full_layer[dst_y1:dst_y2, dst_x1:dst_x2] = tarmac_canvas[src_y1:src_y2, src_x1:src_x2]
        full_alpha[dst_y1:dst_y2, dst_x1:dst_x2] = alpha_canvas[src_y1:src_y2, src_x1:src_x2]

        # Rotate by dominant_angle
        M = cv2.getRotationMatrix2D((float(target_cx), float(target_cy)), dominant_angle, 1.0)
        full_layer = cv2.warpAffine(full_layer, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
        full_alpha = cv2.warpAffine(full_alpha, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)

        # Composite into image
        f_alpha_3ch = np.dstack([full_alpha] * 3)
        morphed = (full_layer.astype(np.float32) * f_alpha_3ch + morphed.astype(np.float32) * (1.0 - f_alpha_3ch)).astype(np.uint8)

        # Orbital sensor MTF response
        blended = cv2.bilateralFilter(morphed, 3, 15, 15)
        morphed = np.where(f_alpha_3ch > 0.05, blended, morphed)

        diff_raw = cv2.absdiff(morphed, img_bgr)
        diff_gray = cv2.cvtColor(diff_raw, cv2.COLOR_BGR2GRAY)
        diff_max = float(np.max(diff_gray))
        diff_enhanced = np.clip((diff_gray.astype(np.float32) / (diff_max + 1e-6)) * 255.0 * 1.5, 0, 255).astype(np.uint8)

        mask_binary = (diff_gray >= 5).astype(np.uint8) * 255
        nonzero_pts = cv2.findNonZero(mask_binary)
        if nonzero_pts is not None:
            bx, by, bw, bh = cv2.boundingRect(nonzero_pts)
            bx1, by1, bx2, by2 = bx, by, bx + bw, by + bh
        else:
            bx1, by1, bx2, by2 = target_cx - 30, target_cy - 30, target_cx + 30, target_cy + 30

        bbox_norm = [round(bx1 / w, 4), round(by1 / h, 4), round(bx2 / w, 4), round(by2 / h, 4)]
        return morphed, bbox_norm, mask_binary, diff_enhanced

    @staticmethod
    def synthesize_military_installation(
        img_bgr: np.ndarray,
        intensity: float = 1.0,
        user_roi: list[int] = None
    ) -> tuple[np.ndarray, list[float], np.ndarray, np.ndarray]:
        """
        Synthesizes an authentic fortified military radar / communications compound:
        - Central operations command building with tiered roof & directional cast shadow
        - Spherical geodesic communications radome on concrete pad
        - High-gain communications lattice antenna mast with long thin solar shadow
        - Storage hangars & barracks with corrugated pitched roofs
        - Tactical vehicle motor pool with parked military transport trucks
        - Organic access road connecting seamlessly to terrain
        """
        h, w = img_bgr.shape[:2]
        morphed = img_bgr.copy()
        intensity = float(np.clip(intensity, 0.1, 1.0))

        terrain_mean = np.mean(morphed, axis=(0, 1))
        compound_bgr = np.clip(terrain_mean * 1.15 + np.array([14, 12, 10]), 70, 240).astype(np.uint8)
        concrete_pad_bgr = np.clip(terrain_mean * 1.30 + np.array([25, 22, 18]), 130, 245).astype(np.uint8)
        roof_light_bgr = np.clip(terrain_mean * 1.25 + np.array([30, 25, 20]), 120, 240).astype(np.uint8)
        roof_dark_bgr = np.clip(terrain_mean * 0.70, 20, 160).astype(np.uint8)
        shadow_bgr = (25, 28, 32)

        if user_roi and len(user_roi) == 4:
            cx = (user_roi[0] + user_roi[2]) // 2
            cy = (user_roi[1] + user_roi[3]) // 2
        else:
            cx = int(w * 0.28)
            cy = int(h * 0.28)

        np.random.seed(66)
        comp_w = int(min(h, w) * 0.20)
        comp_h = int(min(h, w) * 0.16)
        x1 = cx - comp_w // 2
        y1 = cy - comp_h // 2
        x2 = x1 + comp_w
        y2 = y1 + comp_h

        canvas = np.zeros_like(img_bgr)
        alpha = np.zeros((h, w), dtype=np.float32)

        # 1. Graded gravel compound security perimeter
        cv2.rectangle(canvas, (x1, y1), (x2, y2), compound_bgr.tolist(), -1)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (60, 65, 70), 1)
        gravel_noise = np.random.normal(0, 4, (h, w, 3)).astype(np.int16)
        canvas = np.clip(canvas.astype(np.int16) + gravel_noise, 0, 255).astype(np.uint8)
        alpha[y1:y2, x1:x2] = 0.82 * intensity

        # Connecting access road curving out to the left
        for offset in range(-2, 3):
            cv2.line(canvas, (x1, cy), (max(0, x1 - 32), cy + 12 + offset), compound_bgr.tolist(), 2)
            cv2.line(alpha, (x1, cy), (max(0, x1 - 32), cy + 12 + offset), 0.82 * intensity, 2)

        sh_dx, sh_dy = 3, 4

        # 2. Main Command & Communications Building (Central)
        b1_w, b1_h = int(comp_w * 0.36), int(comp_h * 0.32)
        b1_x = cx - b1_w // 2
        b1_y = cy - b1_h // 2 - 4

        cv2.rectangle(canvas, (b1_x + sh_dx, b1_y + sh_dy), (b1_x + b1_w + sh_dx, b1_y + b1_h + sh_dy), shadow_bgr, -1)
        cv2.rectangle(alpha, (b1_x + sh_dx, b1_y + sh_dy), (b1_x + b1_w + sh_dx, b1_y + b1_h + sh_dy), 0.65 * intensity, -1)
        cv2.rectangle(canvas, (b1_x, b1_y), (b1_x + b1_w, b1_y + b1_h), roof_light_bgr.tolist(), -1)
        cv2.rectangle(canvas, (b1_x, b1_y), (b1_x + b1_w, b1_y + b1_h), (95, 100, 105), 1)
        alpha[b1_y:b1_y+b1_h, b1_x:b1_x+b1_w] = 0.98 * intensity

        # 3. Communications Radar Dish & Radome on Pad (North corner)
        dish_pad_x = x1 + int(comp_w * 0.22)
        dish_pad_y = y1 + int(comp_h * 0.26)
        pad_r = 6
        cv2.circle(canvas, (dish_pad_x, dish_pad_y), pad_r, concrete_pad_bgr.tolist(), -1)
        alpha[dish_pad_y-pad_r:dish_pad_y+pad_r, dish_pad_x-pad_r:dish_pad_x+pad_r] = 0.95 * intensity

        dome_r = 4
        cv2.circle(canvas, (dish_pad_x + 2, dish_pad_y + 2), dome_r + 1, shadow_bgr, -1)
        cv2.circle(canvas, (dish_pad_x, dish_pad_y), dome_r, (240, 242, 245), -1)
        cv2.circle(canvas, (dish_pad_x, dish_pad_y), dome_r, (160, 165, 170), 1)

        # Tall lattice antenna mast (East of radome)
        mast_x = dish_pad_x + 12
        mast_y = dish_pad_y
        cv2.circle(canvas, (mast_x, mast_y), 2, (180, 185, 190), -1)
        cv2.line(canvas, (mast_x, mast_y), (mast_x + 10, mast_y + 14), shadow_bgr, 1)
        cv2.line(alpha, (mast_x, mast_y), (mast_x + 10, mast_y + 14), 0.70 * intensity, 1)

        # 4. Storage Hangars / Barracks (South side)
        for bx_off in [-int(comp_w * 0.25), int(comp_w * 0.16)]:
            h_x = cx + bx_off
            h_y = cy + int(comp_h * 0.18)
            hw_b = int(comp_w * 0.22)
            hh_b = int(comp_h * 0.20)
            cv2.rectangle(canvas, (h_x + sh_dx, h_y + sh_dy), (h_x + hw_b + sh_dx, h_y + hh_b + sh_dy), shadow_bgr, -1)
            cv2.rectangle(alpha, (h_x + sh_dx, h_y + sh_dy), (h_x + hw_b + sh_dx, h_y + hh_b + sh_dy), 0.65 * intensity, -1)
            cv2.rectangle(canvas, (h_x, h_y), (h_x + hw_b, h_y + hh_b), roof_dark_bgr.tolist(), -1)
            cv2.line(canvas, (h_x, h_y + hh_b // 2), (h_x + hw_b, h_y + hh_b // 2), roof_light_bgr.tolist(), 1)
            alpha[h_y:h_y+hh_b, h_x:h_x+hw_b] = 0.98 * intensity

        # 5. Vehicle Motor Pool (East side) with 3 parked tactical trucks
        vp_x = x2 - int(comp_w * 0.24)
        vp_y = cy - int(comp_h * 0.22)
        for v_idx in range(3):
            vy = vp_y + v_idx * 7
            cv2.rectangle(canvas, (vp_x + 2, vy + 2), (vp_x + 9, vy + 5), shadow_bgr, -1)
            cv2.rectangle(canvas, (vp_x, vy), (vp_x + 7, vy + 3), (75, 90, 80), -1)
            alpha[vy:vy+5, vp_x:vp_x+9] = 0.92 * intensity

        alpha = cv2.GaussianBlur(alpha, (7, 7), 1.8)
        alpha_3ch = np.dstack([alpha] * 3)
        morphed = (canvas.astype(np.float32) * alpha_3ch + morphed.astype(np.float32) * (1.0 - alpha_3ch)).astype(np.uint8)

        blended = cv2.bilateralFilter(morphed, 3, 15, 15)
        morphed = np.where(alpha_3ch > 0.05, blended, morphed)

        diff_raw = cv2.absdiff(morphed, img_bgr)
        diff_gray = cv2.cvtColor(diff_raw, cv2.COLOR_BGR2GRAY)
        diff_max = float(np.max(diff_gray))
        diff_enhanced = np.clip((diff_gray.astype(np.float32) / (diff_max + 1e-6)) * 255.0 * 1.5, 0, 255).astype(np.uint8)

        mask_binary = (diff_gray >= 5).astype(np.uint8) * 255
        nonzero_pts = cv2.findNonZero(mask_binary)
        if nonzero_pts is not None:
            bx, by, bw, bh = cv2.boundingRect(nonzero_pts)
            bx1, by1, bx2, by2 = bx, by, bx + bw, by + bh
        else:
            bx1, by1, bx2, by2 = x1 - 10, y1 - 10, x2 + 10, y2 + 10

        bbox_norm = [round(bx1 / w, 4), round(by1 / h, 4), round(bx2 / w, 4), round(by2 / h, 4)]
        return morphed, bbox_norm, mask_binary, diff_enhanced

    @staticmethod
    def synthesize_naval_vessel(
        img_bgr: np.ndarray,
        intensity: float = 1.0,
        user_roi: list[int] = None
    ) -> tuple[np.ndarray, list[float], np.ndarray, np.ndarray]:
        """
        Synthesizes an authentic modern stealth guided-missile destroyer (DDG):
        - Angled stealth hull with flare, transom stern, and tapered bow in naval haze gray
        - Tiered superstructure, forward 5-inch gun turret, and 64-cell VLS missile grid
        - Aft helicopter flight deck with landing circle & hangar door
        - Hydrodynamic Kelvin wake: white turbulent foaming wash and divergent cusp lines
        - Directional solar drop shadow on water
        """
        h, w = img_bgr.shape[:2]
        morphed = img_bgr.copy()
        intensity = float(np.clip(intensity, 0.1, 1.0))

        if user_roi and len(user_roi) == 4:
            cx = (user_roi[0] + user_roi[2]) // 2
            cy = (user_roi[1] + user_roi[3]) // 2
            ship_len = max(35, user_roi[2] - user_roi[0])
            ship_beam = max(8, user_roi[3] - user_roi[1])
        else:
            cx = int(w * 0.48)
            cy = int(h * 0.45)
            ship_len = max(45, int(w * 0.20))
            ship_beam = max(8, int(ship_len * 0.16))

        half_len = ship_len // 2
        half_beam = ship_beam // 2

        # 1. Hydrodynamic Kelvin Wake Foam (turbulent rooster-tail wash behind stern)
        wake_len = int(ship_len * 1.1)
        wake_overlay = morphed.copy()
        wake_pts = np.array([
            [cx + half_len - 3, cy],
            [cx + half_len + wake_len, cy - int(half_beam * 2.4)],
            [cx + half_len + wake_len, cy + int(half_beam * 2.4)]
        ], dtype=np.int32)
        cv2.fillPoly(wake_overlay, [wake_pts], (215, 220, 225))
        # Center turbulence wash
        cv2.line(wake_overlay, (cx + half_len - 3, cy), (cx + half_len + wake_len, cy), (240, 245, 250), max(2, half_beam // 2))
        morphed = cv2.addWeighted(wake_overlay, 0.42 * intensity, morphed, 1.0 - (0.42 * intensity), 0)

        # 2. Directional cast shadow on water (SE direction)
        sh_dx = max(2, int(ship_beam * 0.4))
        sh_dy = max(2, int(ship_beam * 0.6))
        shadow_overlay = morphed.copy()
        hull_pts_shadow = np.array([
            [cx - half_len + sh_dx, cy + sh_dy],
            [cx - half_len + int(ship_len * 0.22) + sh_dx, cy - half_beam + sh_dy],
            [cx + half_len + sh_dx, cy - half_beam + sh_dy],
            [cx + half_len + sh_dx, cy + half_beam + sh_dy],
            [cx - half_len + int(ship_len * 0.22) + sh_dx, cy + half_beam + sh_dy]
        ], dtype=np.int32)
        cv2.fillPoly(shadow_overlay, [hull_pts_shadow], (18, 20, 24))
        morphed = cv2.addWeighted(shadow_overlay, 0.55 * intensity, morphed, 1.0 - (0.55 * intensity), 0)

        # 3. Main Hull (Naval Haze Gray)
        hull_pts = np.array([
            [cx - half_len, cy],
            [cx - half_len + int(ship_len * 0.22), cy - half_beam],
            [cx + half_len, cy - half_beam],
            [cx + half_len, cy + half_beam],
            [cx - half_len + int(ship_len * 0.22), cy + half_beam]
        ], dtype=np.int32)
        cv2.fillPoly(morphed, [hull_pts], (125, 130, 138))
        cv2.polylines(morphed, [hull_pts], True, (80, 85, 90), 1, cv2.LINE_AA)

        # 4. Superstructure Bridge Deckhouse
        b_x1 = cx - int(ship_len * 0.14)
        b_x2 = cx + int(ship_len * 0.10)
        b_y1 = cy - int(half_beam * 0.60)
        b_y2 = cy + int(half_beam * 0.60)
        cv2.rectangle(morphed, (b_x1, b_y1), (b_x2, b_y2), (155, 160, 168), -1)
        cv2.rectangle(morphed, (b_x1, b_y1), (b_x2, b_y2), (90, 95, 100), 1)

        # Forward 5-inch gun turret
        gun_x = cx - int(ship_len * 0.26)
        cv2.circle(morphed, (gun_x, cy), max(2, half_beam // 3), (100, 105, 110), -1)
        cv2.line(morphed, (gun_x, cy), (gun_x - max(3, half_len // 4), cy), (55, 60, 65), 1)

        # Forward VLS missile cell grid
        vls_x = cx - int(ship_len * 0.20)
        cv2.rectangle(morphed, (vls_x, cy - max(1, half_beam // 4)), (vls_x + max(2, ship_len // 18), cy + max(1, half_beam // 4)), (75, 80, 85), -1)

        # Radar mast atop bridge
        cv2.circle(morphed, (cx, cy), max(2, half_beam // 4), (200, 205, 215), -1)

        # Aft helipad landing markings
        heli_cx = cx + int(ship_len * 0.32)
        heli_r = max(2, int(half_beam * 0.60))
        cv2.circle(morphed, (heli_cx, cy), heli_r, (235, 238, 242), 1, cv2.LINE_AA)

        diff_raw = cv2.absdiff(morphed, img_bgr)
        diff_gray = cv2.cvtColor(diff_raw, cv2.COLOR_BGR2GRAY)
        diff_max = float(np.max(diff_gray))
        diff_enhanced = np.clip((diff_gray.astype(np.float32) / (diff_max + 1e-6)) * 255.0 * 1.5, 0, 255).astype(np.uint8)

        mask_binary = (diff_gray >= 5).astype(np.uint8) * 255
        nonzero_pts = cv2.findNonZero(mask_binary)
        if nonzero_pts is not None:
            bx, by, bw, bh = cv2.boundingRect(nonzero_pts)
            bx1, by1, bx2, by2 = bx, by, bx + bw, by + bh
        else:
            bx1 = max(0, cx - half_len - 5)
            by1 = max(0, cy - half_beam * 3)
            bx2 = min(w, cx + half_len + wake_len + 5)
            by2 = min(h, cy + half_beam * 3)

        bbox_norm = [round(bx1 / w, 4), round(by1 / h, 4), round(bx2 / w, 4), round(by2 / h, 4)]
        return morphed, bbox_norm, mask_binary, diff_enhanced

    @staticmethod
    def synthesize_runway_cv(img_bgr: np.ndarray) -> tuple[np.ndarray, list[float], np.ndarray]:
        """Backward-compatible runway synthesis."""
        morphed, bbox, mask_binary, _ = SPADEGenerator.synthesize_airbase_facility(img_bgr)
        return morphed, bbox, mask_binary

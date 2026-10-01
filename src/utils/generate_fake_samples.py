import os
import cv2
import numpy as np
from src.models.generators.face_gen import FaceGenerator
from src.models.generators.med_gen import MedGenerator
from src.models.generators.sat_gen import SPADEGenerator
from src.models.generators.gen_gen import GeneralGenerator

def create_synthetic_sample_assets():
    os.makedirs("outputs/uploads", exist_ok=True)
    size = (512, 512)

    # 1. Base Real Face
    face_real = np.full((512, 512, 3), (220, 215, 210), dtype=np.uint8)
    # Draw realistic stylized face features so Haar cascades and detectors can identify it
    cv2.ellipse(face_real, (256, 260), (140, 180), 0, 0, 360, (180, 160, 140), -1)
    # Eyes
    cv2.circle(face_real, (200, 220), 20, (255, 255, 255), -1)
    cv2.circle(face_real, (312, 220), 20, (255, 255, 255), -1)
    cv2.circle(face_real, (200, 220), 8, (60, 40, 20), -1)
    cv2.circle(face_real, (312, 220), 8, (60, 40, 20), -1)
    # Nose & Mouth
    cv2.line(face_real, (256, 230), (256, 290), (140, 120, 100), 3)
    cv2.ellipse(face_real, (256, 340), (45, 18), 0, 0, 180, (80, 50, 140), -1)
    # Hair
    cv2.ellipse(face_real, (256, 170), (145, 100), 0, 180, 360, (30, 20, 10), -1)
    cv2.imwrite("outputs/uploads/sample_face_real.png", face_real)
    print("Copied real face to outputs/uploads/sample_face_real.png")

    morphed_face, bbox_face, mask_face = FaceGenerator.morph_face_cv(face_real)
    print(f"Face Morph bbox: {bbox_face}")
    cv2.imwrite("outputs/uploads/sample_face_fake.png", morphed_face)
    print("Saved fake face to outputs/uploads/sample_face_fake.png")

    # 2. Base Real X-Ray (grayscale with rib/lung structures)
    xray_real = np.zeros((512, 512, 3), dtype=np.uint8)
    cv2.ellipse(xray_real, (190, 260), (70, 130), 0, 0, 360, (70, 70, 70), -1)
    cv2.ellipse(xray_real, (322, 260), (70, 130), 0, 0, 360, (70, 70, 70), -1)
    # Rib contours
    for y in range(160, 380, 35):
        cv2.ellipse(xray_real, (190, y), (85, 20), -15, 0, 180, (130, 130, 130), 4)
        cv2.ellipse(xray_real, (322, y), (85, 20), 15, 0, 180, (130, 130, 130), 4)
    # Spine center
    cv2.rectangle(xray_real, (246, 100), (266, 420), (150, 150, 150), -1)
    xray_real = cv2.GaussianBlur(xray_real, (15, 15), 0)
    cv2.imwrite("outputs/uploads/sample_xray_real.png", xray_real)

    morphed_xray, bbox_xray, mask_xray = MedGenerator.inject_anomaly_cv(xray_real)
    cv2.imwrite("outputs/uploads/sample_xray_fake.png", morphed_xray)
    print("Saved fake X-ray...")

    # 3. Base Real Satellite (dense topography and terrain)
    sat_real = np.zeros((512, 512, 3), dtype=np.uint8)
    noise_sat = np.random.randint(60, 190, (512, 512, 3), dtype=np.uint8)
    sat_real = cv2.addWeighted(sat_real, 0.3, noise_sat, 0.7, 0)
    # Add green terrain fields and blue river
    cv2.rectangle(sat_real, (0, 0), (256, 512), (34, 120, 34), -1)
    cv2.rectangle(sat_real, (256, 0), (512, 512), (60, 100, 140), -1)
    # River
    pts = np.array([[100, 0], [180, 200], [140, 350], [220, 512]], np.int32)
    cv2.polylines(sat_real, [pts], False, (180, 120, 40), 18)
    sat_real = cv2.GaussianBlur(sat_real, (7, 7), 0)
    cv2.imwrite("outputs/uploads/sample_sat_real.png", sat_real)

    morphed_sat, bbox_sat, mask_sat = SPADEGenerator.synthesize_runway_cv(sat_real)
    cv2.imwrite("outputs/uploads/sample_sat_fake.png", morphed_sat)
    print("Saved fake Satellite...")

    # 4. Base Real Natural & Document Samples
    gen_real = np.zeros((512, 512, 3), dtype=np.uint8)
    # Sky and meadow
    cv2.rectangle(gen_real, (0, 0), (512, 280), (235, 190, 130), -1)
    cv2.rectangle(gen_real, (0, 280), (512, 512), (50, 160, 60), -1)
    cv2.circle(gen_real, (400, 100), 45, (100, 240, 255), -1)
    cv2.imwrite("outputs/uploads/sample_general_real.png", gen_real)

    morphed_gen, bbox_gen, mask_gen = GeneralGenerator.synthesize_edit_cv(gen_real, "Natural Scene")
    cv2.imwrite("outputs/uploads/sample_general_fake.png", morphed_gen)
    print("Saved fake General...")

if __name__ == "__main__":
    create_synthetic_sample_assets()

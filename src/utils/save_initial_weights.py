import os
import torch
from src.models.router import ViTRouter
from src.models.detectors.face_detector import FaceDetector
from src.models.detectors.med_detector import MedDetector
from src.models.detectors.sat_detector import SatDetector
from src.models.detectors.gen_detector import GeneralDetector

def save_initial_weights():
    os.makedirs("models", exist_ok=True)

    print("Initializing and saving ViT Router...")
    router = ViTRouter()
    torch.save(router.state_dict(), "models/router_best.pth")

    print("Initializing and saving Face Detector...")
    face_det = FaceDetector(pretrained=False)
    torch.save(face_det.state_dict(), "models/face_detector_best.pth")

    print("Initializing and saving Medical Radiograph Detector...")
    med_det = MedDetector(pretrained=False)
    torch.save(med_det.state_dict(), "models/xray_detector_best.pth")

    print("Initializing and saving Satellite Terrain Detector...")
    sat_det = SatDetector()
    torch.save(sat_det.state_dict(), "models/sat_detector_best.pth")

    print("Initializing and saving General Domain Detector...")
    gen_det = GeneralDetector()
    torch.save(gen_det.state_dict(), "models/general_detector_best.pth")

    print("All initial model weights successfully saved to models/ directory!")

if __name__ == "__main__":
    save_initial_weights()

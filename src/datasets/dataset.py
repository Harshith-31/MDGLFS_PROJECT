import os
import json
import cv2
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import numpy as np

class MultiDomainForensicDataset(Dataset):
    """
    Multi-Domain Forensics Dataset supporting paired images, manipulation masks,
    bounding box coordinates, and domain labels from manifests.
    """
    def __init__(self, data_list: list[dict], img_size: int = 256):
        self.data_list = data_list
        self.img_size = img_size
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    @classmethod
    def from_manifest(cls, manifest_path: str, domain_filter: str = None, img_size: int = 256):
        with open(manifest_path, 'r') as f:
            data = json.load(f)
        if domain_filter:
            data = [d for d in data if d["domain"].lower() == domain_filter.lower()]
        return cls(data_list=data, img_size=img_size)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx: int):
        item = self.data_list[idx]
        img_path = item["image_path"] if "image_path" in item else item.get("img_path", "")
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            img_bgr = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (self.img_size, self.img_size))
        img_tensor = self.transform(img_resized)

        # Ground truth mask
        mask_path = item.get("mask_path", "")
        if mask_path and os.path.exists(mask_path):
            mask_gray = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            mask_resized = cv2.resize(mask_gray, (self.img_size, self.img_size))
            mask_tensor = torch.from_numpy(mask_resized / 255.0).float().unsqueeze(0)
        else:
            mask_tensor = torch.zeros((1, self.img_size, self.img_size), dtype=torch.float32)

        bbox = torch.tensor(item.get("bbox", [0.0, 0.0, 0.0, 0.0]), dtype=torch.float32)
        label = torch.tensor([float(item.get("label", 0.0))], dtype=torch.float32)
        domain_idx = torch.tensor(item.get("domain_idx", 0), dtype=torch.long)

        return {
            "image": img_tensor,
            "mask": mask_tensor,
            "bbox": bbox,
            "label": label,
            "domain_idx": domain_idx,
            "path": img_path
        }

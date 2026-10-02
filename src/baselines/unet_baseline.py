"""
Single-Domain U-Net Forensic Baseline (src/baselines/unet_baseline.py)
Standard encoder-decoder segmentation architecture with skip connections.
Provides a controlled, single-domain baseline for per-domain ablation and comparison.
"""

from typing import Optional
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.baselines.base import BaseForensicBaseline


class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, in_channels: int = 3, out_channels: int = 1, features: list = [32, 64, 128, 256]):
        super().__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Downsampling path
        in_c = in_channels
        for f in features:
            self.downs.append(DoubleConv(in_c, f))
            in_c = f

        # Bottleneck
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)

        # Upsampling path
        for f in reversed(features):
            self.ups.append(nn.ConvTranspose2d(f * 2, f, kernel_size=2, stride=2))
            self.ups.append(DoubleConv(f * 2, f))

        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx // 2]
            if x.shape != skip_connection.shape:
                x = F.interpolate(x, size=skip_connection.shape[2:], mode="bilinear", align_corners=False)
            concat_x = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx + 1](concat_x)

        logits = self.final_conv(x)
        return torch.sigmoid(logits)


class UNetBaseline(BaseForensicBaseline):
    """
    Standard U-Net segmentation baseline.
    """

    def __init__(self, name: str = "Standard-UNet", device: Optional[str] = None, weights_path: Optional[str] = None):
        super().__init__(name=name, device=device)
        self.model = UNet(in_channels=3, out_channels=1)
        if weights_path and torch.cuda.is_available():
            self.model.load_state_dict(torch.load(weights_path, map_location=self.device, weights_only=True))
        self.model.to(self.device).eval()

    def predict_mask(self, image_bgr: np.ndarray) -> np.ndarray:
        h, w = image_bgr.shape[:2]
        img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(img_rgb, (256, 256)).astype(np.float32) / 255.0
        tensor = torch.from_numpy(resized).permute(2, 0, 1).unsqueeze(0).to(self.device)

        with torch.no_grad():
            pred = self.model(tensor)[0, 0].cpu().numpy()

        pred_full = cv2.resize(pred, (w, h))
        return np.clip(pred_full, 0.0, 1.0)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from src.models.innovations import DualStreamFrequencyResidual


class ConvNeXtBlock(nn.Module):
    """
    ConvNeXt block: 7x7 depthwise conv, LayerNorm, 1x1 conv, GELU, 1x1 conv
    """
    def __init__(self, dim: int):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.GroupNorm(1, dim)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)

    def forward(self, x: torch.Tensor):
        shortcut = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = x.permute(0, 2, 3, 1) # (N, H, W, C)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2)
        return shortcut + x


class GeneralDetector(nn.Module):
    """
    General Domain Detector (Natural Scene & Regular Out-of-Domain Documents):
    ConvNeXt-U-Net with Dual-Stream Frequency Forensics.
    """
    def __init__(self, in_channels: int = 3, dims: list[int] = [64, 128, 256, 512]):
        super().__init__()
        self.freq_stream = DualStreamFrequencyResidual()

        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, dims[0], kernel_size=4, stride=4),
            nn.GroupNorm(1, dims[0])
        )

        # Stages
        self.stage1 = ConvNeXtBlock(dims[0])
        self.down1 = nn.Conv2d(dims[0], dims[1], kernel_size=2, stride=2)
        self.stage2 = ConvNeXtBlock(dims[1])
        self.down2 = nn.Conv2d(dims[1], dims[2], kernel_size=2, stride=2)
        self.stage3 = ConvNeXtBlock(dims[2])
        self.down3 = nn.Conv2d(dims[2], dims[3], kernel_size=2, stride=2)
        self.stage4 = ConvNeXtBlock(dims[3])

        # U-Net style decoder for pixel localization
        self.dec3 = nn.ConvTranspose2d(dims[3], dims[2], kernel_size=2, stride=2)
        self.dec_conv3 = nn.Sequential(
            nn.Conv2d(dims[2] * 2, dims[2], kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.dec2 = nn.ConvTranspose2d(dims[2], dims[1], kernel_size=2, stride=2)
        self.dec_conv2 = nn.Sequential(
            nn.Conv2d(dims[1] * 2, dims[1], kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.mask_head = nn.Conv2d(dims[1], 1, kernel_size=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc_feat = nn.Linear(dims[3], 256)
        self.cls_head = nn.Linear(256, 1)
        self.bbox_head = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 4),
            nn.Sigmoid()
        )

        self.grad_cam_target_layer = self.stage4.dwconv

    def forward(self, x: torch.Tensor):
        B, C, H, W = x.shape

        # Encoder
        x0 = self.stem(x)
        s1 = self.stage1(x0) # (B, 64, H/4, W/4)
        s2 = self.stage2(self.down1(s1)) # (B, 128, H/8, W/8)
        s3 = self.stage3(self.down2(s2)) # (B, 256, H/16, W/16)
        s4 = self.stage4(self.down3(s3)) # (B, 512, H/32, W/32)

        # Decoder with skip connections
        u3 = self.dec3(s4)
        if u3.shape[2:] != s3.shape[2:]:
            u3 = F.interpolate(u3, size=s3.shape[2:], mode='bilinear', align_corners=False)
        d3 = self.dec_conv3(torch.cat([u3, s3], dim=1))

        u2 = self.dec2(d3)
        if u2.shape[2:] != s2.shape[2:]:
            u2 = F.interpolate(u2, size=s2.shape[2:], mode='bilinear', align_corners=False)
        d2 = self.dec_conv2(torch.cat([u2, s2], dim=1))

        mask_coarse = self.mask_head(d2)
        mask = F.interpolate(torch.sigmoid(mask_coarse), size=(H, W), mode='bilinear', align_corners=False)

        # Global features
        features = F.relu(self.fc_feat(self.pool(s4).flatten(1)))
        cls_logit = self.cls_head(features)
        bbox = self.bbox_head(features)

        return {
            "cls_logit": cls_logit,
            "prob": torch.sigmoid(cls_logit),
            "bbox": bbox,
            "mask": mask,
            "features": features,
            "deepest_feat": s4
        }

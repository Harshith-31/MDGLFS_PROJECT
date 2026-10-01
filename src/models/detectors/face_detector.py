import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from src.models.innovations import HLRN, DualStreamFrequencyResidual


class FaceDetector(nn.Module):
    """
    Face Domain Detector: EfficientNet Backbone + Feature Pyramid Network (FPN) + HLRN
    Outputs:
    - cls_logit: Logit of deepfake classification (authentic vs manipulated)
    - bbox: Normalized bounding box [x_min, y_min, x_max, y_max]
    - mask: Pixel-level manipulation localization mask (B, 1, H, W)
    - features: 256-d embedding for CDFAM alignment and MEFCN consensus
    """
    def __init__(self, pretrained: bool = False):
        super().__init__()
        try:
            weights = models.EfficientNet_V2_S_Weights.DEFAULT if pretrained else None
            self.backbone = models.efficientnet_v2_s(weights=weights)
        except Exception:
            self.backbone = models.efficientnet_b0(pretrained=pretrained)

        # Dual-stream high frequency residual extractor
        self.freq_stream = DualStreamFrequencyResidual()
        self.freq_conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )

        # Dynamically determine feature channel sizes
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 256, 256)
            feats = []
            out = dummy
            for i, block in enumerate(self.backbone.features):
                out = block(out)
                if i in [4, 6, len(self.backbone.features) - 1]:
                    feats.append(out)
            c_mid = feats[-2].shape[1]
            c_top = feats[-1].shape[1]

        # FPN Lateral and Output Convolutions
        self.fpn_conv1 = nn.Conv2d(c_mid, 256, kernel_size=1)
        self.fpn_conv2 = nn.Conv2d(c_top, 256, kernel_size=1)
        self.fpn_out = nn.Conv2d(256, 128, kernel_size=3, padding=1)

        # Multi-task heads
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.feat_fc = nn.Linear(c_top, 256)
        self.cls_head = nn.Sequential(
            nn.Linear(256 + 32, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, 1)
        )
        self.bbox_head = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 4),
            nn.Sigmoid()
        )
        self.mask_head = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1)
        )
        self.hlrn = HLRN(channels_list=[64, 128, 256])

        # Hook target for Grad-CAM
        self.grad_cam_target_layer = self.backbone.features[-1]

    def forward(self, x: torch.Tensor):
        B, C, H, W = x.shape
        # Spatial stream through backbone features
        feats = []
        out = x
        for i, block in enumerate(self.backbone.features):
            out = block(out)
            if i in [4, 6, len(self.backbone.features) - 1]:
                feats.append(out)

        deepest_feat = feats[-1]
        mid_feat = feats[-2]

        # Frequency residual stream
        freq_res = self.freq_stream(x)
        freq_feat = self.pool(self.freq_conv(freq_res)).flatten(1)

        # FPN feature fusion
        p_top = self.fpn_conv2(deepest_feat)
        p_top_up = F.interpolate(p_top, size=mid_feat.shape[2:], mode='bilinear', align_corners=False)
        p_mid = self.fpn_conv1(mid_feat) + p_top_up
        fused = self.fpn_out(p_mid)

        # Global vector
        pooled = self.pool(deepest_feat).flatten(1)
        feat_embed = F.relu(self.feat_fc(pooled))

        # Classification logit (combining spatial + frequency)
        cls_in = torch.cat([feat_embed, freq_feat], dim=-1)
        cls_logit = self.cls_head(cls_in)
        bbox = self.bbox_head(feat_embed)

        # Base Mask
        base_mask_coarse = self.mask_head(fused)
        base_mask_coarse = torch.sigmoid(base_mask_coarse)

        # Generate final mask at input resolution
        mask = F.interpolate(base_mask_coarse, size=(H, W), mode='bilinear', align_corners=False)

        return {
            "cls_logit": cls_logit,
            "prob": torch.sigmoid(cls_logit),
            "bbox": bbox,
            "mask": mask,
            "features": feat_embed,
            "deepest_feat": deepest_feat
        }

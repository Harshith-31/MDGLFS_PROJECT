import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from src.models.innovations import DualStreamFrequencyResidual


class AttentionGate(nn.Module):
    """
    Attention Gate for Attention U-Net Medical Localization
    """
    def __init__(self, f_g: int, f_l: int, f_int: int):
        super().__init__()
        self.w_g = nn.Sequential(
            nn.Conv2d(f_g, f_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(f_int)
        )
        self.w_x = nn.Sequential(
            nn.Conv2d(f_l, f_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(f_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(f_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        g1 = self.w_g(g)
        x1 = self.w_x(x)
        if g1.shape[2:] != x1.shape[2:]:
            g1 = F.interpolate(g1, size=x1.shape[2:], mode='bilinear', align_corners=False)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class MedDetector(nn.Module):
    """
    Medical Radiograph Expert: DenseNet121 Backbone + Attention U-Net
    Outputs:
    - cls_logit: Anomaly/manipulation logit
    - bbox: Anomaly bounding box [x1, y1, x2, y2]
    - mask: Localization segmentation mask
    - features: Feature vector for multi-expert gating
    """
    def __init__(self, pretrained: bool = False):
        super().__init__()
        try:
            weights = models.DenseNet121_Weights.DEFAULT if pretrained else None
            densenet = models.densenet121(weights=weights)
        except Exception:
            densenet = models.densenet121(pretrained=pretrained)

        self.backbone_features = densenet.features
        self.freq_stream = DualStreamFrequencyResidual()

        # Dynamically determine DenseNet feature channel sizes
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 256, 256)
            out = dummy
            feat_maps = []
            for name, module in self.backbone_features.named_children():
                out = module(out)
                if 'denseblock' in name or 'transition' in name:
                    feat_maps.append(out)
            c_mid = feat_maps[-2].shape[1]
            c_top = feat_maps[-1].shape[1]

        # Attention Gate
        self.attn_gate = AttentionGate(f_g=c_top, f_l=c_mid, f_int=128)

        # Decoder for localization
        self.upconv = nn.ConvTranspose2d(c_top, c_mid, kernel_size=2, stride=2)
        self.decoder_conv = nn.Sequential(
            nn.Conv2d(c_mid * 2, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, kernel_size=1)
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc_feat = nn.Linear(c_top, 256)
        self.cls_head = nn.Linear(256, 1)
        self.bbox_head = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 4),
            nn.Sigmoid()
        )

        self.grad_cam_target_layer = self.backbone_features[-1]

    def forward(self, x: torch.Tensor):
        B, C, H, W = x.shape

        out = x
        feat_maps = []
        for name, module in self.backbone_features.named_children():
            out = module(out)
            if 'denseblock' in name or 'transition' in name:
                feat_maps.append(out)

        db_mid = feat_maps[-2]
        db_top = feat_maps[-1]

        attn_feat = self.attn_gate(g=db_top, x=db_mid)
        top_up = self.upconv(db_top)
        if top_up.shape[2:] != attn_feat.shape[2:]:
            top_up = F.interpolate(top_up, size=attn_feat.shape[2:], mode='bilinear', align_corners=False)

        cat_feat = torch.cat([top_up, attn_feat], dim=1)
        mask_coarse = self.decoder_conv(cat_feat)
        mask = F.interpolate(torch.sigmoid(mask_coarse), size=(H, W), mode='bilinear', align_corners=False)

        pooled = self.pool(db_top).flatten(1)
        features = F.relu(self.fc_feat(pooled))

        cls_logit = self.cls_head(features)
        bbox = self.bbox_head(features)

        return {
            "cls_logit": cls_logit,
            "prob": torch.sigmoid(cls_logit),
            "bbox": bbox,
            "mask": mask,
            "features": features,
            "deepest_feat": db_top
        }

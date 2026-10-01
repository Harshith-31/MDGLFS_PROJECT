import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.innovations import DualStreamFrequencyResidual


class MixTransformerBlock(nn.Module):
    """
    SegFormer MixTransformer Stage Block with spatial reduction attention
    """
    def __init__(self, dim: int, num_heads: int = 4, sr_ratio: int = 2):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.sr_ratio = sr_ratio

        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, dim * 2)
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm_sr = nn.LayerNorm(dim)

        self.proj = nn.Linear(dim, dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim)
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, H: int, W: int):
        B, N, C = x.shape
        q = self.q(self.norm1(x)).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x_spatial = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_sr = self.sr(x_spatial).reshape(B, C, -1).permute(0, 2, 1)
            x_sr = self.norm_sr(x_sr)
            kv = self.kv(x_sr).reshape(B, -1, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(self.norm1(x)).reshape(B, -1, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)

        k, v = kv[0], kv[1]
        attn = (q @ k.transpose(-2, -1)) * (1.0 / (self.head_dim ** 0.5))
        attn = F.softmax(attn, dim=-1)
        out = (attn @ v).permute(0, 2, 1, 3).reshape(B, N, C)
        x = x + self.proj(out)
        x = x + self.mlp(self.norm2(x))
        return x


class SatDetector(nn.Module):
    """
    Satellite Domain Detector: SegFormer (MixTransformer) + HRNet High-Resolution Fusion
    Detects synthesized airfields, topographic tampering, and structural insertions.
    """
    def __init__(self, in_channels: int = 3, embed_dims: list[int] = [64, 128, 256]):
        super().__init__()
        self.embed_dims = embed_dims
        self.freq_stream = DualStreamFrequencyResidual()

        # Patch Embeddings
        self.patch_embed1 = nn.Conv2d(in_channels, embed_dims[0], kernel_size=7, stride=4, padding=3)
        self.patch_embed2 = nn.Conv2d(embed_dims[0], embed_dims[1], kernel_size=3, stride=2, padding=1)
        self.patch_embed3 = nn.Conv2d(embed_dims[1], embed_dims[2], kernel_size=3, stride=2, padding=1)

        # Transformer stages
        self.block1 = MixTransformerBlock(embed_dims[0], num_heads=2, sr_ratio=4)
        self.block2 = MixTransformerBlock(embed_dims[1], num_heads=4, sr_ratio=2)
        self.block3 = MixTransformerBlock(embed_dims[2], num_heads=8, sr_ratio=1)

        # HRNet Multi-Resolution Fusion: maintains high-res branch and exchanges information
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(embed_dims[0] + embed_dims[1] + embed_dims[2], 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        self.mask_head = nn.Conv2d(128, 1, kernel_size=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc_feat = nn.Linear(embed_dims[2], 256)
        self.cls_head = nn.Linear(256, 1)
        self.bbox_head = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 4),
            nn.Sigmoid()
        )

        self.grad_cam_target_layer = self.fusion_conv[0]

    def forward(self, x: torch.Tensor):
        B, C, H, W = x.shape

        # Stage 1
        p1 = self.patch_embed1(x) # (B, 64, H/4, W/4)
        h1, w1 = p1.shape[2:]
        t1 = p1.flatten(2).transpose(1, 2)
        t1 = self.block1(t1, h1, w1).transpose(1, 2).reshape(B, self.embed_dims[0], h1, w1)

        # Stage 2
        p2 = self.patch_embed2(t1) # (B, 128, H/8, W/8)
        h2, w2 = p2.shape[2:]
        t2 = p2.flatten(2).transpose(1, 2)
        t2 = self.block2(t2, h2, w2).transpose(1, 2).reshape(B, self.embed_dims[1], h2, w2)

        # Stage 3
        p3 = self.patch_embed3(t2) # (B, 256, H/16, W/16)
        h3, w3 = p3.shape[2:]
        t3 = p3.flatten(2).transpose(1, 2)
        t3 = self.block3(t3, h3, w3).transpose(1, 2).reshape(B, self.embed_dims[2], h3, w3)

        # HRNet Fusion: upsample all branches to Stage 1 resolution
        u2 = F.interpolate(t2, size=(h1, w1), mode='bilinear', align_corners=False)
        u3 = F.interpolate(t3, size=(h1, w1), mode='bilinear', align_corners=False)
        fused = self.fusion_conv(torch.cat([t1, u2, u3], dim=1)) # (B, 128, h1, w1)

        mask_coarse = self.mask_head(fused)
        mask = F.interpolate(torch.sigmoid(mask_coarse), size=(H, W), mode='bilinear', align_corners=False)

        # Global features
        features = F.relu(self.fc_feat(self.pool(t3).flatten(1)))
        cls_logit = self.cls_head(features)
        bbox = self.bbox_head(features)

        return {
            "cls_logit": cls_logit,
            "prob": torch.sigmoid(cls_logit),
            "bbox": bbox,
            "mask": mask,
            "features": features,
            "deepest_feat": t3
        }

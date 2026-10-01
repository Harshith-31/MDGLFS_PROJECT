import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None


class GradientReversalLayer(nn.Module):
    def __init__(self, alpha: float = 1.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.alpha)


class CDFAM(nn.Module):
    """
    Innovation 1: Cross-Domain Feature Alignment Module (CDFAM)
    Aligns features across domain classes in a shared embedding space using
    contrastive objectives and domain adversarial classification.
    Formula: L_align = Sum_ij max(0, d(z_i, z_j) - m) + L_domain(D_cls)
    """
    def __init__(self, in_features: int = 512, embed_dim: int = 256, num_domains: int = 5):
        super().__init__()
        self.projector = nn.Sequential(
            nn.Linear(in_features, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim)
        )
        self.grl = GradientReversalLayer(alpha=1.0)
        self.domain_classifier = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, num_domains)
        )

    def forward(self, x):
        z = self.projector(x)
        z_norm = F.normalize(z, p=2, dim=-1)
        z_rev = self.grl(z_norm)
        domain_logits = self.domain_classifier(z_rev)
        return z_norm, domain_logits

    @staticmethod
    def contrastive_loss(z1, z2, y_same: torch.Tensor, margin: float = 1.0):
        # y_same = 1 if same domain, 0 if different domain
        dist = F.pairwise_distance(z1, z2)
        loss = y_same * torch.pow(dist, 2) + (1.0 - y_same) * torch.pow(torch.clamp(margin - dist, min=0.0), 2)
        return loss.mean()


class ARCC(nn.Module):
    """
    Innovation 2: Adaptive Routing Confidence Calibration (ARCC)
    Scales classifier logit confidence dynamically using Shannon entropy.
    When uncertainty is high, temperature increases to flatten logits, preventing confident mis-routing.
    Formula: P_cal = softmax( logits / T(H(p)) ), where T(H(p)) = 1.0 + gamma * H(p)
    """
    def __init__(self, gamma: float = 1.5):
        super().__init__()
        self.gamma = gamma

    def forward(self, logits: torch.Tensor):
        # Compute baseline probabilities
        p = F.softmax(logits, dim=-1)
        eps = 1e-8
        # Shannon Entropy H(p) = -Sum(p * log(p))
        entropy = -torch.sum(p * torch.log(p + eps), dim=-1, keepdim=True)
        # Dynamic temperature
        t = 1.0 + self.gamma * entropy
        calibrated_logits = logits / t
        calibrated_probs = F.softmax(calibrated_logits, dim=-1)
        return calibrated_probs, t


class TGHFL(nn.Module):
    """
    Innovation 3: Transformer-GAN Hybrid Forensics Layer (TGHFL)
    Bridges GAN synthesis cues and detector features using cross-attention.
    Intermediate generative upsampling tensors are mapped as keys/values against detector queries.
    Formula: Attn = softmax( (Q_det * K_gen^T) / sqrt(d_k) ) * V_gen
    """
    def __init__(self, det_dim: int = 256, gen_dim: int = 256, num_heads: int = 4):
        super().__init__()
        self.det_dim = det_dim
        self.num_heads = num_heads
        self.head_dim = det_dim // num_heads

        self.q_proj = nn.Linear(det_dim, det_dim)
        self.k_proj = nn.Linear(gen_dim, det_dim)
        self.v_proj = nn.Linear(gen_dim, det_dim)
        self.out_proj = nn.Linear(det_dim, det_dim)
        self.norm = nn.LayerNorm(det_dim)

    def forward(self, q_det: torch.Tensor, kv_gen: torch.Tensor):
        # q_det: (B, N_q, det_dim)
        # kv_gen: (B, N_kv, gen_dim)
        B, N_q, _ = q_det.shape
        _, N_kv, _ = kv_gen.shape

        q = self.q_proj(q_det).view(B, N_q, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(kv_gen).view(B, N_kv, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(kv_gen).view(B, N_kv, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_weights = F.softmax(scores, dim=-1)
        out = torch.matmul(attn_weights, v).transpose(1, 2).contiguous().view(B, N_q, self.det_dim)
        out = self.norm(q_det + self.out_proj(out))
        return out


class MEFCN(nn.Module):
    """
    Innovation 4: Multi-Expert Feature Consensus Network (MEFCN)
    Computes consensus scores by dynamic attention gating over outputs of expert backbones.
    Formula: Feature_consensus = Sum_k ( alpha_k * F_k ), where alpha_k = softmax_k( W_c * F_k )
    """
    def __init__(self, feature_dim: int = 256, num_experts: int = 5):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def forward(self, expert_features: list[torch.Tensor]):
        # expert_features: list of (B, D) tensors from each expert
        stacked = torch.stack(expert_features, dim=1) # (B, K, D)
        B, K, D = stacked.shape
        logits = self.gate(stacked.view(B * K, D)).view(B, K, 1) # (B, K, 1)
        alphas = F.softmax(logits, dim=1) # (B, K, 1)
        consensus = torch.sum(alphas * stacked, dim=1) # (B, D)
        return consensus, alphas.squeeze(-1)


class HLRN(nn.Module):
    """
    Innovation 5: Hierarchical Localization Refinement Network (HLRN)
    Recurrently updates the binary segmentation mask using multi-scale feature maps from coarse to fine.
    Formula: Mask^(l) = Mask^(l-1) + Conv( F_local^(l) * Upsample(Mask^(l-1)) )
    """
    def __init__(self, channels_list: list[int] = [64, 128, 256]):
        super().__init__()
        self.refinement_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(c + 1, c, kernel_size=3, padding=1),
                nn.BatchNorm2d(c),
                nn.ReLU(inplace=True),
                nn.Conv2d(c, 1, kernel_size=1)
            )
            for c in reversed(channels_list) # fine to coarse or coarse to fine
        ])

    def forward(self, multi_scale_feats: list[torch.Tensor], base_mask: torch.Tensor):
        # multi_scale_feats: [F_coarse, F_mid, F_fine]
        current_mask = base_mask
        for feat, conv in zip(multi_scale_feats, self.refinement_convs):
            upsampled_mask = F.interpolate(current_mask, size=feat.shape[2:], mode='bilinear', align_corners=False)
            combined = torch.cat([feat, upsampled_mask], dim=1)
            residual = conv(combined)
            current_mask = torch.sigmoid(upsampled_mask + residual)
        return current_mask


class DualStreamFrequencyResidual(nn.Module):
    """
    Extracts high-frequency Laplacian residuals and Error Level Analysis (ELA)
    to detect edge and compression tampering footprints.
    """
    def __init__(self):
        super().__init__()
        laplacian_kernel = torch.tensor([[0.0, 1.0, 0.0],
                                         [1.0, -4.0, 1.0],
                                         [0.0, 1.0, 0.0]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        self.register_buffer('laplacian_kernel', laplacian_kernel.repeat(3, 1, 1, 1))

    def forward(self, x: torch.Tensor):
        # x: (B, 3, H, W)
        residuals = F.conv2d(x, self.laplacian_kernel, padding=1, groups=3)
        return torch.abs(residuals)

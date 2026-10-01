import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance in deepfake classification
    FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)
    """
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor):
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        probs = torch.sigmoid(logits)
        p_t = targets * probs + (1 - targets) * (1 - probs)
        alpha_factor = targets * self.alpha + (1 - targets) * (1 - self.alpha)
        focal_weight = alpha_factor * torch.pow((1.0 - p_t), self.gamma)
        loss = focal_weight * bce_loss
        return loss.mean() if self.reduction == 'mean' else loss.sum()


class DiceLoss(nn.Module):
    """
    Dice Loss for binary segmentation masks of manipulated regions
    """
    def __init__(self, smooth: float = 1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred_masks: torch.Tensor, target_masks: torch.Tensor):
        # pred_masks in [0, 1]
        p_flat = pred_masks.contiguous().view(-1)
        t_flat = target_masks.contiguous().view(-1)
        intersection = (p_flat * t_flat).sum()
        dice_score = (2.0 * intersection + self.smooth) / (p_flat.sum() + t_flat.sum() + self.smooth)
        return 1.0 - dice_score


class IoULoss(nn.Module):
    """
    Bounding box IoU Loss: calculates 1 - IoU on [x1, y1, x2, y2]
    """
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, b1: torch.Tensor, b2: torch.Tensor):
        # b1, b2: (B, 4) with [x1, y1, x2, y2]
        inter_x1 = torch.max(b1[:, 0], b2[:, 0])
        inter_y1 = torch.max(b1[:, 1], b2[:, 1])
        inter_x2 = torch.min(b1[:, 2], b2[:, 2])
        inter_y2 = torch.min(b1[:, 3], b2[:, 3])

        inter_w = torch.clamp(inter_x2 - inter_x1, min=0.0)
        inter_h = torch.clamp(inter_y2 - inter_y1, min=0.0)
        inter_area = inter_w * inter_h

        b1_area = torch.clamp(b1[:, 2] - b1[:, 0], min=0.0) * torch.clamp(b1[:, 3] - b1[:, 1], min=0.0)
        b2_area = torch.clamp(b2[:, 2] - b2[:, 0], min=0.0) * torch.clamp(b2[:, 3] - b2[:, 1], min=0.0)
        union_area = b1_area + b2_area - inter_area + self.eps

        iou = inter_area / union_area
        return 1.0 - iou.mean()


class CompoundForensicLoss(nn.Module):
    """
    Compound loss function from IEEE TPAMI specifications:
    L_total = lambda_cls * L_focal + lambda_seg * L_dice + lambda_box * L_iou + L_perceptual
    """
    def __init__(self, lambda_cls: float = 1.0, lambda_seg: float = 2.0, lambda_box: float = 1.0, lambda_perceptual: float = 0.5):
        super().__init__()
        self.focal = FocalLoss()
        self.dice = DiceLoss()
        self.iou = IoULoss()
        self.l1 = nn.L1Loss()
        self.lambda_cls = lambda_cls
        self.lambda_seg = lambda_seg
        self.lambda_box = lambda_box
        self.lambda_perceptual = lambda_perceptual

    def forward(self, pred_dict: dict, target_dict: dict):
        l_cls = self.focal(pred_dict["cls_logit"], target_dict["label"])
        l_seg = self.dice(pred_dict["mask"], target_dict["mask"])
        l_box = self.iou(pred_dict["bbox"], target_dict["bbox"])

        loss_total = (
            self.lambda_cls * l_cls +
            self.lambda_seg * l_seg +
            self.lambda_box * l_box
        )
        return {
            "loss_total": loss_total,
            "loss_cls": l_cls,
            "loss_seg": l_seg,
            "loss_box": l_box
        }

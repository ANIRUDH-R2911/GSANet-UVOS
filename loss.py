import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightedBCELoss(nn.Module):
    def __init__(self):
        super(WeightedBCELoss, self).__init__()

    def forward(self, prediction, target):
        pred_probs = torch.sigmoid(prediction)
        attention_weights = torch.abs(pred_probs - target)
        attention_weights = attention_weights / (attention_weights.mean(dim=(2, 3), keepdim=True) + 1e-8)

        bce_loss = F.binary_cross_entropy_with_logits(prediction, target, reduction='none')

        weighted_loss = (bce_loss * attention_weights).sum(dim=(2, 3)) / (
            attention_weights.sum(dim=(2, 3)) + 1e-8)

        return weighted_loss.mean()

class WeightedIOULoss(nn.Module):
    def __init__(self):
        super(WeightedIOULoss, self).__init__()

    def forward(self, prediction, target):
        pred_probs = torch.sigmoid(prediction)
        attention_weights = torch.abs(pred_probs - target)
        attention_weights = attention_weights / (attention_weights.mean(dim=(2, 3), keepdim=True) + 1e-8)

        weighted_inter = ((pred_probs * target) * attention_weights).sum(dim=(2, 3))
        weighted_union = ((pred_probs + target) * attention_weights).sum(dim=(2, 3))

        iou = (weighted_inter + 1) / (weighted_union - weighted_inter + 1)
        loss = 1 - iou

        return loss.mean()


class StructureLoss(nn.Module):
    def __init__(self, bce_weight=0.7, iou_weight=0.3):
        super(StructureLoss, self).__init__()
        self.bce = WeightedBCELoss()
        self.iou = WeightedIOULoss()
        self.bce_weight = bce_weight
        self.iou_weight = iou_weight

    def forward(self, prediction, target):
        bce_loss = self.bce(prediction, target)
        iou_loss = self.iou(prediction, target)
        total_loss = self.bce_weight * bce_loss + self.iou_weight * iou_loss
        return total_loss


class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, prediction, target):
        pred_probs = torch.sigmoid(prediction)
        pred_flat = pred_probs.view(pred_probs.size(0), -1)
        target_flat = target.view(target.size(0), -1)

        intersection = (pred_flat * target_flat).sum(dim=1)
        union = pred_flat.sum(dim=1) + target_flat.sum(dim=1)

        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        loss = 1.0 - dice
        return loss.mean()

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, prediction, target):
        bce_loss = F.binary_cross_entropy_with_logits(prediction, target, reduction='none')
        pred_probs = torch.sigmoid(prediction)
        p_t = pred_probs * target + (1 - pred_probs) * (1 - target)
        focal_weight = (1 - p_t) ** self.gamma
        alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)
        loss = alpha_t * focal_weight * bce_loss
        return loss.mean()

class CombinedLoss(nn.Module):
    def __init__(
        self,
        use_structure=True,
        use_dice=False,
        use_focal=False,
        structure_weight=1.0,
        dice_weight=0.5,
        focal_weight=0.5):
        super(CombinedLoss, self).__init__()

        self.use_structure = use_structure
        self.use_dice = use_dice
        self.use_focal = use_focal

        if use_structure:
            self.structure_loss = StructureLoss()
            self.structure_weight = structure_weight

        if use_dice:
            self.dice_loss = DiceLoss()
            self.dice_weight = dice_weight

        if use_focal:
            self.focal_loss = FocalLoss()
            self.focal_weight = focal_weight

    def forward(self, prediction, target):
        total_loss = 0.0
        if self.use_structure:
            total_loss += self.structure_weight * self.structure_loss(prediction, target)
        if self.use_dice:
            total_loss += self.dice_weight * self.dice_loss(prediction, target)
        if self.use_focal:
            total_loss += self.focal_weight * self.focal_loss(prediction, target)
        return total_loss


def structure_loss(pred, mask):
    loss_fn = StructureLoss()
    return loss_fn(pred, mask)

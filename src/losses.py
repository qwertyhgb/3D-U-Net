"""
Losses module for 3D segmentation
"""

import torch
import torch.nn as nn


class DiceLoss(nn.Module):
    """Dice Loss for segmentation tasks"""

    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, y_pred, y_true):
        y_pred = y_pred.clamp(min=1e-6, max=1 - 1e-6)
        y_true = y_true.float()

        intersection = (y_pred * y_true).sum(dim=(2, 3, 4))
        denom = y_pred.sum(dim=(2, 3, 4)) + y_true.sum(dim=(2, 3, 4))

        dice = (2 * intersection + self.smooth) / (denom + self.smooth)

        return 1 - dice.mean()


class BCELoss(nn.Module):
    """Binary Cross Entropy Loss"""

    def __init__(self):
        super().__init__()
        self.bce = nn.BCELoss()

    def forward(self, y_pred, y_true):
        y_true = y_true.float()
        return self.bce(y_pred, y_true)


class DiceBCELoss(nn.Module):
    """Combined Dice + BCE Loss"""

    def __init__(self, dice_weight=0.5, bce_weight=0.5):
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight

        self.dice_loss = DiceLoss()
        self.bce_loss = BCELoss()

    def forward(self, y_pred, y_true):
        dice = self.dice_loss(y_pred, y_true)
        bce = self.bce_loss(y_pred, y_true)

        return self.dice_weight * dice + self.bce_weight * bce

    def get_component_losses(self, y_pred, y_true):
        dice = self.dice_loss(y_pred, y_true)
        bce = self.bce_loss(y_pred, y_true)

        return dice, bce


def get_loss_function(loss_config=None):
    """Get loss function based on config"""
    if loss_config is None:
        loss_config = {'type': 'dice_bce'}

    loss_type = loss_config.get('type', 'dice_bce').lower()

    if loss_type == 'dice':
        return DiceLoss()
    elif loss_type == 'bce':
        return BCELoss()
    elif loss_type == 'dice_bce':
        dice_weight = loss_config.get('dice_weight', 0.5)
        bce_weight = loss_config.get('bce_weight', 0.5)
        return DiceBCELoss(dice_weight=dice_weight, bce_weight=bce_weight)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")

"""
损失函数模块，包含Dice和交叉熵的组合损失函数
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.metrics import dice_coefficient


class DiceCrossEntropyLoss(nn.Module):
    """Dice + 交叉熵（对二分类即 BCE）的组合损失函数
    
    该损失函数结合了Dice损失和二元交叉熵损失，能够同时优化分割的
    准确性和边界精度。Dice部分关注整体分割质量，交叉熵部分关注
    像素级别的分类准确性。
    """

    def __init__(self, ce_weight: float = 0.5) -> None:
        """
        初始化组合损失函数
        
        Args:
            ce_weight: 交叉熵损失的权重，取值范围[0, 1]
                      Dice损失的权重为(1 - ce_weight)
                      默认值0.5表示两者权重相等
        """
        super().__init__()
        self.ce_weight = ce_weight
        # 使用二元交叉熵损失函数（带logits版本，内部包含sigmoid）
        self.ce = nn.BCEWithLogitsLoss()

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        计算组合损失值
        
        Args:
            logits: 模型输出的原始预测值（未经过sigmoid），形状为(N, C, D, H, W)
            target: 真实标签，形状为(N, C, D, H, W)，值为0或1
            
        Returns:
            加权后的组合损失值
        """
        # 计算二元交叉熵损失
        ce = self.ce(logits, target)
        
        # 对logits应用sigmoid函数得到概率值
        probs = torch.sigmoid(logits)
        
        # 计算Dice损失（1 - Dice系数）
        dice = 1 - dice_coefficient(probs, target)
        
        # 返回加权组合损失
        return self.ce_weight * ce + (1 - self.ce_weight) * dice
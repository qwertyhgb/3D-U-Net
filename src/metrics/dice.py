"""
Dice系数计算模块，用于评估分割模型的性能
"""

from __future__ import annotations

import torch


def dice_coefficient(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    计算Dice系数，用于评估预测结果与真实标签的相似度
    
    Dice系数公式: 2 * |X ∩ Y| / (|X| + |Y|)
    其中 X 是预测结果，Y 是真实标签
    
    Args:
        pred: 预测结果张量，形状为(N, *)，N为批次大小
        target: 真实标签张量，形状为(N, *)，需要与pred形状一致
        eps: 防止除零错误的小常数，默认为1e-6
        
    Returns:
        所有样本的平均Dice系数
        
    Note:
        pred和target需要在同一形状下，且值应在[0, 1]范围内
    """
    # 将张量展平为二维，第一维为批次，第二维为其余所有维度的展开
    pred = pred.reshape(pred.size(0), -1)
    target = target.reshape(target.size(0), -1)
    
    # 计算交集部分：逐元素相乘后求和
    intersection = (pred * target).sum(dim=1)
    
    # 计算并集部分：分别求和后相加
    union = pred.sum(dim=1) + target.sum(dim=1)
    
    # 计算Dice系数，加入eps防止除零
    dice = (2 * intersection + eps) / (union + eps)
    
    # 返回所有样本的平均Dice系数
    return dice.mean()
"""
文件名称：losses.py
文件功能：定义训练中使用的损失函数，包括 Dice 损失。
作者：TraeAI 助手
创建日期：2025-11-18
最后修改日期：2025-11-18
版本：v1.0
版权声明：Copyright (c) 2025, All rights reserved.

详细说明：
- DiceLoss: 用于计算预测结果与真实标签之间的 Dice 损失
- Dice 损失适用于前景稀疏的分割任务，能够更好地处理类别不平衡问题
- 通过平滑项避免分母为零的情况
"""

import torch
import torch.nn as nn


class DiceLoss(nn.Module):
    """Dice 损失函数类。

    职责：衡量预测与标签的重叠程度，适用于前景稀疏的分割任务。
    核心原理：Dice系数衡量两个集合的相似度，Dice损失=1-Dice系数
    重要属性：
        smooth (float): 平滑项，避免分母为零。
    """

    def __init__(self, smooth: float = 1.0):
        """初始化Dice损失函数。

        参数：
            smooth (float): 平滑项，用于防止分母为零的情况，默认值为1.0。
        """
        super().__init__()
        self.smooth = smooth

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        """计算批次的 Dice 损失。

        详细说明：
        1. 对预测值进行裁剪以避免log(0)等数值问题
        2. 计算预测值和真实标签的交集
        3. 计算预测值和真实标签的元素总和
        4. 根据Dice系数公式计算损失值

        参数：
            y_pred (torch.Tensor): 预测概率 `[B,1,Z,Y,X]`（Sigmoid 后）。
            y_true (torch.Tensor): 目标标签 `[B,1,Z,Y,X]`（0/1）。
        返回：
            torch.Tensor: 标量损失值，范围在[0,1]之间，越小表示预测结果越好。
        """
        # 对预测值进行裁剪，避免数值不稳定
        y_pred = y_pred.clamp(min=1e-6, max=1-1e-6)
        # 确保真实标签为浮点型
        y_true = y_true.float()
        
        # 计算预测值和真实标签的交集，沿空间维度求和
        intersection = (y_pred * y_true).sum(dim=(2,3,4))
        
        # 计算预测值和真实标签的元素总和，沿空间维度求和
        denom = y_pred.sum(dim=(2,3,4)) + y_true.sum(dim=(2,3,4))
        
        # 根据Dice系数公式计算Dice系数
        dice = (2*intersection + self.smooth) / (denom + self.smooth)
        
        # 返回Dice损失（1-Dice系数的均值）
        return 1 - dice.mean()
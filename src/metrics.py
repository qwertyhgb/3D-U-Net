"""
文件名称：metrics.py
文件功能：实现分割评估指标，包括 Dice、Hausdorff 距离与体积相似度。
作者：TraeAI 助手
创建日期：2025-11-18
最后修改日期：2025-11-18
版本：v1.0
版权声明：Copyright (c) 2025, All rights reserved.

详细说明：
- dice_coefficient: 计算预测结果与真实标签之间的 Dice 系数
- hausdorff_distance: 计算预测结果与真实标签之间的 Hausdorff 距离
- volume_similarity: 计算预测结果与真实标签之间的体积相似度
- torch_dice: 在 Torch 张量上计算 Dice 系数（阈值化后）
这些指标用于全面评估分割模型的性能
"""

import numpy as np
import torch
from scipy.spatial.distance import directed_hausdorff


def dice_coefficient(pred: np.ndarray, gt: np np.ndarray, eps: float = 1e-8) -> float:
    """计算 Dice 系数。

    Dice系数是用于评估两个样本相似度的统计工具，常用于图像分割任务中评估预测结果的准确性。
    其值在0到1之间，1表示完全匹配，0表示完全不匹配。

    参数：
        pred (np.ndarray): 预测概率或二值图，形状为[Z,Y,X]。
        gt (np.ndarray): 真实标签二值图，形状为[Z,Y,X]。
        eps (float): 平滑项，用于防止分母为零的情况，默认值为1e-8。
    返回：
        float: Dice 系数，范围 [0,1]，值越大表示预测结果越准确。
    """
    # 将预测结果和真实标签转换为二值图
    pred = (pred > 0.5).astype(np.uint8)
    gt = (gt > 0.5).astype(np.uint8)
    
    # 计算预测结果和真实标签的交集
    inter = (pred & gt).sum()
    
    # 计算预测结果和真实标签的元素总和
    denom = pred.sum() + gt.sum()
    
    # 根据Dice系数公式计算结果
    return (2.0 * inter + eps) / (denom + eps)


def hausdorff_distance(pred: np.ndarray, gt: np.ndarray) -> float:
    """计算 Hausdorff 距离（双向取最大）。

    Hausdorff距离是衡量两个点集之间相似度的度量，常用于评估分割边界的一致性。
    算法说明：分别计算预测到标签、标签到预测的定向 Hausdorff 距离，取两者较大值。
    返回：浮点数，值越小表示边界越接近。

    参数：
        pred (np.ndarray): 预测概率或二值图，形状为[Z,Y,X]。
        gt (np.ndarray): 真实标签二值图，形状为[Z,Y,X]。
    返回：
        float: Hausdorff 距离，值越小表示预测边界与真实边界越接近。
              如果其中一个集合为空，则返回无穷大。
    """
    # 获取预测结果和真实标签中非零元素的坐标
    pred_pts = np.argwhere(pred > 0.5)
    gt_pts = np.argwhere(gt > 0.5)
    
    # 如果其中一个集合为空，则返回无穷大
    if len(pred_pts) == 0 or len(gt_pts) == 0:
        return float('inf')
    
    # 计算预测到标签的定向 Hausdorff 距离
    h1 = directed_hausdorff(pred_pts, gt_pts)[0]
    
    # 计算标签到预测的定向 Hausdorff 距离
    h2 = directed_hausdorff(gt_pts, pred_pts)[0]
    
    # 返回两个距离中的较大值
    return float(max(h1, h2))


def volume_similarity(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-8) -> float:
    """计算体积相似度（Volume Similarity）。

    体积相似度衡量预测结果与真实标签在体积上的相似程度。
    其值在0到1之间，1表示体积完全一致，0表示体积差异最大。

    参数：
        pred (np.ndarray): 预测概率或二值图，形状为[Z,Y,X]。
        gt (np.ndarray): 真实标签二值图，形状为[Z,Y,X]。
        eps (float): 平滑项，用于防止分母为零的情况，默认值为1e-8。
    返回：
        float: 体积相似度，范围[0,1]，值越大表示体积越接近。
    """
    # 计算预测结果和真实标签的体积（非零元素个数）
    vp = (pred > 0.5).sum()
    vg = (gt > 0.5).sum()
    
    # 根据体积相似度公式计算结果
    return 1.0 - (abs(vp - vg) / (vp + vg + eps))


@torch.no_grad()
def torch_dice(y_pred: torch.Tensor, y_true: torch.Tensor, eps: float = 1e-6) -> float:
    """在 Torch 张量上计算 Dice（阈值化后）。

    该函数在训练过程中用于实时监控Dice系数，与Dice损失函数相对应。

    参数：
        y_pred (torch.Tensor): 预测概率 `[B,1,Z,Y,X]`（Sigmoid 后）。
        y_true (torch.Tensor): 目标标签 `[B,1,Z,Y,X]`（0/1）。
        eps (float): 平滑项，用于防止分母为零的情况，默认值为1e-6。
    返回：
        float: Dice 系数，范围 [0,1]，值越大表示预测结果越准确。
    """
    # 将预测结果转换为二值图
    y_pred = (y_pred > 0.5).float()
    
    # 确保真实标签为浮点型
    y_true = y_true.float()
    
    # 计算预测结果和真实标签的交集，沿空间维度求和
    inter = (y_pred * y_true).sum(dim=(2,3,4))
    
    # 计算预测结果和真实标签的元素总和，沿空间维度求和
    denom = y_pred.sum(dim=(2,3,4)) + y_true.sum(dim=(2,3,4))
    
    # 根据Dice系数公式计算结果
    dice = (2*inter + eps) / (denom + eps)
    
    # 返回Dice系数的均值
    return dice.mean().item()
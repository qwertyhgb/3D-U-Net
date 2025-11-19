"""
文件名称：metrics.py
文件功能：实现分割评估指标。
作者：TraeAI 助手
创建日期：2025-11-18
最后修改日期：2025-11-19
版本：v1.2
版权声明：Copyright (c) 2025, All rights reserved.

详细说明：
- dice_coefficient: 计算 Dice 系数（重叠度）
- dsc: DSC指标（与dice_coefficient相同，别名）
- hausdorff_distance: 计算 Hausdorff 距离
- nsd: 归一化表面距离（Normalized Surface Distance）
- torch_dice: 在 Torch 张量上计算 Dice 系数
"""

import numpy as np
import torch
from scipy.spatial.distance import directed_hausdorff, cdist


def dice_coefficient(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-8) -> float:
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


def dsc(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-8) -> float:
    """计算 DSC (Dice Similarity Coefficient) - 重叠度指标
    
    DSC是dice_coefficient的别名，用于明确表示这是重叠度指标。
    
    参数：
        pred: 预测概率或二值图，形状为[Z,Y,X]
        gt: 真实标签二值图，形状为[Z,Y,X]
        eps: 平滑项，默认1e-8
    
    返回：
        DSC值，范围[0,1]，越大越好
    """
    return dice_coefficient(pred, gt, eps)


def nsd(pred: np.ndarray, gt: np.ndarray, threshold: float = 2.0) -> float:
    """计算 NSD (Normalized Surface Distance) - 归一化表面距离
    
    NSD衡量预测表面与真实表面之间的距离在给定阈值内的比例。
    该指标对边界精度敏感，值越大表示边界越准确。
    
    参数：
        pred: 预测概率或二值图，形状为[Z,Y,X]
        gt: 真实标签二值图，形状为[Z,Y,X]
        threshold: 距离阈值（单位：像素），默认2.0
    
    返回：
        NSD值，范围[0,1]，越大越好
    """
    # 二值化
    pred_binary = (pred > 0.5).astype(np.uint8)
    gt_binary = (gt > 0.5).astype(np.uint8)
    
    # 提取表面点（边界点）
    from scipy.ndimage import binary_erosion
    
    # 通过腐蚀操作获取边界
    pred_surface = pred_binary ^ binary_erosion(pred_binary)
    gt_surface = gt_binary ^ binary_erosion(gt_binary)
    
    # 获取表面点坐标
    pred_pts = np.argwhere(pred_surface > 0)
    gt_pts = np.argwhere(gt_surface > 0)
    
    # 如果任一表面为空，返回0
    if len(pred_pts) == 0 or len(gt_pts) == 0:
        return 0.0
    
    # 计算预测表面到真实表面的最小距离
    distances_pred_to_gt = cdist(pred_pts, gt_pts, metric='euclidean').min(axis=1)
    
    # 计算真实表面到预测表面的最小距离
    distances_gt_to_pred = cdist(gt_pts, pred_pts, metric='euclidean').min(axis=1)
    
    # 合并所有距离
    all_distances = np.concatenate([distances_pred_to_gt, distances_gt_to_pred])
    
    # 计算在阈值内的点的比例
    nsd_value = (all_distances <= threshold).sum() / len(all_distances)
    
    return float(nsd_value)


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
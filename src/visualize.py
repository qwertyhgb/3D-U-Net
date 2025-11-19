"""
文件名称：visualize.py
文件功能：生成预测与标签的叠加可视化图，便于定性评估。
作者：TraeAI 助手
创建日期：2025-11-18
最后修改日期：2025-11-18
版本：v1.0
版权声明：Copyright (c) 2025, All rights reserved.

详细说明：
- visualize_overlay: 绘制若干切片的预测/标签叠加图并保存
- 通过可视化方式直观展示模型预测效果
- 支持多切片对比显示，便于全面评估模型性能
- 使用不同的颜色区分预测结果和真实标签
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def visualize_overlay(volume: np.ndarray, pred: np.ndarray, gt: np.ndarray, out_png: str, n_slices: int = 6):
    """绘制若干切片的预测/标签叠加图并保存。

    通过将原始图像、预测结果和真实标签叠加显示，直观展示模型的分割效果。
    预测结果用红色显示，真实标签用绿色显示。

    参数：
        volume (np.ndarray): 原始体数据 `[Z,Y,X]`。
        pred (np.ndarray): 预测概率体 `[Z,Y,X]`。
        gt (np.ndarray): 标签体 `[Z,Y,X]`。
        out_png (str): 输出图片路径。
        n_slices (int): 采样可视化的切片数，默认为6。
    """
    # 获取总切片数并均匀采样n_slices个切片
    z = volume.shape[0]
    idxs = np.linspace(0, z-1, n_slices).astype(int)
    
    # 创建图形，设置大小
    plt.figure(figsize=(12, 2*n_slices))
    
    # 对每个选中的切片进行处理
    for i, zi in enumerate(idxs):
        # 获取原始图像切片
        img = volume[zi]
        
        # 将预测结果二值化
        pr = (pred[zi] > 0.5).astype(np.uint8)
        
        # 将真实标签二值化
        gt_ = (gt[zi] > 0.5).astype(np.uint8)
        
        # 绘制预测结果叠加图
        plt.subplot(n_slices, 2, 2*i+1)
        plt.imshow(img, cmap='gray')  # 显示原始图像
        plt.imshow(np.ma.masked_where(pr==0, pr), cmap='Reds', alpha=0.5)  # 红色显示预测结果
        plt.axis('off')  # 关闭坐标轴
        plt.title(f'Slice {zi} Pred')  # 设置标题
        
        # 绘制真实标签叠加图
        plt.subplot(n_slices, 2, 2*i+2)
        plt.imshow(img, cmap='gray')  # 显示原始图像
        plt.imshow(np.ma.masked_where(gt_==0, gt_), cmap='Greens', alpha=0.5)  # 绿色显示真实标签
        plt.axis('off')  # 关闭坐标轴
        plt.title(f'Slice {zi} GT')  # 设置标题
    
    # 调整布局并保存图像
    plt.tight_layout()
    plt.savefig(out_png)
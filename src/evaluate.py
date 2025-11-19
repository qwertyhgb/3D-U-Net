"""
文件名称：evaluate.py
文件功能：使用各折的最佳模型对验证集进行评估，输出量化指标与可视化结果。
作者：TraeAI 助手
创建日期：2025-11-18
最后修改日期：2025-11-18
版本：v1.0
版权声明：Copyright (c) 2025, All rights reserved.

详细说明：
- 加载各折训练得到的最佳模型
- 对验证集进行预测
- 计算Dice系数、Hausdorff距离和体积相似度等评估指标
- 生成预测结果的可视化图像
- 将评估结果保存到CSV文件中
"""

import os
import csv
import numpy as np
import torch

from .config import get_data_config
from .dataset import build_splits, ProstateDataset
from .unet3d import UNet3D
from .metrics import dice_coefficient, hausdorff_distance, volume_similarity
from .utils import save_nifti_like
from .visualize import visualize_overlay


@torch.no_grad()
def evaluate_fold(fold_idx: int, val_idx: list, cases: list, device: torch.device, data_config: dict):
    """评估单个折的验证集并保存结果。

    加载指定折的最佳模型，对验证集进行预测，并计算评估指标。

    参数：
        fold_idx (int): 折编号。
        val_idx (list): 验证集索引列表。
        cases (list): 所有病例信息列表。
        device (torch.device): 计算设备（CPU或GPU）。
        data_config (dict): 数据配置字典。
    """
    modalities = data_config.get('modalities', ['DWI'])
    
    # 创建验证集数据加载器
    ds = ProstateDataset([cases[i] for i in val_idx], augment=False, modalities=modalities)
    loader = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False)
    
    # 初始化模型
    model = UNet3D(in_channels=len(modalities))
    
    # 加载最佳模型权重
    models_dir = os.path.join(data_config['output_dir'], 'models')
    model.load_state_dict(torch.load(os.path.join(models_dir, f'fold_{fold_idx}_best.pth'), map_location=device))
    model.to(device)
    model.eval()

    # 创建评估结果输出文件
    logs_dir = os.path.join(data_config['output_dir'], 'logs')
    preds_dir = os.path.join(data_config['output_dir'], 'preds')
    out_csv = os.path.join(logs_dir, f'fold_{fold_idx}_eval.csv')
    with open(out_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        # 写入表头
        writer.writerow(['id', 'dice', 'hausdorff', 'volume_similarity'])
        
        # 遍历验证集进行评估
        for batch in loader:
            case_id = batch['id'][0]
            x = batch['image'].to(device)
            y = batch['label'].to(device)
            
            # 模型预测
            yhat = model(x)
            
            # 将预测结果和真实标签转换为numpy数组
            pred = yhat.squeeze().cpu().numpy()
            gt = y.squeeze().cpu().numpy()
            arr = x.squeeze().cpu().numpy()
            
            # 处理多模态数据，只保留第一模态用于可视化
            if arr.ndim == 4:
                img = arr[0]
            else:
                img = arr

            # 计算评估指标
            d = dice_coefficient(pred, gt)
            h = hausdorff_distance(pred, gt)
            v = volume_similarity(pred, gt)
            
            # 将评估结果写入CSV文件
            writer.writerow([case_id, d, h, v])

            # 保存预测体与叠加可视化
            out_pred = os.path.join(preds_dir, f'{case_id}_pred.nii')
            # 获取参考路径
            ref_path = cases[val_idx[0]].get('image') or cases[val_idx[0]]['images'][modalities[0]]
            save_nifti_like(ref_path, pred, out_pred)
            out_vis = os.path.join(preds_dir, f'{case_id}_overlay.png')
            visualize_overlay(img, pred, gt, out_vis)


def run_eval():
    """运行所有折的评估流程。

    对5折交叉验证的每一折都进行评估。
    """
    # 获取配置
    data_config = get_data_config()
    modalities = data_config.get('modalities', ['DWI'])
    num_folds = data_config.get('num_folds', 5)
    
    # 设置计算设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 构建数据集划分
    folds, cases = build_splits(num_folds, modalities=modalities)
    
    # 对每一折进行评估
    for i, (_, val_idx) in enumerate(folds):
        print(f"--- Evaluating Fold {i} ---")
        evaluate_fold(i, val_idx, cases, device, data_config)


if __name__ == '__main__':
    run_eval()
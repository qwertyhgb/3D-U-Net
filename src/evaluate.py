"""
文件名称：evaluate.py
文件功能：使用各折的最佳模型对验证集进行评估，输出量化指标与可视化结果。
作者：TraeAI 助手
创建日期：2025-11-18
最后修改日期：2025-11-19
版本：v1.1
版权声明：Copyright (c) 2025, All rights reserved.

详细说明：
- 加载各折训练得到的最佳模型
- 对验证集进行预测
- 计算Dice、DSC、Hausdorff和NSD等评估指标
- 生成预测结果的可视化图像
- 将评估结果保存到CSV文件中
- 使用logging系统记录评估过程
"""

import os
import csv
import logging
import numpy as np
import torch

from .config import get_data_config, get_training_config
from .dataset import build_splits, ProstateDataset
from .unet3d import UNet3D
from .metrics import dice_coefficient, dsc, hausdorff_distance, nsd
from .utils import save_nifti_like
from .visualize import visualize_overlay
from .logger import EvaluationLogger


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
    # 初始化评估日志记录器
    logger = EvaluationLogger(fold_idx, data_config['output_dir'])
    
    modalities = data_config.get('modalities', ['DWI'])
    logger.log_start(len(val_idx))
    
    # 创建验证集数据加载器
    ds = ProstateDataset([cases[i] for i in val_idx], augment=False, modalities=modalities)
    loader = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False)
    
    # 初始化模型
    model = UNet3D(in_channels=len(modalities))
    
    # 加载最佳模型权重
    models_dir = os.path.join(data_config['output_dir'], 'models')
    model_path = os.path.join(models_dir, f'fold_{fold_idx}_best.pth')
    
    try:
        checkpoint = torch.load(model_path, map_location=device)
        # 检查是否是完整的checkpoint还是只有state_dict
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            logger.logger.info(f"加载模型检查点: Epoch {checkpoint.get('epoch', 'N/A')}")
        else:
            model.load_state_dict(checkpoint)
            logger.logger.info("加载模型state_dict")
    except Exception as e:
        logger.logger.warning(f"加载模型时出错: {e}")
        logger.logger.info("尝试直接加载state_dict...")
        model.load_state_dict(torch.load(model_path, map_location=device))
    
    model.to(device)
    model.eval()

    # 创建评估结果输出文件
    logs_dir = os.path.join(data_config['output_dir'], 'logs')
    preds_dir = os.path.join(data_config['output_dir'], 'preds')
    out_csv = os.path.join(logs_dir, f'fold_{fold_idx}_eval.csv')
    with open(out_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        # 写入表头
        writer.writerow(['id', 'dice', 'dsc', 'hausdorff', 'nsd'])
        
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
            dice = dice_coefficient(pred, gt)
            dsc_value = dsc(pred, gt)
            hd = hausdorff_distance(pred, gt)
            nsd_value = nsd(pred, gt)
            
            # 记录样本评估结果
            logger.log_sample(case_id, {
                'dice': dice,
                'dsc': dsc_value,
                'hausdorff': hd,
                'nsd': nsd_value
            })
            
            # 将评估结果写入CSV文件
            writer.writerow([case_id, dice, dsc_value, hd, nsd_value])

            # 保存预测体与叠加可视化
            out_pred = os.path.join(preds_dir, f'{case_id}_pred.nii')
            # 获取参考路径
            ref_path = cases[val_idx[0]].get('image') or cases[val_idx[0]]['images'][modalities[0]]
            save_nifti_like(ref_path, pred, out_pred)
            out_vis = os.path.join(preds_dir, f'{case_id}_overlay.png')
            visualize_overlay(img, pred, gt, out_vis)
    
    logger.log_complete()


def run_eval():
    """运行所有折的评估流程。

    对5折交叉验证的每一折都进行评估。
    """
    from .logger import setup_logger, log_system_info
    
    # 设置主日志记录器
    main_logger = setup_logger(
        'evaluation_main',
        log_file=os.path.join('./outputs/logs', 'evaluation_main.log'),
        level=logging.INFO
    )
    
    # 获取配置
    data_config = get_data_config()
    training_config = get_training_config()
    modalities = data_config.get('modalities', ['DWI'])
    num_folds = training_config.get('num_folds', 5)
    
    # 设置计算设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    main_logger.info("="*60)
    main_logger.info("模型评估")
    main_logger.info("="*60)
    
    # 记录系统信息
    log_system_info()
    
    main_logger.info(f"设备: {device}")
    main_logger.info(f"模态: {modalities}")
    main_logger.info(f"折数: {num_folds}")
    main_logger.info("="*60)
    
    # 构建数据集划分
    folds, cases = build_splits(num_folds, modalities=modalities)
    
    # 对每一折进行评估
    for i, (_, val_idx) in enumerate(folds):
        main_logger.info("="*60)
        main_logger.info(f"评估 Fold {i} (验证集: {len(val_idx)} 个样本)")
        main_logger.info("="*60)
        try:
            evaluate_fold(i, val_idx, cases, device, data_config)
            main_logger.info(f"✓ Fold {i} 评估完成")
        except Exception as e:
            main_logger.error(f"✗ Fold {i} 评估失败: {e}", exc_info=True)


if __name__ == '__main__':
    run_eval()
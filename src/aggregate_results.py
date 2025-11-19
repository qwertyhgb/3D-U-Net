"""
文件名称：aggregate_results.py
文件功能：汇总各折评估 CSV，计算整体均值与标准差并打印。
作者：TraeAI 助手
创建日期：2025-11-18
最后修改日期：2025-11-18
版本：v1.0
版权声明：Copyright (c) 2025, All rights reserved.

详细说明：
- 读取所有折的评估结果文件
- 计算各项指标的均值和标准差
- 输出统计结果以便全面评估模型性能
- 支持5折交叉验证结果的汇总分析
"""

import os
import csv
import numpy as np
from .config import get_data_config, get_training_config

def main():
    """读取评估文件并输出总体统计。

    遍历所有折的评估结果文件，收集Dice系数、Hausdorff距离和体积相似度等指标，
    计算它们的均值和标准差并打印输出。
    """
    # 获取配置
    data_config = get_data_config()
    training_config = get_training_config()
    logs_dir = os.path.join(data_config['output_dir'], 'logs')
    num_folds = training_config.get('num_folds', 5)
    
    # 初始化指标列表
    dices, hausdorffs, vols = [], [], []
    
    # 遍历所有折的评估文件
    for fold in range(num_folds):
        path = os.path.join(logs_dir, f'fold_{fold}_eval.csv')
        # 检查文件是否存在
        if not os.path.exists(path):
            continue
        
        # 读取评估文件
        with open(path, 'r') as f:
            reader = csv.DictReader(f)
            # 遍历每一行数据
            for row in reader:
                # 收集各项指标
                dices.append(float(row['dice']))
                hausdorffs.append(float(row['hausdorff']))
                vols.append(float(row['volume_similarity']))
    
    # 检查是否有数据
    if len(dices) == 0:
        print('No evaluation files found')
        return
    
    # 计算并输出各项指标的统计结果
    print(f'Dice: mean={np.mean(dices):.4f}, std={np.std(dices):.4f}')
    print(f'Hausdorff: mean={np.mean(hausdorffs):.2f}, std={np.std(hausdorffs):.2f}')
    print(f'Volume similarity: mean={np.mean(vols):.4f}, std={np.std(vols):.4f}')

if __name__ == '__main__':
    main()
#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
数据增强示例脚本
演示如何使用新的数据增强功能
"""

import os
import sys
import yaml
import torch
import torchio as tio
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.dataset import ProstateDataset
from src.torchio_transforms import (
    get_train_transforms, 
    get_light_transforms,
    get_intensity_only_transforms,
    get_adaptive_train_transforms
)
from src.transforms3d import adaptive_augmentations

def load_config(config_path):
    """加载配置文件"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def visualize_augmentation(original, augmented, title1="Original", title2="Augmented"):
    """可视化原始图像和增强后的图像"""
    # 选择中间切片
    mid_slice = original.shape[2] // 2
    
    plt.figure(figsize=(12, 6))
    
    # 原始图像
    plt.subplot(1, 2, 1)
    plt.imshow(original[:, :, mid_slice], cmap='gray')
    plt.title(title1)
    plt.axis('off')
    
    # 增强后图像
    plt.subplot(1, 2, 2)
    plt.imshow(augmented[:, :, mid_slice], cmap='gray')
    plt.title(title2)
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

def demo_augmentation_strategies():
    """演示不同的数据增强策略"""
    print("数据增强策略演示")
    print("=" * 50)
    
    # 加载配置
    config_path = "../config.yml"
    if not os.path.exists(config_path):
        print(f"配置文件 {config_path} 不存在，使用默认配置")
        config = {}
    else:
        config = load_config(config_path)
    
    # 创建一个模拟的3D医学图像
    shape = (128, 128, 64)
    image = np.random.rand(*shape).astype(np.float32) * 255
    label = np.zeros_like(image, dtype=np.uint8)
    label[40:80, 40:80, 20:40] = 1  # 添加一个模拟的分割区域
    
    print(f"原始图像形状: {image.shape}")
    print(f"标签形状: {label.shape}")
    
    # 1. 标准增强策略
    print("\n1. 标准增强策略")
    standard_transform = get_train_transforms(config.get('augmentation', {}))
    subject = tio.Subject(
        image=tio.ScalarImage(tensor=torch.from_numpy(image).unsqueeze(0)),
        label=tio.LabelMap(tensor=torch.from_numpy(label).unsqueeze(0))
    )
    transformed = standard_transform(subject)
    visualize_augmentation(
        image, 
        transformed.image.data.squeeze(0).numpy(),
        "原始图像", 
        "标准增强后"
    )
    
    # 2. 轻量级增强策略
    print("\n2. 轻量级增强策略")
    light_transform = get_light_transforms(config.get('augmentation', {}))
    transformed = light_transform(subject)
    visualize_augmentation(
        image, 
        transformed.image.data.squeeze(0).numpy(),
        "原始图像", 
        "轻量级增强后"
    )
    
    # 3. 仅强度变换策略
    print("\n3. 仅强度变换策略")
    intensity_transform = get_intensity_only_transforms(config.get('augmentation', {}))
    transformed = intensity_transform(subject)
    visualize_augmentation(
        image, 
        transformed.image.data.squeeze(0).numpy(),
        "原始图像", 
        "仅强度变换后"
    )
    
    # 4. 自适应增强策略 (不同epoch)
    print("\n4. 自适应增强策略")
    for epoch in [0, 5, 10]:
        adaptive_transform = get_adaptive_train_transforms(
            config.get('augmentation', {}), 
            epoch=epoch
        )
        transformed = adaptive_transform(subject)
        visualize_augmentation(
            image, 
            transformed.image.data.squeeze(0).numpy(),
            "原始图像", 
            f"自适应增强 (epoch {epoch})"
        )

def demo_dataset_with_augmentation():
    """演示如何在数据集中使用数据增强"""
    print("\n数据集增强演示")
    print("=" * 50)
    
    # 加载配置
    config_path = "../config.yml"
    if not os.path.exists(config_path):
        print(f"配置文件 {config_path} 不存在，使用默认配置")
        config = {}
    else:
        config = load_config(config_path)
    
    # 模拟一些病例数据
    cases = [
        {"case_id": "case_001", "image_path": "path/to/image1.nii", "label_path": "path/to/label1.nii"},
        {"case_id": "case_002", "image_path": "path/to/image2.nii", "label_path": "path/to/label2.nii"},
    ]
    
    # 创建使用不同增强策略的数据集
    strategies = ['standard', 'light', 'intensity_only', 'adaptive']
    
    for strategy in strategies:
        print(f"\n创建使用 {strategy} 策略的数据集")
        
        # 更新配置中的策略
        aug_config = config.get('augmentation', {})
        aug_config['strategy'] = strategy
        
        # 创建数据集
        dataset = ProstateDataset(
            cases=cases,
            augment=True,
            augmentation_config=aug_config
        )
        
        # 如果是自适应策略，设置epoch
        if strategy == 'adaptive':
            dataset.set_epoch(5)  # 设置为第5个epoch
        
        print(f"数据集大小: {len(dataset)}")
        print(f"使用的增强策略: {dataset.augmentation_strategy}")
        
        # 创建数据加载器
        dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
        
        # 获取一个批次的数据
        try:
            for batch in dataloader:
                images, labels = batch
                print(f"批次图像形状: {images.shape}")
                print(f"批次标签形状: {labels.shape}")
                break  # 只处理第一个批次
        except Exception as e:
            print(f"数据加载出错: {e}")
            print("这是因为我们使用了模拟的病例数据，实际使用时需要提供真实的图像路径")

def demo_custom_augmentation_config():
    """演示自定义增强配置"""
    print("\n自定义增强配置演示")
    print("=" * 50)
    
    # 自定义增强配置
    custom_config = {
        'strategy': 'light',
        'flip': {
            'axes': [0, 1],
            'probability': 0.7
        },
        'affine': {
            'degrees': 5,
            'translation': 3,
            'scales': 0.05,
            'probability': 0.8
        },
        'elastic': {
            'num_control_points': 5,
            'max_displacement': 3,
            'probability': 0.5
        },
        'intensity': {
            'noise_std': 0.02,
            'blur_std': 0.5
        }
    }
    
    # 创建一个模拟的3D医学图像
    shape = (128, 128, 64)
    image = np.random.rand(*shape).astype(np.float32) * 255
    label = np.zeros_like(image, dtype=np.uint8)
    label[40:80, 40:80, 20:40] = 1  # 添加一个模拟的分割区域
    
    print(f"使用自定义配置创建增强变换")
    
    # 使用自定义配置创建增强变换
    transform = get_train_transforms(custom_config)
    
    subject = tio.Subject(
        image=tio.ScalarImage(tensor=torch.from_numpy(image).unsqueeze(0)),
        label=tio.LabelMap(tensor=torch.from_numpy(label).unsqueeze(0))
    )
    
    transformed = transform(subject)
    
    visualize_augmentation(
        image, 
        transformed.image.data.squeeze(0).numpy(),
        "原始图像", 
        "自定义增强后"
    )

def main():
    """主函数"""
    print("数据增强功能演示")
    print("=" * 50)
    
    try:
        # 演示不同的增强策略
        demo_augmentation_strategies()
        
        # 演示数据集中的增强
        demo_dataset_with_augmentation()
        
        # 演示自定义增强配置
        demo_custom_augmentation_config()
        
        print("\n演示完成！")
        
    except Exception as e:
        print(f"演示过程中出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
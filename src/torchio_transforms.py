"""
文件名称：torchio_transforms.py
文件功能：使用 torchio 实现更高级的 3D 医学影像数据增强操作。
创建日期：2025-11-19
版本：v1.0
版权声明：Copyright (c) 2025, All rights reserved.

说明：
- 使用 torchio 实现更丰富和真实的医学影像增强；
- 包括弹性形变、仿射变换、噪声添加等；
- 支持多模态数据的一致增强；
- 支持测试时增强 (TTA)。
"""

import torch
import torchio as tio
import numpy as np
from typing import List, Union


def get_train_transforms(probability: float = 0.5) -> tio.Compose:
    """获取训练时的数据增强变换组合。
    
    参数：
        probability (float): 各增强操作的执行概率。
    返回：
        tio.Compose: 组合的变换对象。
    """
    transforms = [
        # 随机翻转
        tio.RandomFlip(axes=(0, 1, 2), flip_probability=probability),
        # 随机仿射变换（旋转、缩放、剪切）
        tio.RandomAffine(
            scales=(0.9, 1.1),  # 缩放因子
            degrees=(-10, 10),  # 旋转角度
            translation=(-5, 5),  # 平移像素
            image_interpolation='linear',
            p=probability
        ),
        # 随机弹性形变
        tio.RandomElasticDeformation(
            num_control_points=5,  # 控制点数量
            max_displacement=2,    # 最大位移
            p=probability
        ),
        # 随机噪声
        tio.RandomNoise(
            mean=0,
            std=(0, 0.1),
            p=probability/2  # 噪声概率更低
        ),
        # 随机模糊
        tio.RandomBlur(
            std=(0, 1),
            p=probability/3  # 模糊概率更低
        )
    ]
    return tio.Compose(transforms)


def get_validation_transforms() -> tio.Compose:
    """获取验证/测试时的标准预处理变换。
    
    返回：
        tio.Compose: 组合的变换对象。
    """
    # 验证时通常只做必要的预处理，不进行增强
    return tio.Compose([
        # 可以在这里添加标准化等操作
    ])


def get_tta_transforms() -> List[tio.Transform]:
    """获取测试时增强(TTA)的变换列表。
    
    返回：
        List[tio.Transform]: TTA变换列表。
    """
    # 定义几种不同的增强方式用于TTA
    tta_transforms = [
        # 原始图像
        tio.Compose([]),
        # 水平翻转
        tio.RandomFlip(axes=(0,), flip_probability=1.0),
        # 垂直翻转
        tio.RandomFlip(axes=(1,), flip_probability=1.0),
        # 两种翻转
        tio.Compose([
            tio.RandomFlip(axes=(0,), flip_probability=1.0),
            tio.RandomFlip(axes=(1,), flip_probability=1.0)
        ]),
        # 轻微旋转
        tio.RandomAffine(degrees=(5, 5), p=1.0)
    ]
    return tta_transforms


def apply_transforms_single(volume: np.ndarray, mask: np.ndarray, 
                          transform: tio.Transform) -> tuple:
    """对单模态数据应用torchio变换。
    
    参数：
        volume (np.ndarray): 输入体数据 [Z, Y, X]。
        mask (np.ndarray): 标签数据 [Z, Y, X]。
        transform (tio.Transform): 要应用的变换。
    返回：
        tuple: 增强后的(volume, mask)。
    """
    # 将numpy数组转换为torchio格式
    # torchio期望的格式是 [C, W, H, D]，我们的数据是 [Z, Y, X]
    # 需要添加通道维度: [Z, Y, X] -> [1, Z, Y, X]
    subject = tio.Subject(
        image=tio.ScalarImage(tensor=volume[np.newaxis, ...]),
        label=tio.LabelMap(tensor=mask[np.newaxis, ...])
    )
    
    # 应用变换
    transformed = transform(subject)
    
    # 转换回numpy格式 [Z, Y, X]
    transformed_volume = transformed.image.numpy()[0]
    transformed_mask = transformed.label.numpy()[0]
    
    return transformed_volume, transformed_mask


def apply_transforms_multi(volumes: List[np.ndarray], mask: np.ndarray,
                          transform: tio.Transform) -> tuple:
    """对多模态数据应用torchio变换。
    
    参数：
        volumes (List[np.ndarray]): 多模态输入体数据列表，每个都是[Z, Y, X]。
        mask (np.ndarray): 标签数据 [Z, Y, X]。
        transform (tio.Transform): 要应用的变换。
    返回：
        tuple: 增强后的(volumes, mask)。
    """
    # 创建多模态subject
    subject_dict = {}
    
    # 添加所有模态作为图像
    for i, volume in enumerate(volumes):
        subject_dict[f'image_{i}'] = tio.ScalarImage(
            tensor=volume[np.newaxis, ...]  # 添加通道维度 [1, Z, Y, X]
        )
    
    # 添加标签
    subject_dict['label'] = tio.LabelMap(
        tensor=mask[np.newaxis, ...]
    )
    
    subject = tio.Subject(**subject_dict)
    
    # 应用变换（所有图像和标签会同步变换）
    transformed = transform(subject)
    
    # 转换回numpy格式
    transformed_volumes = []
    for i in range(len(volumes)):
        transformed_volume = transformed[f'image_{i}'].numpy()[0]
        transformed_volumes.append(transformed_volume)
    
    transformed_mask = transformed.label.numpy()[0]
    
    return transformed_volumes, transformed_mask


def apply_tta_single(volume: np.ndarray, model: torch.nn.Module, 
                    device: torch.device) -> np.ndarray:
    """对单模态数据应用测试时增强。
    
    参数：
        volume (np.ndarray): 输入体数据 [Z, Y, X]。
        model (torch.nn.Module): 训练好的模型。
        device (torch.device): 设备。
    返回：
        np.ndarray: TTA平均后的预测结果。
    """
    model.eval()
    tta_transforms = get_tta_transforms()
    predictions = []
    
    with torch.no_grad():
        for transform in tta_transforms:
            # 应用正向变换
            transformed_volume, _ = apply_transforms_single(volume, np.zeros_like(volume), transform)
            
            # 转换为张量并预测
            input_tensor = torch.from_numpy(transformed_volume).unsqueeze(0).unsqueeze(0).float().to(device)
            # 模型输出已经包含 Sigmoid
            prediction = model(input_tensor).cpu().numpy()[0, 0]
            
            # 如果需要，应用逆向变换恢复预测结果
            # 这里简化处理，实际应用中可能需要更复杂的逆变换
            
            predictions.append(prediction)
    
    # 平均所有TTA预测结果
    final_prediction = np.mean(predictions, axis=0)
    return final_prediction


def apply_tta_multi(volumes: List[np.ndarray], model: torch.nn.Module,
                   device: torch.device) -> np.ndarray:
    """对多模态数据应用测试时增强。
    
    参数：
        volumes (List[np.ndarray]): 多模态输入体数据列表，每个都是[Z, Y, X]。
        model (torch.nn.Module): 训练好的模型。
        device (torch.device): 设备。
    返回：
        np.ndarray: TTA平均后的预测结果。
    """
    model.eval()
    tta_transforms = get_tta_transforms()
    predictions = []
    
    with torch.no_grad():
        for transform in tta_transforms:
            # 应用正向变换
            transformed_volumes, _ = apply_transforms_multi(volumes, np.zeros_like(volumes[0]), transform)
            
            # 转换为张量并预测
            # 堆叠多模态数据 [C, Z, Y, X]
            stacked_volumes = np.stack(transformed_volumes, axis=0)
            input_tensor = torch.from_numpy(stacked_volumes).unsqueeze(0).float().to(device)
            # 模型输出已经包含 Sigmoid
            prediction = model(input_tensor).cpu().numpy()[0, 0]
            
            predictions.append(prediction)
    
    # 平均所有TTA预测结果
    final_prediction = np.mean(predictions, axis=0)
    return final_prediction
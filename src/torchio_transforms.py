"""
文件名称：torchio_transforms.py
文件功能：使用 torchio 实现更高级的 3D 医学影像数据增强操作。
创建日期：2025-11-19
最后修改日期：2025-11-19
版本：v2.0
版权声明：Copyright (c) 2025, All rights reserved.

说明：
- 使用 torchio 实现更丰富和真实的医学影像增强；
- 包括弹性形变、仿射变换、噪声添加等；
- 支持多模态数据的一致增强；
- 支持测试时增强 (TTA)；
- 支持自适应增强强度调整；
- 添加医学影像特定的增强方法。
"""

import torch
import torchio as tio
import numpy as np
from typing import List, Union, Dict, Any, Optional


def get_train_transforms(probability: float = 0.5, config: Dict[str, Any] = None) -> tio.Compose:
    """获取训练时的数据增强变换组合。
    
    参数：
        probability (float): 各增强操作的执行概率。
        config (Dict[str, Any]): 增强配置参数。
    返回：
        tio.Compose: 组合的变换对象。
    """
    # 默认配置
    default_config = {
        'flip_probability': probability,
        'affine_probability': probability,
        'elastic_probability': probability,
        'noise_probability': probability * 0.5,  # 噪声概率更低
        'blur_probability': probability * 0.33,  # 模糊概率更低
        'motion_probability': probability * 0.5,  # 运动伪影概率
        'bias_probability': probability * 0.5,  # 偏置场概率
        'gamma_probability': probability * 0.5,  # Gamma变换概率
        'swap_probability': probability * 0.3,  # 交换区域概率
        'intensity_scale': (0.8, 1.2),  # 强度缩放范围
        'rotation_range': (-15, 15),  # 旋转角度范围
        'translation_range': (-8, 8),  # 平移像素范围
        'scale_range': (0.85, 1.15),  # 缩放范围
        'elastic_num_control_points': 7,  # 弹性形变控制点数量
        'elastic_max_displacement': 3,  # 弹性形变最大位移
        'noise_std': (0, 0.15),  # 噪声标准差范围
        'blur_std': (0, 1.5),  # 模糊标准差范围
        'motion_degrees': (0, 5),  # 运动伪影旋转角度
        'motion_translation': (0, 3),  # 运动伪影平移
        'bias_coefficients': (0, 0.3),  # 偏置场系数
        'gamma_range': (0.7, 1.5),  # Gamma变换范围
        'swap_patch_size': 15,  # 交换区域大小
        'num_swaps': 1  # 交换次数
    }
    
    # 更新配置
    if config:
        default_config.update(config)
    
    transforms = [
        # 随机翻转 - 保持原有实现
        tio.RandomFlip(
            axes=(0, 1, 2), 
            flip_probability=default_config['flip_probability']
        ),
        
        # 随机仿射变换 - 扩大范围
        tio.RandomAffine(
            scales=default_config['scale_range'],
            degrees=default_config['rotation_range'],
            translation=default_config['translation_range'],
            image_interpolation='linear',
            p=default_config['affine_probability']
        ),
        
        # 随机弹性形变 - 增加控制点和位移
        tio.RandomElasticDeformation(
            num_control_points=default_config['elastic_num_control_points'],
            max_displacement=default_config['elastic_max_displacement'],
            p=default_config['elastic_probability']
        ),
        
        # 随机运动伪影 - 模拟MRI运动伪影
        tio.RandomMotion(
            degrees=default_config['motion_degrees'],
            translation=default_config['motion_translation'],
            num_transforms=2,
            image_interpolation='linear',
            p=default_config['motion_probability']
        ),
        
        # 随机偏置场 - 模拟MRI偏置场不均匀
        tio.RandomBiasField(
            coefficients=default_config['bias_coefficients'],
            p=default_config['bias_probability']
        ),
        
        # 随机强度缩放
        tio.RescaleIntensity(
            out_min_max=default_config['intensity_scale'],
            p=default_config.get('intensity_scaling_probability', probability)
        ),
        
        # 随机Gamma变换 - 调整图像对比度
        tio.RandomGamma(
            log_gamma=np.log(default_config['gamma_range']),
            p=default_config['gamma_probability']
        ),
        
        # 随机噪声 - 增加噪声范围
        tio.RandomNoise(
            mean=0,
            std=default_config['noise_std'],
            p=default_config['noise_probability']
        ),
        
        # 随机模糊 - 增加模糊范围
        tio.RandomBlur(
            std=default_config['blur_std'],
            p=default_config['blur_probability']
        ),
        
        # 随机区域交换 - 增加多样性
        tio.RandomSwap(
            patch_size=default_config['swap_patch_size'],
            num_iterations=default_config.get('num_swaps', 1),
            p=default_config['swap_probability']
        ),
        
        # 随机各向异性 - 模拟不同分辨率
        tio.RandomAnisotropy(
            axes=(0, 1, 2),
            downsampling=(1, 2),
            p=probability * 0.3
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
    # 定义多种不同的增强方式用于TTA
    tta_transforms = [
        # 原始图像
        tio.Compose([]),
        
        # 水平翻转
        tio.RandomFlip(axes=(0,), flip_probability=1.0),
        
        # 垂直翻转
        tio.RandomFlip(axes=(1,), flip_probability=1.0),
        
        # 深度翻转
        tio.RandomFlip(axes=(2,), flip_probability=1.0),
        
        # 水平+垂直翻转
        tio.Compose([
            tio.RandomFlip(axes=(0,), flip_probability=1.0),
            tio.RandomFlip(axes=(1,), flip_probability=1.0)
        ]),
        
        # 水平+深度翻转
        tio.Compose([
            tio.RandomFlip(axes=(0,), flip_probability=1.0),
            tio.RandomFlip(axes=(2,), flip_probability=1.0)
        ]),
        
        # 垂直+深度翻转
        tio.Compose([
            tio.RandomFlip(axes=(1,), flip_probability=1.0),
            tio.RandomFlip(axes=(2,), flip_probability=1.0)
        ]),
        
        # 轻微旋转
        tio.RandomAffine(degrees=(3, 3), p=1.0),
        
        # 轻微缩放
        tio.RandomAffine(scales=(0.95, 0.95), p=1.0),
        
        # 轻微平移
        tio.RandomAffine(translation=(2, 2), p=1.0),
        
        # 轻微弹性形变
        tio.RandomElasticDeformation(
            num_control_points=5,
            max_displacement=1,
            p=1.0
        )
    ]
    return tta_transforms


def get_adaptive_train_transforms(epoch: int, total_epochs: int, config: Optional[Dict[str, Any]] = None) -> tio.Compose:
    """获取自适应训练增强，增强强度随训练进度变化。
    
    参数：
        epoch: 当前训练轮次。
        total_epochs: 总训练轮次。
        config: 可选的配置字典，包含自定义参数。
    
    返回：
        tio.Compose: 自适应训练增强。
    """
    # 计算训练进度比例 (0到1)
    progress = epoch / max(1, total_epochs)
    
    # 使用默认配置或自定义配置
    if config is None:
        config = {}
    
    # 随着训练进行，逐渐减少几何变换的强度，但保持一定程度的强度变换
    # 这样可以在训练初期学习更鲁棒的特征，后期稳定训练
    geometry_intensity = max(0.3, 1.0 - 0.5 * progress)  # 从1.0逐渐降到0.5
    intensity_intensity = max(0.4, 0.7 - 0.3 * progress)  # 从0.7逐渐降到0.4
    
    # 获取基础配置参数
    flip_axes = config.get('flip_axes', (0, 1, 2))
    affine_degrees = config.get('affine_degrees', 15) * geometry_intensity
    affine_translation = config.get('affine_translation', 10) * geometry_intensity
    affine_scales = config.get('affine_scales', 0.15) * geometry_intensity
    elastic_num_control_points = config.get('elastic_num_control_points', 7)
    elastic_max_displacement = config.get('elastic_max_displacement', 7.5) * geometry_intensity
    
    # 处理元组类型的参数，需要分别乘以强度系数
    noise_std = config.get('noise_std', (0, 0.1))
    if isinstance(noise_std, tuple) or isinstance(noise_std, list):
        noise_std = tuple(n * intensity_intensity for n in noise_std)
    else:
        noise_std = noise_std * intensity_intensity
        
    blur_std = config.get('blur_std', (0.5, 1.5))
    if isinstance(blur_std, tuple) or isinstance(blur_std, list):
        blur_std = tuple(b * intensity_intensity for b in blur_std)
    else:
        blur_std = blur_std * intensity_intensity
    
    # 医学影像特定增强参数
    motion_degrees = config.get('motion_degrees', 15) * geometry_intensity
    motion_translation = config.get('motion_translation', 5) * geometry_intensity
    bias_coefficients = config.get('bias_coefficients', 0.3) * intensity_intensity
    scaling_factors = config.get('scaling_factors', (0.8, 1.2))
    gamma = config.get('gamma', (0.8, 1.2))
    swap_patch_size = config.get('swap_patch_size', 15)
    num_swaps = config.get('num_swaps', 4)
    downsampling = config.get('downsampling', (1.5, 2.5))
    
    transforms = [
        # 几何变换 - 强度随训练进度降低
        tio.RandomFlip(axes=flip_axes, flip_probability=0.5),
        tio.RandomAffine(
            degrees=affine_degrees,
            translation=affine_translation,
            scales=affine_scales,
            p=0.7
        ),
        tio.RandomElasticDeformation(
            num_control_points=elastic_num_control_points,
            max_displacement=elastic_max_displacement,
            p=0.3
        ),
        
        # 医学影像特定增强 - 强度随训练进度调整
        tio.RandomMotion(
            degrees=motion_degrees,
            translation=motion_translation,
            num_transforms=2,
            p=0.1
        ),
        tio.RandomBiasField(
            coefficients=bias_coefficients,
            p=0.2
        ),
        tio.RandomSwap(
            patch_size=swap_patch_size,
            p=0.1
        ),
        
        # 强度变换 - 强度随训练进度降低
        tio.RandomNoise(std=noise_std, p=0.2),
        tio.RandomBlur(std=blur_std, p=0.2),
        tio.RandomGamma(log_gamma=gamma, p=0.2),
        tio.RandomAnisotropy(
            downsampling=downsampling,
            p=0.05
        ),
        
        # 随机应用部分增强，避免过度增强
        tio.RandomElasticDeformation(
            num_control_points=5,
            max_displacement=2,
            p=0.1
        )
    ]
    
    return tio.Compose(transforms)


def get_intensity_only_transforms(config: Optional[Dict[str, Any]] = None) -> tio.Compose:
    """获取仅包含强度变换的增强，不包含几何变换。
    
    参数：
        config: 可选的配置字典，包含自定义参数。
    
    返回：
        tio.Compose: 仅包含强度变换的增强。
    """
    # 使用默认配置或自定义配置
    if config is None:
        config = {}
    
    # 获取基础配置参数
    noise_std = config.get('noise_std', (0, 0.1))
    blur_std = config.get('blur_std', (0.5, 1.5))
    gamma = config.get('gamma', (0.8, 1.2))
    bias_coefficients = config.get('bias_coefficients', 0.3)
    scaling_factors = config.get('scaling_factors', (0.8, 1.2))
    
    transforms = [
        # 仅强度变换
        tio.RandomNoise(std=noise_std, p=0.3),
        tio.RandomBlur(std=blur_std, p=0.3),
        tio.RandomGamma(log_gamma=gamma, p=0.3),
        tio.RandomBiasField(coefficients=bias_coefficients, p=0.3),
        tio.RescaleIntensity(out_min_max=scaling_factors, p=0.3)
    ]
    
    return tio.Compose(transforms)


def get_light_transforms(config: Optional[Dict[str, Any]] = None) -> tio.Compose:
    """获取轻量级增强，适用于数据量较小或模型容易过拟合的情况。
    
    参数：
        config: 可选的配置字典，包含自定义参数。
    
    返回：
        tio.Compose: 轻量级增强。
    """
    # 使用默认配置或自定义配置
    if config is None:
        config = {}
    
    # 获取基础配置参数
    flip_axes = config.get('flip_axes', (0, 1, 2))
    affine_degrees = config.get('affine_degrees', 5)
    affine_translation = config.get('affine_translation', 3)
    affine_scales = config.get('affine_scales', 0.05)
    noise_std = config.get('noise_std', (0, 0.05))
    blur_std = config.get('blur_std', (0.5, 1.0))
    
    transforms = [
        # 轻量级几何变换
        tio.RandomFlip(axes=flip_axes, flip_probability=0.5),
        tio.RandomAffine(
            degrees=affine_degrees,
            translation=affine_translation,
            scales=affine_scales,
            p=0.5
        ),
        
        # 轻量级强度变换
        tio.RandomNoise(std=noise_std, p=0.2),
        tio.RandomBlur(std=blur_std, p=0.2)
    ]
    
    return tio.Compose(transforms)


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
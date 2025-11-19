"""
3D数据增强函数

包含针对3D医学图像的各种数据增强方法，如翻转、旋转、弹性变形等。
这些函数可以单独使用，也可以组合使用，以增加训练数据的多样性。
版本: v2.0
最后修改: 2023-11-15
新增功能:
- 添加更多医学影像特定的增强方法
- 支持自适应增强强度调整
- 增加边界感知增强
"""

import numpy as np
import random
from scipy import ndimage
from scipy.ndimage import gaussian_filter, map_coordinates
from typing import Tuple, Optional, Dict, Any

def random_flip(volume: np.ndarray, mask: np.ndarray):
    """随机在三个轴上翻转体数据与标签。

    在三个维度上分别以50%的概率进行翻转，增加数据多样性。

    参数：
        volume (np.ndarray): 体数据 `[Z,Y,X]`。
        mask (np.ndarray): 标签体 `[Z,Y,X]`。
    返回：
        Tuple[np.ndarray, np.ndarray]: 翻转后的体数据与标签。
    """
    # 在三个轴上分别以50%的概率进行翻转
    for axis in range(3):
        if np.random.rand() < 0.5:
            volume = np.flip(volume, axis=axis)
            mask = np.flip(mask, axis=axis)
    return volume, mask


def random_rotation(volume: np.ndarray, mask: np.ndarray, 
                   angle_range: Tuple[float, float] = (-15, 15),
                   axes: Tuple[int, int] = (0, 1),
                   config: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, np.ndarray]:
    """随机旋转3D体积和掩码。
    
    参数：
        volume: 输入3D体积。
        mask: 输入3D掩码。
        angle_range: 旋转角度范围（度）。
        axes: 旋转的轴。
        config: 可选的配置字典，包含自定义参数。
    
    返回：
        旋转后的体积和掩码。
    """
    # 使用自定义配置或默认配置
    if config is not None:
        angle_range = config.get('angle_range', angle_range)
        axes = config.get('axes', axes)
    
    # 生成随机旋转角度
    angle = np.random.uniform(angle_range[0], angle_range[1])
    
    # 应用旋转
    volume = ndimage.rotate(volume, angle, axes=axes, reshape=False, order=1, mode='nearest')
    mask = ndimage.rotate(mask, angle, axes=axes, reshape=False, order=0, mode='nearest')
    
    return volume, mask


def random_zoom(volume: np.ndarray, mask: np.ndarray,
               zoom_range: Tuple[float, float] = (0.9, 1.1),
               config: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, np.ndarray]:
    """随机缩放3D体积和掩码。
    
    参数：
        volume: 输入3D体积。
        mask: 输入3D掩码。
        zoom_range: 缩放因子范围。
        config: 可选的配置字典，包含自定义参数。
    
    返回：
        缩放后的体积和掩码。
    """
    # 使用自定义配置或默认配置
    if config is not None:
        zoom_range = config.get('zoom_range', zoom_range)
    
    # 生成随机缩放因子
    zoom_factor = np.random.uniform(zoom_range[0], zoom_range[1])
    
    # 应用缩放
    volume = ndimage.zoom(volume, zoom_factor, order=1, mode='nearest')
    mask = ndimage.zoom(mask, zoom_factor, order=0, mode='nearest')
    
    # 如果缩放后尺寸不匹配，裁剪或填充到原始尺寸
    if volume.shape != mask.shape:
        raise ValueError("缩放后体积和掩码尺寸不匹配")
    
    # 如果缩放后尺寸大于原始尺寸，随机裁剪
    if volume.shape[0] > volume.shape[0] / zoom_factor:
        # 计算裁剪起始点
        crop_start = [
            np.random.randint(0, volume.shape[i] - int(volume.shape[i] / zoom_factor) + 1)
            for i in range(3)
        ]
        crop_end = [
            crop_start[i] + int(volume.shape[i] / zoom_factor)
            for i in range(3)
        ]
        
        # 执行裁剪
        volume = volume[crop_start[0]:crop_end[0], crop_start[1]:crop_end[1], crop_start[2]:crop_end[2]]
        mask = mask[crop_start[0]:crop_end[0], crop_start[1]:crop_end[1], crop_start[2]:crop_end[2]]
    
    # 如果缩放后尺寸小于原始尺寸，填充
    elif volume.shape[0] < volume.shape[0] / zoom_factor:
        # 计算填充量
        pad_width = [
            (int((volume.shape[i] / zoom_factor - volume.shape[i]) / 2), 
             int((volume.shape[i] / zoom_factor - volume.shape[i]) / 2))
            for i in range(3)
        ]
        
        # 如果填充量不是整数，调整
        for i in range(3):
            if sum(pad_width[i]) != (volume.shape[i] / zoom_factor - volume.shape[i]):
                pad_width[i] = (pad_width[i][0], pad_width[i][1] + 1)
        
        # 执行填充
        volume = np.pad(volume, pad_width, mode='constant', constant_values=0)
        mask = np.pad(mask, pad_width, mode='constant', constant_values=0)
    
    return volume, mask


def random_intensity_shift(volume: np.ndarray, mask: np.ndarray,
                          shift_range: Tuple[float, float] = (-0.1, 0.1),
                          config: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, np.ndarray]:
    """随机调整3D体积的强度。
    
    参数：
        volume: 输入3D体积。
        mask: 输入3D掩码。
        shift_range: 强度偏移范围（相对于最大强度的比例）。
        config: 可选的配置字典，包含自定义参数。
    
    返回：
        强度调整后的体积和原始掩码。
    """
    # 使用自定义配置或默认配置
    if config is not None:
        shift_range = config.get('shift_range', shift_range)
    
    # 计算强度偏移
    max_intensity = np.max(volume)
    shift = np.random.uniform(shift_range[0], shift_range[1]) * max_intensity
    
    # 应用强度偏移
    volume = volume + shift
    
    # 确保值在合理范围内
    volume = np.clip(volume, 0, max_intensity * 1.2)
    
    return volume, mask


def random_gamma_correction(volume: np.ndarray, mask: np.ndarray,
                           gamma_range: Tuple[float, float] = (0.8, 1.2),
                           config: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, np.ndarray]:
    """对3D体积应用随机Gamma校正。
    
    参数：
        volume: 输入3D体积。
        mask: 输入3D掩码。
        gamma_range: Gamma值范围。
        config: 可选的配置字典，包含自定义参数。
    
    返回：
        Gamma校正后的体积和原始掩码。
    """
    # 使用自定义配置或默认配置
    if config is not None:
        gamma_range = config.get('gamma_range', gamma_range)
    
    # 生成随机Gamma值
    gamma = np.random.uniform(gamma_range[0], gamma_range[1])
    
    # 归一化到[0, 1]
    min_val = np.min(volume)
    max_val = np.max(volume)
    if max_val > min_val:
        normalized = (volume - min_val) / (max_val - min_val)
    else:
        normalized = volume
    
    # 应用Gamma校正
    corrected = np.power(normalized, gamma)
    
    # 恢复原始范围
    volume = corrected * (max_val - min_val) + min_val
    
    return volume, mask


def random_noise(volume: np.ndarray, mask: np.ndarray,
                noise_type: str = 'gaussian',
                noise_range: Tuple[float, float] = (0, 0.05),
                config: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, np.ndarray]:
    """向3D体积添加随机噪声。
    
    参数：
        volume: 输入3D体积。
        mask: 输入3D掩码。
        noise_type: 噪声类型 ('gaussian', 'poisson', 'speckle')。
        noise_range: 噪声强度范围。
        config: 可选的配置字典，包含自定义参数。
    
    返回：
        添加噪声后的体积和原始掩码。
    """
    # 使用自定义配置或默认配置
    if config is not None:
        noise_type = config.get('noise_type', noise_type)
        noise_range = config.get('noise_range', noise_range)
    
    # 计算噪声强度
    noise_intensity = np.random.uniform(noise_range[0], noise_range[1])
    
    if noise_type == 'gaussian':
        # 添加高斯噪声
        noise = np.random.normal(0, noise_intensity, volume.shape)
        volume = volume + noise * np.max(volume)
    
    elif noise_type == 'poisson':
        # 添加泊松噪声
        # 确保值非负
        volume_positive = np.clip(volume, 0, None)
        # 应用泊松噪声
        noise = np.random.poisson(volume_positive * noise_intensity) / noise_intensity - volume_positive
        volume = volume + noise
    
    elif noise_type == 'speckle':
        # 添加斑点噪声
        noise = np.random.normal(0, 1, volume.shape)
        volume = volume * (1 + noise * noise_intensity)
    
    else:
        raise ValueError(f"不支持的噪声类型: {noise_type}")
    
    return volume, mask


def boundary_aware_elastic_deformation(volume: np.ndarray, mask: np.ndarray,
                                     alpha: float = 1000,
                                     sigma: float = 8,
                                     boundary_weight: float = 2.0,
                                     config: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, np.ndarray]:
    """边界感知的弹性变形，在边界区域应用更强的变形。
    
    参数：
        volume: 输入3D体积。
        mask: 输入3D掩码。
        alpha: 变形强度。
        sigma: 高斯滤波的标准差。
        boundary_weight: 边界区域的权重因子。
        config: 可选的配置字典，包含自定义参数。
    
    返回：
        变形后的体积和掩码。
    """
    # 使用自定义配置或默认配置
    if config is not None:
        alpha = config.get('alpha', alpha)
        sigma = config.get('sigma', sigma)
        boundary_weight = config.get('boundary_weight', boundary_weight)
    
    # 计算边界权重图
    # 首先计算掩码的梯度
    grad_x = np.abs(np.gradient(mask.astype(float), axis=0))
    grad_y = np.abs(np.gradient(mask.astype(float), axis=1))
    grad_z = np.abs(np.gradient(mask.astype(float), axis=2))
    grad_magnitude = np.sqrt(grad_x**2 + grad_y**2 + grad_z**2)
    
    # 归一化梯度
    if np.max(grad_magnitude) > 0:
        grad_magnitude = grad_magnitude / np.max(grad_magnitude)
    
    # 创建边界权重图，边界区域权重更高
    boundary_map = 1.0 + boundary_weight * grad_magnitude
    
    # 生成随机位移场
    shape = volume.shape
    dx = np.random.uniform(-1, 1, shape) * alpha
    dy = np.random.uniform(-1, 1, shape) * alpha
    dz = np.random.uniform(-1, 1, shape) * alpha
    
    # 应用高斯平滑
    dx = gaussian_filter(dx, sigma, mode='nearest')
    dy = gaussian_filter(dy, sigma, mode='nearest')
    dz = gaussian_filter(dz, sigma, mode='nearest')
    
    # 应用边界权重
    dx = dx * boundary_map
    dy = dy * boundary_map
    dz = dz * boundary_map
    
    # 创建网格坐标
    x, y, z = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]), indexing='ij')
    
    # 应用位移
    new_x = x + dx
    new_y = y + dy
    new_z = z + dz
    
    # 对体积和掩码进行插值
    volume = map_coordinates(volume, [new_x, new_y, new_z], order=1, mode='nearest')
    mask = map_coordinates(mask, [new_x, new_y, new_z], order=0, mode='nearest')
    
    return volume, mask


def adaptive_augmentations(volume: np.ndarray, mask: np.ndarray,
                          epoch: int, total_epochs: int,
                          config: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, np.ndarray]:
    """自适应数据增强，增强强度随训练进度变化。
    
    参数：
        volume: 输入3D体积。
        mask: 输入3D掩码。
        epoch: 当前训练轮次。
        total_epochs: 总训练轮次。
        config: 可选的配置字典，包含自定义参数。
    
    返回：
        增强后的体积和掩码。
    """
    # 计算训练进度比例 (0到1)
    progress = epoch / max(1, total_epochs)
    
    # 随着训练进行，逐渐减少增强强度
    # 这样可以在训练初期学习更鲁棒的特征，后期稳定训练
    intensity = max(0.3, 1.0 - 0.5 * progress)
    
    # 使用自定义配置或默认配置
    if config is None:
        config = {}
    
    # 调整增强参数
    flip_prob = config.get('flip_prob', 0.5)
    rotation_angle = config.get('rotation_angle', 15) * intensity
    zoom_range = config.get('zoom_range', (0.9, 1.1))
    zoom_range = (1.0 - (1.0 - zoom_range[0]) * intensity, 1.0 + (zoom_range[1] - 1.0) * intensity)
    elastic_alpha = config.get('elastic_alpha', 1000) * intensity
    noise_range = config.get('noise_range', (0, 0.05)) * intensity
    
    # 应用增强
    if np.random.random() < flip_prob:
        volume, mask = random_flip(volume, mask)
    
    if np.random.random() < 0.7:
        volume, mask = random_rotation(volume, mask, angle_range=(-rotation_angle, rotation_angle))
    
    if np.random.random() < 0.5:
        volume, mask = random_zoom(volume, mask, zoom_range=zoom_range)
    
    if np.random.random() < 0.3:
        volume, mask = elastic_deformation(volume, mask, alpha=elastic_alpha)
    
    if np.random.random() < 0.3:
        volume, mask = random_noise(volume, mask, noise_range=noise_range)
    
    if np.random.random() < 0.3:
        volume, mask = random_intensity_shift(volume, mask, shift_range=(-0.05*intensity, 0.05*intensity))
    
    return volume, mask

def random_rotate_90(volume: np.ndarray, mask: np.ndarray):
    """围绕 Z 轴以 90° 步进随机旋转体数据与标签。

    围绕Z轴（切片轴）进行0°、90°、180°或270°的旋转，增加数据角度多样性。

    参数：
        volume (np.ndarray): 体数据 `[Z,Y,X]`。
        mask (np.ndarray): 标签体 `[Z,Y,X]`。
    返回：
        Tuple[np.ndarray, np.ndarray]: 旋转后的体数据与标签。
    """
    # 随机选择旋转次数（0-3次，对应0°-270°）
    k = np.random.randint(0, 4)
    
    # 围绕Z轴（轴1和轴2）进行旋转
    volume = np.rot90(volume, k, axes=(1, 2))
    mask = np.rot90(mask, k, axes=(1, 2))
    return volume, mask

def elastic_deformation(volume: np.ndarray, mask: np.ndarray, alpha=2.0, sigma=0.2):
    """对体数据施加弹性形变（逐切片）。

    算法思路：
        - 在平面上生成随机位移场并用高斯核平滑（控制形变幅度与平滑度）；
        - 将位移场加到网格坐标上，使用 `map_coordinates` 完成采样；
        - 数据采用线性插值，标签采用最近邻插值以保持二值边界。

    弹性形变可以模拟器官的自然变形，提高模型的泛化能力。

    参数：
        volume (np.ndarray): 体数据 `[Z,Y,X]`。
        mask (np.ndarray): 标签体 `[Z,Y,X]`。
        alpha (float): 形变幅度缩放系数，控制形变的强度。
        sigma (float): 高斯平滑标准差，控制形变的平滑程度。
    返回：
        Tuple[np.ndarray, np.ndarray]: 形变后的体数据与标签。
    """
    # 获取数据尺寸
    z, y, x = volume.shape
    
    # 初始化输出数组
    v_out = np.empty_like(volume)
    m_out = np.empty_like(mask)
    
    # 创建网格坐标
    grid_y, grid_x = np.meshgrid(np.arange(y), np.arange(x), indexing='ij')
    
    # 对每个切片进行形变处理
    for i in range(z):
        # 生成并平滑位移场
        # 生成随机位移场 [-1, 1] 并用高斯核平滑
        dy = gaussian_filter((np.random.rand(y, x) * 2 - 1), sigma) * alpha
        dx = gaussian_filter((np.random.rand(y, x) * 2 - 1), sigma) * alpha
        
        # 计算新坐标并进行采样
        indices = np.reshape(grid_y + dy, (-1,)), np.reshape(grid_x + dx, (-1,))
        
        # 对体数据进行线性插值采样
        v_out[i] = map_coordinates(volume[i], indices, order=1, mode='reflect').reshape(y, x)
        
        # 对标签进行最近邻插值采样，保持二值边界
        m_out[i] = map_coordinates(mask[i], indices, order=0, mode='nearest').reshape(y, x)
    
    return v_out, m_out

def apply_augmentations(volume: np.ndarray, mask: np.ndarray):
    """综合应用增强策略并返回增强后的体数据与标签。

    按顺序应用翻转、旋转和弹性形变增强。

    参数：
        volume (np.ndarray): 体数据 `[Z,Y,X]`。
        mask (np.ndarray): 标签体 `[Z,Y,X]`。
    返回：
        Tuple[np.ndarray, np.ndarray]: 增强后的体数据与标签。
    """
    # 应用随机翻转
    volume, mask = random_flip(volume, mask)
    
    # 应用随机旋转
    volume, mask = random_rotate_90(volume, mask)
    
    # 以50%的概率应用弹性形变
    if np.random.rand() < 0.5:
        volume, mask = elastic_deformation(volume, mask)
    
    return volume, mask

def apply_augmentations_multi(volumes, mask):
    """对多模态体数据应用一致增强。

    确保所有模态和标签应用相同的增强变换，保持数据一致性。

    参数：
        volumes (List[np.ndarray]): 多模态体列表，每个为 `[Z,Y,X]`。
        mask (np.ndarray): 标签体 `[Z,Y,X]`。
    返回：
        Tuple[List[np.ndarray], np.ndarray]: 增强后的多模态与标签。
    """
    # 生成随机翻转轴
    flip_axes = [axis for axis in range(3) if np.random.rand() < 0.5]
    
    # 对所有模态和标签应用相同的翻转
    vols = [np.flip(v, axis=flip_axes).copy() if flip_axes else v.copy() for v in volumes]
    m = np.flip(mask, axis=flip_axes).copy() if flip_axes else mask.copy()
    
    # 生成随机旋转次数
    k = np.random.randint(0, 4)
    
    # 对所有模态和标签应用相同的旋转
    vols = [np.rot90(v, k, axes=(1, 2)) for v in vols]
    m = np.rot90(m, k, axes=(1, 2))
    
    # 以50%的概率应用弹性形变（所有模态使用相同的形变场）
    if np.random.rand() < 0.5:
        dy_dx_cache = []
        z, y, x = vols[0].shape
        grid_y, grid_x = np.meshgrid(np.arange(y), np.arange(x), indexing='ij')
        alpha, sigma = 2.0, 0.2
        
        # 定义单个切片的形变函数
        def deform_one(vol, dy, dx):
            indices = np.reshape(grid_y + dy, (-1,)), np.reshape(grid_x + dx, (-1,))
            return map_coordinates(vol, indices, order=1, mode='reflect').reshape(y, x)
        
        # 初始化输出数组
        vol_outs = [np.empty_like(v) for v in vols]
        m_out = np.empty_like(m)
        
        # 对每个切片应用相同的形变
        for i in range(z):
            # 生成一次位移场并应用到所有模态与标签
            dy = gaussian_filter((np.random.rand(y, x) * 2 - 1), sigma) * alpha
            dx = gaussian_filter((np.random.rand(y, x) * 2 - 1), sigma) * alpha
            
            # 对所有模态应用形变
            for k in range(len(vols)):
                vol_outs[k][i] = deform_one(vols[k][i], dy, dx)
            
            # 对标签应用形变（使用最近邻插值）
            indices = np.reshape(grid_y + dy, (-1,)), np.reshape(grid_x + dx, (-1,))
            m_out[i] = map_coordinates(m[i], indices, order=0, mode='nearest').reshape(y, x)
        
        vols = vol_outs
        m = m_out
    
    return vols, m
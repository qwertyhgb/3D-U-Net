"""
文件名称：transforms3d.py
文件功能：实现 3D 医学影像的数据增强操作，包括翻转、旋转与弹性形变。
作者：TraeAI 助手
创建日期：2025-11-18
最后修改日期：2025-11-18
版本：v1.0
版权声明：Copyright (c) 2025, All rights reserved.

详细说明：
- 增强操作在体数据与对应标签上同步执行，确保数据一致性；
- 弹性形变采用高斯平滑的随机位移场并逐切片应用以节省内存；
- 标签插值采用最近邻以避免边缘被软化；
- 所有增强操作都考虑了医学影像的特殊性，避免引入不真实的伪影。
"""

import numpy as np
from scipy.ndimage import map_coordinates, gaussian_filter

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
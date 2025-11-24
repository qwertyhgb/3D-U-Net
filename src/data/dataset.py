"""
医学图像数据集处理模块，用于加载和预处理多模态前列腺MRI数据
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


@dataclass
class CasePaths:
    """记录单个病例的模态与掩膜路径信息的数据类"""

    case_id: str              # 病例ID
    group: str                # 分组信息（如BPH或PCA）
    modalities: Dict[str, Path]  # 各模态MRI图像路径字典
    mask: Path                # 掩膜标签路径


def strip_nii_suffix(path: Path) -> str:
    """
    移除NIfTI文件的后缀名
    
    Args:
        path: NIfTI文件路径对象
        
    Returns:
        不包含后缀的文件名
    """
    name = path.name
    if name.endswith(".nii.gz"):
        return name[: -len(".nii.gz")]
    if name.endswith(".nii"):
        return name[: -len(".nii")]
    return path.stem


def match_modal_file(modal_dir: Path, case_id: str) -> Path:
    """
    在指定模态目录中查找与病例ID匹配的文件
    
    Args:
        modal_dir: 模态图像所在的目录路径
        case_id: 病例ID
        
    Returns:
        匹配的文件路径
        
    Raises:
        FileNotFoundError: 当找不到匹配文件时抛出异常
    """
    # 使用通配符查找匹配的文件（支持.nii和.nii.gz格式）
    candidates = list(modal_dir.glob(f"{case_id}.nii*"))
    if not candidates:
        raise FileNotFoundError(f"模态 {modal_dir.name} 下未找到病例 {case_id} 的文件")
    return candidates[0]


def scan_cases(
    root_dir: str | Path,
    roi_subdir: str,
    groups: Sequence[str],
    modalities: Sequence[str],
) -> List[CasePaths]:
    """
    遍历数据目录，匹配每个病例的模态与掩膜路径
    
    Args:
        root_dir: 数据根目录路径
        roi_subdir: ROI标签子目录名称
        groups: 分组列表（如["BPH", "PCA"]）
        modalities: 模态列表（如["ADC", "DWI", "T2 fs", "T2 not fs"]）
        
    Returns:
        包含所有病例路径信息的列表
        
    Raises:
        FileNotFoundError: 当目录不存在时抛出异常
        RuntimeError: 当未扫描到任何病例时抛出异常
    """

    root = Path(root_dir)
    # 构建ROI标签根目录路径
    roi_root = root / roi_subdir
    cases: List[CasePaths] = []

    # 遍历每个分组
    for group in groups:
        # 构建该分组的掩膜目录路径
        mask_dir = roi_root / group
        if not mask_dir.is_dir():
            raise FileNotFoundError(f"ROI 分组目录不存在: {mask_dir}")
        
        # 遍历该分组下的所有掩膜文件
        for mask_path in mask_dir.glob("*.nii*"):
            # 获取病例ID（去除文件后缀）
            case_id = strip_nii_suffix(mask_path)
            modality_paths = {}
            
            # 为每个模态查找对应的图像文件
            for modal in modalities:
                modal_dir = root / group / modal
                if not modal_dir.is_dir():
                    raise FileNotFoundError(f"模态目录不存在: {modal_dir}")
                # 查找并记录该模态的图像文件路径
                modality_paths[modal] = match_modal_file(modal_dir, case_id)
                
            # 将该病例的信息添加到列表中
            cases.append(
                CasePaths(
                    case_id=case_id,
                    group=group,
                    modalities=modality_paths,
                    mask=mask_path,
                )
            )

    if not cases:
        raise RuntimeError("未扫描到任何病例，请检查数据路径与命名。")
    return cases


def split_cases(
    cases: Sequence[CasePaths],
    val_ratio: float,
    seed: int,
) -> Tuple[List[CasePaths], List[CasePaths]]:
    """
    根据验证比例划分病例列表，确保训练集和验证集的病例不重叠
    
    Args:
        cases: 所有病例的路径信息列表
        val_ratio: 验证集所占比例
        seed: 随机种子，用于保证划分结果可重现
        
    Returns:
        (训练集病例列表, 验证集病例列表)
        
    Raises:
        ValueError: 当val_ratio不在[0, 1)区间时抛出异常
    """

    cases = list(cases)
    # 设置随机种子以确保结果可重现
    rng = random.Random(seed)
    rng.shuffle(cases)

    if not 0 <= val_ratio < 1:
        raise ValueError("val_ratio 需在 [0, 1) 区间")

    # 计算验证集病例数量
    val_count = int(len(cases) * val_ratio)
    # 划分验证集和训练集
    val_cases = cases[:val_count] if val_count > 0 else []
    train_cases = cases[val_count:] if val_cases else cases
    return train_cases, val_cases


class MultiModalProstateDataset(Dataset):
    """多模态前列腺MRI数据集类，继承自PyTorch Dataset"""
    
    def __init__(
        self,
        cases: Sequence[CasePaths],
        modalities: Sequence[str],
        patch_size: Sequence[int],
        augment: bool = False,
    ) -> None:
        """
        初始化数据集
        
        Args:
            cases: 病例路径信息列表
            modalities: 模态列表
            patch_size: 图像块大小（深度、高度、宽度）
            augment: 是否进行数据增强
        """
        self.cases = list(cases)
        self.modalities = list(modalities)
        self.patch_size = tuple(patch_size)
        self.augment = augment

    def __len__(self) -> int:
        """返回数据集大小（病例数量）"""
        return len(self.cases)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        获取单个样本数据
        
        Args:
            idx: 样本索引
            
        Returns:
            包含图像、掩膜、病例ID和分组信息的字典
        """
        case = self.cases[idx]
        volumes = []
        
        # 加载并处理每个模态的图像数据
        for modal in self.modalities:
            # 使用nibabel加载NIfTI格式图像
            img = nib.load(str(case.modalities[modal])).get_fdata().astype(np.float32)
            # 对图像进行标准化处理
            img = self._normalize(img)
            # 转换为PyTorch张量
            volumes.append(torch.from_numpy(img))
            
        # 将多个模态的图像堆叠在一起
        volume = torch.stack(volumes, dim=0)  # (C, H, W, D)
        # 调整维度顺序为(C, D, H, W)
        volume = volume.permute(0, 3, 1, 2)  # -> (C, D, H, W)

        # 加载并处理掩膜标签
        mask = nib.load(str(case.mask)).get_fdata().astype(np.float32)
        # 转换为PyTorch张量并调整维度顺序
        mask = torch.from_numpy(mask).permute(2, 0, 1)  # (D, H, W)
        # 将掩膜转换为二值化标签（大于0.5为前景）
        mask = (mask > 0.5).float()

        # 调整图像和掩膜到指定的图像块大小
        volume = self._resize(volume, self.patch_size, mode="trilinear")
        mask = self._resize(mask.unsqueeze(0), self.patch_size, mode="nearest").squeeze(0)

        # 如果启用数据增强，则进行数据增强处理
        if self.augment:
            volume, mask = self._augment(volume, mask)

        # 返回包含所有必要信息的字典
        return {
            "image": volume,
            "mask": mask,
            "case_id": case.case_id,
            "group": case.group,
        }

    @staticmethod
    def _normalize(volume: np.ndarray) -> np.ndarray:
        """
        对图像进行标准化处理（零均值单位方差）
        
        Args:
            volume: 输入图像数据
            
        Returns:
            标准化后的图像数据
        """
        mean = float(volume.mean())
        std = float(volume.std())
        # 防止除零错误
        if std < 1e-6:
            return volume - mean
        return (volume - mean) / std

    @staticmethod
    def _resize(tensor: torch.Tensor, size: Sequence[int], mode: str) -> torch.Tensor:
        """
        调整张量的空间尺寸
        
        Args:
            tensor: 输入张量
            size: 目标尺寸（深度、高度、宽度）
            mode: 插值模式
            
        Returns:
            调整尺寸后的张量
        """
        size = tuple(size)
        # 如果尺寸已经匹配，则直接返回
        if tensor.shape[-3:] == size:
            return tensor
        # 增加批次维度以便使用插值函数
        x = tensor.unsqueeze(0)
        # 使用插值函数调整尺寸
        x = F.interpolate(x, size=size, mode=mode, align_corners=False)
        # 移除批次维度
        return x.squeeze(0)

    def _augment(self, image: torch.Tensor, mask: torch.Tensor):
        """
        数据增强：随机翻转
        
        Args:
            image: 输入图像张量
            mask: 输入掩膜张量
            
        Returns:
            增强后的图像和掩膜张量
        """
        # 存储需要翻转的维度
        dims_image = []
        dims_mask = []
        
        # 随机决定是否在各个维度上进行翻转
        # 深度维度（dim=1 for image, dim=0 for mask）
        if random.random() < 0.5:
            dims_image.append(2)
            dims_mask.append(1)
        # 高度维度（dim=2 for image, dim=1 for mask）
        if random.random() < 0.5:
            dims_image.append(3)
            dims_mask.append(2)
        # 宽度维度（dim=3 for image, dim=2 for mask）
        if random.random() < 0.5:
            dims_image.append(1)
            dims_mask.append(0)
            
        # 如果有任何维度需要翻转，则执行翻转操作
        if dims_image:
            image = torch.flip(image, dims=dims_image)
            mask = torch.flip(mask, dims=dims_mask)
        return image, mask


def create_data_loaders(
    data_cfg: Dict[str, Any],
    seed: int,
) -> tuple[DataLoader, DataLoader]:
    """
    创建训练和验证数据加载器
    
    Args:
        data_cfg: 数据配置字典
        seed: 随机种子
        
    Returns:
        (训练数据加载器, 验证数据加载器)
    """
    # 扫描所有病例数据
    cases = scan_cases(
        root_dir=data_cfg["root_dir"],
        roi_subdir=data_cfg["roi_subdir"],
        groups=data_cfg["groups"],
        modalities=data_cfg["modalities"],
    )

    # 划分训练集和验证集
    train_cases, val_cases = split_cases(
        cases,
        val_ratio=float(data_cfg.get("val_ratio", 0.2)),
        seed=seed,
    )

    # 创建训练数据集（启用数据增强）
    train_dataset = MultiModalProstateDataset(
        train_cases,
        modalities=data_cfg["modalities"],
        patch_size=data_cfg["patch_size"],
        augment=bool(data_cfg.get("augment", True)),
    )

    # 创建验证数据集（禁用数据增强）
    val_dataset = MultiModalProstateDataset(
        val_cases if val_cases else train_cases,
        modalities=data_cfg["modalities"],
        patch_size=data_cfg["patch_size"],
        augment=False,
    )

    # 数据加载器通用配置
    loader_kwargs = dict(
        batch_size=int(data_cfg.get("batch_size", 1)),
        num_workers=int(data_cfg.get("num_workers", 2)),
        pin_memory=True,
    )

    # 创建训练和验证数据加载器
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    return train_loader, val_loader
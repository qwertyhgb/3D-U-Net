"""
文件名称：dataset.py
文件功能：定义数据收集、预处理与数据集封装，输出统一尺寸的张量供模型训练与评估。
创建日期：2025-11-18
最后修改日期：2025-11-19
版本：v1.1
版权声明：Copyright (c) 2025, All rights reserved.
"""

import os
from typing import List, Tuple, Dict
import numpy as np
import torch
from torch.utils.data import Dataset

from .config import get_data_config
from .utils import read_volume, clip_and_normalize, pad_or_crop_to_shape
# 保留原有的变换模块以保持向后兼容性
from .transforms3d import apply_augmentations, apply_augmentations_multi
# 新增torchio变换模块
try:
    from .torchio_transforms import get_train_transforms, get_validation_transforms, apply_transforms_single, apply_transforms_multi
    TORCHIO_AVAILABLE = True
except ImportError:
    TORCHIO_AVAILABLE = False
    print("Warning: torchio not available. Using legacy transforms.")


def collect_cases(modality: str) -> List[Dict]:
    """收集指定模态的病例并与对应标签配对。

    参数：
        modality (str): 输入影像模态名（如 'DWI'、'ADC' 等）。
    返回：
        List[Dict]: 每个元素包含 `id`、`image`、`label` 的路径信息。
    """
    cases = []
    for cohort in ['BPH', 'PCA']:
        data_config = get_data_config()
        img_dir = os.path.join(data_config['data_root'], cohort, modality)
        if not os.path.isdir(img_dir):
            continue
        label_dir = os.path.join(data_config['label_root'], cohort)
        ids = [os.path.splitext(f)[0] for f in os.listdir(img_dir) if f.lower().endswith('.nii')]
        for case_id in ids:
            img_path = os.path.join(img_dir, f'{case_id}.nii')
            lab_path = os.path.join(label_dir, f'{case_id}.nii')
            if os.path.exists(img_path) and os.path.exists(lab_path):
                cases.append({'id': f'{cohort}-{case_id}', 'image': img_path, 'label': lab_path})
    return sorted(cases, key=lambda x: x['id'])


def collect_cases_multi(modalities: List[str]) -> List[Dict]:
    """收集在所有指定模态中都存在的病例，并与标签配对。

    通过求取不同模态文件夹下病例ID的交集，确保返回的病例
    在每个模态下都有对应的影像文件。

    参数：
        modalities (List[str]): 需要确保同时存在的模态列表。
    返回：
        List[Dict]: 每个元素包含 `id`、`images` (多模态路径字典)、`label` 路径。
    """
    ids_per_mod = {}
    cases = []
    data_config = get_data_config()
    for cohort in ['BPH', 'PCA']:
        present_ids = None
        for m in modalities:
            img_dir = os.path.join(data_config['data_root'], cohort, m)
            if not os.path.isdir(img_dir):
                return []
            ids = set(os.path.splitext(f)[0] for f in os.listdir(img_dir) if f.lower().endswith('.nii'))
            present_ids = ids if present_ids is None else (present_ids & ids)
        label_dir = os.path.join(data_config['label_root'], cohort)
        for case_id in sorted(present_ids):
            paths = {m: os.path.join(data_config['data_root'], cohort, m, f'{case_id}.nii') for m in modalities}
            lab_path = os.path.join(label_dir, f'{case_id}.nii')
            if os.path.exists(lab_path) and all(os.path.exists(p) for p in paths.values()):
                cases.append({'id': f'{cohort}-{case_id}', 'images': paths, 'label': lab_path})
    return sorted(cases, key=lambda x: x['id'])


class ProstateDataset(Dataset):
    """前列腺分割数据集封装。

    职责：
        - 读取影像与标签，完成归一化与尺寸统一；
        - 在训练阶段应用数据增强；
        - 输出符合 3D U-Net 期望的张量形状 `[B,C,Z,Y,X]`。
    重要属性：
        cases (List[Dict]): 病例路径信息列表。
        augment (bool): 是否启用增强。
        normalize_hu (bool): 是否采用 HU 归一化（适用于 CT）。
        use_torchio (bool): 是否使用torchio进行数据增强。
    """

    def __init__(self, cases: List[Dict], augment: bool = False, normalize_hu: bool = False, 
                 modalities: List[str] = None, use_torchio: bool = False):
        """初始化数据集。

        参数：
            cases (List[Dict]): 病例路径信息列表，由 `collect_cases` 或 `collect_cases_multi` 生成。
            augment (bool): 是否在 `__getitem__` 中启用数据增强。
            normalize_hu (bool): 是否按CT的HU值范围进行归一化，否则使用通用的分位数归一化。
            modalities (List[str]): 当前数据集处理的模态列表，用于多模态数据加载。
            use_torchio (bool): 是否使用torchio进行数据增强。
        """
        self.cases = cases
        self.augment = augment
        self.normalize_hu = normalize_hu
        self.modalities = modalities or get_data_config().get('modalities', ['DWI'])
        self.target_shape = tuple(get_data_config().get('target_shape', (256, 256, 32)))
        self.use_torchio = use_torchio and TORCHIO_AVAILABLE
        
        # 如果使用torchio，初始化变换
        if self.use_torchio:
            if augment:
                self.transform = get_train_transforms()
            else:
                self.transform = get_validation_transforms()

    def __len__(self):
        """返回数据集大小。"""
        return len(self.cases)

    def _normalize(self, vol: np.ndarray) -> np.ndarray:
        """对体数据进行强度归一化。

        说明：MRI 默认使用分位数归一化；当 `normalize_hu=True` 时按 HU 范围裁剪到 `[-1000,3000]` 并线性归一化。
        参数：
            vol (np.ndarray): 输入体数据。
        返回：
            np.ndarray: 归一化后的体数据。
        """
        if self.normalize_hu:
            v = np.clip(vol, -1000, 3000)
            v = (v + 1000.0) / 4000.0
            return v.astype(np.float32)
        return clip_and_normalize(vol)

    def __getitem__(self, idx):
        """读取并处理单个样本。

        参数：
            idx (int): 样本索引。
        返回：
            Dict[str, torch.Tensor]: 包含 `image` 与 `label` 张量，以及 `id`。
        """
        item = self.cases[idx]
        lab = read_volume(item['label'])
        if lab.ndim > 3:
            lab = lab.squeeze()
        lab = pad_or_crop_to_shape((lab > 0).astype(np.float32), self.target_shape)

        if 'image' in item:
            imgs = [read_volume(item['image'])]
        else:
            imgs = [read_volume(item['images'][m]) for m in self.modalities]
        imgs = [self._normalize(v) for v in imgs]
        imgs = [pad_or_crop_to_shape(v, self.target_shape) for v in imgs]

        if self.augment and self.use_torchio:
            # 使用torchio进行数据增强
            if len(imgs) == 1:
                img, lab = apply_transforms_single(imgs[0], lab, self.transform)
                imgs = [img]
            else:
                imgs, lab = apply_transforms_multi(imgs, lab, self.transform)
        elif self.augment:
            # 使用原有的数据增强方法
            if len(imgs) == 1:
                img, lab = apply_augmentations(imgs[0], lab)
                imgs = [img]
            else:
                imgs, lab = apply_augmentations_multi(imgs, lab)

        img_stack = np.stack(imgs, axis=0)  # 堆叠成 [C, Z, Y, X] 格式
        img_t = torch.from_numpy(img_stack)
        lab_t = torch.from_numpy(lab).unsqueeze(0)
        return {'id': item['id'], 'image': img_t, 'label': lab_t}


def build_splits(k_folds: int = 5, modalities=None, shuffle: bool = True, random_seed: int = 42) -> Tuple[List[Tuple[List[int], List[int]]], List[Dict]]:
    """构建K折交叉验证的训练/验证集索引，并返回所有病例信息。

    根据提供的模态列表，自动选择单模态或多模态病例收集函数，
    然后将所有病例划分为K个互斥的子集，每次取一个子集作为
    验证集，其余作为训练集。

    参数：
        k_folds (int): 交叉验证的折数。
        modalities (List[str] or str, optional): 目标模态。默认从配置读取。
        shuffle (bool): 是否在划分前打乱数据，默认 True。
        random_seed (int): 随机种子，用于可复现性，默认 42。
    返回：
        Tuple[List[Tuple[List[int], List[int]]], List[Dict]]:
            - folds: 一个列表，每个元素是 `(train_indices, val_indices)` 的元组。
            - cases: 所有收集到的病例信息列表。
    """
    modalities = modalities or get_data_config().get('modalities', ['DWI'])
    if isinstance(modalities, str):
        cases = collect_cases(modalities)
    elif len(modalities) == 1:
        cases = collect_cases(modalities[0])
    else:
        cases = collect_cases_multi(modalities)
    
    n = len(cases)
    if n == 0:
        raise ValueError("未找到任何病例，请检查数据路径和模态配置")
    
    indices = list(range(n))
    
    # 可选的随机打乱
    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    
    folds = []
    fold_size = n // k_folds
    
    for k in range(k_folds):
        start = k * fold_size
        end = (k + 1) * fold_size if k < k_folds - 1 else n
        val_idx = indices[start:end]
        train_idx = [i for i in indices if i not in val_idx]
        folds.append((train_idx, val_idx))
    
    return folds, cases
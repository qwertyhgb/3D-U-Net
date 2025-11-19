"""
文件名称：utils.py
文件功能：提供医学影像读写、强度归一化、尺寸对齐等通用工具方法。
作者：TraeAI 助手
创建日期：2025-11-18
最后修改日期：2025-11-18
版本：v1.0
版权声明：Copyright (c) 2025, All rights reserved.

详细说明：
- 支持 NIfTI 与 DICOM 序列的读取，统一返回为 `[Z, Y, X]` 轴顺序；
- 提供强度裁剪与归一化函数；
- 提供中心裁剪/对称补零的尺寸统一方法；
- 读写 NIfTI 时保持与参考影像一致的仿射与头信息。
- 所有函数都经过优化，以确保医学影像数据的准确处理
"""

import os
import numpy as np

def is_dicom_dir(path: str) -> bool:
    """判断给定路径是否为 DICOM 序列目录。

    通过检查目录中是否存在 `.dcm` 文件来判断是否为 DICOM 序列目录。

    参数：
        path (str): 待判断的目录路径。
    返回：
        bool: 目录中存在 `.dcm` 文件则返回 True，否则返回 False。
    异常：
        无（读取失败时返回 False）。
    """
    try:
        files = os.listdir(path)
    except Exception:
        return False
    return any(f.lower().endswith('.dcm') for f in files)

def safe_import_nib():
    """安全导入 nibabel 库。

    尝试导入 nibabel 库，如果导入失败则返回 None。

    返回：
        module|None: 成功返回 nibabel 模块，否则返回 None。
    """
    try:
        import nibabel as nib  # type: ignore
        return nib
    except Exception:
        return None

def safe_import_sitk():
    """安全导入 SimpleITK 库。

    尝试导入 SimpleITK 库，如果导入失败则返回 None。

    返回：
        module|None: 成功返回 SimpleITK 模块，否则返回 None。
    """
    try:
        import SimpleITK as sitk  # type: ignore
        return sitk
    except Exception:
        return None

def read_volume(path: str) -> np.ndarray:
    """读取医学影像体数据。

    功能：
        - 若为 DICOM 序列目录，使用 SimpleITK 读取并返回 `[Z,Y,X]`；
        - 若为 NIfTI 文件，使用 nibabel 读取并转换为 `[Z,Y,X]`。

    详细处理流程：
        1. 判断输入路径是目录还是文件
        2. 如果是目录且包含 DICOM 文件，则使用 SimpleITK 读取
        3. 如果是文件，则使用 nibabel 读取 NIfTI 格式
        4. 统一数据格式为 `[Z,Y,X]` 轴顺序

    参数：
        path (str): 文件路径或 DICOM 目录路径。
    返回：
        np.ndarray: 三维体数据，轴顺序为 `[Z,Y,X]`，类型为 float32。
    异常：
        RuntimeError: 当缺少必要库时抛出。
    """
    nib = safe_import_nib()
    sitk = safe_import_sitk()
    
    # 判断是否为 DICOM 目录
    if os.path.isdir(path) and is_dicom_dir(path):
        # 检查 SimpleITK 是否可用
        if sitk is None:
            raise RuntimeError('SimpleITK not available to read DICOM series')
        
        # 使用 SimpleITK 读取 DICOM 序列
        reader = sitk.ImageSeriesReader()
        dicom_names = reader.GetGDCMSeriesFileNames(path)
        reader.SetFileNames(dicom_names)
        image = reader.Execute()
        
        # 转换为 numpy 数组，SimpleITK 默认返回 [Z, Y, X] 格式
        arr = sitk.GetArrayFromImage(image)
        return arr.astype(np.float32)
    else:
        # 读取 NIfTI 文件
        if nib is None:
            raise RuntimeError('nibabel not available to read NIfTI file')
        
        # 使用 nibabel 读取 NIfTI 文件
        img = nib.load(path)
        data = img.get_fdata().astype(np.float32)
        
        # nibabel 默认返回 [X, Y, Z]，此处转置为 [Z, Y, X]
        if data.ndim == 3:
            data = np.transpose(data, (2, 1, 0))
        elif data.ndim == 4:
            # 四维数据形状 [X, Y, Z, C]，转置为 [Z, Y, X, C]
            data = np.transpose(data, (2, 1, 0, 3))
        return data

def save_nifti_like(reference_path: str, volume_zyx: np.ndarray, out_path: str):
    """按参考影像的空间信息保存预测体数据为 NIfTI。

    为了保持与原始影像一致的空间信息，使用参考影像的仿射矩阵和头信息。

    参数：
        reference_path (str): 参考 NIfTI 文件路径，用于复用 affine/header。
        volume_zyx (np.ndarray): 预测体数据，轴顺序为 `[Z,Y,X]`。
        out_path (str): 输出 NIfTI 文件路径。
    异常：
        RuntimeError: 当缺少 nibabel 时抛出。
    """
    # 检查 nibabel 是否可用
    nib = safe_import_nib()
    if nib is None:
        raise RuntimeError('nibabel not available to save NIfTI file')
    
    # 加载参考影像
    ref = nib.load(reference_path)
    data = volume_zyx
    
    # 将数据从 [Z, Y, X] 转换回 nibabel 期望的 [X, Y, Z] 格式
    if data.ndim == 3:
        data = np.transpose(data, (2, 1, 0))
    
    # 创建新的 NIfTI 图像对象，保持参考影像的空间信息
    nii = nib.Nifti1Image(data.astype(np.float32), ref.affine, ref.header)
    
    # 保存为 NIfTI 文件
    nib.save(nii, out_path)

def clip_and_normalize(volume: np.ndarray, min_val=None, max_val=None) -> np.ndarray:
    """基于分位数的强度裁剪与归一化。

    通过裁剪极端值并线性映射到 [0,1] 范围来标准化体数据强度值。

    参数：
        volume (np.ndarray): 原始体数据。
        min_val (float|None): 最小裁剪值，默认取 1 分位。
        max_val (float|None): 最大裁剪值，默认取 99 分位。
    返回：
        np.ndarray: 归一化后的体数据，范围约为 [0,1]。
    """
    # 确保数据类型为浮点型
    v = volume.astype(np.float32)
    
    # 如果未指定裁剪范围，则根据分位数计算
    if min_val is None or max_val is None:
        p1, p99 = np.percentile(v, [1, 99])
        min_val = p1
        max_val = p99
    
    # 裁剪极端值
    v = np.clip(v, min_val, max_val)
    
    # 线性归一化到 [0,1] 范围
    v = (v - float(min_val)) / (float(max_val - min_val) + 1e-8)
    return v

def pad_or_crop_to_shape(arr: np.ndarray, target_shape: tuple) -> np.ndarray:
    """通过对称补零与中心裁剪将体数据统一到目标尺寸。

    该函数用于将不同尺寸的医学影像统一到模型期望的输入尺寸。

    参数：
        arr (np.ndarray): 输入体数据 `[Z,Y,X]`。
        target_shape (tuple): 目标尺寸 `(Z,Y,X)`。
    返回：
        np.ndarray: 尺寸为 `target_shape` 的体数据。
    """
    # 获取当前数据尺寸和目标尺寸
    z, y, x = arr.shape
    tz, ty, tx = target_shape
    
    # 计算需要的对称补零量
    pad_z = max(0, tz - z)
    pad_y = max(0, ty - y)
    pad_x = max(0, tx - x)
    
    # 如果需要补零，则进行对称补零
    if pad_z or pad_y or pad_x:
        arr = np.pad(
            arr,
            ((pad_z // 2, pad_z - pad_z // 2),
             (pad_y // 2, pad_y - pad_y // 2),
             (pad_x // 2, pad_x - pad_x // 2)),
            mode='constant', constant_values=0,
        )
        z, y, x = arr.shape
    
    # 中心裁剪得到目标大小
    start_z = max(0, (z - tz) // 2)
    start_y = max(0, (y - ty) // 2)
    start_x = max(0, (x - tx) // 2)
    arr = arr[start_z:start_z+tz, start_y:start_y+ty, start_x:start_x+tx]
    return arr
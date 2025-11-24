#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
3D U-Net推理脚本，用于对新的MRI数据进行前列腺分割预测
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import nibabel as nib
import torch

# 将项目根目录添加到Python路径中，以便正确导入模块
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# 导入项目模块
from src.engine import inference_step
from src.models import UNet3D
from src.utils.common import ensure_dir
from src.utils.config import load_config


def parse_args() -> argparse.Namespace:
    """
    解析命令行参数
    
    Returns:
        解析后的参数命名空间
    """
    parser = argparse.ArgumentParser(description="3D U-Net 推理脚本")
    # 必需参数
    parser.add_argument("--config", required=True, help="配置文件路径")
    parser.add_argument("--ckpt", required=True, help="训练好的模型权重")
    parser.add_argument("--case", required=True, help="待推理病例的多模 MRI NIfTI 序列，多个模态用逗号分隔")
    parser.add_argument("--output", required=True, help="输出 mask 保存目录")
    # 可选参数
    parser.add_argument("--threshold", type=float, default=0.5, help="Sigmoid 阈值")
    return parser.parse_args()


def load_modalities(case_modalities: str) -> torch.Tensor:
    """
    加载并预处理多个模态的MRI数据
    
    Args:
        case_modalities: 逗号分隔的模态文件路径字符串
        
    Returns:
        预处理后的多模态张量，形状为(C, D, H, W)
    """
    tensors = []
    # 分割并处理每个模态的路径
    for modal_path in case_modalities.split(","):
        # 使用nibabel加载NIfTI格式图像
        img = nib.load(modal_path.strip()).get_fdata().astype("float32")
        # 转换为PyTorch张量并调整维度顺序为(D, H, W)
        tensor = torch.from_numpy(img).permute(2, 0, 1)  # (D, H, W)
        # 对图像进行标准化处理（零均值单位方差）
        tensor = (tensor - tensor.mean()) / (tensor.std() + 1e-6)
        tensors.append(tensor)
        
    # 将多个模态的图像堆叠在一起，形成多通道输入
    volume = torch.stack(tensors, dim=0)  # (C, D, H, W)
    return volume


def main() -> None:
    """主推理函数"""
    # 解析命令行参数
    args = parse_args()
    # 加载配置文件
    cfg = load_config(args.config)
    model_cfg = cfg.get("model", {})
    exp_cfg = cfg.get("experiment", {})

    # 设置设备（GPU或CPU）
    device_str = exp_cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)

    # 初始化模型
    model = UNet3D(
        in_channels=model_cfg.get("in_channels", 4),
        out_channels=model_cfg.get("out_channels", 1),
        init_features=model_cfg.get("init_features", 32),
    )
    
    # 加载训练好的模型权重
    state = torch.load(args.ckpt, map_location=device)
    if "model_state" in state:
        # 如果权重文件包含完整状态字典，则加载模型状态
        model.load_state_dict(state["model_state"])
    else:
        # 否则直接加载权重
        model.load_state_dict(state)
    model.to(device)

    # 加载并预处理输入数据
    volume = load_modalities(args.case)
    # 执行推理
    mask = inference_step(model, volume, device, threshold=args.threshold)

    # 保存推理结果
    output_dir = ensure_dir(args.output)
    output_path = output_dir / "prediction.nii.gz"
    # 使用nibabel保存NIfTI格式的分割结果
    nib.save(nib.Nifti1Image(mask.squeeze(0).numpy(), None), str(output_path))
    print(f"保存推理结果到 {output_path}")


if __name__ == "__main__":
    main()
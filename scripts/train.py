#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
3D U-Net训练脚本，用于训练前列腺MRI分割模型
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# 将项目根目录添加到Python路径中，以便正确导入模块
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

# 导入项目模块
from src.data import create_data_loaders
from src.engine import DiceCrossEntropyLoss, train_model
from src.models import UNet3D
from src.utils.common import ensure_dir, set_seed
from src.utils.config import load_config


def parse_args() -> argparse.Namespace:
    """
    解析命令行参数
    
    Returns:
        解析后的参数命名空间
    """
    parser = argparse.ArgumentParser(description="Train 3D U-Net for prostate MRI segmentation")
    # 必需参数：配置文件路径
    parser.add_argument("--config", required=True, help="路径到 YAML 配置文件")
    return parser.parse_args()


def main() -> None:
    """主训练函数"""
    # 解析命令行参数
    args = parse_args()
    # 加载配置文件
    cfg = load_config(args.config)

    # 提取配置信息
    exp_cfg = cfg.get("experiment", {})
    data_cfg = cfg.get("data", {})
    model_cfg = cfg.get("model", {})
    train_cfg = cfg.get("training", {})

    # 设置随机种子以确保结果可重现
    seed = int(exp_cfg.get("seed", 42))
    set_seed(seed)

    # 设置设备（GPU或CPU）
    device_str = exp_cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)

    # 创建数据加载器
    train_loader, val_loader = create_data_loaders(data_cfg, seed)

    # 初始化模型
    model = UNet3D(
        in_channels=model_cfg.get("in_channels", 4),
        out_channels=model_cfg.get("out_channels", 1),
        init_features=model_cfg.get("init_features", 32),
    ).to(device)

    # 定义损失函数
    loss_fn = DiceCrossEntropyLoss()
    
    # 定义优化器（AdamW）
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg.get("lr", 2e-4)),  # 学习率
        weight_decay=float(train_cfg.get("weight_decay", 1e-5)),  # 权重衰减
    )
    
    # 定义学习率调度器（基于验证损失的ReduceLROnPlateau）
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",      # 当监测值不再减小时降低学习率
        factor=0.5,      # 学习率衰减因子
        patience=5,      # 等待改善的epoch数
    )

    # 创建输出目录结构
    output_root = ensure_dir(Path(exp_cfg.get("output_dir", "outputs")) / exp_cfg.get("name", "experiment"))
    log_dir = ensure_dir(output_root / "logs")        # 日志目录
    ckpt_dir = ensure_dir(output_root / "checkpoints")  # 检查点目录

    # 开始训练模型
    best_path = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=loss_fn,
        device=device,
        epochs=int(train_cfg.get("epochs", 100)),     # 训练轮数
        log_dir=log_dir,
        use_amp=bool(train_cfg.get("amp", True)),     # 是否使用混合精度训练
        grad_clip=train_cfg.get("grad_clip"),         # 梯度裁剪值
        save_every=int(train_cfg.get("save_every", 5)),  # 每多少轮保存一次模型
        ckpt_dir=ckpt_dir,
    )

    print(f"Best checkpoint saved at: {best_path}")


if __name__ == "__main__":
    main()
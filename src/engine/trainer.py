"""
模型训练和推理模块，包含完整的训练循环、验证和推理功能
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Tuple

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.metrics import dice_coefficient


logger = logging.getLogger(__name__)


def _move_batch(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    """
    将批次数据移动到指定设备（CPU或GPU）
    
    Args:
        batch: 包含图像、掩膜等数据的字典
        device: 目标设备
        
    Returns:
        移动到目标设备后的数据字典
    """
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}


def train_model(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    loss_fn,
    device: torch.device,
    epochs: int,
    log_dir: Path,
    use_amp: bool = True,
    grad_clip: float | None = None,
    save_every: int = 5,
    ckpt_dir: Path | None = None,
) -> Path:
    """
    训练模型的主函数，包含完整的训练和验证循环
    
    Args:
        model: 待训练的神经网络模型
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        optimizer: 优化器
        scheduler: 学习率调度器
        loss_fn: 损失函数
        device: 计算设备（CPU或GPU）
        epochs: 训练轮数
        log_dir: 日志保存目录
        use_amp: 是否使用自动混合精度训练，默认为True
        grad_clip: 梯度裁剪阈值，None表示不使用梯度裁剪
        save_every: 每多少轮保存一次模型，默认为5轮
        ckpt_dir: 检查点保存目录，None表示使用log_dir
        
    Returns:
        最佳模型权重文件的路径
    """
    # 创建日志和检查点目录
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(log_dir))
    
    # 初始化自动混合精度缩放器
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    
    # 初始化最佳Dice系数和相关路径
    best_dice = 0.0
    ckpt_dir = ckpt_dir or log_dir
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_path = ckpt_dir / "best_model.pt"

    logger.info("开始训练，总计 %d 轮", epochs)

    # 训练循环
    for epoch in range(1, epochs + 1):
        # 设置模型为训练模式
        model.train()
        running_loss = 0.0
        
        # 初始化训练进度条
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [train]", leave=False)
        
        # 遍历训练数据
        for step, batch in enumerate(train_bar, 1):
            # 将数据移动到指定设备
            batch = _move_batch(batch, device)
            
            # 清零梯度
            optimizer.zero_grad(set_to_none=True)
            
            # 使用自动混合精度进行前向和反向传播
            with torch.autocast(device_type=device.type, enabled=use_amp):
                # 前向传播
                logits = model(batch["image"])
                # 计算损失
                loss = loss_fn(logits, batch["mask"].unsqueeze(1))
                
            # 反向传播
            scaler.scale(loss).backward()
            
            # 如果设置了梯度裁剪，则执行梯度裁剪
            if grad_clip:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                
            # 更新参数
            scaler.step(optimizer)
            scaler.update()
            
            # 累计损失
            running_loss += loss.item()
            
            # 更新进度条显示
            train_bar.set_postfix({"loss": f"{loss.item():.4f}"})

        # 计算平均训练损失
        avg_train_loss = running_loss / max(1, len(train_loader))
        
        # 在验证集上评估模型
        val_loss, val_dice = evaluate_model(
            model,
            val_loader,
            loss_fn,
            device,
            desc=f"Epoch {epoch}/{epochs} [val]",
        )

        # 记录TensorBoard日志
        writer.add_scalar("Loss/train", avg_train_loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("Dice/val", val_dice, epoch)

        # 更新学习率
        scheduler.step(val_loss)

        # 获取当前学习率
        current_lr = optimizer.param_groups[0]["lr"]
        
        # 记录训练日志
        logger.info(
            "Epoch %d/%d | train_loss: %.4f | val_loss: %.4f | val_dice: %.4f | lr: %.2e",
            epoch,
            epochs,
            avg_train_loss,
            val_loss,
            val_dice,
            current_lr,
        )

        # 如果当前Dice系数更好，则保存最佳模型
        if val_dice > best_dice:
            best_dice = val_dice
            torch.save({
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "epoch": epoch,
                "best_dice": best_dice,
            }, best_path)
            logger.info("验证 Dice 提升至 %.4f，已保存最佳权重 -> %s", best_dice, best_path)

        # 定期保存模型检查点
        if epoch % save_every == 0:
            torch.save(model.state_dict(), ckpt_dir / f"epoch-{epoch}.pt")
            logger.info("周期性保存权重 -> %s", ckpt_dir / f"epoch-{epoch}.pt")

    # 关闭TensorBoard写入器
    writer.close()
    logger.info("训练完成，最佳权重位于 %s", best_path)
    
    # 返回最佳模型路径
    return best_path


def evaluate_model(
    model: torch.nn.Module,
    loader: DataLoader,
    loss_fn,
    device: torch.device,
) -> Tuple[float, float]:
    """
    在数据加载器上评估模型性能
    
    Args:
        model: 待评估的模型
        loader: 数据加载器
        loss_fn: 损失函数
        device: 计算设备
        
    Returns:
        (平均损失值, 平均Dice系数)
    """
    # 设置模型为评估模式
    model.eval()
    total_loss = 0.0
    total_dice = 0.0
    
    # 关闭梯度计算
    with torch.no_grad():
        # 初始化验证进度条
        val_bar = loader
        if isinstance(loader, DataLoader):
            val_bar = tqdm(loader, desc="Validating", leave=False)
            
        # 遍历验证数据
        for batch in val_bar:
            # 将数据移动到指定设备
            batch = _move_batch(batch, device)
            
            # 前向传播
            logits = model(batch["image"])
            
            # 计算损失
            loss = loss_fn(logits, batch["mask"].unsqueeze(1))
            
            # 计算概率和Dice系数
            probs = torch.sigmoid(logits)
            dice = dice_coefficient((probs > 0.5).float(), batch["mask"].unsqueeze(1))
            
            # 累计损失和Dice系数
            total_loss += loss.item()
            total_dice += dice.item()
            
    # 返回平均损失和平均Dice系数
    return total_loss / len(loader), total_dice / len(loader)


def inference_step(
    model: torch.nn.Module,
    volume: torch.Tensor,
    device: torch.device,
    threshold: float = 0.5,
) -> torch.Tensor:
    """
    执行单次推理步骤
    
    Args:
        model: 训练好的模型
        volume: 输入的医学图像张量
        device: 计算设备
        threshold: 分割阈值，默认为0.5
        
    Returns:
        分割结果掩膜张量
    """
    # 设置模型为评估模式
    model.eval()
    
    # 关闭梯度计算
    with torch.no_grad():
        # 将输入数据移动到指定设备
        volume = volume.to(device)
        
        # 前向传播
        logits = model(volume.unsqueeze(0))
        
        # 计算概率
        probs = torch.sigmoid(logits)
        
        # 根据阈值生成二值化掩膜
        mask = (probs > threshold).float()
        
    # 记录推理日志
    logger.info(
        "完成推理，输入体素=%s，阈值=%.2f，输出掩膜形状=%s",
        tuple(volume.shape),
        threshold,
        tuple(mask.shape),
    )
    
    # 将结果移回CPU并返回
    return mask.squeeze(0).cpu()
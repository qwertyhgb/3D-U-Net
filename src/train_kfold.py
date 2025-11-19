"""
文件名称：train_kfold.py
文件功能：实现 5 折交叉验证训练流程，包含早停与学习率衰减，并保存模型与曲线。
创建日期：2025-11-18
最后修改日期：2025-11-18
版本：v1.2
版权声明：Copyright (c) 2025, All rights reserved.
"""

import csv
import os
import logging
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
import torch
from torch.cuda.amp import GradScaler, autocast
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from .config import get_data_config, get_training_config
from .dataset import build_splits, ProstateDataset
from .losses import get_loss_function
from .metrics import torch_dice
from .unet3d import UNet3D
from .optimizers import get_optimizer, apply_gradient_clipping
from .schedulers import get_scheduler
from .logger import TrainingLogger, log_system_info, log_data_info


def train_one_fold(fold_idx: int, train_idx: list, val_idx: list, cases: list, device: torch.device, data_config: dict, training_config: dict):
    """训练单个折并保存最佳模型与训练曲线。

    参数：
        fold_idx (int): 折编号。
        train_idx (list): 训练集索引。
        val_idx (list): 验证集索引。
        cases (list): 病例路径信息列表。
        device (torch.device): 训练设备。
        data_config (dict): 数据相关配置。
        training_config (dict): 训练相关配置。
    返回：
        None
    """
    # 初始化训练日志记录器
    logger = TrainingLogger(fold_idx, data_config['output_dir'])
    
    logger.logger.info("="*60)
    logger.logger.info(f"Fold {fold_idx}: 训练集 {len(train_idx)} 样本, 验证集 {len(val_idx)} 样本")
    logger.logger.info("="*60)
    
    # 检查是否启用torchio增强
    use_torchio = data_config.get('use_torchio', False)
    
    train_ds = ProstateDataset([cases[i] for i in train_idx], augment=True, modalities=data_config['modalities'], use_torchio=use_torchio)
    val_ds = ProstateDataset([cases[i] for i in val_idx], augment=False, modalities=data_config['modalities'], use_torchio=use_torchio)

    # 优化 DataLoader 配置
    num_workers = training_config.get('num_workers', 2)
    # Windows 系统建议使用较少的 workers
    if os.name == 'nt' and num_workers > 0:
        num_workers = min(num_workers, 2)
    
    train_loader = DataLoader(
        train_ds, 
        batch_size=training_config['batch_size'], 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True if device.type == 'cuda' else False,
        persistent_workers=True if num_workers > 0 else False
    )
    val_loader = DataLoader(
        val_ds, 
        batch_size=training_config['batch_size'], 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True if device.type == 'cuda' else False,
        persistent_workers=True if num_workers > 0 else False
    )

    model = UNet3D(in_channels=len(data_config['modalities'])).to(device)
    
    # 获取损失函数
    loss_config = training_config.get('loss', {'type': 'dice_bce'})
    criterion = get_loss_function(loss_config)
    loss_type = loss_config.get('type', 'dice_bce')
    
    logger.logger.info(f"使用损失函数: {loss_type}")
    
    # 使用高级优化器
    optimizer_config = training_config.get('optimizer', {'type': 'adam', 'lr': training_config['learning_rate']})
    optimizer = get_optimizer(model.parameters(), optimizer_config)
    
    # 使用高级学习率调度器
    scheduler_config = training_config.get('scheduler', {'type': 'plateau', 'mode': 'min', 'factor': 0.5, 'patience': 3})
    
    # 对于非plateau类型的调度器，需要特殊处理step调用
    scheduler_type = scheduler_config.get('type', 'plateau').lower()
    scheduler = get_scheduler(optimizer, scheduler_config)

    # 初始化混合精度训练
    use_amp = training_config.get('amp', False)
    scaler = GradScaler() if use_amp and device.type == 'cuda' else None

    # 初始化 TensorBoard
    tensorboard_dir = os.path.join(data_config['output_dir'], 'runs', f'fold_{fold_idx}')
    os.makedirs(tensorboard_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=tensorboard_dir)
    best_val = np.inf
    best_epoch = -1
    no_improve = 0
    train_losses, val_losses, val_dices = [], [], []

    # 确保日志目录存在
    logs_dir = os.path.join(data_config['output_dir'], 'logs')
    os.makedirs(logs_dir, exist_ok=True)
    log_path = os.path.join(logs_dir, f'fold_{fold_idx}_log.csv')

    with open(log_path, 'w', newline='') as f:
        writer_csv = csv.writer(f)
        if loss_type == 'dice_bce':
            writer_csv.writerow(['epoch', 'train_loss', 'val_loss', 'val_dice', 'train_dice_loss', 'train_bce_loss', 'val_dice_loss', 'val_bce_loss'])
        else:
            writer_csv.writerow(['epoch', 'train_loss', 'val_loss', 'val_dice'])

        for epoch in range(1, training_config['num_epochs'] + 1):
            logger.log_epoch_start(epoch, training_config['num_epochs'])
            
            # 如果使用自适应增强策略，更新数据集的epoch
            if hasattr(train_ds, 'set_epoch'):
                train_ds.set_epoch(epoch)
                
            model.train()
            tl = 0.0
            
            # 如果使用组合损失，记录各组件
            if loss_type == 'dice_bce':
                tl_dice, tl_bce = 0.0, 0.0
            
            # 添加进度条
            from tqdm import tqdm
            train_pbar = tqdm(train_loader, desc=f'Epoch {epoch}/{training_config["num_epochs"]} [Train]', leave=False)
            
            for batch in train_pbar:
                x = batch['image'].to(device, non_blocking=True)
                y = batch['label'].to(device, non_blocking=True)
                optimizer.zero_grad()
                
                # 混合精度训练
                if use_amp and device.type == 'cuda':
                    with autocast():
                        yhat = model(x)
                        loss = criterion(yhat, y)
                        
                        # 如果使用组合损失，记录各组件
                        if loss_type == 'combined':
                            dice_loss, focal_loss, boundary_loss = criterion.get_component_losses(yhat, y)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    yhat = model(x)
                    loss = criterion(yhat, y)
                    
                    # 如果使用组合损失，记录各组件
                    if loss_type == 'dice_bce':
                        dice_loss, bce_loss = criterion.get_component_losses(yhat, y)
                    
                    loss.backward()
                    
                    # 应用梯度裁剪
                    clip_config = training_config.get('gradient_clipping', None)
                    if clip_config:
                        apply_gradient_clipping(model, clip_config)
                    
                    optimizer.step()
                
                batch_loss = loss.item()
                tl += batch_loss * x.size(0)
                
                # 如果使用组合损失，记录各组件
                if loss_type == 'dice_bce':
                    tl_dice += dice_loss.item() * x.size(0)
                    tl_bce += bce_loss.item() * x.size(0)
                
                train_pbar.set_postfix({'loss': f'{batch_loss:.4f}'})
            
            tl /= len(train_loader.dataset)
            
            # 如果使用组合损失，计算各组件平均值
            if loss_type == 'dice_bce':
                tl_dice /= len(train_loader.dataset)
                tl_bce /= len(train_loader.dataset)

            # 验证阶段
            model.eval()
            vl = 0.0
            vd = 0.0
            
            # 如果使用组合损失，记录各组件
            if loss_type == 'dice_bce':
                vl_dice, vl_bce = 0.0, 0.0
            
            val_pbar = tqdm(val_loader, desc=f'Epoch {epoch}/{training_config["num_epochs"]} [Val]', leave=False)
            
            with torch.no_grad():
                for batch in val_pbar:
                    x = batch['image'].to(device, non_blocking=True)
                    y = batch['label'].to(device, non_blocking=True)
                    yhat = model(x)
                    loss = criterion(yhat, y)
                    dice = torch_dice(yhat, y)
                    vl += loss.item() * x.size(0)
                    vd += dice * x.size(0)
                    
                    # 如果使用组合损失，记录各组件
                    if loss_type == 'dice_bce':
                        dice_loss, bce_loss = criterion.get_component_losses(yhat, y)
                        vl_dice += dice_loss.item() * x.size(0)
                        vl_bce += bce_loss.item() * x.size(0)
                    
                    val_pbar.set_postfix({'loss': f'{loss.item():.4f}', 'dice': f'{dice:.4f}'})
            
            vl /= len(val_loader.dataset)
            vd /= len(val_loader.dataset)
            
            # 如果使用组合损失，计算各组件平均值
            if loss_type == 'dice_bce':
                vl_dice /= len(val_loader.dataset)
                vl_bce /= len(val_loader.dataset)

            # 根据调度器类型调用step
            if scheduler_type == 'plateau':
                scheduler.step(vl)
            else:
                scheduler.step()

            train_losses.append(tl)
            val_losses.append(vl)
            val_dices.append(vd)
            
            # 记录到CSV
            if loss_type == 'dice_bce':
                writer_csv.writerow([epoch, tl, vl, vd, tl_dice, tl_bce, vl_dice, vl_bce])
            else:
                writer_csv.writerow([epoch, tl, vl, vd])

            # 记录到 TensorBoard
            writer.add_scalar('Loss/train', tl, epoch)
            writer.add_scalar('Loss/validation', vl, epoch)
            writer.add_scalar('Dice/validation', vd, epoch)
            writer.add_scalar('Learning Rate', optimizer.param_groups[0]['lr'], epoch)
            
            # 如果使用组合损失，记录各组件
            if loss_type == 'dice_bce':
                writer.add_scalar('Loss/train_dice', tl_dice, epoch)
                writer.add_scalar('Loss/train_bce', tl_bce, epoch)
                writer.add_scalar('Loss/val_dice', vl_dice, epoch)
                writer.add_scalar('Loss/val_bce', vl_bce, epoch)
            
            if use_amp and scaler:
                writer.add_scalar('AMP/Scale', scaler.get_scale(), epoch)

            # 记录当前 epoch 结果
            metrics = {
                'train_loss': tl,
                'val_loss': vl,
                'val_dice': vd,
                'lr': optimizer.param_groups[0]['lr']
            }
            
            if loss_type == 'dice_bce':
                metrics.update({
                    'train_dice_loss': tl_dice,
                    'train_bce_loss': tl_bce,
                    'val_dice_loss': vl_dice,
                    'val_bce_loss': vl_bce
                })
            
            logger.log_epoch_end(epoch, metrics)
            
            # 早停与最佳模型保存
            if vl < best_val:
                best_val = vl
                best_epoch = epoch
                no_improve = 0
                # 确保模型目录存在
                models_dir = os.path.join(data_config['output_dir'], 'models')
                os.makedirs(models_dir, exist_ok=True)
                model_path = os.path.join(models_dir, f'fold_{fold_idx}_best.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': vl,
                    'val_dice': vd,
                }, model_path)
                logger.log_best_model(epoch, 'val_loss', vl)
            else:
                no_improve += 1

            if no_improve >= training_config['patience']:
                logger.log_early_stopping(epoch, training_config['patience'])
                break

    # 绘制并保存训练/验证曲线
    plots_dir = os.path.join(data_config['output_dir'], 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    plt.figure(figsize=(15, 5))
    
    # 损失曲线
    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label='Train Loss', linewidth=2)
    plt.plot(val_losses, label='Val Loss', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Fold {fold_idx} - Loss Curves')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Dice 曲线
    plt.subplot(1, 3, 2)
    plt.plot(val_dices, label='Val Dice', linewidth=2, color='green')
    plt.xlabel('Epoch')
    plt.ylabel('Dice Score')
    plt.title(f'Fold {fold_idx} - Dice Score')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 学习率曲线（如果有记录）
    plt.subplot(1, 3, 3)
    epochs = list(range(1, len(train_losses) + 1))
    plt.plot(epochs, train_losses, label='Train Loss', alpha=0.7)
    plt.plot(epochs, val_losses, label='Val Loss', alpha=0.7)
    plt.axvline(x=best_epoch, color='r', linestyle='--', label=f'Best Epoch ({best_epoch})')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Fold {fold_idx} - Best Model at Epoch {best_epoch}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f'fold_{fold_idx}_curves.png'), dpi=150)
    plt.close()
    
    writer.close()
    
    logger.log_training_complete(best_epoch, best_val)
    logger.logger.info(f"最佳验证 Dice: {max(val_dices):.4f}")

def run_kfold():
    """运行 K 折训练流程。"""
    from .logger import get_logger, setup_logger
    
    # 设置主日志记录器
    main_logger = setup_logger(
        'training_main',
        log_file=os.path.join('./outputs/logs', 'training_main.log'),
        level=logging.INFO
    )
    
    data_config = get_data_config()
    training_config = get_training_config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    main_logger.info("="*60)
    main_logger.info("前列腺癌 3D U-Net 分割 - K折交叉验证训练")
    main_logger.info("="*60)
    
    # 记录系统信息
    log_system_info()
    
    # 记录训练配置
    main_logger.info(f"设备: {device}")
    if device.type == 'cuda':
        main_logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        main_logger.info(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    main_logger.info(f"混合精度训练 (AMP): {training_config.get('amp', False)}")
    main_logger.info(f"模态: {data_config['modalities']}")
    main_logger.info(f"目标尺寸: {data_config['target_shape']}")
    main_logger.info(f"批次大小: {training_config['batch_size']}")
    main_logger.info(f"学习率: {training_config['learning_rate']}")
    main_logger.info(f"最大 Epoch: {training_config['num_epochs']}")
    main_logger.info(f"早停耐心: {training_config['patience']}")
    main_logger.info(f"折数: {training_config['num_folds']}")
    main_logger.info(f"TorchIO 增强: {data_config.get('use_torchio', False)}")
    main_logger.info("="*60)
    
    # 从新的数据配置中获取模态信息
    try:
        folds, cases = build_splits(training_config['num_folds'], modalities=data_config['modalities'])
        main_logger.info(f"✓ 成功加载 {len(cases)} 个病例")
        main_logger.info(f"✓ 构建 {len(folds)} 折交叉验证")
        
        # 记录数据信息
        log_data_info(len(cases), training_config['num_folds'], data_config['modalities'])
    except Exception as e:
        main_logger.error(f"✗ 数据加载失败: {e}", exc_info=True)
        return
    
    # 训练所有折
    import time
    start_time = time.time()
    
    for i, (train_idx, val_idx) in enumerate(folds):
        fold_start = time.time()
        try:
            train_one_fold(i, train_idx, val_idx, cases, device, data_config, training_config)
            fold_time = time.time() - fold_start
            main_logger.info(f"Fold {i} 耗时: {fold_time/60:.2f} 分钟")
        except Exception as e:
            main_logger.error(f"Fold {i} 训练失败: {e}", exc_info=True)
            continue
    
    total_time = time.time() - start_time
    main_logger.info("="*60)
    main_logger.info("所有折训练完成！")
    main_logger.info(f"总耗时: {total_time/60:.2f} 分钟 ({total_time/3600:.2f} 小时)")
    main_logger.info("="*60)

if __name__ == '__main__':
    run_kfold()
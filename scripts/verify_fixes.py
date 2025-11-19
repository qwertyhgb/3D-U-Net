#!/usr/bin/env python
"""
修复验证脚本

验证所有修复是否正确工作
"""

import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import numpy as np


def test_combined_loss():
    """测试组合损失函数的get_component_losses方法"""
    print("="*60)
    print("测试 1: 组合损失函数")
    print("="*60)
    
    try:
        from src.losses import CombinedLoss
        
        # 创建组合损失
        criterion = CombinedLoss(dice_weight=0.5, focal_weight=0.4, boundary_weight=0.1)
        print("✓ 组合损失函数创建成功")
        
        # 创建测试数据
        y_pred = torch.rand(2, 1, 32, 32, 32)
        y_true = torch.randint(0, 2, (2, 1, 32, 32, 32)).float()
        
        # 测试forward
        loss = criterion(y_pred, y_true)
        print(f"✓ forward方法工作正常，损失值: {loss.item():.4f}")
        
        # 测试get_component_losses
        dice_loss, focal_loss, boundary_loss = criterion.get_component_losses(y_pred, y_true)
        print(f"✓ get_component_losses方法工作正常")
        print(f"  - Dice损失: {dice_loss.item():.4f}")
        print(f"  - Focal损失: {focal_loss.item():.4f}")
        print(f"  - Boundary损失: {boundary_loss.item():.4f}")
        
        return True
        
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_optimizer_api():
    """测试优化器API更新"""
    print("\n" + "="*60)
    print("测试 2: 优化器API")
    print("="*60)
    
    try:
        from src.optimizers import RAdam, get_optimizer
        
        # 创建一个简单的模型
        model = torch.nn.Linear(10, 1)
        
        # 测试RAdam
        optimizer = RAdam(model.parameters(), lr=0.001)
        print("✓ RAdam优化器创建成功")
        
        # 测试一次优化步骤
        x = torch.randn(5, 10)
        y = torch.randn(5, 1)
        
        output = model(x)
        loss = torch.nn.functional.mse_loss(output, y)
        loss.backward()
        optimizer.step()
        print("✓ RAdam优化器step方法工作正常")
        
        # 测试get_optimizer
        optimizer_config = {
            'type': 'adamw',
            'lr': 0.001,
            'weight_decay': 0.01
        }
        optimizer = get_optimizer(model.parameters(), optimizer_config)
        print("✓ get_optimizer函数工作正常")
        
        return True
        
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_scheduler():
    """测试学习率调度器"""
    print("\n" + "="*60)
    print("测试 3: 学习率调度器")
    print("="*60)
    
    try:
        from src.schedulers import CosineAnnealingWarmRestarts, get_scheduler
        
        # 创建一个简单的模型和优化器
        model = torch.nn.Linear(10, 1)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        # 测试CosineAnnealingWarmRestarts（使用浮点数参数）
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10.0, T_mult=1.5)
        print("✓ CosineAnnealingWarmRestarts创建成功（支持浮点数参数）")
        
        # 测试step
        scheduler.step()
        print("✓ 调度器step方法工作正常")
        
        # 测试get_scheduler
        scheduler_config = {
            'type': 'cosine',
            'T_0': 10,
            'T_mult': 1,
            'eta_min': 1e-6
        }
        scheduler = get_scheduler(optimizer, scheduler_config)
        print("✓ get_scheduler函数工作正常")
        
        # 测试plateau调度器
        scheduler_config = {
            'type': 'plateau',
            'mode': 'min',
            'factor': 0.5,
            'patience': 3
        }
        scheduler = get_scheduler(optimizer, scheduler_config)
        scheduler.step(0.5)  # plateau需要传入metric
        print("✓ ReduceLROnPlateau调度器工作正常")
        
        return True
        
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_config():
    """测试配置加载"""
    print("\n" + "="*60)
    print("测试 4: 配置加载")
    print("="*60)
    
    try:
        from src.config import get_data_config, get_training_config, get_augmentation_config
        
        data_config = get_data_config()
        print("✓ 数据配置加载成功")
        print(f"  - 模态: {data_config.get('modalities')}")
        print(f"  - 目标尺寸: {data_config.get('target_shape')}")
        
        training_config = get_training_config()
        print("✓ 训练配置加载成功")
        print(f"  - 批次大小: {training_config.get('batch_size')}")
        print(f"  - 学习率: {training_config.get('learning_rate')}")
        print(f"  - 折数: {training_config.get('num_folds')}")
        
        # 检查损失函数配置
        loss_config = training_config.get('loss', {})
        print(f"  - 损失函数类型: {loss_config.get('type')}")
        
        # 检查调度器配置
        scheduler_config = training_config.get('scheduler', {})
        print(f"  - 调度器类型: {scheduler_config.get('type')}")
        if scheduler_config.get('type') == 'cosine':
            print(f"  - T_0: {scheduler_config.get('T_0')}")
            print(f"  - T_mult: {scheduler_config.get('T_mult')}")
        
        augmentation_config = get_augmentation_config()
        print("✓ 增强配置加载成功")
        print(f"  - 增强策略: {augmentation_config.get('strategy')}")
        
        return True
        
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model_loading():
    """测试模型加载逻辑"""
    print("\n" + "="*60)
    print("测试 5: 模型加载逻辑")
    print("="*60)
    
    try:
        from src.unet3d import UNet3D
        import tempfile
        
        # 创建模型
        model = UNet3D(in_channels=1)
        print("✓ 模型创建成功")
        
        # 创建临时文件测试两种保存格式
        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as f:
            temp_path1 = f.name
        
        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as f:
            temp_path2 = f.name
        
        try:
            # 格式1: 完整checkpoint
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'epoch': 10,
                'val_loss': 0.5
            }
            torch.save(checkpoint, temp_path1)
            print("✓ 保存完整checkpoint成功")
            
            # 格式2: 仅state_dict
            torch.save(model.state_dict(), temp_path2)
            print("✓ 保存state_dict成功")
            
            # 测试加载格式1
            model1 = UNet3D(in_channels=1)
            checkpoint = torch.load(temp_path1, map_location='cpu')
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model1.load_state_dict(checkpoint['model_state_dict'])
            print("✓ 加载完整checkpoint成功")
            
            # 测试加载格式2
            model2 = UNet3D(in_channels=1)
            state_dict = torch.load(temp_path2, map_location='cpu')
            model2.load_state_dict(state_dict)
            print("✓ 加载state_dict成功")
            
            return True
            
        finally:
            # 清理临时文件
            if os.path.exists(temp_path1):
                os.remove(temp_path1)
            if os.path.exists(temp_path2):
                os.remove(temp_path2)
        
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """运行所有验证测试"""
    print("\n" + "="*60)
    print("修复验证测试")
    print("="*60)
    
    results = []
    
    # 运行所有测试
    results.append(("组合损失函数", test_combined_loss()))
    results.append(("优化器API", test_optimizer_api()))
    results.append(("学习率调度器", test_scheduler()))
    results.append(("配置加载", test_config()))
    results.append(("模型加载逻辑", test_model_loading()))
    
    # 总结
    print("\n" + "="*60)
    print("测试总结")
    print("="*60)
    
    for name, result in results:
        status = "✓ 通过" if result else "✗ 失败"
        print(f"{name}: {status}")
    
    all_passed = all(result for _, result in results)
    
    print("\n" + "="*60)
    if all_passed:
        print("✓ 所有修复验证通过！")
        print("\n建议下一步:")
        print("1. 运行冒烟测试: python scripts/smoke_test.py")
        print("2. 运行快速训练测试: python run_training.py --quick-test")
        print("3. 如果有已训练的模型，运行评估测试")
    else:
        print("✗ 部分测试失败，请检查错误信息")
    print("="*60)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())

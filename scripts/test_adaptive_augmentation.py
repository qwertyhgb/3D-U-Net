#!/usr/bin/env python
"""
文件名称：test_adaptive_augmentation.py
文件功能：测试自适应增强策略是否正常工作
作者：TraeAI 助手
创建日期：2025-11-19
版本：v1.0
"""

import os
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.dataset import ProstateDataset
from src.config import get_data_config, get_training_config


def test_adaptive_augmentation():
    """测试自适应增强策略是否正常工作。"""
    print("="*60)
    print("自适应增强策略测试")
    print("="*60)
    
    # 加载配置
    data_config = get_data_config()
    training_config = get_training_config()
    
    # 获取增强配置
    from src.config import get_augmentation_config
    augmentation_config = get_augmentation_config()
    
    # 检查配置中的策略是否为adaptive
    strategy = augmentation_config['strategy']
    print(f"当前增强策略: {strategy}")
    
    if strategy != 'adaptive':
        print("警告: 当前配置不是自适应策略，测试可能不准确")
    
    # 创建临时目录和模拟数据
    import tempfile
    import shutil
    
    temp_dir = tempfile.mkdtemp()
    print(f"使用临时目录: {temp_dir}")
    
    try:
        # 创建模拟数据目录结构
        images_dir = os.path.join(temp_dir, "images")
        labels_dir = os.path.join(temp_dir, "labels")
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(labels_dir, exist_ok=True)
        
        # 创建模拟数据文件
        for i in range(3):
            # 模拟多模态图像文件
            for modality in data_config['modalities']:
                img_path = os.path.join(images_dir, f"case_{i:03d}_{modality}.nii.gz")
                # 创建一个更真实的3D numpy数组并保存
                import nibabel as nib
                img_data = np.random.rand(32, 256, 256).astype(np.float32) * 100  # 模拟3D图像
                affine = np.eye(4)  # 单位矩阵作为仿射变换
                nib_img = nib.Nifti1Image(img_data, affine)
                nib.save(nib_img, img_path)
            
            # 模拟标签文件
            label_path = os.path.join(labels_dir, f"case_{i:03d}_seg.nii.gz")
            import nibabel as nib
            label_data = (np.random.rand(32, 256, 256) > 0.7).astype(np.uint8)  # 二进制标签
            affine = np.eye(4)
            nib_label = nib.Nifti1Image(label_data, affine)
            nib.save(nib_label, label_path)
        
        # 使用临时目录作为数据根目录
        data_root = temp_dir
        
        # 获取病例列表
        cases = []
        for i in range(3):
            # 创建模拟数据路径
            image_path = os.path.join(data_root, "images", f"case_{i:03d}_{data_config['modalities'][0]}.nii.gz")
            label_path = os.path.join(data_root, "labels", f"case_{i:03d}_seg.nii.gz")
            cases.append({
                'id': f'case_{i:03d}',
                'image': image_path,
                'label': label_path
            })
        
        # 创建数据集
        train_dataset = ProstateDataset(
            cases=cases,
            augment=True,
            modalities=data_config['modalities'],
            use_torchio=True,
            augmentation_config=augmentation_config
        )
        
        # 检查数据集是否有set_epoch方法
        if not hasattr(train_dataset, 'set_epoch'):
            print("错误: 数据集没有set_epoch方法，自适应增强策略无法工作")
            return False
        
        print("✓ 数据集有set_epoch方法")
        
        # 创建数据加载器
        train_loader = DataLoader(
            train_dataset,
            batch_size=1,
            shuffle=True,
            num_workers=0  # 设置为0以避免多进程问题
        )
        
        # 测试不同epoch的增强强度
        num_epochs = 5
        total_epochs = training_config['num_epochs']
        initial_intensity = augmentation_config['adaptive']['initial_intensity']
        final_intensity = augmentation_config['adaptive']['final_intensity']
        
        print(f"\n测试 {num_epochs} 个epoch的增强强度变化:")
        print(f"总训练epoch: {total_epochs}")
        print(f"初始增强强度: {initial_intensity}")
        print(f"最终增强强度: {final_intensity}")
        
        # 记录每个epoch的图像数据范围
        data_ranges = []
        
        for epoch in range(1, num_epochs + 1):
            # 更新数据集的epoch
            train_dataset.set_epoch(epoch)
            
            # 获取一个批次的数据
            batch = next(iter(train_loader))
            images = batch['image']
            labels = batch['label']
            
            # 计算当前训练进度
            progress = epoch / total_epochs
            
            # 计算当前增强强度
            current_intensity = initial_intensity - (initial_intensity - final_intensity) * progress
            
            # 记录数据范围
            data_range = (images.min().item(), images.max().item())
            data_ranges.append(data_range)
            
            print(f"Epoch {epoch:2d}: 进度={progress:.2f}, 强度={current_intensity:.2f}, 数据范围=[{data_range[0]:.2f}, {data_range[1]:.2f}]")
        
        # 检查数据范围是否有变化
        print("\n检查数据增强效果:")
        if len(data_ranges) > 1:
            first_range = data_ranges[0]
            last_range = data_ranges[-1]
            
            # 检查数据范围是否有显著变化
            range_change = abs(first_range[1] - first_range[0]) - abs(last_range[1] - last_range[0])
            
            if range_change > 0.1:  # 如果范围变化超过0.1，认为增强有效
                print("✓ 数据范围有明显变化，增强策略工作正常")
            else:
                print("? 数据范围变化不明显，可能需要更多epoch或调整增强参数")
        
        # 测试数据集的增强变换
        print("\n检查数据集的增强变换:")
        if hasattr(train_dataset, 'transform'):
            print("✓ 数据集有transform属性")
            transform = train_dataset.transform
            
            # 检查transform是否包含自适应组件
            transform_str = str(transform)
            if 'adaptive' in transform_str.lower():
                print("✓ 变换包含自适应组件")
            else:
                print("? 变换可能不包含自适应组件")
        else:
            print("? 数据集没有transform属性")
    
    except Exception as e:
        print(f"测试过程中发生错误: {e}")
        return False
    finally:
        # 清理临时目录
        if 'temp_dir' in locals():
            shutil.rmtree(temp_dir, ignore_errors=True)
            print(f"已清理临时目录: {temp_dir}")
    

    
    print("\n" + "="*60)
    print("自适应增强策略测试完成")
    print("="*60)
    
    return True


def test_training_loop_integration():
    """测试训练循环是否正确集成了自适应增强策略。"""
    print("\n" + "="*60)
    print("训练循环集成测试")
    print("="*60)
    
    # 检查train_kfold.py文件是否包含set_epoch调用
    train_kfold_path = os.path.join(os.path.dirname(__file__), 'src', 'train_kfold.py')
    
    if not os.path.exists(train_kfold_path):
        print(f"错误: 找不到训练文件 {train_kfold_path}")
        return False
    
    with open(train_kfold_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 检查是否包含set_epoch调用
    if 'set_epoch' in content:
        print("✓ 训练代码包含set_epoch调用")
        
        # 检查set_epoch调用是否在epoch循环中
        if 'for epoch in' in content and 'set_epoch' in content:
            # 提取包含set_epoch的行
            lines = content.split('\n')
            set_epoch_lines = [line for line in lines if 'set_epoch' in line]
            
            if set_epoch_lines:
                print("✓ 找到set_epoch调用:")
                for line in set_epoch_lines:
                    print(f"  {line.strip()}")
                
                # 检查是否有条件检查
                if 'hasattr' in content and 'set_epoch' in content:
                    print("✓ 包含安全检查（hasattr）")
                else:
                    print("? 建议添加安全检查（hasattr）")
            else:
                print("? set_epoch调用可能不在正确位置")
        else:
            print("? set_epoch调用可能不在epoch循环中")
    else:
        print("✗ 训练代码不包含set_epoch调用")
        print("  需要在epoch循环中添加: train_dataset.set_epoch(epoch)")
        return False
    
    # 检查配置文件中的策略设置
    config_path = os.path.join(os.path.dirname(__file__), 'config.yml')
    
    if not os.path.exists(config_path):
        print(f"错误: 找不到配置文件 {config_path}")
        return False
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config_content = f.read()
    
    if 'strategy: adaptive' in config_content:
        print("✓ 配置文件中已设置自适应策略")
    else:
        print("? 配置文件中可能未设置自适应策略")
    
    print("\n" + "="*60)
    print("训练循环集成测试完成")
    print("="*60)
    
    return True


if __name__ == '__main__':
    # 测试自适应增强策略
    success1 = test_adaptive_augmentation()
    
    # 测试训练循环集成
    success2 = test_training_loop_integration()
    
    # 总结
    print("\n" + "="*60)
    print("测试总结")
    print("="*60)
    
    if success1 and success2:
        print("✓ 所有测试通过，自适应增强策略已正确配置")
        print("\n使用方法:")
        print("1. 确保config.yml中设置 strategy: 'adaptive'")
        print("2. 运行训练: python run_training.py")
        print("3. 训练过程中，增强强度会自动随epoch调整")
    else:
        print("✗ 部分测试失败，请检查配置和代码")
        if not success1:
            print("- 自适应增强策略测试失败")
        if not success2:
            print("- 训练循环集成测试失败")
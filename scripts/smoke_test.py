"""
文件名称：smoke_test.py
文件功能：快速验证数据读取、预处理和模型前向传播是否正常工作。
创建日期：2025-11-19
版本：v1.0
版权声明：Copyright (c) 2025, All rights reserved.

说明：
- 验证数据集能否正确加载
- 检查数据预处理流程
- 测试模型前向传播
- 确认输出张量形状正确
"""

import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import numpy as np
from src.config import get_data_config, get_training_config
from src.dataset import collect_cases, collect_cases_multi, ProstateDataset
from src.unet3d import UNet3D


def test_data_loading():
    """测试数据加载功能。"""
    print("=" * 60)
    print("测试 1: 数据加载")
    print("=" * 60)
    
    data_config = get_data_config()
    modalities = data_config.get('modalities', ['DWI'])
    
    print(f"配置的模态: {modalities}")
    
    # 收集病例
    if len(modalities) == 1:
        cases = collect_cases(modalities[0])
    else:
        cases = collect_cases_multi(modalities)
    
    print(f"✓ 找到 {len(cases)} 个病例")
    
    if len(cases) == 0:
        print("✗ 错误: 未找到任何病例！")
        print(f"  请检查数据目录: {data_config['data_root']}")
        return False
    
    # 显示前几个病例
    print(f"\n前 3 个病例:")
    for i, case in enumerate(cases[:3]):
        print(f"  {i+1}. {case['id']}")
    
    return True, cases


def test_dataset():
    """测试数据集类。"""
    print("\n" + "=" * 60)
    print("测试 2: 数据集预处理")
    print("=" * 60)
    
    data_config = get_data_config()
    modalities = data_config.get('modalities', ['DWI'])
    target_shape = tuple(data_config.get('target_shape', (256, 256, 32)))
    
    # 收集病例
    if len(modalities) == 1:
        cases = collect_cases(modalities[0])
    else:
        cases = collect_cases_multi(modalities)
    
    if len(cases) == 0:
        return False
    
    # 创建数据集
    dataset = ProstateDataset(cases[:5], augment=False, modalities=modalities)
    
    print(f"✓ 数据集创建成功，包含 {len(dataset)} 个样本")
    
    # 读取第一个样本
    try:
        sample = dataset[0]
        print(f"✓ 成功读取样本: {sample['id']}")
        print(f"  - 图像形状: {sample['image'].shape}")
        print(f"  - 标签形状: {sample['label'].shape}")
        print(f"  - 图像数据类型: {sample['image'].dtype}")
        print(f"  - 标签数据类型: {sample['label'].dtype}")
        
        # 验证形状
        expected_shape = (len(modalities), *target_shape)
        if sample['image'].shape == expected_shape:
            print(f"✓ 图像形状正确: {expected_shape}")
        else:
            print(f"✗ 图像形状错误: 期望 {expected_shape}, 实际 {sample['image'].shape}")
            return False
        
        # 验证数值范围
        img_min, img_max = sample['image'].min(), sample['image'].max()
        print(f"  - 图像值范围: [{img_min:.4f}, {img_max:.4f}]")
        
        label_unique = torch.unique(sample['label'])
        print(f"  - 标签唯一值: {label_unique.tolist()}")
        
        return True
        
    except Exception as e:
        print(f"✗ 读取样本失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model():
    """测试模型前向传播。"""
    print("\n" + "=" * 60)
    print("测试 3: 模型前向传播")
    print("=" * 60)
    
    data_config = get_data_config()
    modalities = data_config.get('modalities', ['DWI'])
    target_shape = tuple(data_config.get('target_shape', (256, 256, 32)))
    
    # 创建模型
    model = UNet3D(in_channels=len(modalities))
    print(f"✓ 模型创建成功")
    
    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  - 总参数量: {total_params:,}")
    print(f"  - 可训练参数: {trainable_params:,}")
    
    # 创建随机输入
    batch_size = 2
    input_shape = (batch_size, len(modalities), *target_shape)
    dummy_input = torch.randn(input_shape)
    
    print(f"\n测试输入形状: {input_shape}")
    
    # 前向传播
    try:
        model.eval()
        with torch.no_grad():
            output = model(dummy_input)
        
        print(f"✓ 前向传播成功")
        print(f"  - 输出形状: {output.shape}")
        print(f"  - 输出值范围: [{output.min():.4f}, {output.max():.4f}]")
        
        # 验证输出形状
        expected_output_shape = (batch_size, 1, *target_shape)
        if output.shape == expected_output_shape:
            print(f"✓ 输出形状正确: {expected_output_shape}")
        else:
            print(f"✗ 输出形状错误: 期望 {expected_output_shape}, 实际 {output.shape}")
            return False
        
        # 验证输出范围（应该在 [0, 1] 之间，因为有 Sigmoid）
        if output.min() >= 0 and output.max() <= 1:
            print(f"✓ 输出值范围正确 (Sigmoid 激活)")
        else:
            print(f"⚠ 警告: 输出值超出 [0, 1] 范围")
        
        return True
        
    except Exception as e:
        print(f"✗ 前向传播失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_augmentation():
    """测试数据增强。"""
    print("\n" + "=" * 60)
    print("测试 4: 数据增强")
    print("=" * 60)
    
    data_config = get_data_config()
    modalities = data_config.get('modalities', ['DWI'])
    use_torchio = data_config.get('use_torchio', False)
    
    # 收集病例
    if len(modalities) == 1:
        cases = collect_cases(modalities[0])
    else:
        cases = collect_cases_multi(modalities)
    
    if len(cases) == 0:
        return False
    
    # 创建带增强的数据集
    try:
        dataset = ProstateDataset(cases[:3], augment=True, modalities=modalities, use_torchio=use_torchio)
        print(f"✓ 增强数据集创建成功 (use_torchio={use_torchio})")
        
        # 读取样本测试增强
        sample = dataset[0]
        print(f"✓ 增强样本读取成功")
        print(f"  - 图像形状: {sample['image'].shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ 数据增强测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """运行所有测试。"""
    print("\n" + "=" * 60)
    print("前列腺癌分割项目 - 冒烟测试")
    print("=" * 60)
    
    results = []
    
    # 测试 1: 数据加载
    result = test_data_loading()
    if isinstance(result, tuple):
        success, cases = result
        results.append(success)
    else:
        results.append(result)
        if not result:
            print("\n✗ 数据加载失败，跳过后续测试")
            return
    
    # 测试 2: 数据集
    results.append(test_dataset())
    
    # 测试 3: 模型
    results.append(test_model())
    
    # 测试 4: 数据增强
    results.append(test_augmentation())
    
    # 总结
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)
    
    test_names = ["数据加载", "数据集预处理", "模型前向传播", "数据增强"]
    for name, result in zip(test_names, results):
        status = "✓ 通过" if result else "✗ 失败"
        print(f"{name}: {status}")
    
    all_passed = all(results)
    print("\n" + "=" * 60)
    if all_passed:
        print("✓ 所有测试通过！项目配置正确，可以开始训练。")
    else:
        print("✗ 部分测试失败，请检查配置和数据。")
    print("=" * 60)


if __name__ == "__main__":
    main()

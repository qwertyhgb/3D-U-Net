#!/usr/bin/env python3
"""
自适应增强策略演示脚本

这个脚本演示了如何使用自适应增强策略进行训练。
自适应增强策略会随着训练进度自动调整增强强度，
在训练初期使用较强的增强，后期逐渐减弱。

使用方法:
    python demo_adaptive_augmentation.py
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

# 添加项目路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.config import get_data_config, get_training_config, get_augmentation_config
from src.dataset import ProstateDataset, collect_cases_multi
from src.torchio_transforms import get_adaptive_train_transforms, get_train_transforms
from src.utils import read_volume, clip_and_normalize, pad_or_crop_to_shape


def visualize_augmentation_intensity():
    """可视化不同epoch的增强强度变化"""
    print("="*60)
    print("自适应增强强度可视化")
    print("="*60)
    
    # 获取配置
    augmentation_config = get_augmentation_config()
    training_config = get_training_config()
    
    # 获取自适应增强参数
    initial_intensity = augmentation_config['adaptive']['initial_intensity']
    final_intensity = augmentation_config['adaptive']['final_intensity']
    total_epochs = training_config['num_epochs']
    
    # 计算每个epoch的增强强度
    epochs = np.arange(1, total_epochs + 1)
    progress = epochs / total_epochs
    intensities = initial_intensity - (initial_intensity - final_intensity) * progress
    
    # 绘制增强强度变化曲线
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, intensities, 'b-', linewidth=2)
    plt.title('自适应增强强度随训练进度变化', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('增强强度', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.xlim(0, total_epochs)
    plt.ylim(0, initial_intensity + 0.1)
    
    # 添加关键点标注
    plt.scatter([1, total_epochs], [initial_intensity, final_intensity], color='red', s=100)
    plt.annotate(f'初始强度: {initial_intensity}', xy=(1, initial_intensity), 
                xytext=(20, initial_intensity + 0.05), fontsize=10)
    plt.annotate(f'最终强度: {final_intensity}', xy=(total_epochs, final_intensity), 
                xytext=(total_epochs - 40, final_intensity - 0.1), fontsize=10)
    
    # 保存图像
    output_dir = os.path.join(os.getcwd(), 'outputs', 'plots')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'adaptive_augmentation_intensity.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"增强强度变化曲线已保存至: {output_path}")
    
    plt.show()
    
    return epochs, intensities


def compare_augmentation_strategies():
    """比较不同增强策略的效果"""
    print("\n" + "="*60)
    print("不同增强策略效果比较")
    print("="*60)
    
    # 获取配置
    augmentation_config = get_augmentation_config()
    data_config = get_data_config()
    
    # 创建模拟数据
    np.random.seed(42)
    volume = np.random.rand(32, 256, 256).astype(np.float32)
    mask = (volume > 0.7).astype(np.uint8)
    
    # 获取不同策略的变换
    standard_transform = get_train_transforms(probability=0.5, config=augmentation_config)
    adaptive_transform_early = get_adaptive_train_transforms(1, 100, augmentation_config)
    adaptive_transform_mid = get_adaptive_train_transforms(50, 100, augmentation_config)
    adaptive_transform_late = get_adaptive_train_transforms(100, 100, augmentation_config)
    
    # 应用变换
    from src.torchio_transforms import apply_transforms_single
    
    # 原始数据
    original_volume, original_mask = volume.copy(), mask.copy()
    
    # 标准增强
    standard_volume, standard_mask = apply_transforms_single(
        volume.copy(), mask.copy(), standard_transform
    )
    
    # 自适应增强 - 早期
    adaptive_early_volume, adaptive_early_mask = apply_transforms_single(
        volume.copy(), mask.copy(), adaptive_transform_early
    )
    
    # 自适应增强 - 中期
    adaptive_mid_volume, adaptive_mid_mask = apply_transforms_single(
        volume.copy(), mask.copy(), adaptive_transform_mid
    )
    
    # 自适应增强 - 后期
    adaptive_late_volume, adaptive_late_mask = apply_transforms_single(
        volume.copy(), mask.copy(), adaptive_transform_late
    )
    
    # 可视化结果
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    
    # 选择中间切片进行可视化
    slice_idx = 16
    
    # 原始数据
    axes[0, 0].imshow(original_volume[slice_idx], cmap='gray')
    axes[0, 0].set_title('原始图像', fontsize=12)
    axes[0, 0].axis('off')
    
    axes[1, 0].imshow(original_mask[slice_idx], cmap='jet')
    axes[1, 0].set_title('原始标签', fontsize=12)
    axes[1, 0].axis('off')
    
    # 标准增强
    axes[0, 1].imshow(standard_volume[slice_idx], cmap='gray')
    axes[0, 1].set_title('标准增强', fontsize=12)
    axes[0, 1].axis('off')
    
    axes[1, 1].imshow(standard_mask[slice_idx], cmap='jet')
    axes[1, 1].set_title('标准增强标签', fontsize=12)
    axes[1, 1].axis('off')
    
    # 自适应增强 - 早期
    axes[0, 2].imshow(adaptive_early_volume[slice_idx], cmap='gray')
    axes[0, 2].set_title('自适应增强 (早期)', fontsize=12)
    axes[0, 2].axis('off')
    
    axes[1, 2].imshow(adaptive_early_mask[slice_idx], cmap='jet')
    axes[1, 2].set_title('自适应增强标签 (早期)', fontsize=12)
    axes[1, 2].axis('off')
    
    # 自适应增强 - 后期
    axes[0, 3].imshow(adaptive_late_volume[slice_idx], cmap='gray')
    axes[0, 3].set_title('自适应增强 (后期)', fontsize=12)
    axes[0, 3].axis('off')
    
    axes[1, 3].imshow(adaptive_late_mask[slice_idx], cmap='jet')
    axes[1, 3].set_title('自适应增强标签 (后期)', fontsize=12)
    axes[1, 3].axis('off')
    
    plt.tight_layout()
    
    # 保存图像
    output_dir = os.path.join(os.getcwd(), 'outputs', 'plots')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'augmentation_strategies_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"增强策略比较图已保存至: {output_path}")
    
    plt.show()


def demo_adaptive_augmentation():
    """演示自适应增强策略在不同epoch的效果变化。"""
    print("="*60)
    print("自适应增强策略演示")
    print("="*60)
    
    # 加载配置
    data_config = get_data_config()
    training_config = get_training_config()
    augmentation_config = get_augmentation_config()
    
    # 检查是否有真实数据
    data_root = data_config['data_root']
    if not os.path.exists(data_root):
        print(f"警告: 数据路径 {data_root} 不存在，使用模拟数据")
        # 创建模拟数据目录结构
        os.makedirs(os.path.join(data_root, "images"), exist_ok=True)
        os.makedirs(os.path.join(data_root, "labels"), exist_ok=True)
        
        # 创建模拟数据文件
        for i in range(5):
            # 模拟多模态图像文件
            for modality in data_config['modalities']:
                img_path = os.path.join(data_root, "images", f"case_{i:03d}_{modality}.nii.gz")
                if not os.path.exists(img_path):
                    # 创建一个简单的3D numpy数组并保存
                    import nibabel as nib
                    img_data = np.random.rand(32, 32, 32) * 100  # 模拟3D图像
                    affine = np.eye(4)  # 单位矩阵作为仿射变换
                    nib_img = nib.Nifti1Image(img_data, affine)
                    nib.save(nib_img, img_path)
            
            # 模拟标签文件
            label_path = os.path.join(data_root, "labels", f"case_{i:03d}_seg.nii.gz")
            if not os.path.exists(label_path):
                import nibabel as nib
                label_data = np.random.randint(0, 2, size=(32, 32, 32))  # 二进制标签
                affine = np.eye(4)
                nib_label = nib.Nifti1Image(label_data, affine)
                nib.save(nib_label, label_path)
    
    # 尝试收集真实病例
    try:
        modalities = data_config['modalities']
        cases = collect_cases_multi(modalities)
        
        if len(cases) == 0:
            print("警告: 未找到有效病例，使用模拟数据")
            # 获取病例列表
            case_ids = []
            for i in range(5):
                case_ids.append(f"case_{i:03d}")
            
            # 创建数据集
            train_dataset = ProstateDataset(
                data_root=data_root,
                case_ids=case_ids,
                phase='train',
                config=data_config,
                modalities=modalities
            )
        else:
            # 只使用前几个病例进行演示
            demo_cases = cases[:min(5, len(cases))]
            
            # 创建数据集
            train_dataset = ProstateDataset(
                cases=demo_cases,
                augment=True,
                modalities=modalities,
                use_torchio=True,
                augmentation_config=augmentation_config
            )
    except Exception as e:
        print(f"使用真实数据时出错: {e}，使用模拟数据")
        # 获取病例列表
        case_ids = []
        for i in range(5):
            case_ids.append(f"case_{i:03d}")
        
        # 创建数据集
        train_dataset = ProstateDataset(
            data_root=data_root,
            case_ids=case_ids,
            phase='train',
            config=data_config,
            modalities=data_config['modalities']
        )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=2,
        shuffle=True,
        num_workers=0  # 设置为0以避免多进程问题
    )
    
    # 模拟训练过程，展示不同epoch的增强强度
    num_epochs = 10
    total_epochs = training_config['num_epochs']
    
    print(f"模拟训练 {num_epochs} 个epoch（总训练epoch: {total_epochs}）")
    print("观察自适应增强策略如何随训练进度调整增强强度\n")
    
    for epoch in range(1, num_epochs + 1):
        # 更新数据集的epoch（关键步骤）
        if hasattr(train_dataset, 'set_epoch'):
            train_dataset.set_epoch(epoch)
        
        # 获取一个批次的数据
        batch = next(iter(train_loader))
        images = batch['image']
        labels = batch['label']
        
        # 计算当前训练进度
        progress = epoch / total_epochs
        
        # 计算当前增强强度
        initial_intensity = augmentation_config['adaptive']['initial_intensity']
        final_intensity = augmentation_config['adaptive']['final_intensity']
        
        # 使用线性插值计算当前强度
        current_intensity = initial_intensity - (initial_intensity - final_intensity) * progress
        
        print(f"Epoch {epoch:2d}/{num_epochs} | 进度: {progress:.2f} | 增强强度: {current_intensity:.2f}")
        
        # 如果是第一个或最后一个epoch，显示更详细的信息
        if epoch == 1 or epoch == num_epochs:
            print(f"  - 图像批次形状: {images.shape}")
            print(f"  - 标签批次形状: {labels.shape}")
            print(f"  - 图像数据范围: [{images.min():.2f}, {images.max():.2f}]")
            print(f"  - 标签数据范围: [{labels.min():.2f}, {labels.max():.2f}]")
            
            # 展示不同模态的数据范围
            if len(images.shape) > 4:  # 多模态数据
                for i, modality in enumerate(data_config['modalities']):
                    modality_data = images[:, i, ...]  # 取第一个样本的第i个模态
                    print(f"  - {modality}模态数据范围: [{modality_data.min():.2f}, {modality_data.max():.2f}]")
    
    print("\n" + "="*60)
    print("自适应增强策略演示完成")
    print("="*60)
    print("\n关键点:")
    print("1. 在训练循环中，每个epoch开始时调用 train_dataset.set_epoch(epoch)")
    print("2. 自适应策略会根据训练进度自动调整增强强度")
    print("3. 初期增强强度高，后期逐渐降低，有助于模型稳定收敛")
    print("4. 这种策略特别适合长时间训练和数据量有限的情况")


def compare_strategies():
    """比较不同增强策略的效果。"""
    print("\n" + "="*60)
    print("不同增强策略对比")
    print("="*60)
    
    # 加载配置
    data_config = get_data_config()
    training_config = get_training_config()
    augmentation_config = get_augmentation_config()
    
    # 模拟数据路径（如果实际数据不存在）
    data_root = data_config['data_root']
    if not os.path.exists(data_root):
        print(f"警告: 数据路径 {data_root} 不存在，使用模拟数据")
        # 创建模拟数据目录结构
        os.makedirs(os.path.join(data_root, "BPH", "DWI"), exist_ok=True)
        os.makedirs(os.path.join(data_root, "PCA", "DWI"), exist_ok=True)
        os.makedirs(os.path.join(data_root, "ROI(BPH+PCA)", "BPH"), exist_ok=True)
        os.makedirs(os.path.join(data_root, "ROI(BPH+PCA)", "PCA"), exist_ok=True)
        
        # 创建模拟数据文件
        for cohort in ['BPH', 'PCA']:
            for i in range(3):
                case_id = f"{i:04d}"
                
                # 模拟图像文件
                img_path = os.path.join(data_root, cohort, "DWI", f"{case_id}.nii")
                if not os.path.exists(img_path):
                    # 创建一个简单的3D numpy数组并保存
                    import nibabel as nib
                    img_data = np.random.rand(32, 32, 32) * 100  # 模拟3D图像
                    affine = np.eye(4)  # 单位矩阵作为仿射变换
                    nib_img = nib.Nifti1Image(img_data, affine)
                    nib.save(nib_img, img_path)
                
                # 模拟标签文件
                label_path = os.path.join(data_root, "ROI(BPH+PCA)", cohort, f"{case_id}.nii")
                if not os.path.exists(label_path):
                    import nibabel as nib
                    label_data = np.random.randint(0, 2, size=(32, 32, 32))  # 二进制标签
                    affine = np.eye(4)
                    nib_label = nib.Nifti1Image(label_data, affine)
                    nib.save(nib_label, label_path)
    
    # 使用collect_cases函数收集病例
    from src.dataset import collect_cases
    cases = collect_cases('DWI')
    
    # 如果仍然没有病例，创建模拟病例
    if len(cases) == 0:
        cases = []
        for cohort in ['BPH', 'PCA']:
            for i in range(3):
                case_id = f"{i:04d}"
                case_dict = {
                    'id': f'{cohort}-{case_id}',
                    'image': os.path.join(data_root, cohort, "DWI", f"{case_id}.nii"),
                    'label': os.path.join(data_root, "ROI(BPH+PCA)", cohort, f"{case_id}.nii")
                }
                cases.append(case_dict)
    
    # 测试不同策略
    strategies = ['standard', 'light', 'intensity_only', 'adaptive']
    
    for strategy in strategies:
        print(f"\n测试策略: {strategy}")
        print("-" * 40)
        
        # 临时修改配置中的策略
        original_strategy = augmentation_config.get('strategy', 'standard')
        augmentation_config['strategy'] = strategy
        
        # 创建数据集
        train_dataset = ProstateDataset(
            cases=cases,
            augment=True,
            modalities=data_config['modalities'],
            use_torchio=True,
            augmentation_config=augmentation_config
        )
        
        # 创建数据加载器
        train_loader = DataLoader(
            train_dataset,
            batch_size=1,
            shuffle=True,
            num_workers=0  # 设置为0以避免多进程问题
        )
        
        # 获取一个批次的数据
        batch = next(iter(train_loader))
        images = batch['image']
        labels = batch['label']
        
        print(f"  - 图像批次形状: {images.shape}")
        print(f"  - 标签批次形状: {labels.shape}")
        print(f"  - 图像数据范围: [{images.min():.2f}, {images.max():.2f}]")
        
        # 如果是自适应策略，展示不同epoch的变化
        if strategy == 'adaptive':
            print("  - 自适应策略不同epoch的增强强度:")
            for epoch in [1, 10, 50, 100]:
                if epoch <= training_config.get('num_epochs', 100):
                    train_dataset.set_epoch(epoch)
                    batch = next(iter(train_loader))
                    images = batch['image']
                    progress = epoch / training_config.get('num_epochs', 100)
                    initial_intensity = augmentation_config.get('adaptive', {}).get('initial_intensity', 1.0)
                    final_intensity = augmentation_config.get('adaptive', {}).get('final_intensity', 0.5)
                    current_intensity = initial_intensity - (initial_intensity - final_intensity) * progress
                    print(f"    Epoch {epoch:3d}: 强度 {current_intensity:.2f}, 数据范围 [{images.min():.2f}, {images.max():.2f}]")
        
        # 恢复原始策略
        augmentation_config['strategy'] = original_strategy
    
    print("\n" + "="*60)
    print("策略对比完成")
    print("="*60)


def main():
    """主函数"""
    print("自适应增强策略演示")
    print("="*60)
    
    # 检查配置
    augmentation_config = get_augmentation_config()
    
    strategy = augmentation_config.get('strategy', 'standard')
    print(f"当前增强策略: {strategy}")
    
    if strategy != 'adaptive':
        print("警告: 当前配置不是自适应策略")
        print("请确保config.yml中设置:")
        print("```yaml")
        print("augmentation:")
        print("  strategy: 'adaptive'")
        print("```")
        # 仍然继续演示，但使用临时配置
        print("将临时使用自适应策略进行演示")
        augmentation_config['strategy'] = 'adaptive'
    
    # 1. 可视化增强强度变化
    epochs, intensities = visualize_augmentation_intensity()
    
    # 2. 比较不同增强策略
    compare_augmentation_strategies()
    
    # 3. 演示自适应增强策略
    demo_adaptive_augmentation()
    
    # 4. 比较不同策略
    compare_strategies()
    
    print("\n" + "="*60)
    print("演示完成")
    print("="*60)
    print("\n自适应增强策略的主要优势:")
    print("1. 训练初期使用较强增强，提高模型鲁棒性")
    print("2. 训练后期逐渐减弱增强，帮助模型收敛")
    print("3. 自动调整，无需手动干预")
    print("4. 可配置初始和最终强度，适应不同需求")
    print("\n配置参数说明:")
    print("- initial_intensity: 初始增强强度 (默认: 1.0)")
    print("- final_intensity: 最终增强强度 (默认: 0.5)")
    print("- decay_rate: 强度衰减速率 (默认: 0.5)")
    print("\n更多详情请参考:")
    print("- README.md: 项目文档和配置说明")
    print("- doc/AUGMENTATION_OPTIMIZATION.md: 数据增强优化指南")


if __name__ == "__main__":
    main()
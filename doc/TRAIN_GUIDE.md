# 3D U-Net 前列腺癌分割训练指南

## 概述

本指南详细介绍了如何使用 3D U-Net 模型训练前列腺癌分割模型，包括环境准备、数据准备、训练配置、训练执行和结果分析等完整流程。

## 环境准备

### 系统要求

- Python 3.9 或更高版本
- CUDA 11.6+ (可选，用于 GPU 加速)
- 8GB+ RAM (16GB+ 推荐)
- 10GB+ 可用磁盘空间

### 安装依赖

```bash
# 创建虚拟环境（推荐）
python -m venv venv

# 激活虚拟环境
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# 安装依赖
pip install -r requirements.txt
```

### 验证环境

```bash
python scripts/smoke_test.py
```

如果看到 "✓ 所有测试通过！"，说明环境配置正确。

## 数据准备

### 数据结构

将您的数据按以下结构组织：

```
data/BPH-PCA/
├── BPH/
│   ├── ADC/
│   │   ├── 00001234.nii
│   │   └── ...
│   ├── DWI/
│   │   ├── 00001234.nii
│   │   └── ...
│   └── T2 fs/
│       ├── 00001234.nii
│       └── ...
├── PCA/
│   ├── ADC/
│   │   ├── 00005678.nii
│   │   └── ...
│   ├── DWI/
│   │   ├── 00005678.nii
│   │   └── ...
│   └── T2 fs/
│       ├── 00005678.nii
│       └── ...
└── ROI(BPH+PCA)/
    ├── BPH/
    │   ├── 00001234.nii
    │   └── ...
    └── PCA/
        ├── 00005678.nii
        └── ...
```

### 数据格式

- 支持 NIfTI (.nii) 和 DICOM 格式
- 标签文件应为二值化分割掩码
- 所有文件名应一一对应（影像和标签）

## 配置调整

### config.yml 配置文件

编辑 `config.yml` 文件调整训练参数：

```yaml
# 数据配置
data:
  data_root: './data/BPH-PCA'              # 数据根目录
  label_root: './data/BPH-PCA/ROI(BPH+PCA)' # 标签目录
  output_dir: './outputs'                   # 输出目录
  modalities: ['DWI']                       # 输入模态
  target_shape: [256, 256, 32]              # 目标尺寸 [Z, Y, X]
  use_torchio: true                         # 使用 TorchIO 增强

# 数据增强配置
augmentation:
  # 增强策略: 'standard', 'light', 'intensity_only', 'adaptive'
  strategy: 'adaptive'
  
  # 自适应增强参数
  adaptive:
    initial_intensity: 1.0   # 初始增强强度
    final_intensity: 0.5     # 最终增强强度
    decay_rate: 0.5          # 强度衰减率

# 训练配置
training:
  batch_size: 4                             # 批次大小
  num_epochs: 100                           # 最大 epoch
  learning_rate: 0.0001                     # 学习率
  patience: 10                              # 早停耐心
  num_folds: 5                              # K 折数
  num_workers: 2                            # 数据加载线程
  amp: true                                 # 混合精度训练
  
  # 损失函数配置
  loss:
    type: 'dice_bce'      # Dice损失 + BCE损失
    dice_weight: 0.5      # Dice损失权重
    bce_weight: 0.5       # BCE损失权重
  
  # 优化器配置
  optimizer:
    type: 'adamw'  # 使用AdamW优化器
    lr: 0.0001  # 学习率
    weight_decay: 0.01  # 权重衰减
  
  # 学习率调度器配置
  scheduler:
    type: 'cosine'  # 使用余弦退火调度器
    T_0: 10  # 第一次重启的周期长度
    eta_min: 1e-6  # 最小学习率
  
  # 梯度裁剪配置
  gradient_clipping:
    type: 'norm'  # 裁剪类型: 'norm' 或 'value'
    max_norm: 1.0  # 最大范数或值
```

### 关键配置说明

1. **模态选择**：
   - 单模态：`modalities: ['DWI']`
   - 多模态：`modalities: ['ADC', 'DWI', 'T2 fs']`

2. **增强策略**：
   - `standard`：标准增强，适用于数据量充足的情况
   - `light`：轻量级增强，适用于数据量有限的情况
   - `intensity_only`：仅强度变换，不改变几何结构
   - `adaptive`：自适应增强，训练初期强增强，后期弱增强

3. **混合精度训练**：
   - `amp: true` 启用混合精度训练，可减少显存使用并加速训练

## 训练执行

### 标准训练

```bash
# 标准训练（5 折交叉验证）
python run_training.py
```

### 快速测试

```bash
# 快速测试（1 折，5 epoch）
python run_training.py --quick-test
```

### 自定义配置

```bash
# 使用自定义配置文件
python run_training.py --config my_config.yml
```

### 指定设备

```bash
# 使用 CPU 训练
python run_training.py --device cpu

# 使用 GPU 训练
python run_training.py --device cuda
```

## 训练监控

### TensorBoard 可视化

```bash
# 启动 TensorBoard
tensorboard --logdir outputs/runs

# 在浏览器中打开 http://localhost:6006
```

### 日志文件

训练日志保存在 `outputs/logs/` 目录：

- `training_main.log`：主训练日志
- `fold_{i}_training.log`：各折详细训练日志
- `fold_{i}_log.csv`：各折训练过程 CSV 记录

## 结果分析

### 输出文件结构

训练完成后，结果保存在 `outputs/` 目录：

```
outputs/
├── models/
│   ├── fold_0_best.pth      # Fold 0 最佳模型
│   └── ...
├── logs/
│   ├── fold_0_log.csv       # 训练日志（每个 epoch）
│   ├── fold_0_eval.csv      # 评估结果（每个样本）
│   └── ...
├── plots/
│   ├── fold_0_curves.png    # 训练/验证曲线
│   └── ...
└── preds/
    ├── BPH-00001234_pred.nii    # 预测结果
    ├── BPH-00001234_overlay.png # 可视化
    └── ...
```

### 评估模型

```bash
# 评估所有折
python run_evaluation.py

# 评估并汇总结果
python run_evaluation.py --aggregate

# 只汇总已有结果
python run_evaluation.py --aggregate-only
```

### 性能指标

项目使用以下指标评估分割性能：

- **Dice系数**：衡量区域重叠度（0-1，越大越好）
- **DSC**：Dice Similarity Coefficient，重叠度指标
- **Hausdorff距离**：衡量最大边界偏差（越小越好）
- **NSD**：归一化表面距离，衡量边界精度（0-1，越大越好）

## 高级功能

### 测试时增强 (TTA)

```bash
# 使用测试时增强进行预测
python -m src.predict --model_path outputs/models/fold_0_best.pth --use_tta
```

### 独立预测

```bash
# 标准预测
python -m src.predict --model_path outputs/models/fold_0_best.pth

# 多模态预测
python -m src.predict --model_path outputs/models/fold_0_best.pth --modality ADC DWI T2
```

## 性能调优

### GPU 内存不足

```yaml
# 减小批次大小
batch_size: 2

# 减小目标尺寸
target_shape: [128, 128, 24]

# 关闭混合精度训练
amp: false
```

### 训练速度慢

```yaml
# 启用混合精度训练
amp: true

# 增加批次大小（如果显存允许）
batch_size: 8

# 增加数据加载线程（非 Windows）
num_workers: 4
```

### 过拟合

```yaml
# 启用 TorchIO 增强
use_torchio: true

# 增加早停耐心
patience: 15

# 减小学习率
learning_rate: 0.00005
```

## 故障排除

### 常见错误

1. **找不到数据**：
   - 检查 `config.yml` 中的 `data_root` 路径
   - 确保数据文件是 `.nii` 格式
   - 确保文件名匹配（影像和标签）

2. **CUDA 错误**：
   ```bash
   # 减小批次大小
   batch_size: 2

   # 或使用 CPU
   python run_training.py --device cpu
   ```

3. **OpenMP 冲突 (Windows)**：
   ```powershell
   # PowerShell
   $env:KMP_DUPLICATE_LIB_OK="TRUE"
   python run_training.py
   ```

4. **数据加载慢**：
   ```yaml
   # Windows 用户
   num_workers: 0  # 或 1-2

   # Linux/Mac 用户
   num_workers: 4  # 或更多
   ```

## 最佳实践

1. **首次运行建议使用 `--quick-test`** 快速验证流程
2. **使用 TensorBoard** 实时监控训练过程
3. **定期备份** `outputs/models/` 目录
4. **尝试不同的超参数** 找到最佳配置
5. **使用 TTA** 可以提升预测精度 2-3%

## 参考文档

- [项目结构说明](PROJECT_STRUCTURE.md)
- [数据增强优化](AUGMENTATION_OPTIMIZATION.md)
- [损失函数优化](LOSS_OPTIMIZATION.md)
- [训练策略优化](TRAINING_OPTIMIZATION.md)
- [日志系统指南](LOGGING_GUIDE.md)
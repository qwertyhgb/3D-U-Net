# 前列腺癌 3D U-Net 分割训练指南

> 本指南介绍如何在本项目中进行数据准备、训练、评估与可视化，并提供参数配置与常见问题说明。

## 环境准备

### 系统要求
- **Python**: 3.9+
- **操作系统**: Windows / Linux / macOS
- **GPU**: 推荐使用CUDA支持的GPU（可选，CPU也可运行）

### 安装步骤

1. **安装依赖**
   ```bash
   pip install -r requirements.txt
   ```

2. **PyTorch安装**
   - `requirements.txt`包含适用于CUDA 11.6的PyTorch版本
   - 如果CUDA版本不同，请访问[PyTorch官网](https://pytorch.org/get-started/locally/)获取对应安装命令

3. **环境验证**
   ```bash
   python scripts/smoke_test.py
   ```
   应该看到成功加载病例和模型前向传播的信息

4. **Windows用户注意**
   如遇到OpenMP冲突错误，设置环境变量：
   ```powershell
   $env:KMP_DUPLICATE_LIB_OK="TRUE"
   ```

## 数据准备

期望目录结构：
```
data/BPH-PCA/
├── BPH/
│   ├── ADC/
│   ├── DWI/
│   └── T2 fs/
├── PCA/
│   ├── ADC/
│   ├── DWI/
│   └── T2 fs/
└── ROI(BPH+PCA)/
    ├── BPH/
    └── PCA/
```

配对规则：按病例ID将影像与标签一一匹配。

## 快速开始

### 1. 验证环境
```bash
python scripts/smoke_test.py
```

### 2. 快速训练测试
```bash
python run_training.py --quick-test
```
快速测试模式：1折，5个epoch。

### 3. 完整训练
```bash
python run_training.py
```
标准5折交叉验证训练。

## 训练配置

### 核心配置（config.yml）

#### 数据配置
```yaml
data:
  modalities: ['DWI']              # 输入模态
  target_shape: [256, 256, 32]    # 目标尺寸
  use_torchio: true                # 使用TorchIO增强
```

#### 训练配置
```yaml
training:
  batch_size: 4
  num_epochs: 100
  learning_rate: 0.0001
  patience: 10
  num_folds: 5
  amp: true                        # 混合精度训练
```

#### 损失函数配置
```yaml
training:
  loss:
    type: 'combined'               # dice/focal/boundary/combined
    weights: [0.5, 0.4, 0.1]
```

#### 数据增强配置
```yaml
augmentation:
  strategy: 'adaptive'             # standard/light/intensity_only/adaptive
  adaptive:
    initial_intensity: 1.0
    final_intensity: 0.5
```

### 多模态训练

```yaml
data:
  modalities: ['ADC', 'DWI', 'T2 fs']
```

## 模型评估

### 评估所有折
```bash
python run_evaluation.py --aggregate
```

### 评估单个折
```bash
python run_evaluation.py --fold 0
```

### 仅汇总结果
```bash
python run_evaluation.py --aggregate-only
```

## 独立预测

### 基本预测
```bash
python -m src.predict --model_path outputs/models/fold_0_best.pth
```

### 使用TTA
```bash
python -m src.predict --model_path outputs/models/fold_0_best.pth --use_tta
```

### 多模态预测
```bash
python -m src.predict --model_path outputs/models/fold_0_best.pth --modality ADC DWI T2
```

## 输出目录结构

```
outputs/
├── models/          # 训练好的模型
├── logs/            # 训练和评估日志
├── plots/           # 训练曲线
├── preds/           # 预测结果
└── runs/            # TensorBoard日志
```

## 高级配置

### 损失函数选择

**Dice Loss**（默认）
```yaml
loss:
  type: 'dice'
```

**BCE Loss**
```yaml
loss:
  type: 'bce'
```

**Dice + BCE Loss**（推荐）
```yaml
loss:
  type: 'dice_bce'
  dice_weight: 0.5
  bce_weight: 0.5
```

### 数据增强策略

- **standard**: 全面增强，数据量充足时使用
- **light**: 轻量级增强，数据量有限时使用
- **intensity_only**: 仅强度变换
- **adaptive**: 自适应增强（推荐）

### 优化器选择

- **Adam**: 标准选择
- **AdamW**: 改进的权重衰减（推荐）
- **RAdam**: 修正的Adam，更稳定
- **SGD**: 经典优化器
- **RMSprop**: 特殊场景使用

### 学习率调度器

- **plateau**: 基于验证损失调整
- **cosine**: 余弦退火
- **polynomial**: 多项式衰减
- **exponential**: 指数衰减

## 常见问题

### CUDA内存不足
1. 减小`batch_size`
2. 减小`target_shape`
3. 关闭混合精度：`amp: false`

### 训练速度慢
1. 启用混合精度：`amp: true`
2. 增加`num_workers`
3. 使用SSD存储数据

### OpenMP冲突
```powershell
$env:KMP_DUPLICATE_LIB_OK="TRUE"
```

### 训练不收敛
1. 调整学习率
2. 检查数据归一化
3. 调整批次大小
4. 使用轻量级增强

## 监控训练

### TensorBoard
```bash
tensorboard --logdir outputs/runs
```

### 查看日志
```bash
cat outputs/logs/fold_0_log.csv
```

## 性能优化

### 训练速度
- 启用混合精度训练
- 优化数据加载
- 使用高效增强策略

### 内存优化
- 减小批次大小
- 减小输入尺寸
- 使用梯度累积

## 实验复现

### 设置随机种子
```python
import random, numpy as np, torch

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
```

## 进阶功能

- 自定义损失函数：编辑`src/losses.py`
- 自定义数据增强：编辑`src/transforms3d.py`
- 修改模型架构：编辑`src/unet3d.py`
- 添加评估指标：编辑`src/metrics.py`

## 相关文档

- **README.md** - 项目概述
- **PROJECT_STRUCTURE.md** - 项目结构
- **LOSS_OPTIMIZATION.md** - 损失函数优化
- **AUGMENTATION_OPTIMIZATION.md** - 数据增强优化
- **PROJECT_STATUS.md** - 项目状态

## 技术支持

1. 查看相关文档
2. 运行`python scripts/smoke_test.py`
3. 运行`python scripts/verify_fixes.py`
4. 检查GitHub Issues

---

**最后更新**: 2025-11-19  
**版本**: v2.0  
**状态**: 生产就绪 ✅

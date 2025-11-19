# 项目结构说明

## 目录结构

```
3D-U-Net/
├── config.yml                  # 全局配置文件
├── requirements.txt            # Python 依赖
├── README.md                   # 项目说明
├── .gitignore                  # Git 忽略文件
│
├── run_training.py             # 训练启动脚本 ⭐
├── run_evaluation.py           # 评估启动脚本 ⭐
│
├── src/                        # 核心源代码
│   ├── __init__.py
│   ├── config.py               # 配置加载
│   ├── dataset.py              # 数据集定义
│   ├── unet3d.py               # 3D U-Net 模型
│   ├── losses.py               # 损失函数
│   ├── metrics.py              # 评估指标
│   ├── utils.py                # 工具函数
│   ├── transforms3d.py         # 传统数据增强
│   ├── torchio_transforms.py   # TorchIO 增强
│   ├── visualize.py            # 可视化工具
│   ├── train_kfold.py          # K折训练
│   ├── evaluate.py             # 模型评估
│   ├── aggregate_results.py    # 结果汇总
│   └── predict.py              # 独立预测
│
├── scripts/                    # 辅助脚本
│   └── smoke_test.py           # 冒烟测试
│
├── doc/                        # 文档
│   ├── TRAINING_GUIDE.md       # 训练指南
│   └── OPTIMIZATION_SUMMARY.md # 优化总结
│
├── data/                       # 数据目录
│   └── BPH-PCA/
│       ├── BPH/
│       │   ├── ADC/
│       │   ├── DWI/
│       │   └── T2 fs/
│       ├── PCA/
│       │   ├── ADC/
│       │   ├── DWI/
│       │   └── T2 fs/
│       └── ROI(BPH+PCA)/
│           ├── BPH/
│           └── PCA/
│
└── outputs/                    # 输出目录
    ├── models/                 # 训练好的模型
    ├── logs/                   # 训练日志
    ├── plots/                  # 训练曲线
    ├── preds/                  # 预测结果
    └── runs/                   # TensorBoard 日志
```

## 核心模块说明

### 1. 配置系统 (`src/config.py`)
- 从 `config.yml` 加载配置
- 提供 `get_data_config()` 和 `get_training_config()` 接口
- 自动创建输出目录
- 处理 Windows OpenMP 冲突

### 2. 数据处理 (`src/dataset.py`)
- `collect_cases()`: 收集单模态病例
- `collect_cases_multi()`: 收集多模态病例
- `ProstateDataset`: PyTorch 数据集类
- `build_splits()`: 构建 K 折交叉验证

### 3. 模型定义 (`src/unet3d.py`)
- 标准 3D U-Net 架构
- 4 层编码器和解码器
- 跳跃连接
- Sigmoid 输出激活

### 4. 训练流程 (`src/train_kfold.py`)
- K 折交叉验证
- 早停机制
- 学习率自适应衰减
- 混合精度训练 (AMP)
- TensorBoard 日志
- 进度条显示

### 5. 评估系统 (`src/evaluate.py`, `src/aggregate_results.py`)
- 计算 Dice、Hausdorff、体积相似度
- 生成预测可视化
- 汇总所有折的统计结果

### 6. 数据增强
- **传统方法** (`src/transforms3d.py`):
  - 随机翻转
  - 90° 旋转
  - 弹性形变
  
- **TorchIO 方法** (`src/torchio_transforms.py`):
  - 随机仿射变换
  - 弹性形变
  - 噪声添加
  - 随机模糊
  - 测试时增强 (TTA)

## 工作流程

### 1. 环境准备
```bash
# 安装依赖
pip install -r requirements.txt

# 验证环境
python scripts/smoke_test.py
```

### 2. 配置调整
编辑 `config.yml`:
```yaml
data:
  modalities: ['DWI']          # 选择模态
  target_shape: [256, 256, 32] # 目标尺寸
  use_torchio: true            # 使用 TorchIO 增强

training:
  batch_size: 4
  num_epochs: 100
  learning_rate: 0.0001
  patience: 10
  num_folds: 5
  amp: true                    # 混合精度训练
```

### 3. 训练模型
```bash
# 标准训练
python run_training.py

# 快速测试
python run_training.py --quick-test

# 指定配置文件
python run_training.py --config my_config.yml
```

### 4. 评估模型
```bash
# 评估所有折
python run_evaluation.py

# 评估并汇总
python run_evaluation.py --aggregate

# 只汇总结果
python run_evaluation.py --aggregate-only
```

### 5. 独立预测
```bash
# 标准预测
python -m src.predict --model_path outputs/models/fold_0_best.pth

# 使用 TTA
python -m src.predict --model_path outputs/models/fold_0_best.pth --use_tta

# 多模态预测
python -m src.predict --model_path outputs/models/fold_0_best.pth --modality ADC DWI T2
```

## 配置文件详解

### config.yml
```yaml
# 数据配置
data:
  data_root: './data/BPH-PCA'              # 数据根目录
  label_root: './data/BPH-PCA/ROI(BPH+PCA)' # 标签目录
  output_dir: './outputs'                   # 输出目录
  modalities: ['DWI']                       # 输入模态
  target_shape: [256, 256, 32]              # 目标尺寸 [Z, Y, X]
  use_torchio: true                         # 使用 TorchIO 增强

# 训练配置
training:
  batch_size: 4                             # 批次大小
  num_epochs: 100                           # 最大 epoch
  learning_rate: 0.0001                     # 学习率
  patience: 10                              # 早停耐心
  num_folds: 5                              # K 折数
  num_workers: 2                            # 数据加载线程
  amp: true                                 # 混合精度训练
```

## 输出文件说明

### 模型文件 (`outputs/models/`)
- `fold_0_best.pth`: Fold 0 最佳模型
- 包含模型权重、优化器状态、验证指标

### 日志文件 (`outputs/logs/`)
- `fold_0_log.csv`: 训练日志（每个 epoch）
- `fold_0_eval.csv`: 评估结果（每个样本）

### 曲线图 (`outputs/plots/`)
- `fold_0_curves.png`: 训练/验证曲线

### 预测结果 (`outputs/preds/`)
- `{case_id}_pred.nii`: 预测的 NIfTI 文件
- `{case_id}_overlay.png`: 预测/标签叠加图

### TensorBoard (`outputs/runs/`)
```bash
# 查看训练过程
tensorboard --logdir outputs/runs
```

## 常见问题

### 1. CUDA 内存不足
- 减小 `batch_size`
- 减小 `target_shape`
- 关闭 `amp`

### 2. 数据加载慢
- 增加 `num_workers`（Windows 建议 ≤2）
- 使用 SSD 存储数据

### 3. 训练不收敛
- 检查数据归一化
- 调整学习率
- 增加数据增强

### 4. OpenMP 冲突
```bash
# Windows PowerShell
$env:KMP_DUPLICATE_LIB_OK="TRUE"
```

## 扩展功能

### 添加新模态
1. 在 `data/BPH-PCA/` 下添加新模态文件夹
2. 修改 `config.yml` 中的 `modalities`
3. 重新训练

### 自定义数据增强
编辑 `src/transforms3d.py` 或 `src/torchio_transforms.py`

### 修改模型架构
编辑 `src/unet3d.py`

### 添加新的评估指标
编辑 `src/metrics.py`

## 性能优化建议

1. **使用混合精度训练** (`amp: true`)
2. **启用 TorchIO 增强** (`use_torchio: true`)
3. **调整 batch_size** 以充分利用 GPU
4. **使用 pin_memory** (已自动启用)
5. **合理设置 num_workers**

## 引用

如果本项目对您的研究有帮助，请引用：
```bibtex
@software{prostate_cancer_multimodal_segmentation,
  author = {TraeAI Assistant},
  title = {3D U-Net 前列腺癌多模态分割},
  year = {2025},
  version = {1.1}
}
```

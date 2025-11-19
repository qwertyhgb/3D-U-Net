# 快速开始指南

本指南将帮助您在 10 分钟内开始使用本项目。

## 📋 前置要求

- Python 3.9 或更高版本
- CUDA 11.6+ (可选，用于 GPU 加速)
- 8GB+ RAM (16GB+ 推荐)
- 10GB+ 可用磁盘空间

## 🚀 5 步快速开始

### 步骤 1: 克隆项目
```bash
git clone <repository-url>
cd 3D-U-Net
```

### 步骤 2: 安装依赖
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

### 步骤 3: 准备数据
将您的数据按以下结构组织：
```
data/BPH-PCA/
├── BPH/
│   └── DWI/
│       ├── 00001234.nii
│       └── ...
├── PCA/
│   └── DWI/
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

### 步骤 4: 验证环境
```bash
python scripts/smoke_test.py
```

如果看到 "✓ 所有测试通过！"，说明环境配置正确。

### 步骤 5: 开始训练
```bash
# 快速测试（1 折，5 epoch）
python run_training.py --quick-test

# 完整训练（5 折，100 epoch）
python run_training.py
```

## 🎯 常用命令

### 训练相关
```bash
# 标准训练
python run_training.py

# 快速测试
python run_training.py --quick-test

# 使用自定义配置
python run_training.py --config my_config.yml

# 只使用 CPU
python run_training.py --device cpu
```

### 评估相关
```bash
# 评估所有折
python run_evaluation.py

# 评估并汇总结果
python run_evaluation.py --aggregate

# 只汇总已有结果
python run_evaluation.py --aggregate-only
```

### 预测相关
```bash
# 标准预测
python -m src.predict --model_path outputs/models/fold_0_best.pth

# 使用测试时增强
python -m src.predict --model_path outputs/models/fold_0_best.pth --use_tta

# 指定输出目录
python -m src.predict \
    --model_path outputs/models/fold_0_best.pth \
    --output_dir ./my_predictions
```

### 可视化相关
```bash
# 启动 TensorBoard
tensorboard --logdir outputs/runs

# 在浏览器中打开 http://localhost:6006
```

## ⚙️ 配置调整

编辑 `config.yml` 文件：

```yaml
# 基础配置
data:
  modalities: ['DWI']          # 单模态
  # modalities: ['ADC', 'DWI', 'T2 fs']  # 多模态
  target_shape: [256, 256, 32] # 目标尺寸
  use_torchio: true            # 使用高级增强

training:
  batch_size: 4                # 根据显存调整
  num_epochs: 100              # 训练轮数
  learning_rate: 0.0001        # 学习率
  patience: 10                 # 早停耐心
  num_folds: 5                 # K 折数
  num_workers: 2               # 数据加载线程
  amp: true                    # 混合精度训练
```

## 📊 查看结果

训练完成后，结果保存在 `outputs/` 目录：

```
outputs/
├── models/
│   ├── fold_0_best.pth      # 模型权重
│   └── ...
├── logs/
│   ├── fold_0_log.csv       # 训练日志
│   ├── fold_0_eval.csv      # 评估结果
│   └── ...
├── plots/
│   ├── fold_0_curves.png    # 训练曲线
│   └── ...
└── preds/
    ├── BPH-00001234_pred.nii    # 预测结果
    ├── BPH-00001234_overlay.png # 可视化
    └── ...
```

## 🔍 性能调优

### GPU 内存不足？
```yaml
# 减小批次大小
batch_size: 2

# 减小目标尺寸
target_shape: [128, 128, 24]

# 关闭混合精度训练
amp: false
```

### 训练太慢？
```yaml
# 启用混合精度训练
amp: true

# 增加批次大小（如果显存允许）
batch_size: 8

# 增加数据加载线程（非 Windows）
num_workers: 4
```

### 过拟合？
```yaml
# 启用 TorchIO 增强
use_torchio: true

# 增加早停耐心
patience: 15

# 减小学习率
learning_rate: 0.00005
```

## 🐛 常见问题

### 1. 找不到数据
**错误**: `未找到任何病例`

**解决**:
- 检查 `config.yml` 中的 `data_root` 路径
- 确保数据文件是 `.nii` 格式
- 确保文件名匹配（影像和标签）

### 2. CUDA 错误
**错误**: `CUDA out of memory`

**解决**:
```yaml
# 减小批次大小
batch_size: 2

# 或使用 CPU
python run_training.py --device cpu
```

### 3. OpenMP 冲突 (Windows)
**错误**: `OMP: Error #15`

**解决**:
```powershell
# PowerShell
$env:KMP_DUPLICATE_LIB_OK="TRUE"
python run_training.py
```

### 4. 数据加载慢
**解决**:
```yaml
# Windows 用户
num_workers: 0  # 或 1-2

# Linux/Mac 用户
num_workers: 4  # 或更多
```

## 📚 下一步

- 阅读 [训练指南](doc/TRAINING_GUIDE.md) 了解详细配置
- 查看 [项目结构](PROJECT_STRUCTURE.md) 了解代码组织
- 阅读 [更新日志](CHANGELOG.md) 了解最新改进

## 💡 提示

1. **首次运行建议使用 `--quick-test`** 快速验证流程
2. **使用 TensorBoard** 实时监控训练过程
3. **定期备份** `outputs/models/` 目录
4. **尝试不同的超参数** 找到最佳配置
5. **使用 TTA** 可以提升预测精度 2-3%

## 🆘 获取帮助

- 查看 [常见问题](doc/TRAINING_GUIDE.md#常见问题)
- 提交 [Issue](https://github.com/your-repo/issues)
- 查看 [文档](doc/)

---

祝您使用愉快！🎉

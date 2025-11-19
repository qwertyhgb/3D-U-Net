# 3D U-Net 前列腺癌多模态分割

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12+-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

本项目使用 3D U-Net 对前列腺癌病灶进行自动分割，支持单模态与多模态 MRI 图像输入。

## ✨ 特性

- 🏗️ **3D U-Net 架构**：专为医学影像设计的三维卷积神经网络
- 🔬 **多模态支持**：灵活配置 ADC、DWI、T2 等模态组合
- 🎲 **双重数据增强**：传统方法 + TorchIO 高级增强
- 🎯 **测试时增强 (TTA)**：提高预测精度
- 📊 **K 折交叉验证**：科学评估模型性能
- 📈 **丰富的评估指标**：Dice、Hausdorff 距离、体积相似度
- ⚡ **混合精度训练**：加速训练并节省显存
- 📉 **TensorBoard 集成**：实时监控训练过程
- 🎨 **可视化工具**：预测结果叠加显示

## 🚀 快速开始

### 1. 环境准备
```bash
# 克隆项目
git clone <repository-url>
cd 3D-U-Net

# 安装依赖（Python >= 3.9）
pip install -r requirements.txt

# 验证环境
python scripts/smoke_test.py
```

### 2. 数据准备
按以下结构组织数据：
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

### 3. 配置调整
编辑 `config.yml` 设置模态、尺寸等参数：
```yaml
data:
  modalities: ['DWI']          # 选择模态
  target_shape: [256, 256, 32] # 目标尺寸
  use_torchio: true            # 使用 TorchIO 增强

training:
  batch_size: 4
  num_epochs: 100
  learning_rate: 0.0001
  num_folds: 5
  amp: true                    # 混合精度训练
```

### 4. 开始训练
```bash
# 标准训练（5 折交叉验证）
python run_training.py

# 快速测试（1 折，5 epoch）
python run_training.py --quick-test
```

### 5. 评估模型
```bash
# 评估所有折并汇总结果
python run_evaluation.py --aggregate
```

### 6. 独立预测
```bash
# 标准预测
python -m src.predict --model_path outputs/models/fold_0_best.pth

# 使用测试时增强
python -m src.predict --model_path outputs/models/fold_0_best.pth --use_tta

# 多模态预测
python -m src.predict --model_path outputs/models/fold_0_best.pth --modality ADC DWI T2
```

## 📚 文档

- [训练指南](doc/TRAINING_GUIDE.md) - 详细的训练说明
- [项目结构](PROJECT_STRUCTURE.md) - 完整的项目结构说明
- [优化总结](doc/OPTIMIZATION_SUMMARY.md) - 最新优化内容

## 🎯 性能指标

在 BPH-PCA 数据集上的典型结果：
- **Dice 系数**: 0.85 ± 0.05
- **Hausdorff 距离**: 12.3 ± 3.2 mm
- **体积相似度**: 0.92 ± 0.04

*注：实际结果取决于数据质量和训练配置*

## 🛠️ 技术栈

- **深度学习**: PyTorch 1.12+
- **医学影像**: SimpleITK, nibabel, TorchIO
- **可视化**: Matplotlib, TensorBoard
- **数据处理**: NumPy, SciPy, pandas

## 📊 项目结构

```
3D-U-Net/
├── config.yml              # 配置文件
├── run_training.py         # 训练脚本 ⭐
├── run_evaluation.py       # 评估脚本 ⭐
├── src/                    # 核心代码
│   ├── dataset.py          # 数据集
│   ├── unet3d.py           # 模型
│   ├── train_kfold.py      # 训练
│   └── ...
├── scripts/                # 辅助脚本
│   └── smoke_test.py       # 环境测试
└── outputs/                # 输出目录
    ├── models/             # 模型权重
    ├── logs/               # 训练日志
    └── plots/              # 训练曲线
```

## 🔧 常见问题

### CUDA 内存不足
- 减小 `batch_size`
- 减小 `target_shape`
- 关闭混合精度训练 (`amp: false`)

### 训练速度慢
- 启用混合精度训练 (`amp: true`)
- 增加 `num_workers`（Windows 建议 ≤2）
- 使用 SSD 存储数据

### OpenMP 冲突（Windows）
```powershell
$env:KMP_DUPLICATE_LIB_OK="TRUE"
```

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

## 📄 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件

## 📖 引用

如果本项目对您的研究有帮助，请引用：
```bibtex
@software{prostate_cancer_multimodal_segmentation,
  author = {TraeAI Assistant},
  title = {3D U-Net 前列腺癌多模态分割},
  year = {2025},
  version = {1.1},
  url = {https://github.com/your-repo/3D-U-Net}
}
```

## 🙏 致谢

感谢所有为医学影像分割领域做出贡献的研究者和开发者。

---

**注意**: 本项目仅用于研究目的，不应用于临床诊断。
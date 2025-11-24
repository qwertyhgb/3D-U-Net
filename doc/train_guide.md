# 3D U-Net 前列腺多模态 MRI 训练指南

本指南面向需要使用 `data/BPH-PCA` 数据集并基于本仓库训练 3D U-Net 的开发者，覆盖环境搭建、数据准备、配置说明、训练与验证流程、常见问题及进阶建议。

## 1. 环境准备

1. **Python 版本**：建议 Python 3.10+
2. **虚拟环境**（推荐）：
   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # Windows
   ```
3. **安装依赖**：
   ```bash
   pip install -r requirements.txt
   ```
4. **GPU 驱动**：若使用 CUDA，请确保驱动与 PyTorch 版本兼容（requirements 中默认为 `torch>=2.1`）。

## 2. 数据准备

项目默认数据目录为 `data/BPH-PCA`，结构与原始数据一致：

```
BPH-PCA/
├── BPH/
│   ├── ADC/
│   ├── DWI/
│   ├── T2 fs/
│   ├── T2 not fs/
│   └── gaoqing-T2/
├── PCA/
│   ├── ADC/
│   ├── DWI/
│   ├── T2 fs/
│   ├── T2 not fs/
│   └── gaoqing-T2/
└── ROI(BPH+PCA)/
    ├── BPH/
    └── PCA/
```

准备要点：

1. **命名一致**：确保各模态与 ROI 文件同名（例如 `0001234567.nii.gz`）。
2. **ROI 预处理**：`ROI(BPH+PCA)` 中掩膜默认为单通道，数据集会在读取时转换为二值 mask。
3. **训练/验证划分**：默认按 `data.val_ratio` 随机切分，可通过配置调整或修改 `src/data/dataset.py` 自定义划分列表。
4. **模态组合**：`data.modalities` 默认 `ADC/DWI/T2 fs/T2 not fs` 四模态，若新增 `gaoqing-T2`，需同步调整 `model.in_channels`。

## 3. 配置文件（`configs/default.yaml`）

| 区块 | 关键字段 | 说明 |
| --- | --- | --- |
| `experiment` | `name`, `output_dir`, `seed`, `device` | 控制实验命名、输出目录、随机种子与设备（cuda/cpu） |
| `data` | `root_dir`, `roi_subdir`, `groups`, `modalities`, `patch_size`, `batch_size`, `val_ratio`, `augment`, `num_workers` | 数据根路径、ROI 子路径、患者分组、模态列表、输入尺寸、批大小、验证占比、是否增强、DataLoader 线程数 |
| `model` | `in_channels`, `out_channels`, `init_features` | 3D U-Net 输入/输出通道及初始特征宽度 |
| `training` | `epochs`, `lr`, `weight_decay`, `amp`, `grad_clip`, `save_every` | 训练轮数、学习率、权重衰减、AMP、梯度裁剪、checkpoint 间隔 |

根据显存可调整 `patch_size`、`batch_size`，较大 patch 捕获更多上下文但消耗更多显存。

## 4. 启动训练

运行脚本：
```bash
python scripts/train.py --config configs/default.yaml
```

流程概览：

1. **加载配置**（`src/utils/config.py`）。
2. **设置随机种子**（`src/utils/common.py#set_seed`）。
3. **构建 DataLoader**（`src/data/dataset.py#create_data_loaders`），扫描病例并按 `val_ratio` 划分。
4. **实例化模型/优化器/调度器**（`scripts/train.py`）。
5. **训练循环**（`src/engine/trainer.py#train_model`）：
   - AMP + GradScaler（可在配置中关闭）。
   - 损失为 Dice+交叉熵组合（`src/engine/losses.py`）。
   - TensorBoard 日志写入 `outputs/<experiment_name>/logs`。
   - `ReduceLROnPlateau` 根据验证 loss 调整学习率。
   - 保存最佳权重至 `outputs/<experiment>/checkpoints/best_model.pt`，并按 `save_every` 备份。

## 5. 训练监控与评估

1. **TensorBoard**：
   ```bash
   tensorboard --logdir outputs
   ```
   重点关注 `Loss/train`, `Loss/val`, `Dice/val`。

2. **验证指标**：默认使用阈值 0.5 计算 Dice，若需软 Dice 或多阈值评估，可在 `evaluate_model` 中自定义。

3. **过拟合监控**：若 `Loss/train` 下降但 `Loss/val` 上升，可尝试：
   - 提高 `val_ratio`；
   - 增强数据（或禁用增强做对照）；
   - 调整学习率或启用 `weight_decay`。

## 6. 推理与可视化

推理脚本：
```bash
python scripts/predict.py \
    --config configs/default.yaml \
    --ckpt outputs/baseline-3d-unet/checkpoints/best_model.pt \
    --case "data/BPH-PCA/BPH/ADC/0001234567.nii.gz,data/BPH-PCA/BPH/DWI/0001234567.nii.gz,..." \
    --output outputs/inference \
    --threshold 0.5
```

注意事项：

1. `--case` 应按训练时的模态顺序提供同一病例的 NIfTI 文件，使用逗号分隔。
2. 推理时每个模态会做均值/方差归一化，并堆叠为 `(C, D, H, W)`。
3. 默认输出 `prediction.nii.gz` 不包含仿射矩阵，可通过 `nib.Nifti1Image(mask, affine)` 保存与原图一致的空间信息。

## 7. 常见问题排查

| 症状 | 排查建议 |
| --- | --- |
| `scan_cases` 报缺少文件 | 核对 `data.modalities` 与目录名称；确保 ROI 与各模态存在同名文件 |
| 显存不足 | 减小 `patch_size` 或 `batch_size`，或开启梯度累积（可自行扩展）；确认 AMP 已启用 |
| Dice 长期低于 0.5 | 检查 ROI 是否为二值、输入是否正确对齐；可在 `dataset.py` 打印样本统计或进行可视化 |
| 训练太慢 | 提高 `data.num_workers`（Windows 下需在 `if __name__ == "__main__":` 内创建 DataLoader）；使用 SSD 存储 |
| 推理输出全零 | 确认阈值、权重加载是否成功；可打印 `torch.sigmoid(logits).mean()` 检查模型输出分布 |

## 8. 进阶建议

1. **增强策略**：在 `MultiModalProstateDataset._augment` 中加入随机旋转、Gamma 校正、模态 Dropout 等 3D 增强。
2. **多类别分割**：若需要前列腺亚区/病灶区分，可将掩膜处理为多通道，并将 `model.out_channels` 调整至类别数，损失函数改用多类 Dice 或 CE。
3. **分布式训练**：可基于 `torch.distributed` 对训练循环封装，或迁移到 PyTorch Lightning / MONAI Engine，提升大规模训练效率。
4. **实验追踪**：结合 `Weights & Biases`、`MLflow` 等工具记录超参与指标，便于复现实验。

## 9. 关键源码索引

- 项目概览：`README.md`
- 模型：`src/models/unet3d.py`
- 数据管线：`src/data/dataset.py`
- 损失/指标：`src/engine/losses.py`, `src/metrics/dice.py`
- 训练引擎：`src/engine/trainer.py`
- 推理脚本：`scripts/predict.py`

若在训练过程中遇到其他问题，可结合上述文件定位源头或进一步自定义。祝训练顺利！

# 前列腺多模态 MRI 分割（3D U-Net 基线）

本项目基于 3D U-Net 提供一个从数据准备、训练到推理的完整 baseline，实现对 BPH/PCA 数据集中多模态 MRI 的前列腺分割。

## 目录结构

```
3D-U-Net/
├── configs/            # YAML 配置文件
├── data/               # 数据集（BPH-PCA 原始结构）
├── scripts/            # 训练 / 推理脚本
├── src/                # Python 包（数据、模型、训练引擎等）
├── requirements.txt    # 依赖
└── README.md
```

`src/` 进一步划分：
- `data/`：数据集定义、预处理与增强
- `models/`：3D U-Net 及其组件
- `engine/`：训练与验证循环
- `metrics/`：指标（Dice）
- `utils/`：日志、随机种子、配置加载等

## 数据准备

默认假设数据目录为 `data/BPH-PCA`，内部结构与提供的一致：
```
BPH-PCA/
├── BPH/
│   ├── ADC/
│   ├── DWI/
│   ├── T2 fs/
│   ├── T2 not fs/
│   └── gaoqing-T2/
├── PCA/
│   ├── ...
└── ROI(BPH+PCA)/
    ├── BPH/
    └── PCA/
```

1. 确保同一病例在各模态与 ROI 中的文件名一致（如 `0001234567.nii`）。
2. 如果需要，可将 ROI 转换为二值 mask（项目默认在读取时进行处理）。
3. 若想自定义训练/验证划分，可在 `configs/*.yaml` 中修改 `data.val_ratio` 或提供自定义列表。

## 快速开始

1. 创建虚拟环境并安装依赖：
   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # Windows
   pip install -r requirements.txt
   ```

2. 启动训练（默认配置 `configs/default.yaml`）：
   ```bash
   python scripts/train.py --config configs/default.yaml
   ```

   - 输出目录位于 `outputs/<experiment_name>`
   - 每轮训练会在 TensorBoard 日志中记录 loss / Dice

3. 推理示例：
   ```bash
   python scripts/predict.py \
       --config configs/default.yaml \
       --ckpt outputs/baseline-3d-unet/best_model.pt \
       --case-path data/BPH-PCA/BPH \
       --output outputs/inference
   ```

## 配置

`configs/default.yaml` 通过以下部分组织：
- `experiment`: 结果输出、日志等
- `data`: 根目录、模态列表、裁剪尺寸、线程数等
- `model`: 通道数、基准通道数
- `training`: epoch、学习率、AMP、梯度裁剪等

如需多实验管理，可复制默认文件并根据需要覆盖字段。

## 结果与监控

- 指标：训练过程同时计算 Dice+交叉熵组合损失与 Dice 系数。
- 可使用 `tensorboard --logdir outputs` 实时查看。

## 下一步

- 根据病例数量调整 `batch_size` 与 `patch_size`，避免显存不足。
- 如需更丰富的数据增强，可在 `src/data/transforms.py` 中扩展。
- 可接入 MONAI、nnU-Net 等更强框架进行对比。

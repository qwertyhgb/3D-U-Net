# 前列腺癌 3D U-Net 分割训练指南

> 本指南介绍如何在本项目中进行数据准备、训练、评估与可视化，并提供参数配置与常见问题说明。适用于 Windows 环境，其他系统亦可参考命令调整。

## 环境准备
- Python 版本：建议 3.9+
- 依赖库：请使用 `requirements.txt` 文件安装所有必需的库，以确保环境一致性。
- 安装命令：
  ```bash
  pip install -r requirements.txt
  ```
- PyTorch 安装说明：`requirements.txt` 中包含了适用于 CUDA 11.6 的 PyTorch 版本。如果您的 CUDA 版本不同，请访问 [PyTorch 官网](https://pytorch.org/get-started/locally/) 获取对应的安装命令，并手动安装后再运行上述 `pip install -r` 命令。
- PowerShell/OpenMP 冲突提示：如运行时出现 libiomp 重复加载错误，临时设置环境变量后再执行：
  ```powershell
  $env:KMP_DUPLICATE_LIB_OK="TRUE"
  ```

## 数据准备
- 期望目录结构（已包含于 `data/BPH-PCA/`）
  ```
  data/
    BPH-PCA/
      BPH/
        ADC/ DWI/ T2 fs/ T2 not fs/ gaoqing-T2/
      PCA/
        ADC/ DWI/ T2 fs/ T2 not fs/ gaoqing-T2/
      ROI(BPH+PCA)/
        BPH/
        PCA/
  ```
- 配对规则：按病例 ID 将影像（如 `DWI/0000xxxx.nii`）与标签（`ROI(BPH+PCA)/BPH|PCA/0000xxxx.nii`）一一匹配。

## 快速验证
- 运行冒烟测试，确认读取、归一化与尺寸对齐正常：
  ```bash
  python scripts/smoke_test.py
  ```
  输出应包含找到的病例数与张量形状 `1×256×256×32`。

## 训练（5 折交叉验证）
- 启动训练：
  ```bash
  python -m src.train_kfold
  ```
- 训练配置在 `config.yml`：
  - `modalities`: 默认 `['DWI']`（设置多模态如 `['ADC','DWI','T2 fs']`）
  - `target_shape`: `256×256×32`
  - `batch_size`: 4
  - `num_epochs`: 100
  - `learning_rate`: 1e-4
  - `patience`: 10（早停）
  - `num_folds`: 5
  - `use_torchio`: true（使用torchio进行数据增强）
- 训练过程：
  - 损失函数：DiceLoss
  - 优化器：Adam
  - 学习率：ReduceLROnPlateau 按验证损失自适应衰减
  - 数据增强：
    - 传统方法：随机翻转、90°旋转、弹性形变（逐切片）
    - TorchIO增强（推荐）：弹性形变、仿射变换、噪声添加、随机模糊等

### 多模态训练
- 启用方式：在 `src/config.py` 中设置 `INPUT_MODALITIES` 为多个模态，例如：
  ```python
  INPUT_MODALITIES = ['ADC', 'DWI', 'T2 fs']
  ```
- 模型会自动将输入通道设为模态数，并在数据加载阶段将各模态按通道维堆叠为 `[C,Z,Y,X]`。
- 增强操作在所有模态上保持一致的几何变换（翻转、旋转、弹性形变）。

## 评估与可视化
- 对各折的验证集评估并保存预测示例：
  ```bash
  python -m src.evaluate
  ```
- 汇总所有折的整体统计（均值/标准差）：
  ```bash
  python -m src.aggregate_results
  ```
- 可视化说明：将生成若干切片的叠加图（红色为预测，绿色为标签）。

## 独立预测与测试时增强 (TTA)
项目现在包含一个独立的预测脚本 `src/predict.py`，支持测试时增强 (TTA) 功能。

- 基本预测：
  ```bash
  python -m src.predict --model_path outputs/models/fold_0_best.pth
  ```

- 使用测试时增强 (TTA) 进行预测：
  ```bash
  python -m src.predict --model_path outputs/models/fold_0_best.pth --use_tta
  ```

- 多模态预测：
  ```bash
  python -m src.predict --model_path outputs/models/fold_0_best.pth --modality ADC DWI T2
  ```

- 指定输出目录和设备：
  ```bash
  python -m src.predict --model_path outputs/models/fold_0_best.pth --output_dir ./predictions --device cpu
  ```

TTA功能通过在预测时应用多种数据增强变换并对结果进行平均，可以提高模型的泛化能力和预测精度。

## 输出目录
- `outputs/models/fold_{i}_best.pth`：各折最佳权重
- `outputs/logs/fold_{i}_log.csv`：训练/验证日志（逐 epoch）
- `outputs/plots/fold_{i}_curves.png`：损失与 Dice 曲线
- `outputs/logs/fold_{i}_eval.csv`：验证集量化评估
- `outputs/preds/{case_id}_pred.nii`：预测体 NIfTI
- `outputs/preds/{case_id}_overlay.png`：预测/标签叠加图

## 配置与自定义
- 修改输入模态：编辑 `src/config.py` 中 `INPUT_MODALITY`
- 调整尺寸：编辑 `TARGET_SHAPE`（保持与模型下采样步长兼容）
- 更改训练参数：`BATCH_SIZE`、`NUM_EPOCHS`、`LEARNING_RATE`、`PATIENCE`、`NUM_FOLDS`
- 切换归一化：在数据集初始化时传递 `normalize_hu=True`（用于 CT HU 归一化）
- 增强策略：在 `src/transforms3d.py` 调整 `alpha/sigma` 或增删增强函数

## 常见问题
- OpenMP 冲突（libiomp 重复加载）：
  - 解决：在 PowerShell 设置 `KMP_DUPLICATE_LIB_OK` 环境变量后再运行训练或评估。
- CUDA 未检测到：
  - 模型将自动使用 CPU（`torch.cuda.is_available()`）；如需 GPU，请正确安装匹配版本的 CUDA 与 PyTorch。
- 读取 DICOM 失败：
  - 确认 `SimpleITK` 已安装；DICOM 目录需包含 `.dcm` 文件。

## 文件索引与关键实现
- 数据集与预处理：`src/dataset.py:12` 病例收集、`src/dataset.py:63` 5 折划分
- 工具：`src/utils.py:25` 读取 NIfTI/DICOM、`src/utils.py:71` 尺寸统一
- 增强：
  - 传统方法：`src/transforms3d.py:32` 综合增强
  - TorchIO增强：`src/torchio_transforms.py:15` 高级增强与TTA
- 模型：`src/unet3d.py:16` 3D U-Net 实现
- 训练：`src/train_kfold.py:22` 单折训练、曲线与权重保存
- 指标：`src/metrics.py:6` Dice、`src/metrics.py:14` Hausdorff、`src/metrics.py:24` 体积相似度
- 评估与可视化：`src/evaluate.py:15` 验证评估与示例保存
- 结果汇总：`src/aggregate_results.py:1` 汇总脚本
- 独立预测：`src/predict.py:15` 支持TTA的预测脚本

## 复现实验建议
- 随机种子：可在训练脚本中添加固定种子设置以增强可复现性
- 记录工具：可集成 `tensorboard` 或 `wandb` 进行更丰富的训练监控
- 数据划分：当前为顺序 K 折切分，如需严格分层/打乱，可改为 `sklearn.model_selection.KFold`

---
如需增加“测试集”独立评估或导出更详细的可视化报告，请告知具体需求，我将补充相应脚本与说明。
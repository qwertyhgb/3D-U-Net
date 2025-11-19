# Logging系统使用指南

## 概述

项目已经从使用`print`语句升级到使用Python的`logging`模块，提供更专业和灵活的日志记录功能。

## 主要特性

### ✅ 已实现的功能

1. **彩色控制台输出** - 不同级别的日志使用不同颜色
2. **文件日志** - 自动保存到文件，便于后续分析
3. **分级日志** - DEBUG, INFO, WARNING, ERROR, CRITICAL
4. **专用日志记录器** - 训练和评估有专门的日志类
5. **自动格式化** - 包含时间戳、文件名、行号等信息
6. **异常追踪** - 自动记录完整的异常堆栈

## 日志级别

```python
DEBUG    # 详细的调试信息
INFO     # 一般信息
WARNING  # 警告信息
ERROR    # 错误信息
CRITICAL # 严重错误
```

## 基本使用

### 1. 简单使用

```python
from src.logger import get_logger

logger = get_logger()

logger.debug("这是调试信息")
logger.info("这是普通信息")
logger.warning("这是警告信息")
logger.error("这是错误信息")
logger.critical("这是严重错误")
```

### 2. 带文件输出

```python
from src.logger import setup_logger

logger = setup_logger(
    name='my_module',
    log_file='outputs/logs/my_module.log',
    level=logging.INFO
)

logger.info("这条信息会同时输出到控制台和文件")
```

### 3. 训练日志

```python
from src.logger import TrainingLogger

# 初始化训练日志记录器
logger = TrainingLogger(fold_idx=0, output_dir='./outputs')

# 记录配置
logger.log_config({
    'batch_size': 4,
    'learning_rate': 0.0001,
    'num_epochs': 100
})

# 记录epoch开始
logger.log_epoch_start(epoch=1, total_epochs=100)

# 记录epoch结束和指标
logger.log_epoch_end(epoch=1, metrics={
    'train_loss': 0.5,
    'val_loss': 0.4,
    'val_dice': 0.85
})

# 记录最佳模型保存
logger.log_best_model(epoch=1, metric_name='val_loss', metric_value=0.4)

# 记录早停
logger.log_early_stopping(epoch=50, patience=10)

# 记录训练完成
logger.log_training_complete(best_epoch=45, best_metric=0.35)
```

### 4. 评估日志

```python
from src.logger import EvaluationLogger

# 初始化评估日志记录器
logger = EvaluationLogger(fold_idx=0, output_dir='./outputs')

# 记录评估开始
logger.log_start(num_samples=50)

# 记录单个样本
logger.log_sample('case_001', {
    'dice': 0.85,
    'dsc': 0.85,
    'hausdorff': 12.3,
    'nsd': 0.92
})

# 记录评估总结
logger.log_summary({
    'dice': (0.85, 0.05),
    'dsc': (0.85, 0.05),
    'hausdorff': (12.3, 3.2),
    'nsd': (0.92, 0.04)
})

# 记录评估完成
logger.log_complete()
```

### 5. 系统信息日志

```python
from src.logger import log_system_info, log_data_info

# 记录系统信息
log_system_info()

# 记录数据信息
log_data_info(
    num_cases=100,
    num_folds=5,
    modalities=['DWI', 'ADC']
)
```

## 日志输出示例

### 控制台输出（带颜色）

```
2025-11-19 10:30:15 - INFO - 开始训练...
2025-11-19 10:30:16 - INFO - 设备: cuda
2025-11-19 10:30:16 - INFO - GPU: NVIDIA GeForce RTX 3090
2025-11-19 10:30:20 - INFO - Epoch 1/100 开始
2025-11-19 10:35:42 - INFO - Epoch 1 完成 - train_loss: 0.5234 | val_loss: 0.4123 | val_dice: 0.8567
2025-11-19 10:35:42 - INFO - ✓ 保存最佳模型 (Epoch 1, val_loss: 0.4123)
2025-11-19 10:35:43 - WARNING - 学习率降低: 0.0001 -> 0.00005
2025-11-19 11:20:15 - WARNING - 早停触发！在Epoch 50停止训练（耐心值: 10）
```

### 文件输出（详细信息）

```
2025-11-19 10:30:15 - training_fold_0 - INFO - train_kfold.py:45 - 开始训练...
2025-11-19 10:30:16 - training_fold_0 - INFO - train_kfold.py:52 - 设备: cuda
2025-11-19 10:30:16 - training_fold_0 - INFO - train_kfold.py:54 - GPU: NVIDIA GeForce RTX 3090
2025-11-19 10:30:20 - training_fold_0 - INFO - train_kfold.py:120 - Epoch 1/100 开始
2025-11-19 10:35:42 - training_fold_0 - INFO - train_kfold.py:185 - Epoch 1 完成 - train_loss: 0.5234 | val_loss: 0.4123 | val_dice: 0.8567
```

## 日志文件位置

```
outputs/
└── logs/
    ├── training_main.log              # 主训练日志
    ├── fold_0_training.log            # Fold 0训练详细日志
    ├── fold_1_training.log            # Fold 1训练详细日志
    ├── evaluation_main.log            # 主评估日志
    ├── fold_0_evaluation.log          # Fold 0评估详细日志
    └── fold_1_evaluation.log          # Fold 1评估详细日志
```

## 高级用法

### 1. 异常记录

```python
try:
    # 一些可能出错的代码
    result = risky_operation()
except Exception as e:
    logger.error(f"操作失败: {e}", exc_info=True)
    # exc_info=True 会自动记录完整的异常堆栈
```

### 2. 条件日志

```python
if logger.isEnabledFor(logging.DEBUG):
    # 只在DEBUG级别时执行昂贵的日志操作
    expensive_debug_info = compute_expensive_info()
    logger.debug(f"详细信息: {expensive_debug_info}")
```

### 3. 自定义格式

```python
import logging

# 创建自定义格式化器
formatter = logging.Formatter(
    '%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)

# 应用到handler
handler = logging.StreamHandler()
handler.setFormatter(formatter)

logger = logging.getLogger('custom')
logger.addHandler(handler)
```

### 4. 多个日志文件

```python
# 错误单独记录到文件
error_handler = logging.FileHandler('outputs/logs/errors.log')
error_handler.setLevel(logging.ERROR)
logger.addHandler(error_handler)

# 所有日志记录到另一个文件
all_handler = logging.FileHandler('outputs/logs/all.log')
all_handler.setLevel(logging.DEBUG)
logger.addHandler(all_handler)
```

## 最佳实践

### ✅ 推荐做法

1. **使用合适的日志级别**
   ```python
   logger.debug("变量值: x=5, y=10")  # 调试信息
   logger.info("训练开始")            # 一般信息
   logger.warning("学习率过大")       # 警告
   logger.error("模型加载失败")       # 错误
   ```

2. **记录关键信息**
   ```python
   logger.info(f"Epoch {epoch}/{total_epochs}")
   logger.info(f"Loss: {loss:.4f}")
   logger.info(f"保存模型到: {model_path}")
   ```

3. **使用异常追踪**
   ```python
   try:
       train_model()
   except Exception as e:
       logger.error("训练失败", exc_info=True)
   ```

4. **避免敏感信息**
   ```python
   # ❌ 不要记录密码、密钥等
   logger.info(f"API Key: {api_key}")
   
   # ✅ 只记录必要信息
   logger.info("API认证成功")
   ```

### ❌ 避免的做法

1. **不要在循环中过度记录**
   ```python
   # ❌ 避免
   for i in range(10000):
       logger.info(f"处理第 {i} 个样本")
   
   # ✅ 推荐
   for i in range(10000):
       if i % 1000 == 0:
           logger.info(f"已处理 {i}/10000 个样本")
   ```

2. **不要记录大量数据**
   ```python
   # ❌ 避免
   logger.info(f"数据: {huge_array}")
   
   # ✅ 推荐
   logger.info(f"数据形状: {huge_array.shape}")
   ```

## 从print迁移

### 迁移对照表

| print语句 | logging等价 |
|-----------|------------|
| `print("开始训练")` | `logger.info("开始训练")` |
| `print(f"错误: {e}")` | `logger.error(f"错误: {e}")` |
| `print("警告: 内存不足")` | `logger.warning("警告: 内存不足")` |
| `print(f"调试: x={x}")` | `logger.debug(f"调试: x={x}")` |

### 迁移示例

**之前（使用print）：**
```python
print("="*60)
print("开始训练")
print(f"Epoch: {epoch}")
print(f"Loss: {loss:.4f}")
print("="*60)
```

**之后（使用logging）：**
```python
logger.info("="*60)
logger.info("开始训练")
logger.info(f"Epoch: {epoch}")
logger.info(f"Loss: {loss:.4f}")
logger.info("="*60)
```

## 配置建议

### 开发环境

```python
logger = setup_logger(
    name='dev',
    log_file='dev.log',
    level=logging.DEBUG  # 显示所有日志
)
```

### 生产环境

```python
logger = setup_logger(
    name='prod',
    log_file='prod.log',
    level=logging.INFO,  # 只显示重要信息
    console=False        # 不输出到控制台
)
```

## 故障排除

### 问题1: 日志重复输出

**原因**: 多次调用`setup_logger`导致重复添加handler

**解决**:
```python
# 使用get_logger而不是setup_logger
logger = get_logger('my_module')
```

### 问题2: 日志文件没有创建

**原因**: 目录不存在

**解决**: logging系统会自动创建目录，但确保有写权限

### 问题3: 颜色不显示

**原因**: 不是在终端环境中运行

**解决**: 颜色只在终端中显示，文件和重定向输出不会有颜色

## 总结

使用logging系统的优势：

1. ✅ **专业** - 行业标准的日志记录方式
2. ✅ **灵活** - 可以轻松调整日志级别和输出位置
3. ✅ **可追溯** - 自动记录时间、文件、行号
4. ✅ **可分析** - 日志文件便于后续分析和调试
5. ✅ **可扩展** - 容易添加新的日志处理器

---

**最后更新**: 2025-11-19  
**版本**: v1.0

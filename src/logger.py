"""
文件名称：logger.py
文件功能：统一的日志记录系统，替代print语句。
创建日期：2025-11-19
版本：v1.0
"""

import logging
import sys
import os
from pathlib import Path
from typing import Optional


class ColoredFormatter(logging.Formatter):
    """带颜色的日志格式化器（用于控制台输出）"""
    
    # ANSI颜色代码
    COLORS = {
        'DEBUG': '\033[36m',      # 青色
        'INFO': '\033[32m',       # 绿色
        'WARNING': '\033[33m',    # 黄色
        'ERROR': '\033[31m',      # 红色
        'CRITICAL': '\033[35m',   # 紫色
    }
    RESET = '\033[0m'
    
    def format(self, record):
        # 添加颜色
        if record.levelname in self.COLORS:
            record.levelname = f"{self.COLORS[record.levelname]}{record.levelname}{self.RESET}"
        return super().format(record)


def setup_logger(
    name: str = 'prostate_segmentation',
    log_file: Optional[str] = None,
    level: int = logging.INFO,
    console: bool = True,
    file_level: Optional[int] = None
) -> logging.Logger:
    """设置日志记录器
    
    参数:
        name: 日志记录器名称
        log_file: 日志文件路径（可选）
        level: 控制台日志级别
        console: 是否输出到控制台
        file_level: 文件日志级别（如果为None，使用level）
    
    返回:
        配置好的日志记录器
    """
    # 创建日志记录器
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)  # 设置为最低级别，由handler控制实际输出
    
    # 避免重复添加handler
    if logger.handlers:
        return logger
    
    # 日志格式
    console_format = '%(asctime)s - %(levelname)s - %(message)s'
    file_format = '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
    
    date_format = '%Y-%m-%d %H:%M:%S'
    
    # 控制台处理器
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        
        # 使用带颜色的格式化器
        if sys.stdout.isatty():  # 如果是终端，使用颜色
            console_formatter = ColoredFormatter(console_format, datefmt=date_format)
        else:
            console_formatter = logging.Formatter(console_format, datefmt=date_format)
        
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
    
    # 文件处理器
    if log_file:
        # 确保日志目录存在
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(file_level if file_level is not None else level)
        file_formatter = logging.Formatter(file_format, datefmt=date_format)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str = 'prostate_segmentation') -> logging.Logger:
    """获取已配置的日志记录器
    
    参数:
        name: 日志记录器名称
    
    返回:
        日志记录器
    """
    logger = logging.getLogger(name)
    
    # 如果logger还没有配置，使用默认配置
    if not logger.handlers:
        return setup_logger(name)
    
    return logger


class TrainingLogger:
    """训练专用日志记录器，提供便捷的训练日志方法"""
    
    def __init__(self, fold_idx: int, output_dir: str = './outputs'):
        """初始化训练日志记录器
        
        参数:
            fold_idx: 折索引
            output_dir: 输出目录
        """
        self.fold_idx = fold_idx
        self.output_dir = output_dir
        
        # 创建日志文件路径
        log_dir = os.path.join(output_dir, 'logs')
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f'fold_{fold_idx}_training.log')
        
        # 设置日志记录器
        self.logger = setup_logger(
            name=f'training_fold_{fold_idx}',
            log_file=log_file,
            level=logging.INFO
        )
    
    def log_config(self, config: dict):
        """记录配置信息"""
        self.logger.info("="*60)
        self.logger.info(f"Fold {self.fold_idx} 训练配置")
        self.logger.info("="*60)
        for key, value in config.items():
            self.logger.info(f"{key}: {value}")
        self.logger.info("="*60)
    
    def log_epoch_start(self, epoch: int, total_epochs: int):
        """记录epoch开始"""
        self.logger.info(f"Epoch {epoch}/{total_epochs} 开始")
    
    def log_epoch_end(self, epoch: int, metrics: dict):
        """记录epoch结束和指标"""
        metrics_str = " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        self.logger.info(f"Epoch {epoch} 完成 - {metrics_str}")
    
    def log_best_model(self, epoch: int, metric_name: str, metric_value: float):
        """记录最佳模型保存"""
        self.logger.info(f"✓ 保存最佳模型 (Epoch {epoch}, {metric_name}: {metric_value:.4f})")
    
    def log_early_stopping(self, epoch: int, patience: int):
        """记录早停"""
        self.logger.warning(f"早停触发！在Epoch {epoch}停止训练（耐心值: {patience}）")
    
    def log_training_complete(self, best_epoch: int, best_metric: float):
        """记录训练完成"""
        self.logger.info("="*60)
        self.logger.info(f"Fold {self.fold_idx} 训练完成")
        self.logger.info(f"最佳Epoch: {best_epoch}")
        self.logger.info(f"最佳指标: {best_metric:.4f}")
        self.logger.info("="*60)
    
    def log_error(self, error: Exception):
        """记录错误"""
        self.logger.error(f"训练出错: {str(error)}", exc_info=True)


class EvaluationLogger:
    """评估专用日志记录器"""
    
    def __init__(self, fold_idx: int, output_dir: str = './outputs'):
        """初始化评估日志记录器
        
        参数:
            fold_idx: 折索引
            output_dir: 输出目录
        """
        self.fold_idx = fold_idx
        
        # 创建日志文件路径
        log_dir = os.path.join(output_dir, 'logs')
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f'fold_{fold_idx}_evaluation.log')
        
        # 设置日志记录器
        self.logger = setup_logger(
            name=f'evaluation_fold_{fold_idx}',
            log_file=log_file,
            level=logging.INFO
        )
    
    def log_start(self, num_samples: int):
        """记录评估开始"""
        self.logger.info("="*60)
        self.logger.info(f"Fold {self.fold_idx} 评估开始")
        self.logger.info(f"样本数量: {num_samples}")
        self.logger.info("="*60)
    
    def log_sample(self, case_id: str, metrics: dict):
        """记录单个样本的评估结果"""
        metrics_str = " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        self.logger.info(f"{case_id} - {metrics_str}")
    
    def log_summary(self, metrics: dict):
        """记录评估总结"""
        self.logger.info("="*60)
        self.logger.info("评估总结")
        self.logger.info("="*60)
        for metric_name, (mean, std) in metrics.items():
            self.logger.info(f"{metric_name}: {mean:.4f} ± {std:.4f}")
        self.logger.info("="*60)
    
    def log_complete(self):
        """记录评估完成"""
        self.logger.info(f"Fold {self.fold_idx} 评估完成")


# 便捷函数
def log_system_info():
    """记录系统信息"""
    import torch
    import platform
    
    logger = get_logger()
    
    logger.info("="*60)
    logger.info("系统信息")
    logger.info("="*60)
    logger.info(f"Python版本: {platform.python_version()}")
    logger.info(f"PyTorch版本: {torch.__version__}")
    logger.info(f"CUDA可用: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        logger.info(f"CUDA版本: {torch.version.cuda}")
        logger.info(f"GPU数量: {torch.cuda.device_count()}")
        logger.info(f"GPU名称: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    logger.info(f"操作系统: {platform.system()} {platform.release()}")
    logger.info("="*60)


def log_data_info(num_cases: int, num_folds: int, modalities: list):
    """记录数据信息"""
    logger = get_logger()
    
    logger.info("="*60)
    logger.info("数据信息")
    logger.info("="*60)
    logger.info(f"病例数量: {num_cases}")
    logger.info(f"交叉验证折数: {num_folds}")
    logger.info(f"模态: {modalities}")
    logger.info("="*60)


# 示例使用
if __name__ == '__main__':
    # 基本使用
    logger = setup_logger('test', log_file='test.log')
    logger.debug("这是调试信息")
    logger.info("这是普通信息")
    logger.warning("这是警告信息")
    logger.error("这是错误信息")
    logger.critical("这是严重错误信息")
    
    # 训练日志使用
    train_logger = TrainingLogger(fold_idx=0)
    train_logger.log_config({'batch_size': 4, 'lr': 0.0001})
    train_logger.log_epoch_start(1, 100)
    train_logger.log_epoch_end(1, {'train_loss': 0.5, 'val_loss': 0.4, 'val_dice': 0.85})
    train_logger.log_best_model(1, 'val_loss', 0.4)
    
    # 评估日志使用
    eval_logger = EvaluationLogger(fold_idx=0)
    eval_logger.log_start(50)
    eval_logger.log_sample('case_001', {'dice': 0.85, 'hausdorff': 12.3})
    eval_logger.log_summary({'dice': (0.85, 0.05), 'hausdorff': (12.3, 3.2)})
    eval_logger.log_complete()

"""训练与推理引擎。"""

from .losses import DiceCrossEntropyLoss
from .trainer import train_model, evaluate_model, inference_step

__all__ = [
    "DiceCrossEntropyLoss",
    "train_model",
    "evaluate_model",
    "inference_step",
]

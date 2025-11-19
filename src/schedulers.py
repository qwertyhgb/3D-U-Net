"""
文件名称：schedulers.py
文件功能：实现多种学习率调度策略，用于优化训练过程。
创建日期：2025-11-19
版本：v1.0
"""

import torch
from torch.optim.lr_scheduler import _LRScheduler
import math


class CosineAnnealingWarmRestarts(_LRScheduler):
    """余弦退火预热重启调度器。
    
    在训练过程中周期性地重置学习率，有助于跳出局部最优解。
    """
    
    def __init__(self, optimizer, T_0, T_mult=1, eta_min=0, last_epoch=-1):
        """
        参数:
            optimizer: 优化器
            T_0: 第一次重启的周期长度
            T_mult: 每次重启后周期长度的乘数因子
            eta_min: 最小学习率
            last_epoch: 上一个epoch的索引
        """
        if T_0 <= 0:
            raise ValueError("Expected positive T_0, but got {}".format(T_0))
        if T_mult < 1:
            raise ValueError("Expected T_mult >= 1, but got {}".format(T_mult))
        self.T_0 = int(T_0)
        self.T_i = int(T_0)
        self.T_mult = int(T_mult)
        self.eta_min = eta_min
        self.T_cur = last_epoch
        super(CosineAnnealingWarmRestarts, self).__init__(optimizer, last_epoch)
    
    def get_lr(self):
        return [self.eta_min + (base_lr - self.eta_min) * 
                (1 + math.cos(math.pi * self.T_cur / self.T_i)) / 2
                for base_lr in self.base_lrs]
    
    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.T_cur = self.T_cur + 1
            if self.T_cur >= self.T_i:
                self.T_cur = self.T_cur - self.T_i
                self.T_i = self.T_i * self.T_mult
        else:
            if epoch < 0:
                raise ValueError("Expected non-negative epoch, but got {}".format(epoch))
            if epoch >= self.T_0:
                if self.T_mult == 1:
                    self.T_cur = epoch % self.T_0
                else:
                    n = int(math.log((epoch / self.T_0 * (self.T_mult - 1) + 1), self.T_mult))
                    self.T_cur = epoch - self.T_0 * (self.T_mult ** n - 1) / (self.T_mult - 1)
                    self.T_i = self.T_0 * self.T_mult ** (n)
            else:
                self.T_i = self.T_0
                self.T_cur = epoch
        self.last_epoch = epoch
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr


class PolynomialLR(_LRScheduler):
    """多项式衰减学习率调度器。
    
    学习率按照多项式函数衰减，适合需要平滑衰减的场景。
    """
    
    def __init__(self, optimizer, max_iter=100, power=0.9, last_epoch=-1):
        """
        参数:
            optimizer: 优化器
            max_iter: 最大迭代次数
            power: 多项式幂次
            last_epoch: 上一个epoch的索引
        """
        self.max_iter = max_iter
        self.power = power
        super(PolynomialLR, self).__init__(optimizer, last_epoch)
    
    def get_lr(self):
        return [base_lr * (1 - self.last_epoch / self.max_iter) ** self.power 
                for base_lr in self.base_lrs]


class ExponentialLR(_LRScheduler):
    """指数衰减学习率调度器。
    
    学习率按照指数函数衰减，适合需要快速降低学习率的场景。
    """
    
    def __init__(self, optimizer, gamma=0.95, last_epoch=-1):
        """
        参数:
            optimizer: 优化器
            gamma: 衰减因子
            last_epoch: 上一个epoch的索引
        """
        self.gamma = gamma
        super(ExponentialLR, self).__init__(optimizer, last_epoch)
    
    def get_lr(self):
        return [base_lr * self.gamma ** self.last_epoch 
                for base_lr in self.base_lrs]


def get_scheduler(optimizer, scheduler_config):
    """根据配置创建学习率调度器。
    
    参数:
        optimizer: 优化器
        scheduler_config: 调度器配置字典
        
    返回:
        学习率调度器
    """
    scheduler_type = scheduler_config.get('type', 'plateau').lower()
    
    if scheduler_type == 'plateau':
        from torch.optim.lr_scheduler import ReduceLROnPlateau
        return ReduceLROnPlateau(
            optimizer, 
            mode=scheduler_config.get('mode', 'min'),
            factor=scheduler_config.get('factor', 0.5),
            patience=scheduler_config.get('patience', 5)
        )
    elif scheduler_type == 'cosine':
        return CosineAnnealingWarmRestarts(
            optimizer,
            T_0=scheduler_config.get('T_0', 10),
            T_mult=scheduler_config.get('T_mult', 1),
            eta_min=scheduler_config.get('eta_min', 1e-6)
        )
    elif scheduler_type == 'polynomial':
        return PolynomialLR(
            optimizer,
            max_iter=scheduler_config.get('max_iter', 100),
            power=scheduler_config.get('power', 0.9)
        )
    elif scheduler_type == 'exponential':
        return ExponentialLR(
            optimizer,
            gamma=scheduler_config.get('gamma', 0.95)
        )
    elif scheduler_type == 'step':
        from torch.optim.lr_scheduler import StepLR
        return StepLR(
            optimizer,
            step_size=scheduler_config.get('step_size', 30),
            gamma=scheduler_config.get('gamma', 0.1)
        )
    elif scheduler_type == 'multistep':
        from torch.optim.lr_scheduler import MultiStepLR
        return MultiStepLR(
            optimizer,
            milestones=scheduler_config.get('milestones', [30, 60, 90]),
            gamma=scheduler_config.get('gamma', 0.1)
        )
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")
"""
文件名称：optimizers.py
文件功能：实现多种优化器策略，包括权重衰减、梯度裁剪等高级技术。
创建日期：2025-11-19
版本：v1.0
"""

import torch
from torch.optim import Adam, AdamW, SGD, RMSprop
from torch.optim.optimizer import Optimizer
import math


class AdamWWithWeightDecay(AdamW):
    """带权重衰减的AdamW优化器。
    
    实现了Decoupled Weight Decay Regularization，将权重衰减与梯度更新解耦，
    有助于提高模型的泛化能力。
    """
    
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=1e-2, amsgrad=False):
        super(AdamWWithWeightDecay, self).__init__(
            params, lr=lr, betas=betas, eps=eps,
            weight_decay=weight_decay, amsgrad=amsgrad
        )


class RAdam(Optimizer):
    """Rectified Adam优化器。
    
    修正了Adam在训练初期方差过大的问题，提供更稳定的训练过程。
    """
    
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, degenerated_to_sgd=True):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        
        self.degenerated_to_sgd = degenerated_to_sgd
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        self.buffer = [[None, None, None] for _ in range(10)]
        super(RAdam, self).__init__(params, defaults)
    
    def __setstate__(self, state):
        super(RAdam, self).__setstate__(state)
    
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data.float()
                if grad.is_sparse:
                    raise RuntimeError('RAdam does not support sparse gradients')
                
                p_data_fp32 = p.data.float()
                
                state = self.state[p]
                
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p_data_fp32)
                    state['exp_avg_sq'] = torch.zeros_like(p_data_fp32)
                else:
                    state['exp_avg'] = state['exp_avg'].type_as(p_data_fp32)
                    state['exp_avg_sq'] = state['exp_avg_sq'].type_as(p_data_fp32)
                
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']
                
                state['step'] += 1
                
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                
                buffered = self.buffer[int(state['step'] % 10)]
                
                if state['step'] == buffered[0]:
                    N_sma, step_size = buffered[1], buffered[2]
                else:
                    buffered[0] = state['step']
                    beta2_t = beta2 ** state['step']
                    N_sma_max = 2 / (1 - beta2) - 1
                    N_sma = N_sma_max - 2 * state['step'] * beta2_t / (1 - beta2_t)
                    buffered[1] = N_sma
                    
                    # more conservative since it's an approximated value
                    if N_sma >= 5:
                        step_size = math.sqrt((1 - beta2_t) * (N_sma - 4) / (N_sma_max - 4) * (N_sma - 2) / N_sma * N_sma_max / (N_sma_max - 2)) / (1 - beta1 ** state['step'])
                    elif self.degenerated_to_sgd:
                        step_size = 1.0 / (1 - beta1 ** state['step'])
                    else:
                        step_size = -1
                    
                    buffered[2] = step_size
                
                # more conservative since it's an approximated value
                if N_sma >= 5:
                    if group['weight_decay'] != 0:
                        p_data_fp32.add_(p_data_fp32, alpha=-group['weight_decay'] * group['lr'])
                    denom = exp_avg_sq.sqrt().add_(group['eps'])
                    p_data_fp32.addcdiv_(exp_avg, denom, value=-step_size * group['lr'])
                elif step_size > 0:
                    if group['weight_decay'] != 0:
                        p_data_fp32.add_(p_data_fp32, alpha=-group['weight_decay'] * group['lr'])
                    p_data_fp32.add_(exp_avg, alpha=-step_size * group['lr'])
                
                p.data.copy_(p_data_fp32)
        
        return loss


class Lookahead(Optimizer):
    """Lookahead优化器包装器。
    
    通过周期性地同步"慢权重"和"快权重"，提高训练稳定性和泛化能力。
    """
    
    def __init__(self, base_optimizer, alpha=0.5, k=6):
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f'Invalid slow update rate: {alpha}')
        if not 1 <= k:
            raise ValueError(f'Invalid lookahead steps: {k}')
        
        self.optimizer = base_optimizer
        self.param_groups = self.optimizer.param_groups
        self.alpha = alpha
        self.k = k
        for group in self.param_groups:
            group["step_counter"] = 0
        self.slow_weights = [[p.clone().detach() for p in group['params']] for group in self.param_groups]
        
        for w in self.slow_weights:
            for w_i in w:
                w_i.requires_grad = False
        
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = self.optimizer.step(closure)
        else:
            loss = self.optimizer.step()
        
        for group, slow_weights in zip(self.param_groups, self.slow_weights):
            for p, slow_p in zip(group['params'], slow_weights):
                if p.grad is None:
                    continue
                group["step_counter"] += 1
                if group["step_counter"] % self.k != 0:
                    continue
                # 更新慢权重
                slow_p.add_(self.alpha, p.data - slow_p.data)
                p.data.copy_(slow_p.data)
        
        return loss
    
    def state_dict(self):
        fast_dict = self.optimizer.state_dict()
        fast_state = fast_dict['state']
        param_groups = fast_dict['param_groups']
        
        # 添加慢权重状态
        slow_state = {}
        for group, slow_weights in zip(param_groups, self.slow_weights):
            for p, slow_p in zip(group['params'], slow_weights):
                slow_state[p] = slow_p.data
        
        return {
            'fast_state': fast_state,
            'slow_state': slow_state,
            'param_groups': param_groups
        }
    
    def load_state_dict(self, state_dict):
        slow_state = state_dict.pop('slow_state')
        fast_dict = state_dict
        fast_state = fast_dict['state']
        param_groups = fast_dict['param_groups']
        
        # 加载慢权重
        for group, slow_weights in zip(param_groups, self.slow_weights):
            for p, slow_p in zip(group['params'], slow_weights):
                slow_p.data.copy_(slow_state[p])
        
        # 加载优化器状态
        fast_dict['state'] = fast_state
        self.optimizer.load_state_dict(fast_dict)
        
        # 重置step counter
        for group in self.param_groups:
            group["step_counter"] = 0


def get_optimizer(model_parameters, optimizer_config):
    """根据配置创建优化器。
    
    参数:
        model_parameters: 模型参数
        optimizer_config: 优化器配置字典
        
    返回:
        优化器
    """
    optimizer_type = optimizer_config.get('type', 'adam').lower()
    lr = optimizer_config.get('lr', 1e-3)
    weight_decay = optimizer_config.get('weight_decay', 1e-4)
    
    if optimizer_type == 'adam':
        betas = optimizer_config.get('betas', (0.9, 0.999))
        eps = optimizer_config.get('eps', 1e-8)
        return Adam(model_parameters, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
    
    elif optimizer_type == 'adamw':
        betas = optimizer_config.get('betas', (0.9, 0.999))
        eps = optimizer_config.get('eps', 1e-8)
        return AdamWWithWeightDecay(model_parameters, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
    
    elif optimizer_type == 'radam':
        betas = optimizer_config.get('betas', (0.9, 0.999))
        eps = optimizer_config.get('eps', 1e-8)
        return RAdam(model_parameters, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
    
    elif optimizer_type == 'sgd':
        momentum = optimizer_config.get('momentum', 0.9)
        return SGD(model_parameters, lr=lr, momentum=momentum, weight_decay=weight_decay)
    
    elif optimizer_type == 'rmsprop':
        alpha = optimizer_config.get('alpha', 0.99)
        return RMSprop(model_parameters, lr=lr, alpha=alpha, weight_decay=weight_decay)
    
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")


def apply_gradient_clipping(model, clip_config):
    """应用梯度裁剪。
    
    参数:
        model: 模型
        clip_config: 裁剪配置字典
    """
    clip_type = clip_config.get('type', 'norm')
    max_norm = clip_config.get('max_norm', 1.0)
    
    if clip_type == 'norm':
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
    elif clip_type == 'value':
        torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=max_norm)
    else:
        raise ValueError(f"Unknown clip type: {clip_type}")
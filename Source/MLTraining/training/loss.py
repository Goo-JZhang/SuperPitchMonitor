#!/usr/bin/env python3
"""
损失函数模块

提供训练使用的各种损失函数:
- ConfidenceLoss: 置信度损失
- EnergyLoss: 能量分布损失
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConfidenceLoss(nn.Module):
    """
    置信度损失函数
    
    使用二元交叉熵损失评估音高存在性的预测
    """
    
    def __init__(self, reduction='mean'):
        """
        Args:
            reduction: 'mean', 'sum', 或 'none'
        """
        super().__init__()
        self.reduction = reduction
    
    def forward(self, pred_conf, target_conf):
        """
        Args:
            pred_conf: [B, 2048] 预测置信度
            target_conf: [B, 2048] 目标置信度
        
        Returns:
            损失值
        """
        return F.binary_cross_entropy(pred_conf, target_conf, reduction=self.reduction)


class EnergyLoss(nn.Module):
    """
    能量损失函数 - 用于概率分布（已归一化）
    
    支持 KL 散度、JS 散度或 MSE 来度量能量分布差异
    """
    
    def __init__(self, loss_type='kl', reduction='mean'):
        """
        Args:
            loss_type: 'kl' (KL散度), 'js' (JS散度), 'mse' (MSE)
            reduction: 'mean', 'sum', 或 'none'
        """
        super().__init__()
        self.loss_type = loss_type
        self.reduction = reduction
    
    def forward(self, pred_energy, target_energy):
        """
        Args:
            pred_energy: [B, 2048] 预测能量 (已通过softmax归一化)
            target_energy: [B, 2048] 目标能量 (需要归一化)
        
        Returns:
            损失值
        """
        # 确保目标也是概率分布
        target_energy = target_energy / (target_energy.sum(dim=-1, keepdim=True) + 1e-8)
        
        if self.loss_type == 'kl':
            # KL散度: KL(target || pred) = sum(target * log(target/pred))
            loss = target_energy * (torch.log(target_energy + 1e-8) - torch.log(pred_energy + 1e-8))
            if self.reduction == 'mean':
                return loss.sum(dim=-1).mean()
            elif self.reduction == 'sum':
                return loss.sum()
            else:
                return loss.sum(dim=-1)
        
        elif self.loss_type == 'js':
            # JS散度: 更对称，对零值更友好
            m = 0.5 * (pred_energy + target_energy)
            kl_pm = (pred_energy * (torch.log(pred_energy + 1e-8) - torch.log(m + 1e-8))).sum(dim=-1)
            kl_qm = (target_energy * (torch.log(target_energy + 1e-8) - torch.log(m + 1e-8))).sum(dim=-1)
            js = 0.5 * (kl_pm + kl_qm)
            if self.reduction == 'mean':
                return js.mean()
            elif self.reduction == 'sum':
                return js.sum()
            else:
                return js
        
        else:  # 'mse'
            # 简单MSE
            return F.mse_loss(pred_energy, target_energy, reduction=self.reduction)


class SparsityLoss(nn.Module):
    """
    稀疏性损失函数
    
    鼓励置信度输出稀疏（大多数 bin 接近 0）
    """
    
    def __init__(self, target_sparsity=0.05):
        """
        Args:
            target_sparsity: 目标稀疏度（非零比例），默认 5%
        """
        super().__init__()
        self.target_sparsity = target_sparsity
    
    def forward(self, pred_conf):
        """
        Args:
            pred_conf: [B, 2048] 预测置信度
        
        Returns:
            损失值（预测置信度的平均值与目标稀疏度的差异）
        """
        mean_conf = pred_conf.mean()
        # 鼓励置信度接近目标稀疏度
        return (mean_conf - self.target_sparsity) ** 2


class PitchDetectionLoss(nn.Module):
    """
    完整的音高检测损失函数
    
    组合 confidence + energy + sparsity 三种损失
    """
    
    def __init__(self, 
                 conf_weight=1.0, 
                 energy_weight=0.3, 
                 sparsity_weight=0.01,
                 energy_loss_type='kl'):
        """
        Args:
            conf_weight: 置信度损失权重
            energy_weight: 能量损失权重
            sparsity_weight: 稀疏性损失权重
            energy_loss_type: 能量损失类型 ('kl', 'js', 'mse')
        """
        super().__init__()
        self.conf_loss_fn = ConfidenceLoss()
        self.energy_loss_fn = EnergyLoss(loss_type=energy_loss_type)
        self.sparsity_loss_fn = SparsityLoss()
        
        self.conf_weight = conf_weight
        self.energy_weight = energy_weight
        self.sparsity_weight = sparsity_weight
    
    def forward(self, pred_conf, pred_energy, target_conf, target_energy):
        """
        Args:
            pred_conf: [B, 2048] 预测置信度
            pred_energy: [B, 2048] 预测能量
            target_conf: [B, 2048] 目标置信度
            target_energy: [B, 2048] 目标能量
        
        Returns:
            dict: {
                'total': 总损失,
                'confidence': 置信度损失,
                'energy': 能量损失,
                'sparsity': 稀疏性损失
            }
        """
        loss_conf = self.conf_loss_fn(pred_conf, target_conf)
        loss_energy = self.energy_loss_fn(pred_energy, target_energy)
        loss_sparsity = self.sparsity_loss_fn(pred_conf)
        
        total = (self.conf_weight * loss_conf + 
                self.energy_weight * loss_energy + 
                self.sparsity_weight * loss_sparsity)
        
        return {
            'total': total,
            'confidence': loss_conf.detach(),
            'energy': loss_energy.detach(),
            'sparsity': loss_sparsity.detach()
        }


__all__ = ['ConfidenceLoss', 'EnergyLoss', 'SparsityLoss', 'PitchDetectionLoss']

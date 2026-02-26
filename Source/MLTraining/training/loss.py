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
    置信度损失函数 - Focal Loss + Sharpness
    
    改进点：
    - Focal Loss: 自动处理类别不平衡，关注难样本
    - Sharpness Loss: 鼓励尖锐的 peak (可微分，梯度友好)
    - 无位置感知: 避免复杂索引，保持梯度流畅
    """
    
    def __init__(self, reduction='mean', 
                 use_focal=True, 
                 use_sharpness=True,
                 focal_gamma=2.0,  # Focal loss focusing parameter
                 focal_alpha=0.25,  # Focal loss positive weight
                 sharpness_weight=0.5):
        """
        Args:
            reduction: 'mean', 'sum', 或 'none'
            use_focal: 是否使用 Focal Loss (替代加权 BCE)
            use_sharpness: 是否使用 sharpness 损失
            focal_gamma: Focal loss gamma，越大越关注难样本
            focal_alpha: Focal loss alpha，正样本权重
            sharpness_weight: sharpness 损失权重
        """
        super().__init__()
        self.reduction = reduction
        self.use_focal = use_focal
        self.use_sharpness = use_sharpness
        self.focal_gamma = focal_gamma
        self.focal_alpha = focal_alpha
        self.sharpness_weight = sharpness_weight
    
    def forward(self, pred_conf, target_conf):
        """
        Args:
            pred_conf: [B, 2048] 预测置信度
            target_conf: [B, 2048] 目标置信度
        
        Returns:
            损失值
        """
        eps = 1e-7
        pred_conf = torch.clamp(pred_conf, eps, 1 - eps)
        
        # 1. Focal Loss (替代 BCE)
        if self.use_focal:
            ce_loss = -target_conf * torch.log(pred_conf) - (1 - target_conf) * torch.log(1 - pred_conf)
            p_t = (target_conf * pred_conf) + ((1 - target_conf) * (1 - pred_conf))
            focal_weight = (1 - p_t) ** self.focal_gamma
            alpha_t = self.focal_alpha * target_conf + (1 - self.focal_alpha) * (1 - target_conf)
            loss = alpha_t * focal_weight * ce_loss
            
            if self.reduction == 'mean':
                base_loss = loss.mean()
            elif self.reduction == 'sum':
                base_loss = loss.sum()
            else:
                base_loss = loss
        else:
            # 标准 BCE
            base_loss = F.binary_cross_entropy(pred_conf, target_conf, reduction=self.reduction)
        
        # 2. Sharpness Loss (梯度友好版)
        if self.use_sharpness:
            sharp_loss = self._sharpness_loss(pred_conf, target_conf)
            if self.reduction == 'mean':
                sharp_loss = sharp_loss.mean()
            elif self.reduction == 'sum':
                sharp_loss = sharp_loss.sum()
            
            return base_loss + self.sharpness_weight * sharp_loss
        
        return base_loss
    
    def _sharpness_loss(self, pred_conf, target_conf):
        """
        Sharpness 损失 - 梯度友好版
        
        原理：
        - 使用拉普拉斯算子检测 "peak 锐利度"
        - 鼓励正样本位置的二阶梯度大 (曲率高 = sharp)
        - 完全可微分，无复杂索引
        
        Args:
            pred_conf: [B, 2048]
            target_conf: [B, 2048]
        Returns:
            loss: [B] 或标量
        """
        threshold = 0.3
        
        # 找到正样本位置
        positive_mask = (target_conf > threshold).float()
        
        if positive_mask.sum() == 0:
            return torch.tensor(0.0, device=pred_conf.device)
        
        # 拉普拉斯算子 (二阶差分) 检测曲率
        # L(x) = x[i-1] - 2*x[i] + x[i+1]
        # 对于 sharp peak: L(x) 在 peak 处为很大的负数
        left = torch.cat([pred_conf[:, :1], pred_conf[:, :-1]], dim=1)
        right = torch.cat([pred_conf[:, 1:], pred_conf[:, -1:]], dim=1)
        
        laplacian = left - 2 * pred_conf + right  # [B, 2048]
        
        # 在正样本位置，我们希望 laplacian 是大的负数 (sharp peak)
        # 即 -laplacian 应该是大的正数
        # 惩罚：如果正样本位置不够 sharp (laplacian 不够负)
        sharpness = -laplacian * positive_mask  # [B, 2048]
        
        # 我们希望 sharpness 大，所以用负的作为损失
        # 或者直接惩罚小的 sharpness
        loss = F.relu(1.0 - sharpness) * positive_mask  # 希望 sharpness > 1
        
        return loss.sum(dim=-1)  # [B]


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
    
    增强模式 (enhanced=True):
    - 使用 Focal Loss 替代 BCE，自动处理类别不平衡和难易样本
    - 使用 Sharpness 损失，鼓励尖锐的 peak (梯度友好)
    """
    
    def __init__(self, 
                 conf_weight=1.0, 
                 energy_weight=0.3, 
                 sparsity_weight=0.01,
                 energy_loss_type='kl',
                 enhanced=True,
                 focal_gamma=2.0):
        """
        Args:
            conf_weight: 置信度损失权重
            energy_weight: 能量损失权重
            sparsity_weight: 稀疏性损失权重
            energy_loss_type: 能量损失类型 ('kl', 'js', 'mse')
            enhanced: 是否使用增强版 confidence 损失 (Focal + Sharpness)
            focal_gamma: Focal loss gamma，越大越关注难样本
        """
        super().__init__()
        self.conf_loss_fn = ConfidenceLoss(
            use_focal=enhanced,
            use_sharpness=enhanced,
            focal_gamma=focal_gamma,
            focal_alpha=0.25 if enhanced else 0.5,
            sharpness_weight=0.5
        )
        self.energy_loss_fn = EnergyLoss(loss_type=energy_loss_type)
        self.sparsity_loss_fn = SparsityLoss()
        
        self.conf_weight = conf_weight
        self.energy_weight = energy_weight
        self.sparsity_weight = sparsity_weight
        self.enhanced = enhanced
    
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

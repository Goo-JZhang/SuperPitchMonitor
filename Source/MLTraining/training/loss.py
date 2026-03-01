#!/usr/bin/env python3
"""
损失函数模块

提供训练使用的各种损失函数:
- ConfidenceLoss: 置信度损失 (Focal + Tversky + Sharpness)
- EnergyLoss: 能量分布损失
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConfidenceLoss(nn.Module):
    """
    置信度损失函数 - Focal Loss + Tversky Loss + Sharpness
    
    各损失项典型取值范围（供权重参考）:
    - Focal Loss: 0.01 ~ 0.5 (训练初期高，后期低)
    - Tversky Loss: 0.0 ~ 1.0 (标准化良好)
    - Sharpness Loss: 0.0 ~ 2.0 (取决于 peak 锐利度)
    
    推荐权重配置:
    - focal_weight=1.0: 作为主要分类损失
    - tversky_weight=0.3 ~ 0.5: 辅助位置精确性
    - sharpness_weight=0.1 ~ 0.2: 轻微鼓励尖锐 peak
    """
    
    def __init__(self, reduction='mean', 
                 use_focal=True,
                 use_tversky=True,
                 use_sharpness=True,
                 focal_gamma=2.0,
                 focal_alpha=0.25,
                 tversky_weight=0.3,
                 tversky_alpha=0.3,
                 tversky_beta=0.7,
                 sharpness_weight=0.1):
        """
        Args:
            reduction: 'mean', 'sum', 或 'none'
            use_focal: 是否使用 Focal Loss
            use_tversky: 是否使用 Tversky Loss (对位置偏移强惩罚)
            use_sharpness: 是否使用 sharpness 损失
            focal_gamma: Focal loss gamma
            focal_alpha: Focal loss alpha
            tversky_weight: Tversky 损失权重 (推荐 0.3-0.5)
            tversky_alpha: Tversky 假阳性惩罚系数 (0-1)
            tversky_beta: Tversky 假阴性惩罚系数 (0-1), 建议 > alpha
            sharpness_weight: sharpness 损失权重 (推荐 0.1-0.2，较低)
        """
        super().__init__()
        self.reduction = reduction
        self.use_focal = use_focal
        self.use_tversky = use_tversky
        self.use_sharpness = use_sharpness
        
        self.focal_gamma = focal_gamma
        self.focal_alpha = focal_alpha
        
        self.tversky_weight = tversky_weight
        self.tversky_alpha = tversky_alpha
        self.tversky_beta = tversky_beta
        
        self.sharpness_weight = sharpness_weight
    
    def forward(self, pred_conf, target_conf):
        """
        Args:
            pred_conf: [B, 2048] 预测置信度
            target_conf: [B, 2048] 目标置信度
        
        Returns:
            loss_dict: {
                'total': 总损失,
                'focal': Focal损失 (detach),
                'tversky': Tversky损失 (detach),
                'sharpness': Sharpness损失 (detach)
            }
        """
        eps = 1e-7
        pred_conf = torch.clamp(pred_conf, eps, 1 - eps)
        
        loss_dict = {}
        total_loss = 0.0
        
        # 1. Focal Loss (主要分类损失)
        if self.use_focal:
            focal_loss = self._focal_loss(pred_conf, target_conf)
            total_loss = total_loss + focal_loss
            loss_dict['focal'] = focal_loss.detach()
        
        # 2. Tversky Loss (位置精确性损失)
        if self.use_tversky:
            tversky_loss = self._tversky_loss(pred_conf, target_conf)
            total_loss = total_loss + self.tversky_weight * tversky_loss
            loss_dict['tversky'] = tversky_loss.detach()
        
        # 3. Sharpness Loss (锐利度损失，低权重)
        if self.use_sharpness:
            sharp_loss = self._sharpness_loss(pred_conf, target_conf)
            total_loss = total_loss + self.sharpness_weight * sharp_loss
            loss_dict['sharpness'] = sharp_loss.detach()
        
        loss_dict['total'] = total_loss
        return loss_dict
    
    def _focal_loss(self, pred, target):
        """Focal Loss - 处理类别不平衡"""
        ce_loss = -target * torch.log(pred) - (1 - target) * torch.log(1 - pred)
        p_t = (target * pred) + ((1 - target) * (1 - pred))
        focal_weight = (1 - p_t) ** self.focal_gamma
        alpha_t = self.focal_alpha * target + (1 - self.focal_alpha) * (1 - target)
        loss = alpha_t * focal_weight * ce_loss
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
    
    def _tversky_loss(self, pred, target, smooth=1e-7):
        """
        Tversky Loss - 对峰位置偏离提供强惩罚
        
        适用于软标签 (高斯分布的目标)。
        Tversky 系数范围 [0, 1]，损失范围 [0, 1]。
        
        Args:
            pred: [B, 2048] 预测置信度
            target: [B, 2048] 目标置信度 (高斯软标签)
        
        Returns:
            loss: Tversky 损失，标量
        """
        # 计算各类像素
        tp = (pred * target).sum(dim=-1)           # True Positive
        fp = (pred * (1 - target)).sum(dim=-1)     # False Positive
        fn = ((1 - pred) * target).sum(dim=-1)     # False Negative
        
        # Tversky 系数
        tversky = (tp + smooth) / (tp + self.tversky_alpha * fp + self.tversky_beta * fn + smooth)
        
        if self.reduction == 'mean':
            return (1 - tversky).mean()
        elif self.reduction == 'sum':
            return (1 - tversky).sum()
        else:
            return 1 - tversky
    
    def _sharpness_loss(self, pred_conf, target_conf):
        """
        Sharpness 损失 - 梯度友好版 (修复版)
        
        典型取值: 0.0 ~ 1.0 (已归一化)
        权重建议: 0.1 ~ 0.2
        
        Args:
            pred_conf: [B, 2048]
            target_conf: [B, 2048] (高斯软标签)
        Returns:
            loss: 标量
        """
        # 使用软权重而非硬阈值，与高斯标签兼容
        # 目标值越高，该位置的 sharpness 权重越大
        weight = target_conf  # [B, 2048]
        
        if weight.sum() == 0:
            return torch.tensor(0.0, device=pred_conf.device)
        
        # 拉普拉斯算子 (二阶差分) 检测曲率
        left = torch.cat([pred_conf[:, :1], pred_conf[:, :-1]], dim=1)
        right = torch.cat([pred_conf[:, 1:], pred_conf[:, -1:]], dim=1)
        laplacian = left - 2 * pred_conf + right  # [B, 2048]
        
        # 在目标位置，我们希望 laplacian 是大的负数 (sharp peak)
        # sharpness 越大表示越 sharp
        sharpness = -laplacian  # [B, 2048]
        
        # 惩罚不够 sharp 的位置 (sharpness < 1.0)
        # 使用软权重：目标值高的位置惩罚权重更大
        loss_per_bin = F.relu(1.0 - sharpness) * weight  # [B, 2048]
        
        # 归一化：按权重和归一，避免数值过大
        loss = loss_per_bin.sum(dim=-1) / (weight.sum(dim=-1) + 1e-7)  # [B]
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class EnergyLoss(nn.Module):
    """
    能量损失函数 - 用于概率分布（已归一化）
    
    各类型典型取值范围:
    - KL 散度: 0.1 ~ 2.0 (分布差异大时较高)
    - JS 散度: 0.05 ~ 1.0 (比 KL 更对称，通常更低)
    - MSE: 0.001 ~ 0.1 (数值小，因为能量已归一化)
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
            # KL散度: 典型范围 0.1 ~ 2.0
            loss = target_energy * (torch.log(target_energy + 1e-8) - torch.log(pred_energy + 1e-8))
            if self.reduction == 'mean':
                return loss.sum(dim=-1).mean()
            elif self.reduction == 'sum':
                return loss.sum()
            else:
                return loss.sum(dim=-1)
        
        elif self.loss_type == 'js':
            # JS散度: 典型范围 0.05 ~ 1.0
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
            # MSE: 典型范围 0.001 ~ 0.1
            return F.mse_loss(pred_energy, target_energy, reduction=self.reduction)


class SparsityLoss(nn.Module):
    """
    稀疏性损失函数
    
    典型取值范围: 0.0001 ~ 0.04
    (取决于 mean_conf 与 target_sparsity 的差距)
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
        return (mean_conf - self.target_sparsity) ** 2


class PitchDetectionLoss(nn.Module):
    """
    完整的音高检测损失函数
    
    组合 confidence + energy + sparsity 三种损失
    
    各损失项典型值及推荐权重（参考）:
    ==========================================================
    损失项          典型范围        推荐权重        说明
    ==========================================================
    Focal Loss      0.1 ~ 0.5       10.0 (主导)     主要分类损失
    Tversky Loss    0.1 ~ 0.5       3.0 (强辅助)    位置精确性
    Sharpness Loss  0.1 ~ 0.5       0.1 (轻微)      peak 锐利度
    Energy (KL)     0.1 ~ 1.0       0.3 (次要)      能量分布
    Sparsity        0.001 ~ 0.01    0.1 (正则化)    防止全0预测
    ==========================================================
    
    设计理念:
    - 所有配置都以 confidence 预测为核心（分类问题，必须准确）
    - Energy 预测为次要任务（回归问题，容错较高）
    - default 配置已经是最强的 confidence 配置
    
    配置选择:
    - default: 最强的 confidence 预测（推荐）
    - position_focused: 在 default 基础上强调位置精度
    - peak_focused: 在 default 基础上强调 peak 质量
    - balanced: 在 default 基础上稍微平衡 energy
    - simple: 去掉 sharpness，保留核心 conf 损失
    """
    
    def __init__(self, 
                 # 置信度损失权重（以 confidence 为核心）
                 focal_weight=10.0,
                 tversky_weight=3.0,
                 sharpness_weight=0.1,
                 # Energy 损失权重（次要）
                 energy_weight=0.3,
                 energy_loss_type='kl',
                 # Sparsity 损失权重
                 sparsity_weight=0.1,
                 # 各损失组件开关
                 use_focal=True,
                 use_tversky=True,
                 use_sharpness=True,
                 # Tversky 参数
                 tversky_alpha=0.3,
                 tversky_beta=0.7,
                 # Focal 参数
                 focal_gamma=0.5,
                 focal_alpha=0.8):
        """
        Args:
            focal_weight: Focal 损失权重 (默认 10.0, confidence 主导)
            tversky_weight: Tversky 损失权重 (默认 3.0, 强辅助)
            sharpness_weight: Sharpness 损失权重 (默认 0.1, 轻微)
            energy_weight: 能量损失权重 (默认 0.3, 次要)
            energy_loss_type: 能量损失类型 ('kl', 'js', 'mse')
            sparsity_weight: 稀疏性损失权重 (默认 0.1)
            use_focal: 是否使用 Focal Loss
            use_tversky: 是否使用 Tversky Loss
            use_sharpness: 是否使用 Sharpness Loss
            tversky_alpha: Tversky 假阳性惩罚
            tversky_beta: Tversky 假阴性惩罚 (建议 > alpha)
            focal_gamma: Focal loss gamma (默认 0.5，让更多样本参与)
            focal_alpha: Focal loss alpha (默认 0.8，强调正样本)
        """
        super().__init__()
        
        self.conf_loss_fn = ConfidenceLoss(
            use_focal=use_focal,
            use_tversky=use_tversky,
            use_sharpness=use_sharpness,
            focal_gamma=focal_gamma,
            focal_alpha=focal_alpha,
            tversky_weight=1.0,  # 内部已乘权重，这里用 1.0
            tversky_alpha=tversky_alpha,
            tversky_beta=tversky_beta,
            sharpness_weight=1.0  # 内部已乘权重，这里用 1.0
        )
        self.energy_loss_fn = EnergyLoss(loss_type=energy_loss_type)
        self.sparsity_loss_fn = SparsityLoss()
        
        self.focal_weight = focal_weight
        self.tversky_weight = tversky_weight
        self.sharpness_weight = sharpness_weight
        self.energy_weight = energy_weight
        self.sparsity_weight = sparsity_weight
        
        self.use_focal = use_focal
        self.use_tversky = use_tversky
        self.use_sharpness = use_sharpness
    
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
                'focal': Focal损失,
                'tversky': Tversky损失,
                'sharpness': Sharpness损失,
                'energy': 能量损失,
                'sparsity': 稀疏性损失
            }
        """
        # 置信度损失（返回 dict）
        conf_losses = self.conf_loss_fn(pred_conf, target_conf)
        
        # 分别加权
        total_conf_loss = 0.0
        if self.use_focal and 'focal' in conf_losses:
            total_conf_loss += self.focal_weight * conf_losses['focal']
        if self.use_tversky and 'tversky' in conf_losses:
            total_conf_loss += self.tversky_weight * conf_losses['tversky']
        if self.use_sharpness and 'sharpness' in conf_losses:
            total_conf_loss += self.sharpness_weight * conf_losses['sharpness']
        
        # 能量和稀疏性损失
        loss_energy = self.energy_loss_fn(pred_energy, target_energy)
        loss_sparsity = self.sparsity_loss_fn(pred_conf)
        
        # 总损失
        total = total_conf_loss + self.energy_weight * loss_energy + self.sparsity_weight * loss_sparsity
        
        return {
            'total': total,
            'confidence': total_conf_loss.detach(),  # 总置信度损失（用于兼容）
            'focal': conf_losses.get('focal', torch.tensor(0.0)).detach(),
            'tversky': conf_losses.get('tversky', torch.tensor(0.0)).detach(),
            'sharpness': conf_losses.get('sharpness', torch.tensor(0.0)).detach(),
            'energy': loss_energy.detach(),
            'sparsity': loss_sparsity.detach()
        }


# 预定义的推荐配置
def get_loss_config(config_name='default'):
    """
    获取推荐的损失函数配置
    
    Args:
        config_name: 'default', 'conf_focused', 'position_focused', 'peak_focused', 'balanced'
    
    Returns:
        dict: 配置参数字典
    """
    # 所有配置都以 confidence 预测为核心
    # default 是最强的 confidence 配置，其他在此基础上调整侧重点
    configs = {
        'default': {
            # 最强的 confidence 预测配置（推荐）
            'focal_weight': 10.0,       # 主导：confidence 分类
            'tversky_weight': 3.0,      # 强辅助：peak 位置精度
            'sharpness_weight': 0.1,    # 轻微：peak 形状
            'energy_weight': 0.3,       # 次要：能量分布
            'sparsity_weight': 0.1,     # 防止全 0 预测
            'use_focal': True,
            'use_tversky': True,
            'use_sharpness': True,
            'tversky_alpha': 0.3,
            'tversky_beta': 0.7,
            'focal_gamma': 0.5,         # 让更多样本参与学习
            'focal_alpha': 0.8,         # 高正样本权重（正样本很少）
        },
        'position_focused': {
            # 在 default 基础上，进一步提高位置精度要求
            'focal_weight': 8.0,
            'tversky_weight': 5.0,      # 大幅提高位置权重
            'sharpness_weight': 0.1,
            'energy_weight': 0.3,
            'sparsity_weight': 0.1,
            'use_focal': True,
            'use_tversky': True,
            'use_sharpness': True,
            'tversky_alpha': 0.3,
            'tversky_beta': 0.7,
            'focal_gamma': 0.5,
            'focal_alpha': 0.8,
        },
        'peak_focused': {
            # 在 default 基础上，强调 peak 质量
            'focal_weight': 8.0,
            'tversky_weight': 2.5,
            'sharpness_weight': 1.0,    # 大幅提高锐利度
            'energy_weight': 0.3,
            'sparsity_weight': 0.1,
            'use_focal': True,
            'use_tversky': True,
            'use_sharpness': True,
            'tversky_alpha': 0.3,
            'tversky_beta': 0.7,
            'focal_gamma': 0.5,
            'focal_alpha': 0.8,
        },
        'balanced': {
            # 在 default 基础上，稍微平衡 energy 权重
            'focal_weight': 8.0,
            'tversky_weight': 2.5,
            'sharpness_weight': 0.1,
            'energy_weight': 0.8,       # 提高 energy 权重
            'sparsity_weight': 0.08,
            'use_focal': True,
            'use_tversky': True,
            'use_sharpness': True,
            'tversky_alpha': 0.3,
            'tversky_beta': 0.7,
            'focal_gamma': 0.5,
            'focal_alpha': 0.8,
        },
        'simple': {
            # 简化版：去掉 sharpness，保留核心 conf 损失
            'focal_weight': 10.0,
            'tversky_weight': 3.0,
            'sharpness_weight': 0.0,    # 关闭
            'energy_weight': 0.3,
            'sparsity_weight': 0.1,
            'use_focal': True,
            'use_tversky': True,
            'use_sharpness': False,
            'tversky_alpha': 0.3,
            'tversky_beta': 0.7,
            'focal_gamma': 0.5,
            'focal_alpha': 0.8,
        }
    }
    return configs.get(config_name, configs['default'])


__all__ = [
    'ConfidenceLoss', 
    'EnergyLoss', 
    'SparsityLoss', 
    'PitchDetectionLoss',
    'get_loss_config'
]

#!/usr/bin/env python3
"""
PitchNetEnhancedV2 - 在 PitchNetEnhanced 基础上添加门控融合

Phase 3 改进:
- 在双分支 Backbone 后添加 TaskGatedFusion
- 让 confidence 和 energy 任务互相协作，共享有用信息
- 解决：conf 预测偏差但 energy 峰位置正确的问题
- 利用能量信息辅助音高检测（能量峰与置信度峰通常位置一致）

架构:
    输入: [B, 1, 4096]
    → Frontend: [B, 64, 64]
    → Dual Backbone (Conf/Energy): [B, 128, 32] each
    → TaskGatedFusion: 信息交换与融合
    → Freq Attention: [B, 128, 32]
    → Upsample Head: [B, 16, 2048]
    → Output: [B, 2048, 2]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FrequencyAttention(nn.Module):
    """频域自注意力 - 与 PitchNetEnhanced 相同"""
    
    def __init__(self, num_bins: int = 32, num_heads: int = 4, 
                 channels: int = 128, dropout: float = 0.1):
        super().__init__()
        assert channels % num_heads == 0, "channels must be divisible by num_heads"
        
        self.num_bins = num_bins
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv_proj = nn.Conv1d(channels, channels * 3, kernel_size=1)
        self.out_proj = nn.Conv1d(channels, channels, kernel_size=1)
        
        self.rel_pos_emb = nn.Parameter(
            torch.randn(num_bins * 2 - 1, num_heads) * 0.02
        )
        
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.BatchNorm1d(channels)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, N = x.shape
        assert N == self.num_bins, f"Expected num_bins={self.num_bins}, got {N}"
        
        residual = x
        
        qkv = self.qkv_proj(x)
        q, k, v = qkv.chunk(3, dim=1)
        
        q = q.view(B, self.num_heads, self.head_dim, N).transpose(-1, -2)
        k = k.view(B, self.num_heads, self.head_dim, N).transpose(-1, -2)
        v = v.view(B, self.num_heads, self.head_dim, N).transpose(-1, -2)
        
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        pos_bias = self._get_rel_pos_bias(N)
        pos_bias = pos_bias.permute(2, 0, 1).unsqueeze(0)
        attn = attn + pos_bias
        
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        out = torch.matmul(attn, v)
        out = out.transpose(-1, -2).contiguous().view(B, C, N)
        
        out = self.out_proj(out)
        out = self.norm(out + residual)
        
        return out
    
    def _get_rel_pos_bias(self, n: int) -> torch.Tensor:
        pos_idx = torch.arange(n, device=self.rel_pos_emb.device).unsqueeze(0) - \
                  torch.arange(n, device=self.rel_pos_emb.device).unsqueeze(1)
        pos_idx = pos_idx + self.num_bins - 1
        pos_idx = pos_idx.clamp(0, self.num_bins * 2 - 2)
        return self.rel_pos_emb[pos_idx]


class TaskGatedFusion(nn.Module):
    """
    任务门控融合模块
    
    让 confidence 和 energy 任务互相协作:
    - 能量峰位置可以提示置信度应该关注的位置
    - 置信度检测到音高的位置应该有对应的能量支持
    """
    
    def __init__(self, channels: int = 128):
        super().__init__()
        
        # Confidence 门控
        self.conf_gate = nn.Sequential(
            nn.Conv1d(channels * 2, channels, kernel_size=1),
            nn.BatchNorm1d(channels),
            nn.Sigmoid()
        )
        
        # Energy 门控
        self.energy_gate = nn.Sequential(
            nn.Conv1d(channels * 2, channels, kernel_size=1),
            nn.BatchNorm1d(channels),
            nn.Sigmoid()
        )
        
        # 特征变换层
        self.conf_transform = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=1),
            nn.BatchNorm1d(channels),
            nn.GELU()
        )
        
        self.energy_transform = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=1),
            nn.BatchNorm1d(channels),
            nn.GELU()
        )
    
    def forward(self, conf_feat: torch.Tensor, energy_feat: torch.Tensor) -> tuple:
        """
        Args:
            conf_feat: [B, channels, N]
            energy_feat: [B, channels, N]
        Returns:
            (conf_enhanced, energy_enhanced)
        """
        combined = torch.cat([conf_feat, energy_feat], dim=1)
        
        conf_gate = self.conf_gate(combined)
        energy_gate = self.energy_gate(combined)
        
        energy_to_conf = self.energy_transform(energy_feat)
        conf_to_energy = self.conf_transform(conf_feat)
        
        conf_enhanced = conf_feat + conf_gate * energy_to_conf
        energy_enhanced = energy_feat + energy_gate * conf_to_energy
        
        return conf_enhanced, energy_enhanced


class PitchNetEnhancedV2(nn.Module):
    """
    PitchNetEnhancedV2 - 带门控融合的改进版
    
    固定配置:
    - 输入: 4096 samples
    - 输出: 2048 bins
    - 使用门控融合 + 频域注意力
    """
    
    def __init__(self, input_samples: int = 4096, num_bins: int = 2048):
        super().__init__()
        
        self.input_samples = input_samples
        self.num_bins = num_bins
        
        # 前端: [B, 64, 64]
        self.frontend = nn.Conv1d(
            in_channels=1,
            out_channels=64,
            kernel_size=256,
            stride=64,
            padding=96
        )
        
        # Confidence Backbone: [B, 128, 32]
        self.conf_backbone = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Conv1d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.GELU(),
        )
        
        # Energy Backbone: [B, 128, 32]
        self.energy_backbone = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Conv1d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.GELU(),
        )
        
        # 门控融合
        self.gate_fusion = TaskGatedFusion(channels=128)
        
        # 频域注意力
        self.conf_attention = FrequencyAttention(
            num_bins=32, num_heads=4, channels=128
        )
        self.energy_attention = FrequencyAttention(
            num_bins=32, num_heads=4, channels=128
        )
        
        # 上采样
        self.conf_upsample = nn.Sequential(
            nn.ConvTranspose1d(128, 64, kernel_size=4, stride=4, padding=0),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.ConvTranspose1d(64, 32, kernel_size=4, stride=4, padding=0),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.ConvTranspose1d(32, 16, kernel_size=4, stride=4, padding=0),
            nn.BatchNorm1d(16),
            nn.GELU(),
        )
        
        self.energy_upsample = nn.Sequential(
            nn.ConvTranspose1d(128, 64, kernel_size=4, stride=4, padding=0),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.ConvTranspose1d(64, 32, kernel_size=4, stride=4, padding=0),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.ConvTranspose1d(32, 16, kernel_size=4, stride=4, padding=0),
            nn.BatchNorm1d(16),
            nn.GELU(),
        )
        
        # 最终输出层
        self.conf_final = nn.Sequential(
            nn.Conv1d(16, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        self.energy_final = nn.Sequential(
            nn.Conv1d(16, 1, kernel_size=1)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """权重初始化"""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        # 前端
        x = self.frontend(x)
        
        # Backbone
        conf_feat = self.conf_backbone(x)
        energy_feat = self.energy_backbone(x)
        
        # 门控融合
        conf_feat, energy_feat = self.gate_fusion(conf_feat, energy_feat)
        
        # 频域注意力
        conf_feat = self.conf_attention(conf_feat)
        energy_feat = self.energy_attention(energy_feat)
        
        # 上采样
        conf_up = self.conf_upsample(conf_feat)
        energy_up = self.energy_upsample(energy_feat)
        
        # 最终输出
        conf = self.conf_final(conf_up).squeeze(1)
        energy = self.energy_final(energy_up).squeeze(1)
        energy = F.softmax(energy, dim=-1)
        
        return torch.stack([conf, energy], dim=-1)
    
    def count_parameters(self) -> int:
        """计算可训练参数数量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    print("=" * 60)
    print("PitchNetEnhancedV2 - Model Test")
    print("=" * 60)
    
    model = PitchNetEnhancedV2()
    total_params = model.count_parameters()
    print(f"Total parameters: {total_params:,} ({total_params/1e6:.2f}M)")
    
    # 前向测试
    x = torch.randn(2, 1, 4096)
    model.eval()
    with torch.no_grad():
        y = model(x)
    
    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {y.shape}")
    print(f"  - Confidence range: [{y[..., 0].min():.3f}, {y[..., 0].max():.3f}]")
    print(f"  - Energy sum: {y[..., 1].sum(dim=-1).tolist()}")
    
    # 验证
    energy_sum = y[..., 1].sum(dim=-1)
    assert torch.allclose(energy_sum, torch.ones_like(energy_sum), atol=1e-5)
    assert (y[..., 0] >= 0).all() and (y[..., 0] <= 1).all()
    
    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)

#!/usr/bin/env python3
"""
PitchNetEnhanced - 改进版音高检测网络

针对 PitchNetBaseline 的缺陷进行优化:
1. 移除全局池化，保留频率分辨率
2. 任务分离的 Backbone，减少 confidence 和 energy 任务干扰
3. 频域注意力模块，建模 bin 间的频域关系（谐波、邻近性）
4. Conv-based Head 替代 Linear Head，更好地建模局部频域关系

架构:
    输入: [B, 1, 4096]
    → 前端 (Conv1d): [B, 64, 64]
    → 双分支 Backbone: [B, 128, 32] each
    → 频域注意力: [B, 128, 32]
    → ConvTranspose 上采样: [B, 16, 2048]
    → 双头输出: [B, 2048, 2]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FrequencyAttention(nn.Module):
    """
    频域自注意力 - 让模型学习频率 bin 之间的关系
    
    关键作用:
    1. 捕捉谐波关系（基频与倍频的关联）
    2. 学习频率邻近性（平滑约束）
    3. 为后续多音训练做准备（分离不同音高的特征）
    """
    
    def __init__(self, num_bins: int = 32, num_heads: int = 4, 
                 channels: int = 128, dropout: float = 0.1):
        super().__init__()
        assert channels % num_heads == 0, "channels must be divisible by num_heads"
        
        self.num_bins = num_bins
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Q, K, V 投影
        self.qkv_proj = nn.Conv1d(channels, channels * 3, kernel_size=1)
        self.out_proj = nn.Conv1d(channels, channels, kernel_size=1)
        
        # 相对位置编码（捕捉频率距离）
        self.rel_pos_emb = nn.Parameter(
            torch.randn(num_bins * 2 - 1, num_heads) * 0.02
        )
        
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.BatchNorm1d(channels)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, channels, num_bins]
        Returns:
            [B, channels, num_bins]
        """
        B, C, N = x.shape
        assert N == self.num_bins, f"Expected num_bins={self.num_bins}, got {N}"
        
        # 残差连接
        residual = x
        
        # 生成 Q, K, V
        qkv = self.qkv_proj(x)
        q, k, v = qkv.chunk(3, dim=1)
        
        # 多头拆分
        q = q.view(B, self.num_heads, self.head_dim, N).transpose(-1, -2)
        k = k.view(B, self.num_heads, self.head_dim, N).transpose(-1, -2)
        v = v.view(B, self.num_heads, self.head_dim, N).transpose(-1, -2)
        
        # 注意力计算
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # 添加相对位置偏置
        pos_bias = self._get_rel_pos_bias(N)
        pos_bias = pos_bias.permute(2, 0, 1).unsqueeze(0)
        attn = attn + pos_bias
        
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        # 应用注意力
        out = torch.matmul(attn, v)
        out = out.transpose(-1, -2).contiguous().view(B, C, N)
        
        out = self.out_proj(out)
        out = self.norm(out + residual)
        
        return out
    
    def _get_rel_pos_bias(self, n: int) -> torch.Tensor:
        """获取相对位置偏置矩阵"""
        pos_idx = torch.arange(n, device=self.rel_pos_emb.device).unsqueeze(0) - \
                  torch.arange(n, device=self.rel_pos_emb.device).unsqueeze(1)
        pos_idx = pos_idx + self.num_bins - 1
        pos_idx = pos_idx.clamp(0, self.num_bins * 2 - 2)
        return self.rel_pos_emb[pos_idx]


class PitchNetEnhanced(nn.Module):
    """
    PitchNetEnhanced - 改进版音高检测网络
    
    固定配置:
    - 输入: 4096 samples
    - 输出: 2048 bins
    - 使用频域注意力
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
        
        # 频域注意力
        self.conf_attention = FrequencyAttention(
            num_bins=32, num_heads=4, channels=128
        )
        self.energy_attention = FrequencyAttention(
            num_bins=32, num_heads=4, channels=128
        )
        
        # 上采样到 2048 bins
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
        """
        Args:
            x: [B, 1, 4096] 输入音频
        Returns:
            [B, 2048, 2] - confidence, energy
        """
        # 前端特征提取
        x = self.frontend(x)
        
        # 任务分离的 backbone
        conf_feat = self.conf_backbone(x)
        energy_feat = self.energy_backbone(x)
        
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
    print("PitchNetEnhanced - Model Test")
    print("=" * 60)
    
    model = PitchNetEnhanced()
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

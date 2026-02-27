#!/usr/bin/env python3
"""
PitchNetEnhancedV3 - 大容量版本

主要改进:
1. 通道扩展: 128 -> 256
2. 残差堆叠 Backbone: 固定 4 个残差块
3. 渐进上采样: 平滑特征恢复
4. Transformer-style Attention Block: Attention + FFN + 残差

架构:
    输入: [B, 1, 4096]
    → Frontend (2层): [B, 256, 64]
    → Downsample: [B, 256, 32]
    → ResBlocks (x4): [B, 256, 32]
    → Transformer Block: [B, 256, 32]
    → Progressive Upsample: [B, 16, 2048]
    → Output: [B, 2048, 2]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResConvBlock(nn.Module):
    """残差卷积块"""
    
    def __init__(self, channels: int, kernel_size: int = 3):
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm1d(channels)
        self.gelu = nn.GELU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.gelu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return self.gelu(x + residual)


class TransformerBlock(nn.Module):
    """
    Transformer Block: Attention + FFN + 残差连接
    
    结构:
        Input -> LayerNorm -> Attention -> Residual -> 
        LayerNorm -> FFN -> Residual -> Output
    """
    
    def __init__(self, num_bins: int = 32, num_heads: int = 8, 
                 channels: int = 256, ffn_ratio: float = 2.0, dropout: float = 0.1):
        super().__init__()
        
        self.num_bins = num_bins
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Self-Attention
        self.qkv_proj = nn.Conv1d(channels, channels * 3, kernel_size=1)
        self.out_proj = nn.Conv1d(channels, channels, kernel_size=1)
        self.rel_pos_emb = nn.Parameter(torch.randn(num_bins * 2 - 1, num_heads) * 0.02)
        self.attn_dropout = nn.Dropout(dropout)
        self.attn_norm = nn.BatchNorm1d(channels)
        
        # FFN
        ffn_hidden = int(channels * ffn_ratio)
        self.ffn = nn.Sequential(
            nn.Conv1d(channels, ffn_hidden, kernel_size=1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(ffn_hidden, channels, kernel_size=1),
            nn.Dropout(dropout),
        )
        self.ffn_norm = nn.BatchNorm1d(channels)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, N = x.shape
        
        # Attention 分支
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
        attn = self.attn_dropout(attn)
        
        out = torch.matmul(attn, v)
        out = out.transpose(-1, -2).contiguous().view(B, C, N)
        out = self.out_proj(out)
        
        x = self.attn_norm(out + residual)
        
        # FFN 分支
        ffn_out = self.ffn(x)
        x = self.ffn_norm(ffn_out + x)
        
        return x
    
    def _get_rel_pos_bias(self, n: int) -> torch.Tensor:
        pos_idx = torch.arange(n, device=self.rel_pos_emb.device).unsqueeze(0) - \
                  torch.arange(n, device=self.rel_pos_emb.device).unsqueeze(1)
        pos_idx = pos_idx + self.num_bins - 1
        pos_idx = pos_idx.clamp(0, self.num_bins * 2 - 2)
        return self.rel_pos_emb[pos_idx]


class PitchNetEnhancedV3(nn.Module):
    """
    PitchNetEnhancedV3 - 大容量版本
    
    固定配置:
    - 通道数: 256
    - 残差块: 4 个
    - Attention heads: 8
    - FFN 扩展比: 2.0
    """
    
    def __init__(self, input_samples: int = 4096, num_bins: int = 2048):
        super().__init__()
        
        self.input_samples = input_samples
        self.num_bins = num_bins
        
        # 前端: 2层 [B, 256, 64]
        self.frontend = nn.Sequential(
            nn.Conv1d(1, 96, kernel_size=256, stride=64, padding=96),
            nn.BatchNorm1d(96),
            nn.GELU(),
            nn.Conv1d(96, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.GELU(),
        )
        
        # 下采样: [B, 256, 32]
        self.conf_downsample = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(256),
            nn.GELU(),
        )
        self.energy_downsample = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(256),
            nn.GELU(),
        )
        
        # 残差堆叠: 固定 4 个块
        self.conf_blocks = nn.ModuleList([
            ResConvBlock(256) for _ in range(4)
        ])
        self.energy_blocks = nn.ModuleList([
            ResConvBlock(256) for _ in range(4)
        ])
        
        # Transformer Block
        self.conf_transformer = TransformerBlock(
            num_bins=32, num_heads=8, channels=256, ffn_ratio=2.0
        )
        self.energy_transformer = TransformerBlock(
            num_bins=32, num_heads=8, channels=256, ffn_ratio=2.0
        )
        
        # 渐进上采样: 5层
        self.conf_upsample = nn.Sequential(
            nn.ConvTranspose1d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.ConvTranspose1d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.ConvTranspose1d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.ConvTranspose1d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(16),
            nn.GELU(),
            nn.ConvTranspose1d(16, 16, kernel_size=8, stride=4, padding=2),
            nn.BatchNorm1d(16),
            nn.GELU(),
        )
        
        self.energy_upsample = nn.Sequential(
            nn.ConvTranspose1d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.ConvTranspose1d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.ConvTranspose1d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.ConvTranspose1d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(16),
            nn.GELU(),
            nn.ConvTranspose1d(16, 16, kernel_size=8, stride=4, padding=2),
            nn.BatchNorm1d(16),
            nn.GELU(),
        )
        
        # 最终输出
        self.conf_final = nn.Sequential(
            nn.Conv1d(16, 1, kernel_size=1),
            nn.Sigmoid()
        )
        self.energy_final = nn.Sequential(
            nn.Conv1d(16, 1, kernel_size=1)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d)):
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
        # 前端
        x = self.frontend(x)
        
        # Backbone
        conf_feat = self.conf_downsample(x)
        energy_feat = self.energy_downsample(x)
        
        # 残差块
        for block in self.conf_blocks:
            conf_feat = block(conf_feat)
        for block in self.energy_blocks:
            energy_feat = block(energy_feat)
        
        # Transformer Block
        conf_feat = self.conf_transformer(conf_feat)
        energy_feat = self.energy_transformer(energy_feat)
        
        # 上采样
        conf_up = self.conf_upsample(conf_feat)
        energy_up = self.energy_upsample(energy_feat)
        
        # 输出
        conf = self.conf_final(conf_up).squeeze(1)
        energy = self.energy_final(energy_up).squeeze(1)
        energy = F.softmax(energy, dim=-1)
        
        return torch.stack([conf, energy], dim=-1)
    
    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    print("=" * 60)
    print("PitchNetEnhancedV3 - Model Test")
    print("=" * 60)
    
    model = PitchNetEnhancedV3()
    total_params = model.count_parameters()
    print(f"Total parameters: {total_params:,} ({total_params/1e6:.2f}M)")
    
    x = torch.randn(2, 1, 4096)
    model.eval()
    with torch.no_grad():
        y = model(x)
    
    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {y.shape}")
    print(f"  - Confidence range: [{y[..., 0].min():.3f}, {y[..., 0].max():.3f}]")
    print(f"  - Energy sum: {y[..., 1].sum(dim=-1).tolist()}")
    
    energy_sum = y[..., 1].sum(dim=-1)
    assert torch.allclose(energy_sum, torch.ones_like(energy_sum), atol=1e-5)
    assert (y[..., 0] >= 0).all() and (y[..., 0] <= 1).all()
    
    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)

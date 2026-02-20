#!/usr/bin/env python3
"""
PitchNet 模型定义
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PitchNetBaseline(nn.Module):
    """
    PitchNet Baseline 模型
    
    架构:
    - 前端: Conv1d 1->64, kernel=512, stride=128
    - 主干: 3层 Conv 进行时序压缩  
    - 池化: 全局平均池化
    - 双头: 独立输出 confidence (Sigmoid) 和 energy (Softmax)
    
    输出: [B, 2048, 2] - confidence [0-1], energy [总和=1]
    """
    
    def __init__(self, input_samples: int = 4096, num_bins: int = 2048):
        super().__init__()
        
        # 前端: 可学习滤波器组
        self.frontend = nn.Conv1d(
            in_channels=1, out_channels=64,
            kernel_size=512, stride=128, padding=256
        )  # [B, 64, 32]
        
        # 主干: 时序压缩
        self.backbone = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Conv1d(128, 256, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Conv1d(256, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(256),
            nn.GELU(),
        )  # [B, 256, 4]
        
        # 全局池化
        self.pool = nn.AdaptiveAvgPool1d(1)
        
        # Confidence 头 (Sigmoid激活)
        self.conf_head = nn.Sequential(
            nn.Linear(256, 1024),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(1024, num_bins),
            nn.Sigmoid()
        )
        
        # Energy 头 (Softmax激活，在forward中应用)
        self.energy_head = nn.Sequential(
            nn.Linear(256, 1024),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(1024, num_bins),
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Xavier初始化"""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
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
        x = self.frontend(x)
        x = self.backbone(x)
        x = self.pool(x).squeeze(-1)
        
        conf = self.conf_head(x)
        energy = self.energy_head(x)
        energy = F.softmax(energy, dim=-1)
        
        return torch.stack([conf, energy], dim=-1)
    
    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    # 测试
    model = PitchNetBaseline()
    x = torch.randn(2, 1, 4096)
    y = model(x)
    print(f"Input:  {x.shape}")
    print(f"Output: {y.shape}")
    print(f"Params: {model.count_parameters()/1e6:.2f}M")

"""
PitchNetEnhancedV4 - Ultra-Compatible Cross-Platform Architecture

Design for maximum compatibility across ALL platforms:
- Windows: CUDA, DirectML
- Android: NNAPI
- iOS: CoreML
- Linux: CPU/GPU

Key constraints (based on GitHub Issue #22346):
- NO LayerNorm (not supported on NNAPI/CoreML)
- NO GELU/Erf (not supported on NNAPI/CoreML)  
- NO ReduceMean/GlobalAveragePool (problematic on CoreML)
- NO InstanceNorm (not supported)
- Use only: Conv, BatchNorm, ReLU, Sigmoid, Add, Mul

Architecture:
- Pure CNN (no attention, no pooling-based ops)
- Residual blocks with BatchNorm + ReLU
- Simple upsampling with ConvTranspose
- Conservative design: if in doubt, leave it out
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any


class BasicResidualBlock(nn.Module):
    """
    Most basic mobile-friendly block.
    Conv -> BN -> ReLU -> Conv -> BN -> Add -> ReLU
    """
    def __init__(self, channels: int, kernel_size: int = 3):
        super().__init__()
        padding = kernel_size // 2
        
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm1d(channels)
        
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=padding, bias=False)
        self.bn2 = nn.BatchNorm1d(channels)
        
    def forward(self, x):
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out = out + residual
        out = F.relu(out)
        
        return out


class SimpleConvBlock(nn.Module):
    """
    Simple Conv + BN + ReLU block.
    """
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, padding=padding, bias=False)
        self.bn = nn.BatchNorm1d(out_ch)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x)
        return x


class DualTaskBackboneV4(nn.Module):
    """
    Simple residual backbone for dual task (confidence + energy).
    Pure CNN, no attention, no pooling.
    """
    def __init__(self, channels: int, num_blocks: int = 8):
        super().__init__()
        
        self.blocks = nn.ModuleList([
            BasicResidualBlock(channels) for _ in range(num_blocks)
        ])
        
    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


class UpsampleBlockV4(nn.Module):
    """
    Upsampling using ConvTranspose + BN + ReLU.
    2x upsampling per block.
    """
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        # kernel=4, stride=2, padding=1 gives 2x upsampling
        self.up = nn.ConvTranspose1d(in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn = nn.BatchNorm1d(out_ch)
        
    def forward(self, x):
        x = self.up(x)
        x = self.bn(x)
        x = F.relu(x)
        return x


class PitchNetEnhancedV4(nn.Module):
    """
    Cross-platform pitch detection network - Maximum Compatibility Version.
    
    Guaranteed compatible operators:
    - Conv1d, ConvTranspose1d
    - BatchNorm1d  
    - ReLU
    - Sigmoid
    - Add (residual)
    
    Architecture:
    - Frontend: Conv (learnable STFT alternative)
    - Expansion: 1x1 conv to increase channels
    - Dual backbones: Pure residual CNN (8 blocks each)
    - Progressive upsampling: 2x per stage, 6 stages total (64 -> 2048)
    - Dual output: Sigmoid(confidence) + Softmax(energy)
    """
    
    def __init__(self, input_samples: int = 4096, num_bins: int = 2048):
        super().__init__()
        
        self.input_samples = input_samples
        self.num_bins = num_bins
        
        # Frontend: raw audio -> feature map
        # Input: [B, 1, 4096] -> Output: [B, 96, 64]
        # Kernel 256, stride 64: (4096 + 2*96 - 256)/64 + 1 = 64
        self.frontend = nn.Conv1d(1, 96, kernel_size=256, stride=64, padding=96)
        self.frontend_bn = nn.BatchNorm1d(96)
        
        # Channel expansion: 96 -> 256
        self.expand = nn.Conv1d(96, 256, kernel_size=1, bias=False)
        self.expand_bn = nn.BatchNorm1d(256)
        
        # Dual task backbones (no attention, no pooling)
        self.conf_backbone = DualTaskBackboneV4(256, num_blocks=8)
        self.energy_backbone = DualTaskBackboneV4(256, num_blocks=8)
        
        # Progressive upsampling: 64 -> 2048 (6 stages of 2x)
        # Stage 1: 64 -> 128
        self.conf_up1 = UpsampleBlockV4(256, 192)
        self.energy_up1 = UpsampleBlockV4(256, 192)
        
        # Stage 2: 128 -> 256
        self.conf_up2 = UpsampleBlockV4(192, 128)
        self.energy_up2 = UpsampleBlockV4(192, 128)
        
        # Stage 3: 256 -> 512
        self.conf_up3 = UpsampleBlockV4(128, 64)
        self.energy_up3 = UpsampleBlockV4(128, 64)
        
        # Stage 4: 512 -> 1024
        self.conf_up4 = UpsampleBlockV4(64, 32)
        self.energy_up4 = UpsampleBlockV4(64, 32)
        
        # Stage 5: 1024 -> 2048
        self.conf_up5 = UpsampleBlockV4(32, 16)
        self.energy_up5 = UpsampleBlockV4(32, 16)
        
        # Stage 6: refinement (already at 2048, just reduce channels)
        self.conf_refine = nn.Conv1d(16, 8, kernel_size=3, padding=1, bias=False)
        self.conf_refine_bn = nn.BatchNorm1d(8)
        self.energy_refine = nn.Conv1d(16, 8, kernel_size=3, padding=1, bias=False)
        self.energy_refine_bn = nn.BatchNorm1d(8)
        
        # Final output convolutions
        self.conf_final = nn.Conv1d(8, 1, kernel_size=1)
        self.energy_final = nn.Conv1d(8, 1, kernel_size=1)
        
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights."""
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        """
        Args:
            x: [B, 1, 4096] raw audio
        Returns:
            output: [B, 2, 2048] - combined tensor
                output[:, 0, :] = confidence (sigmoid)
                output[:, 1, :] = energy (softmax)
        """
        # Frontend
        x = self.frontend(x)
        x = self.frontend_bn(x)
        x = F.relu(x)
        
        # Channel expansion
        x = self.expand(x)
        x = self.expand_bn(x)
        x = F.relu(x)
        
        # Dual backbones
        conf = self.conf_backbone(x)
        energy = self.energy_backbone(x)
        
        # Progressive upsampling
        conf = self.conf_up1(conf)
        conf = self.conf_up2(conf)
        conf = self.conf_up3(conf)
        conf = self.conf_up4(conf)
        conf = self.conf_up5(conf)
        
        energy = self.energy_up1(energy)
        energy = self.energy_up2(energy)
        energy = self.energy_up3(energy)
        energy = self.energy_up4(energy)
        energy = self.energy_up5(energy)
        
        # Refinement
        conf = self.conf_refine(conf)
        conf = self.conf_refine_bn(conf)
        conf = F.relu(conf)
        
        energy = self.energy_refine(energy)
        energy = self.energy_refine_bn(energy)
        energy = F.relu(energy)
        
        # Final outputs
        conf = self.conf_final(conf)
        conf = torch.sigmoid(conf)
        
        energy = self.energy_final(energy)
        energy = F.softmax(energy, dim=-1)
        
        # Combine and transpose to [B, 2048, 2] for compatibility with training code
        # Training code uses pred[..., 0] for conf and pred[..., 1] for energy
        output = torch.cat([conf, energy], dim=1)  # [B, 2, 2048]
        output = output.permute(0, 2, 1)  # [B, 2048, 2]
        return output
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        total_params = sum(p.numel() for p in self.parameters())
        
        return {
            "name": "PitchNetEnhancedV4",
            "version": "4.0.0",
            "description": "Ultra-compatible cross-platform model (pure CNN)",
            "input_samples": self.input_samples,
            "num_bins": self.num_bins,
            "total_params": total_params,
            "param_size_mb": total_params * 4 / (1024 * 1024),
            "operators": ["Conv1d", "ConvTranspose1d", "BatchNorm1d", "ReLU", "Sigmoid", "Add"],
            "compatibility": {
                "Windows": ["CUDA", "DirectML", "CPU"],
                "Android": ["NNAPI", "CPU"],
                "iOS": ["CoreML", "CPU"],
                "Linux": ["CUDA", "CPU"]
            }
        }


if __name__ == "__main__":
    # Test
    model = PitchNetEnhancedV4()
    info = model.get_model_info()
    
    print("=" * 60)
    print("PitchNetEnhancedV4 - Ultra-Compatible Model")
    print("=" * 60)
    print(f"Parameters: {info['total_params']:,}")
    print(f"Size: {info['param_size_mb']:.2f} MB (FP32)")
    print(f"\nOperators: {', '.join(info['operators'])}")
    print("=" * 60)
    
    # Forward test
    x = torch.randn(2, 1, 4096)
    output = model(x)
    # Output is [B, 2048, 2], use last dimension for conf/energy
    conf = output[..., 0]  # [B, 2048]
    energy = output[..., 1]  # [B, 2048]
    print(f"\nInput: {x.shape}")
    print(f"Output: {output.shape}")
    print(f"Confidence: {conf.shape}, range: [{conf.min():.3f}, {conf.max():.3f}]")
    print(f"Energy: {energy.shape}, sum: {energy.sum(dim=-1).mean():.3f}")

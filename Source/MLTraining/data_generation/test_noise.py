#!/usr/bin/env python3
"""
测试噪声生成
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from generate_note import generate_noise, NOISE_TYPES

print('Checking noise energy profiles:')
print('=' * 60)

fig, axes = plt.subplots(len(NOISE_TYPES), 1, figsize=(12, 10))

for idx, color in enumerate(NOISE_TYPES):
    sample = generate_noise(color=color, num_bins=2048)
    energy = sample['energies']
    confs = sample['confs']
    
    print(f'{color:10s}: sum={energy.sum():.6f}, max={energy.max():.6f}, mean={energy.mean():.6f}')
    print(f'           confs_sum={confs.sum():.6f} (should be 0)')
    
    # 显示能量分布特点
    low_energy = energy[:512].mean()   # 低频
    mid_energy = energy[512:1536].mean()  # 中频
    high_energy = energy[1536:].mean()  # 高频
    print(f'           low_freq_avg={low_energy:.6f}, mid_freq_avg={mid_energy:.6f}, high_freq_avg={high_energy:.6f}')
    
    # 绘制能量分布
    ax = axes[idx]
    ax.plot(energy, label=f'{color} noise', linewidth=1)
    ax.set_ylabel('Energy')
    ax.set_title(f'{color.capitalize()} Noise Energy Profile')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    print()

axes[-1].set_xlabel('Frequency Bin')
plt.tight_layout()
plt.savefig('noise_profiles.png', dpi=150, bbox_inches='tight')
print('Saved noise_profiles.png')

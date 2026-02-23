#!/usr/bin/env python3
"""
Sanity + Noise 混合数据集生成器

每个样本包含：
- 单音正弦波（log尺度均匀采样频率，随机相位，随机幅度）
- 五种最强噪声混合（white, pink, brown, blue, violet，权重随机）
- 正弦波和噪声各自有随机幅度比例

总样本数：204,800
"""

import numpy as np
import sys
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from dataset_writer import DatasetWriter
from generate_note import _generate_colored_noise


def generate_sanity_with_noise_dataset(
    output_dir: str = '../../../TrainingData/SanityWithNoise',
    total_samples: int = 204800,
    num_bins: int = 2048,
    min_freq: float = 20.0,
    max_freq: float = 5000.0,
    sigma: float = 1.5,
    seed: int = 42
):
    """
    生成 Sanity + Noise 混合数据集
    
    Args:
        output_dir: 输出目录
        total_samples: 总样本数（默认204800）
        num_bins: 频率bin数量
        min_freq: 最低频率
        max_freq: 最高频率
        sigma: 高斯平滑参数
        seed: 随机种子
    """
    print(f"Generating Sanity + Noise Mixed Dataset")
    print(f"  Output: {output_dir}")
    print(f"  Total samples: {total_samples:,}")
    print(f"  Frequency range: {min_freq}-{max_freq} Hz")
    print(f"  Bins: {num_bins}")
    print()
    
    rng = np.random.RandomState(seed)
    
    # 预生成五种噪声的独立RNG种子（确保每个样本噪声独立）
    noise_colors = ['white', 'pink', 'brown', 'blue', 'violet']
    
    with DatasetWriter(output_dir, shard_size_gb=1.0, save_indices=False) as writer:
        for i in tqdm(range(total_samples), desc="Generating"):
            # ===== 1. 随机选择频率（log尺度均匀）=====
            log_min = np.log2(min_freq)
            log_max = np.log2(max_freq)
            log_freq = rng.uniform(log_min, log_max)
            freq = 2 ** log_freq
            
            # 计算对应的bin索引
            bin_idx = int((log_freq - log_min) / (log_max - log_min) * (num_bins - 1))
            bin_idx = np.clip(bin_idx, 0, num_bins - 1)
            
            # ===== 2. 生成正弦波（随机相位，随机幅度）=====
            phase = rng.uniform(0, 2 * np.pi)
            sine_amp = rng.uniform(0.3, 1.0)  # 正弦波幅度 0.3-1.0
            
            sample_rate = 44100
            t = np.arange(4096) / sample_rate
            sine_wave = sine_amp * np.sin(2 * np.pi * freq * t + phase).astype(np.float32)
            
            # ===== 3. 生成五种噪声混合（权重随机）=====
            noise_weights = rng.rand(5)  # 5种噪声的权重
            noise_weights = noise_weights / noise_weights.sum()  # 归一化
            
            mixed_noise = np.zeros(4096, dtype=np.float32)
            for color, weight in zip(noise_colors, noise_weights):
                # 每种噪声使用独立RNG
                noise_rng = np.random.RandomState(rng.randint(0, 2**31))
                noise = _generate_colored_noise(4096, color, noise_rng)
                mixed_noise += weight * noise
            
            # 噪声也添加随机幅度缩放
            noise_amp = rng.uniform(0.1, 0.5)  # 噪声幅度 0.1-0.5
            mixed_noise = mixed_noise * noise_amp
            
            # ===== 4. 混合波形 =====
            waveform = sine_wave + mixed_noise
            
            # 归一化到标准差为1.0（与之前数据处理一致）
            waveform = waveform / (np.std(waveform) + 1e-10)
            
            # ===== 5. 生成真值 =====
            # Confidence：高斯平滑（只在单音位置）
            confs = np.zeros(num_bins, dtype=np.float32)
            
            range_size = int(3 * sigma) + 1
            for j in range(max(0, bin_idx - range_size), min(num_bins, bin_idx + range_size + 1)):
                dist = abs(j - bin_idx)
                val = np.exp(-0.5 * (dist / sigma) ** 2)
                if val > 0.01:
                    confs[j] = min(val, 1.0)
            
            # Energy：单音能量 + 噪声能量
            # 单音能量（高斯平滑）
            sine_energy = np.zeros(num_bins, dtype=np.float32)
            for j in range(max(0, bin_idx - range_size), min(num_bins, bin_idx + range_size + 1)):
                dist = abs(j - bin_idx)
                val = np.exp(-0.5 * (dist / sigma) ** 2)
                if val > 0.01:
                    sine_energy[j] = val
            
            # 归一化单音能量
            sine_sum = sine_energy.sum()
            if sine_sum > 0:
                sine_energy = sine_energy / sine_sum
            
            # 噪声能量（5种混合的加权，weight**2 因为能量是平方关系）
            from generate_note import _get_noise_energy_profile
            noise_energy = np.zeros(num_bins, dtype=np.float32)
            for color, weight in zip(noise_colors, noise_weights):
                energy_profile = _get_noise_energy_profile(color, num_bins)
                noise_energy += (weight ** 2) * energy_profile
            
            # 混合能量 = 单音能量 × sine_amp² + 噪声能量 × noise_amp²（能量是平方关系）
            # 然后归一化
            total_energy = (sine_energy * sine_amp**2 + noise_energy * noise_amp**2)
            energy_sum = total_energy.sum()
            if energy_sum > 0:
                energies = total_energy / energy_sum
            else:
                energies = total_energy
            
            # ===== 6. 写入数据集 =====
            writer.add_sample(
                waveform=waveform,
                confs=confs,
                energies=energies
            )
        
        # 添加元数据
        writer.finalize({
            'total_samples': total_samples,
            'num_bins': num_bins,
            'frequency_range': [min_freq, max_freq],
            'sigma': sigma,
            'seed': seed,
            'description': 'Mixed dataset: sine wave + 5-color noise (white, pink, brown, blue, violet)',
            'sine_amplitude_range': [0.3, 1.0],
            'noise_amplitude_range': [0.1, 0.5],
        })
    
    print(f"\nDataset generation complete!")
    print(f"  Total samples: {total_samples:,}")
    print(f"  Output: {output_dir}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate Sanity + Noise Mixed Dataset")
    parser.add_argument('--output', type=str,
                       default='../../../TrainingData/SanityWithNoise',
                       help='Output directory')
    parser.add_argument('--samples', type=int, default=204800,
                       help='Total number of samples (default: 204800)')
    parser.add_argument('--num-bins', type=int, default=2048,
                       help='Number of frequency bins')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    generate_sanity_with_noise_dataset(
        output_dir=args.output,
        total_samples=args.samples,
        num_bins=args.num_bins,
        seed=args.seed
    )

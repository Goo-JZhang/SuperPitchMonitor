#!/usr/bin/env python3
"""
纯噪声数据集生成器

生成各种彩色噪声样本及其组合，用于训练负样本（无音高）

包含:
- 单一噪声: white, pink, brown, blue, violet (各1000个)
- 双重组合: C(5,2)=10种 (各1000个)
- 三重组合: C(5,3)=10种 (各1000个)
- 四重组合: C(5,4)=5种 (各1000个)
- 五重组合: 1种 (1000个)

总计: 31种 × 1000 = 31,000个样本
"""

import numpy as np
import sys
from pathlib import Path
from tqdm import tqdm
from itertools import combinations

sys.path.insert(0, str(Path(__file__).parent))
from dataset_writer import DatasetWriter
from generate_note import generate_noise, generate_mixed_noise, NOISE_TYPES


def generate_noise_dataset(output_dir: str = '../../../TrainingData/NoiseDataset',
                           samples_per_type: int = 1000,
                           shard_size_gb: float = 1.0,
                           num_bins: int = 2048,
                           seed: int = 42):
    """
    生成纯噪声数据集（包含所有组合）
    
    Args:
        output_dir: 输出目录
        samples_per_type: 每种噪声/组合的样本数
        shard_size_gb: 触发分片的大小
        num_bins: 频率bin数量
        seed: 随机种子
    """
    rng = np.random.RandomState(seed)
    
    # 计算所有组合
    noise_types = []
    
    # 1. 单一噪声
    for color in NOISE_TYPES:
        noise_types.append((f'single_{color}', [color]))
    
    # 2. 双重组合
    for combo in combinations(NOISE_TYPES, 2):
        noise_types.append((f'mix_{"_".join(combo)}', list(combo)))
    
    # 3. 三重组合
    for combo in combinations(NOISE_TYPES, 3):
        noise_types.append((f'mix_{"_".join(combo)}', list(combo)))
    
    # 4. 四重组合
    for combo in combinations(NOISE_TYPES, 4):
        noise_types.append((f'mix_{"_".join(combo)}', list(combo)))
    
    # 5. 五重组合
    noise_types.append(('mix_all', NOISE_TYPES.copy()))
    
    total_samples = len(noise_types) * samples_per_type
    
    print(f"Generating Noise Dataset V2")
    print(f"  Output: {output_dir}")
    print(f"  Noise types: {len(noise_types)}")
    print(f"    - Single (1-color): 5 types")
    print(f"    - Double (2-color): 10 types")
    print(f"    - Triple (3-color): 10 types")
    print(f"    - Quad (4-color): 5 types")
    print(f"    - All (5-color): 1 type")
    print(f"  Samples per type: {samples_per_type}")
    print(f"  Total samples: {total_samples:,}")
    print(f"  Seed: {seed}")
    print()
    
    with DatasetWriter(output_dir, shard_size_gb=shard_size_gb, save_indices=False) as writer:
        for type_name, colors in tqdm(noise_types, desc="Generating noise types"):
            for _ in range(samples_per_type):
                # 随机权重（对于单一噪声也是随机的，增加多样性）
                if len(colors) == 1:
                    # 单一噪声：直接生成
                    sample = generate_noise(color=colors[0], num_bins=num_bins, rng=rng)
                else:
                    # 混合噪声：随机权重
                    weights = rng.rand(len(colors))
                    sample = generate_mixed_noise(
                        colors=colors,
                        num_bins=num_bins,
                        weights=weights,
                        rng=rng
                    )
                
                writer.add_sample(
                    waveform=sample['waveform'],
                    confs=sample['confs'],
                    energies=sample['energies']
                )
        
        # 添加元数据
        writer.finalize({
            'noise_types': len(noise_types),
            'single_types': 5,
            'double_types': 10,
            'triple_types': 10,
            'quad_types': 5,
            'all_types': 1,
            'samples_per_type': samples_per_type,
            'total_samples': total_samples,
            'num_bins': num_bins,
            'seed': seed,
            'shard_size_gb': shard_size_gb,
            'description': 'Colored noise dataset with all combinations (single to 5-mix)'
        })
    
    print(f"\nNoise dataset generation complete!")
    print(f"  Total types: {len(noise_types)}")
    print(f"  Total samples: {total_samples:,}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate Noise Dataset")
    parser.add_argument('--output', type=str,
                       default='../../../TrainingData/NoiseDataset',
                       help='Output directory')
    parser.add_argument('--samples-per-type', type=int, default=1000,
                       help='Samples per noise type/combination')
    parser.add_argument('--shard-size-gb', type=float, default=1.0,
                       help='Trigger shard when waveform data exceeds this size (GB)')
    parser.add_argument('--num-bins', type=int, default=2048,
                       help='Number of frequency bins')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    generate_noise_dataset(
        output_dir=args.output,
        samples_per_type=args.samples_per_type,
        shard_size_gb=args.shard_size_gb,
        num_bins=args.num_bins,
        seed=args.seed
    )

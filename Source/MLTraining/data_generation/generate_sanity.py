#!/usr/bin/env python3
"""
Sanity Check 数据集生成器 V2

使用 DatasetWriter 进行数据存储，本文件只关注数据生成逻辑
"""

import numpy as np
import sys
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from dataset_writer import DatasetWriter
from generate_note import generate_single_note


def generate_sanity_dataset(output_dir: str = '../../../TrainingData/SingleSanity',
                            samples_per_bin: int = 20,
                            shard_size_gb: float = 1.0,
                            num_bins: int = 2048,
                            seed: int = 42):
    """
    生成 Sanity Check 数据集
    
    数据特点:
    - 单音 (单bin激活)
    - 正弦波 (无泛音)
    - 每个bin多个随机相位
    
    Args:
        output_dir: 输出目录
        samples_per_bin: 每个bin的样本数（不同相位）
        shard_size_gb: 触发分片的大小
        num_bins: 频率bin数量
        seed: 随机种子
    """
    total_samples = num_bins * samples_per_bin
    
    print(f"Generating Sanity Check Dataset")
    print(f"  Output: {output_dir}")
    print(f"  Samples: {total_samples} ({num_bins} bins x {samples_per_bin})")
    print(f"  Seed: {seed}")
    print()
    
    rng = np.random.RandomState(seed)
    
    with DatasetWriter(output_dir, shard_size_gb=shard_size_gb, save_indices=False) as writer:
        for bin_idx in tqdm(range(num_bins), desc="Generating"):
            for _ in range(samples_per_bin):
                # 生成单音样本
                sample = generate_single_note(bin_idx=bin_idx, num_bins=num_bins, rng=rng)
                
                # 添加到数据集
                writer.add_sample(
                    waveform=sample['waveform'],
                    confs=sample['confs'],
                    energies=sample['energies']
                )
        
        # 添加元数据
        writer.finalize({
            'samples_per_bin': samples_per_bin,
            'num_bins': num_bins,
            'seed': seed,
            'shard_size_gb': shard_size_gb,
            'description': 'Single note sine wave dataset with random phases'
        })


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate Sanity Check Dataset")
    parser.add_argument('--output', type=str,
                       default='../../../TrainingData/SingleSanity',
                       help='Output directory')
    parser.add_argument('--samples-per-bin', type=int, default=10,
                       help='Samples per frequency bin')
    parser.add_argument('--shard-size-gb', type=float, default=1.0,
                       help='Trigger shard when waveform data exceeds this size (GB)')
    parser.add_argument('--num-bins', type=int, default=2048,
                       help='Number of frequency bins')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    generate_sanity_dataset(
        output_dir=args.output,
        samples_per_bin=args.samples_per_bin,
        shard_size_gb=args.shard_size_gb,
        num_bins=args.num_bins,
        seed=args.seed
    )

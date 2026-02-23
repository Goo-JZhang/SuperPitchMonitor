#!/usr/bin/env python3
"""
多音数据集生成器

使用 DatasetWriter 存储，本文件只关注多音数据生成逻辑
"""

import numpy as np
import sys
from pathlib import Path
from tqdm import tqdm
from typing import List

sys.path.insert(0, str(Path(__file__).parent))
from dataset_writer import DatasetWriter
from energy_calculator import FrequencyDomainEnergyCalculator, Note
from sample_generator import WindowedSampleGenerator


def generate_polyphony_dataset(output_dir: str = '../../../TrainingData/Polyphony',
                               num_samples: int = 10000,
                               shard_size_gb: float = 1.0,
                               seed: int = 42):
    """
    生成多音数据集
    
    使用 WindowedSampleGenerator 生成随机多音样本
    
    Args:
        output_dir: 输出目录
        num_samples: 样本数量
        shard_size_gb: 触发分片的大小
        seed: 随机种子
    """
    print(f"Generating Polyphony Dataset")
    print(f"  Output: {output_dir}")
    print(f"  Samples: {num_samples}")
    print(f"  Seed: {seed}")
    print()
    
    # 初始化生成器
    sample_gen = WindowedSampleGenerator()
    energy_calc = FrequencyDomainEnergyCalculator()
    
    with DatasetWriter(output_dir, shard_size_gb=shard_size_gb, save_indices=False) as writer:
        for i in tqdm(range(num_samples), desc="Generating"):
            # 生成随机多音样本
            sample = sample_gen.generate_sample()
            
            # 添加到数据集
            writer.add_sample(
                waveform=sample['waveform'],
                confs=sample['target_confidence'],
                energies=sample['target_energy']
            )
        
        # 添加元数据
        writer.finalize({
            'num_samples': num_samples,
            'seed': seed,
            'shard_size_gb': shard_size_gb,
            'description': 'Polyphony dataset with random notes'
        })


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate Polyphony Dataset")
    parser.add_argument('--output', type=str,
                       default='../../../TrainingData/Polyphony',
                       help='Output directory')
    parser.add_argument('--num-samples', type=int, default=10000,
                       help='Number of samples')
    parser.add_argument('--shard-size-gb', type=float, default=1.0,
                       help='Trigger shard when waveform data exceeds this size (GB)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    generate_polyphony_dataset(
        output_dir=args.output,
        num_samples=args.num_samples,
        shard_size_gb=args.shard_size_gb,
        seed=args.seed
    )

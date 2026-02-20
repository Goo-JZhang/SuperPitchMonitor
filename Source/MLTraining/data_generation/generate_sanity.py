#!/usr/bin/env python3
"""
Sanity Check 数据集生成器

特点:
- 波形与真值分开存储 (NumPy格式)
- 真值数据无损压缩 (稀疏表示)
- 自动分片: 波形数据>1GB时触发
- 波形与真值分片一一对应

存储结构:
TrainingData/SingleSanity/
├── meta.pkl
├── waveforms/
│   ├── shard_00000.npy  # [N, 4096] float32
│   ├── shard_00001.npy
│   └── ...
└── targets/
    ├── indices_00000.npy    # [N, K] int16 - 非零bin索引
    ├── confs_00000.npy      # [N, K] float16 - confidence值
    ├── energies_00000.npy   # [N, K] float16 - energy值
    └── ...
"""

import numpy as np
import sys
from pathlib import Path
from tqdm import tqdm
import pickle

# 添加模型路径
sys.path.insert(0, str(Path(__file__).parent.parent / 'Model'))


def bin_to_freq(bin_idx: int, num_bins: int = 2048, min_freq: float = 20.0, max_freq: float = 5000.0) -> float:
    """bin索引转换为频率（对数分布）"""
    log_min = np.log2(min_freq)
    log_max = np.log2(max_freq)
    log_freq = log_min + (bin_idx / (num_bins - 1)) * (log_max - log_min)
    return 2 ** log_freq


def get_bin_freq_range(bin_idx: int, num_bins: int = 2048, min_freq: float = 20.0, max_freq: float = 5000.0):
    """
    获取bin的频率范围 [low, high)
    
    Returns:
        (low_freq, high_freq): 该bin的频率上下界
    """
    # 计算该bin的中心频率
    center = bin_to_freq(bin_idx, num_bins, min_freq, max_freq)
    
    # 计算相邻bin的频率来确定边界
    if bin_idx == 0:
        # 第一个bin: 下限就是min_freq
        low = min_freq
        high = (bin_to_freq(1, num_bins, min_freq, max_freq) + center) / 2
    elif bin_idx == num_bins - 1:
        # 最后一个bin: 上限就是max_freq
        low = (bin_to_freq(num_bins - 2, num_bins, min_freq, max_freq) + center) / 2
        high = max_freq
    else:
        # 中间的bin: 取相邻bin中心的中点
        low = (bin_to_freq(bin_idx - 1, num_bins, min_freq, max_freq) + center) / 2
        high = (bin_to_freq(bin_idx + 1, num_bins, min_freq, max_freq) + center) / 2
    
    return low, high


def create_sparse_target(bin_idx: int, num_bins: int = 2048, sigma: float = 1.5):
    """
    创建稀疏表示的目标频谱
    
    Returns:
        indices: 非零bin索引列表
        conf_values: 对应confidence值
        energy_values: 对应energy值
    """
    # 计算非零范围 (|i - bin_idx| < 3*sigma)
    range_size = int(3 * sigma) + 1
    
    indices = []
    conf_values = []
    energy_values = []
    
    for i in range(max(0, bin_idx - range_size), min(num_bins, bin_idx + range_size + 1)):
        dist = abs(i - bin_idx)
        val = np.exp(-0.5 * (dist / sigma) ** 2)
        if val > 0.01:
            indices.append(i)
            conf_values.append(min(val, 1.0))
            energy_values.append(val)
    
    return indices, conf_values, energy_values


class SanityGenerator:
    """Sanity数据集生成器，自动分片管理"""
    
    def __init__(self, output_dir: str, shard_size_gb: float = 1.0):
        """
        Args:
            output_dir: 输出目录
            shard_size_gb: 触发分片的波形数据大小(GB)
        """
        self.output_path = Path(output_dir)
        self.wave_dir = self.output_path / 'waveforms'
        self.target_dir = self.output_path / 'targets'
        
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.wave_dir.mkdir(exist_ok=True)
        self.target_dir.mkdir(exist_ok=True)
        
        self.shard_size_bytes = int(shard_size_gb * 1024 * 1024 * 1024)
        self.current_shard = 0
        
        # 当前分片缓冲
        self.waveforms = []
        self.indices_list = []
        self.confs_list = []
        self.energies_list = []
        self.current_waveform_bytes = 0
        
        self.total_samples = 0
        
    def _save_current_shard(self):
        """保存当前分片到磁盘"""
        if not self.waveforms:
            return
        
        n_samples = len(self.waveforms)
        
        # 1. 保存波形 [N, 4096] float32
        wave_path = self.wave_dir / f"shard_{self.current_shard:05d}.npy"
        wave_array = np.array(self.waveforms, dtype=np.float32)
        np.save(wave_path, wave_array)
        
        # 2. 保存稀疏真值
        # 找出最大非零数 (统一长度)
        max_nonzero = max(len(idxs) for idxs in self.indices_list)
        
        # 填充到统一长度
        indices_padded = []
        confs_padded = []
        energies_padded = []
        
        for idxs, confs, energies in zip(self.indices_list, self.confs_list, self.energies_list):
            # 填充
            idxs_pad = idxs + [0] * (max_nonzero - len(idxs))
            confs_pad = confs + [0.0] * (max_nonzero - len(confs))
            energies_pad = energies + [0.0] * (max_nonzero - len(energies))
            
            indices_padded.append(idxs_pad[:max_nonzero])
            confs_padded.append(confs_pad[:max_nonzero])
            energies_padded.append(energies_pad[:max_nonzero])
        
        # 保存
        idx_path = self.target_dir / f"indices_{self.current_shard:05d}.npy"
        conf_path = self.target_dir / f"confs_{self.current_shard:05d}.npy"
        energy_path = self.target_dir / f"energies_{self.current_shard:05d}.npy"
        
        np.save(idx_path, np.array(indices_padded, dtype=np.int16))
        np.save(conf_path, np.array(confs_padded, dtype=np.float16))
        np.save(energy_path, np.array(energies_padded, dtype=np.float16))
        
        print(f"  Saved shard {self.current_shard}: {n_samples} samples "
              f"(wave: {wave_path.stat().st_size/1024/1024:.1f}MB, "
              f"target: {idx_path.stat().st_size/1024:.0f}KB)")
        
        self.total_samples += n_samples
        self.current_shard += 1
        
        # 清空缓冲
        self.waveforms = []
        self.indices_list = []
        self.confs_list = []
        self.energies_list = []
        self.current_waveform_bytes = 0
    
    def add_sample(self, waveform: np.ndarray, bin_idx: int):
        """
        添加样本到缓冲，自动触发分片
        
        Args:
            waveform: [4096] float32 波形
            bin_idx: 目标bin索引
        """
        # 生成稀疏真值
        indices, confs, energies = create_sparse_target(bin_idx)
        
        # 添加到缓冲
        self.waveforms.append(waveform)
        self.indices_list.append(indices)
        self.confs_list.append(confs)
        self.energies_list.append(energies)
        
        # 累计波形数据大小
        self.current_waveform_bytes += waveform.nbytes
        
        # 检查是否触发分片
        if self.current_waveform_bytes >= self.shard_size_bytes:
            self._save_current_shard()
    
    def finalize(self, samples_per_bin: int, num_bins: int = 2048):
        """完成生成，保存最后一个分片和元数据"""
        self._save_current_shard()
        
        # 保存元数据
        meta = {
            'total_samples': self.total_samples,
            'num_shards': self.current_shard,
            'num_bins': num_bins,
            'samples_per_bin': samples_per_bin,
            'shard_size_gb': self.shard_size_bytes / 1024 / 1024 / 1024,
            'format': 'sanity_separate',
        }
        
        with open(self.output_path / 'meta.pkl', 'wb') as f:
            pickle.dump(meta, f)
        
        # 统计
        wave_total = sum(f.stat().st_size for f in self.wave_dir.glob('*.npy'))
        target_total = sum(f.stat().st_size for f in self.target_dir.glob('*.npy'))
        
        print(f"\nDataset saved to: {self.output_path}")
        print(f"  Total samples: {self.total_samples}")
        print(f"  Num shards: {self.current_shard}")
        print(f"  Waveforms: {wave_total / 1024 / 1024:.1f} MB")
        print(f"  Targets: {target_total / 1024 / 1024:.2f} MB")
        print(f"  Total: {(wave_total + target_total) / 1024 / 1024:.1f} MB")
        print(f"  Size per sample: {(wave_total + target_total) / self.total_samples / 1024:.2f} KB")


def generate_sanity_dataset(output_dir: str = '../../../TrainingData/SingleSanity',
                            samples_per_bin: int = 10,
                            shard_size_gb: float = 1.0):
    """
    生成Sanity Check数据集
    
    Args:
        output_dir: 输出目录
        samples_per_bin: 每个bin的样本数（不同相位）
        shard_size_gb: 触发分片的波形数据大小(GB)
    """
    num_bins = 2048
    total_samples = num_bins * samples_per_bin
    
    print(f"Generating Sanity Check Dataset")
    print(f"  Output: {output_dir}")
    print(f"  Total samples: {total_samples} ({num_bins} bins × {samples_per_bin})")
    print(f"  Shard trigger: {shard_size_gb} GB waveform data")
    print(f"  Phase: Random (0-2π)")
    print(f"  Frequency: Random within each bin")
    
    # 创建生成器
    generator = SanityGenerator(output_dir, shard_size_gb)
    
    # 随机数生成器
    rng = np.random.RandomState(42)
    
    # 生成所有样本
    for bin_idx in tqdm(range(num_bins), desc="Generating bins"):
        # 获取该bin的频率范围
        freq_low, freq_high = get_bin_freq_range(bin_idx)
        
        for _ in range(samples_per_bin):
            # 在该bin的频率范围内随机选择频率（对数均匀分布）
            # 使用对数分布以匹配频率感知
            log_freq = rng.uniform(np.log2(freq_low), np.log2(freq_high))
            freq = 2 ** log_freq
            
            # 随机相位
            phase = rng.uniform(0, 2 * np.pi)
            
            # 生成正弦波
            t = np.arange(4096) / 44100.0
            waveform = np.sin(2 * np.pi * freq * t + phase).astype(np.float32) * 0.8
            
            # 添加样本 (真值仍使用该bin_idx)
            generator.add_sample(waveform, bin_idx)
    
    # 完成
    generator.finalize(samples_per_bin, num_bins)


class SanityDataset:
    """Sanity数据集读取器"""
    
    def __init__(self, data_dir: str, preload: bool = False):
        self.data_dir = Path(data_dir)
        self.wave_dir = self.data_dir / 'waveforms'
        self.target_dir = self.data_dir / 'targets'
        
        with open(self.data_dir / 'meta.pkl', 'rb') as f:
            self.meta = pickle.load(f)
        
        self.total_samples = self.meta['total_samples']
        self.num_shards = self.meta['num_shards']
        
        print(f"SanityDataset: {self.data_dir}")
        print(f"  Samples: {self.total_samples}, Shards: {self.num_shards}")
        
        self.preload = preload
        
        if preload:
            print("  Preloading to memory...")
            self.waves = []
            self.indices = []
            self.confs = []
            self.energies = []
            
            for i in range(self.num_shards):
                self.waves.append(np.load(self.wave_dir / f"shard_{i:05d}.npy"))
                self.indices.append(np.load(self.target_dir / f"indices_{i:05d}.npy"))
                self.confs.append(np.load(self.target_dir / f"confs_{i:05d}.npy"))
                self.energies.append(np.load(self.target_dir / f"energies_{i:05d}.npy"))
            
            self.waves = np.concatenate(self.waves, axis=0)
            self.indices = np.concatenate(self.indices, axis=0)
            self.confs = np.concatenate(self.confs, axis=0)
            self.energies = np.concatenate(self.energies, axis=0)
            
            mem = (self.waves.nbytes + self.indices.nbytes + 
                   self.confs.nbytes + self.energies.nbytes)
            print(f"  Memory: {mem / 1024 / 1024:.1f} MB")
        else:
            # 加载分片元数据
            self.shard_sizes = []
            for i in range(self.num_shards):
                wave_path = self.wave_dir / f"shard_{i:05d}.npy"
                with open(wave_path, 'rb') as f:
                    version = np.lib.format.read_magic(f)
                    shape, _, _ = np.lib.format._read_array_header(f, version)
                    self.shard_sizes.append(shape[0])
            
            self.shard_offsets = [0]
            for size in self.shard_sizes[:-1]:
                self.shard_offsets.append(self.shard_offsets[-1] + size)
    
    def __len__(self):
        return self.total_samples
    
    def _expand_sparse(self, indices, confs, energies):
        """展开稀疏表示为完整频谱"""
        target_conf = np.zeros(2048, dtype=np.float32)
        target_energy = np.zeros(2048, dtype=np.float32)
        
        for idx, c, e in zip(indices, confs, energies):
            if c > 0:
                target_conf[idx] = c
                target_energy[idx] = e
        
        return target_conf, target_energy
    
    def __getitem__(self, idx):
        import torch
        
        if self.preload:
            waveform = self.waves[idx]
            indices = self.indices[idx]
            confs = self.confs[idx]
            energies = self.energies[idx]
        else:
            # 找到分片
            for shard_idx, offset in enumerate(self.shard_offsets):
                if idx < offset + self.shard_sizes[shard_idx]:
                    local_idx = idx - offset
                    
                    wave_path = self.wave_dir / f"shard_{shard_idx:05d}.npy"
                    idx_path = self.target_dir / f"indices_{shard_idx:05d}.npy"
                    conf_path = self.target_dir / f"confs_{shard_idx:05d}.npy"
                    energy_path = self.target_dir / f"energies_{shard_idx:05d}.npy"
                    
                    # 使用内存映射
                    waveform = np.load(wave_path, mmap_mode='r')[local_idx]
                    indices = np.load(idx_path, mmap_mode='r')[local_idx]
                    confs = np.load(conf_path, mmap_mode='r')[local_idx]
                    energies = np.load(energy_path, mmap_mode='r')[local_idx]
                    break
        
        target_conf, target_energy = self._expand_sparse(indices, confs, energies)
        
        return {
            'waveform': torch.from_numpy(waveform).unsqueeze(0),
            'target_confidence': torch.from_numpy(target_conf),
            'target_energy': torch.from_numpy(target_energy)
        }


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
    
    args = parser.parse_args()
    
    generate_sanity_dataset(args.output, args.samples_per_bin, args.shard_size_gb)

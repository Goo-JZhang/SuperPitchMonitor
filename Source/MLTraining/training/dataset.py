#!/usr/bin/env python3
"""
数据集读取器 - 支持多种格式

支持的格式:
1. Sanity Separate (推荐): TrainingData/SingleSanity
   - 波形和真值分开存储
   - 真值稀疏压缩
   - 自动分片管理

2. NumPy Shards: TrainingData/sanity_shards
   - 每个分片包含波形和真值
   - 适用于小规模数据集

3. HDF5 (已废弃): 仅用于兼容旧数据
"""

import numpy as np
import torch
from pathlib import Path
from typing import Optional
import pickle


class SanityDataset:
    """
    Sanity数据集读取器 (分离存储格式)
    
    文件结构:
        data_dir/
        ├── meta.pkl
        ├── waveforms/shard_*.npy
        └── targets/{indices,confs,energies}_*.npy
    """
    
    def __init__(self, data_dir: str, preload: bool = True):
        """
        Args:
            data_dir: 数据目录 (如 TrainingData/SingleSanity)
            preload: 是否预加载到内存
        """
        self.data_dir = Path(data_dir)
        self.wave_dir = self.data_dir / 'waveforms'
        self.target_dir = self.data_dir / 'targets'
        
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Dataset not found: {data_dir}")
        
        # 加载元数据
        with open(self.data_dir / 'meta.pkl', 'rb') as f:
            self.meta = pickle.load(f)
        
        self.total_samples = self.meta['total_samples']
        self.num_shards = self.meta['num_shards']
        
        print(f"SanityDataset: {self.data_dir}")
        print(f"  Samples: {self.total_samples}, Shards: {self.num_shards}")
        
        self.preload = preload
        
        if preload:
            self._preload_to_memory()
        else:
            self._setup_mmap()
    
    def _preload_to_memory(self):
        """预加载所有数据到内存"""
        print("  Preloading to memory...")
        
        waves_list = []
        indices_list = []
        confs_list = []
        energies_list = []
        
        for i in range(self.num_shards):
            waves_list.append(np.load(self.wave_dir / f"shard_{i:05d}.npy"))
            indices_list.append(np.load(self.target_dir / f"indices_{i:05d}.npy"))
            confs_list.append(np.load(self.target_dir / f"confs_{i:05d}.npy"))
            energies_list.append(np.load(self.target_dir / f"energies_{i:05d}.npy"))
        
        self.waves = np.concatenate(waves_list, axis=0)
        self.indices = np.concatenate(indices_list, axis=0)
        self.confs = np.concatenate(confs_list, axis=0)
        self.energies = np.concatenate(energies_list, axis=0)
        
        mem = (self.waves.nbytes + self.indices.nbytes + 
               self.confs.nbytes + self.energies.nbytes)
        print(f"  Memory: {mem / 1024 / 1024:.1f} MB")
    
    def _setup_mmap(self):
        """设置内存映射"""
        print("  Using memory mapping...")
        
        # 加载分片大小
        self.shard_sizes = []
        for i in range(self.num_shards):
            wave_path = self.wave_dir / f"shard_{i:05d}.npy"
            with open(wave_path, 'rb') as f:
                version = np.lib.format.read_magic(f)
                shape, _, _ = np.lib.format._read_array_header(f, version)
                self.shard_sizes.append(shape[0])
        
        # 计算偏移
        self.shard_offsets = [0]
        for size in self.shard_sizes[:-1]:
            self.shard_offsets.append(self.shard_offsets[-1] + size)
        
        # 简单缓存
        self.cache = {}
        self.cache_keys = []
        self.cache_size = 100
    
    def __len__(self):
        return self.total_samples
    
    def _expand_sparse(self, indices, confs, energies):
        """展开稀疏表示为完整频谱 [2048]"""
        target_conf = np.zeros(2048, dtype=np.float32)
        target_energy = np.zeros(2048, dtype=np.float32)
        
        for idx, c, e in zip(indices, confs, energies):
            if c > 0:  # 只展开非零值
                target_conf[idx] = c
                target_energy[idx] = e
        
        return target_conf, target_energy
    
    def __getitem__(self, idx):
        if idx < 0 or idx >= self.total_samples:
            raise IndexError(f"Index {idx} out of range [0, {self.total_samples})")
        
        if self.preload:
            # 直接从内存读取
            waveform = self.waves[idx]
            indices = self.indices[idx]
            confs = self.confs[idx]
            energies = self.energies[idx]
        else:
            # 检查缓存
            if idx in self.cache:
                return self.cache[idx]
            
            # 找到分片
            for shard_idx, offset in enumerate(self.shard_offsets):
                if idx < offset + self.shard_sizes[shard_idx]:
                    local_idx = idx - offset
                    
                    # 内存映射读取
                    wave_path = self.wave_dir / f"shard_{shard_idx:05d}.npy"
                    idx_path = self.target_dir / f"indices_{shard_idx:05d}.npy"
                    conf_path = self.target_dir / f"confs_{shard_idx:05d}.npy"
                    energy_path = self.target_dir / f"energies_{shard_idx:05d}.npy"
                    
                    waveform = np.load(wave_path, mmap_mode='r')[local_idx]
                    indices = np.load(idx_path, mmap_mode='r')[local_idx]
                    confs = np.load(conf_path, mmap_mode='r')[local_idx]
                    energies = np.load(energy_path, mmap_mode='r')[local_idx]
                    break
        
        # 展开稀疏表示
        target_conf, target_energy = self._expand_sparse(indices, confs, energies)
        
        result = {
            'waveform': torch.from_numpy(waveform).unsqueeze(0),
            'target_confidence': torch.from_numpy(target_conf),
            'target_energy': torch.from_numpy(target_energy)
        }
        
        # 更新缓存
        if not self.preload:
            self.cache[idx] = result
            self.cache_keys.append(idx)
            if len(self.cache_keys) > self.cache_size:
                del self.cache[self.cache_keys.pop(0)]
        
        return result


class MemoryCachedDataset(torch.utils.data.Dataset):
    """
    通用内存缓存数据集
    自动识别格式并加载
    """
    
    def __init__(self, data_dir: str):
        """
        Args:
            data_dir: 数据目录
        """
        self.data_dir = Path(data_dir)
        
        # 检测格式
        if (self.data_dir / 'waveforms').exists() and (self.data_dir / 'targets').exists():
            # Sanity Separate 格式
            self.dataset = SanityDataset(data_dir, preload=True)
            self.format = 'sanity_separate'
        elif (self.data_dir / 'meta.pkl').exists():
            # 可能是shards格式
            with open(self.data_dir / 'meta.pkl', 'rb') as f:
                meta = pickle.load(f)
            if meta.get('format') == 'sanity_separate':
                self.dataset = SanityDataset(data_dir, preload=True)
                self.format = 'sanity_separate'
            else:
                raise NotImplementedError(f"Unsupported format: {meta.get('format')}")
        else:
            raise ValueError(f"Unknown dataset format in: {data_dir}")
        
        self.total_samples = len(self.dataset)
    
    def __len__(self):
        return self.total_samples
    
    def __getitem__(self, idx):
        return self.dataset[idx]


# 保持向后兼容
PitchDataset = MemoryCachedDataset

#!/usr/bin/env python3
"""
NumPy Shards Dataset - 高效大数据集存储格式
"""

import numpy as np
import torch
from pathlib import Path
from typing import Iterator, Optional
import pickle


class ShardDataset(torch.utils.data.Dataset):
    """
    NumPy分片数据集
    
    特点:
    - 分片存储，每个shard几百MB
    - 内存映射，按需加载
    - 适合20-30GB级数据集
    """
    
    def __init__(self, data_dir: str, cache_size: int = 100):
        """
        Args:
            data_dir: 数据目录 (包含shard_*.npz和meta.pkl)
            cache_size: 样本缓存数量
        """
        self.data_dir = Path(data_dir)
        
        # 加载元数据
        meta_path = self.data_dir / 'meta.pkl'
        if not meta_path.exists():
            raise FileNotFoundError(f"Metadata not found: {meta_path}")
        
        with open(meta_path, 'rb') as f:
            self.meta = pickle.load(f)
        
        self.total_samples = self.meta['total_samples']
        self.num_shards = self.meta['num_shards']
        
        print(f"Loading shards from: {self.data_dir}")
        print(f"  Total samples: {self.total_samples}")
        print(f"  Num shards: {self.num_shards}")
        
        # 加载所有分片（内存映射）
        self.shards = []
        self.shard_sizes = []
        
        for i in range(self.num_shards):
            shard_path = self.data_dir / f"shard_{i:05d}.npz"
            if not shard_path.exists():
                raise FileNotFoundError(f"Shard not found: {shard_path}")
            
            # mmap_mode='r' 只读内存映射，不加载到RAM
            shard = np.load(shard_path, mmap_mode='r')
            self.shards.append(shard)
            self.shard_sizes.append(len(shard['waveforms']))
        
        # 计算每个分片的起始索引
        self.shard_offsets = [0]
        for size in self.shard_sizes[:-1]:
            self.shard_offsets.append(self.shard_offsets[-1] + size)
        
        # 简单LRU缓存
        self.cache = {}
        self.cache_keys = []
        self.cache_size = cache_size
    
    def __len__(self):
        return self.total_samples
    
    def _get_shard_and_local_idx(self, global_idx: int):
        """获取分片索引和本地索引"""
        for shard_idx, offset in enumerate(self.shard_offsets):
            if global_idx < offset + self.shard_sizes[shard_idx]:
                return shard_idx, global_idx - offset
        raise IndexError(f"Index {global_idx} out of range [0, {self.total_samples})")
    
    def __getitem__(self, idx: int):
        """获取单个样本"""
        if idx < 0 or idx >= self.total_samples:
            raise IndexError(f"Index {idx} out of range")
        
        # 检查缓存
        if idx in self.cache:
            return self.cache[idx]
        
        # 从分片读取
        shard_idx, local_idx = self._get_shard_and_local_idx(idx)
        shard = self.shards[shard_idx]
        
        sample = {
            'waveform': np.array(shard['waveforms'][local_idx]),  # 复制到内存
            'target_confidence': np.array(shard['confs'][local_idx]),
            'target_energy': np.array(shard['energies'][local_idx]),
        }
        
        # 更新缓存
        self.cache[idx] = sample
        self.cache_keys.append(idx)
        if len(self.cache_keys) > self.cache_size:
            old_key = self.cache_keys.pop(0)
            del self.cache[old_key]
        
        return sample


class MemoryCachedShardDataset(torch.utils.data.Dataset):
    """
    完全载入内存的ShardDataset (适合小数据集<5GB)
    """
    
    def __init__(self, data_dir: str, device='cpu'):
        """
        Args:
            data_dir: 数据目录
            device: 预加载到指定设备
        """
        self.data_dir = Path(data_dir)
        
        with open(self.data_dir / 'meta.pkl', 'rb') as f:
            self.meta = pickle.load(f)
        
        self.total_samples = self.meta['total_samples']
        self.num_shards = self.meta['num_shards']
        
        print(f"Loading dataset into memory: {self.data_dir}")
        print(f"  Total samples: {self.total_samples}")
        
        # 加载所有数据到内存
        waveforms = []
        confs = []
        energies = []
        
        for i in range(self.num_shards):
            shard_path = self.data_dir / f"shard_{i:05d}.npz"
            shard = np.load(shard_path)
            waveforms.append(shard['waveforms'])
            confs.append(shard['confs'])
            energies.append(shard['energies'])
        
        # 合并
        self.waveforms = np.concatenate(waveforms, axis=0)
        self.confs = np.concatenate(confs, axis=0)
        self.energies = np.concatenate(energies, axis=0)
        
        # 计算内存占用
        mem_mb = (self.waveforms.nbytes + self.confs.nbytes + self.energies.nbytes) / 1024 / 1024
        print(f"  Memory usage: {mem_mb:.1f} MB")
        
        # 可选：预加载到GPU
        if device != 'cpu' and torch.cuda.is_available():
            print(f"  Preloading to GPU...")
            self.waveforms = torch.from_numpy(self.waveforms).to(device)
            self.confs = torch.from_numpy(self.confs).to(device)
            self.energies = torch.from_numpy(self.energies).to(device)
    
    def __len__(self):
        return self.total_samples
    
    def __getitem__(self, idx):
        if isinstance(self.waveforms, torch.Tensor):
            return {
                'waveform': self.waveforms[idx].unsqueeze(0),
                'target_confidence': self.confs[idx],
                'target_energy': self.energies[idx]
            }
        else:
            return {
                'waveform': torch.from_numpy(self.waveforms[idx]).unsqueeze(0),
                'target_confidence': torch.from_numpy(self.confs[idx]),
                'target_energy': torch.from_numpy(self.energies[idx])
            }

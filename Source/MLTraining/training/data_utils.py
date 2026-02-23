#!/usr/bin/env python3
"""
数据集加载工具 - 支持多种数据格式

支持的格式:
1. 当前格式 (推荐):
   - meta.json
   - waveforms/shard_*.npy
   - targets/shard_*.npz (包含 confs, energies)

2. 旧格式 (兼容):
   - meta.json/pkl
   - waveforms/shard_*.npy
   - targets/{indices,confs,energies}_*.npy (分开存储)
"""

import json
import numpy as np
import torch
from pathlib import Path
from typing import Optional, Dict, Any


class DatasetReader(torch.utils.data.Dataset):
    """
    通用数据集读取器
    
    自动检测数据格式并加载
    """
    
    def __init__(self, 
                 data_dir: str, 
                 preload: bool = False, 
                 device: str = 'cpu',
                 cache_size: int = 100):
        """
        Args:
            data_dir: 数据目录
            preload: 是否预加载到内存
            device: 预加载到的设备 ('cpu' 或 'cuda')
            cache_size: 内存映射模式下的缓存大小
        """
        self.data_dir = Path(data_dir)
        self.wave_dir = self.data_dir / 'waveforms'
        self.target_dir = self.data_dir / 'targets'
        self.preload = preload
        self.device = device
        self.cache_size = cache_size
        
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Dataset not found: {data_dir}")
        
        # 加载元数据并检测格式
        self._load_metadata()
        
        print(f"DatasetReader: {self.data_dir}")
        print(f"  Format: {self.format}")
        print(f"  Samples: {self.total_samples}, Shards: {self.num_shards}")
        
        if preload:
            self._preload_to_memory()
        else:
            self._setup_mmap()
    
    def _load_metadata(self):
        """加载元数据并检测格式"""
        # 尝试 JSON 格式
        meta_json = self.data_dir / 'meta.json'
        if meta_json.exists():
            with open(meta_json, 'r', encoding='utf-8') as f:
                self.meta = json.load(f)
        else:
            # 尝试旧 pickle 格式
            meta_pkl = self.data_dir / 'meta.pkl'
            if meta_pkl.exists():
                import pickle
                with open(meta_pkl, 'rb') as f:
                    self.meta = pickle.load(f)
            else:
                raise FileNotFoundError(f"Metadata not found in {self.data_dir}")
        
        self.total_samples = self.meta['total_samples']
        self.num_shards = self.meta['num_shards']
        
        # 检测数据格式
        if self._check_current_format():
            self.format = 'current'
        elif self._check_legacy_format():
            self.format = 'legacy'
        else:
            raise ValueError(f"Unknown data format in {self.data_dir}")
    
    def _check_current_format(self) -> bool:
        """检查是否为当前格式 (targets/*.npz)"""
        if not self.target_dir.exists():
            return False
        npz_files = list(self.target_dir.glob('shard_*.npz'))
        return len(npz_files) > 0
    
    def _check_legacy_format(self) -> bool:
        """检查是否为旧格式 (targets/{confs,energies}_*.npy)"""
        if not self.target_dir.exists():
            return False
        confs_files = list(self.target_dir.glob('confs_*.npy'))
        return len(confs_files) > 0
    
    def _preload_to_memory(self):
        """预加载所有数据到内存"""
        print("  Preloading to memory...")
        
        self.waves = []
        self.confs = []
        self.energies = []
        
        for i in range(self.num_shards):
            # 加载波形
            wave_path = self.wave_dir / f"shard_{i:05d}.npy"
            self.waves.append(np.load(wave_path))
            
            # 加载真值
            if self.format == 'current':
                target_path = self.target_dir / f"shard_{i:05d}.npz"
                target = np.load(target_path)
                self.confs.append(target['confs'])
                self.energies.append(target['energies'])
            else:  # legacy
                conf_path = self.target_dir / f"confs_{i:05d}.npy"
                energy_path = self.target_dir / f"energies_{i:05d}.npy"
                self.confs.append(np.load(conf_path))
                self.energies.append(np.load(energy_path))
        
        # 合并
        self.waves = np.concatenate(self.waves, axis=0)
        self.confs = np.concatenate(self.confs, axis=0)
        self.energies = np.concatenate(self.energies, axis=0)
        
        mem_mb = (self.waves.nbytes + self.confs.nbytes + self.energies.nbytes) / 1024 / 1024
        print(f"  Memory usage: {mem_mb:.1f} MB")
        
        # 移动到 GPU
        if self.device != 'cpu' and torch.cuda.is_available():
            print(f"  Moving to {self.device}...")
            self.waves = torch.from_numpy(self.waves).to(self.device)
            self.confs = torch.from_numpy(self.confs).to(self.device)
            self.energies = torch.from_numpy(self.energies).to(self.device)
    
    def _setup_mmap(self):
        """设置内存映射模式"""
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
        
        # 缓存
        self.cache = {}
        self.cache_keys = []
    
    def __len__(self):
        return self.total_samples
    
    def __getitem__(self, idx):
        if idx < 0 or idx >= self.total_samples:
            raise IndexError(f"Index {idx} out of range [0, {self.total_samples})")
        
        if self.preload:
            # 从内存直接读取
            if isinstance(self.waves, torch.Tensor):
                return {
                    'waveform': self.waves[idx].unsqueeze(0),
                    'target_confidence': self.confs[idx],
                    'target_energy': self.energies[idx]
                }
            else:
                return {
                    'waveform': torch.from_numpy(self.waves[idx]).unsqueeze(0),
                    'target_confidence': torch.from_numpy(self.confs[idx]),
                    'target_energy': torch.from_numpy(self.energies[idx])
                }
        
        # 检查缓存
        if idx in self.cache:
            return self.cache[idx]
        
        # 找到分片
        for shard_idx, offset in enumerate(self.shard_offsets):
            if idx < offset + self.shard_sizes[shard_idx]:
                local_idx = idx - offset
                
                # 加载波形
                wave_path = self.wave_dir / f"shard_{shard_idx:05d}.npy"
                waveform = np.load(wave_path, mmap_mode='r')[local_idx]
                
                # 加载真值
                if self.format == 'current':
                    target_path = self.target_dir / f"shard_{shard_idx:05d}.npz"
                    target = np.load(target_path)
                    confs = target['confs'][local_idx]
                    energies = target['energies'][local_idx]
                else:  # legacy
                    conf_path = self.target_dir / f"confs_{shard_idx:05d}.npy"
                    energy_path = self.target_dir / f"energies_{shard_idx:05d}.npy"
                    confs = np.load(conf_path, mmap_mode='r')[local_idx]
                    energies = np.load(energy_path, mmap_mode='r')[local_idx]
                
                result = {
                    'waveform': torch.from_numpy(waveform.copy()).unsqueeze(0),
                    'target_confidence': torch.from_numpy(confs.astype(np.float32).copy()),
                    'target_energy': torch.from_numpy(energies.astype(np.float32).copy())
                }
                
                # 更新缓存
                self.cache[idx] = result
                self.cache_keys.append(idx)
                if len(self.cache_keys) > self.cache_size:
                    del self.cache[self.cache_keys.pop(0)]
                
                return result
        
        raise IndexError(f"Index {idx} out of range")


def load_dataset(data_dir: str, 
                 preload: bool = False, 
                 device: str = 'cpu') -> torch.utils.data.Dataset:
    """
    便捷函数：加载数据集
    
    Args:
        data_dir: 数据目录
        preload: 是否预加载到内存
        device: 预加载到的设备
    
    Returns:
        Dataset 对象
    """
    return DatasetReader(data_dir, preload=preload, device=device)


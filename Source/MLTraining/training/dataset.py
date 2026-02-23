#!/usr/bin/env python3
"""
数据集读取器 - 支持多目录和自动流式加载

功能:
- 多目录数据合并加载
- 自动内存检测（超过阈值使用流式）
- 支持大数据集的迭代器模式
"""

import json
import numpy as np
import torch
from pathlib import Path
from typing import Optional, Dict, Any, List, Union
import psutil


class DatasetReader(torch.utils.data.Dataset):
    """
    通用数据集读取器 - 单目录
    
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
    
    def _normalize_waveform(self, waveform):
        """
        归一化波形：Z-score 归一化 (均值为0，方差为1)
        与 C++ 推理代码保持一致
        """
        if isinstance(waveform, torch.Tensor):
            # Step 1: 去直流偏移 (mean = 0)
            mean = torch.mean(waveform)
            waveform = waveform - mean
            
            # Step 2: Z-score 归一化 (std = 1)
            std = torch.std(waveform)
            if std > 1e-8:
                waveform = waveform / std
        else:
            # Step 1: 去直流偏移 (mean = 0)
            waveform = waveform - np.mean(waveform)
            
            # Step 2: Z-score 归一化 (std = 1)
            std = np.std(waveform)
            if std > 1e-8:
                waveform = waveform / std
        return waveform
    
    def __getitem__(self, idx):
        if idx < 0 or idx >= self.total_samples:
            raise IndexError(f"Index {idx} out of range [0, {self.total_samples})")
        
        if self.preload:
            # 从内存直接读取
            if isinstance(self.waves, torch.Tensor):
                waveform = self.waves[idx]
                # 峰值归一化（与 C++ 推理一致）
                max_amp = torch.max(torch.abs(waveform))
                if max_amp > 1e-8:
                    waveform = waveform / max_amp
                return {
                    'waveform': waveform.unsqueeze(0),
                    'target_confidence': self.confs[idx],
                    'target_energy': self.energies[idx]
                }
            else:
                waveform = self.waves[idx]
                waveform = self._normalize_waveform(waveform)
                return {
                    'waveform': torch.from_numpy(waveform).unsqueeze(0),
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
                
                # 峰值归一化（与 C++ 推理一致）
                waveform = self._normalize_waveform(waveform)
                
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


class MultiDataset(torch.utils.data.Dataset):
    """
    多目录数据集合并加载器
    
    自动检测内存使用，决定是否流式加载
    """
    
    def __init__(self,
                 root_dir: str,
                 subdirs: Optional[List[str]] = None,
                 preload: Optional[bool] = None,
                 max_memory_gb: float = 4.0,
                 device: str = 'cpu'):
        """
        Args:
            root_dir: 数据根目录
            subdirs: 子目录列表，None表示遍历所有
            preload: 是否预加载到内存，None表示自动决定
            max_memory_gb: 超过此内存使用流式加载
            device: 预加载到的设备
        """
        self.root_dir = Path(root_dir)
        self.max_memory_gb = max_memory_gb
        self.device = device
        
        # 发现数据集
        self.data_dirs = self._discover_datasets(subdirs)
        if not self.data_dirs:
            raise ValueError(f"No datasets found in {root_dir}")
        
        print(f"\nMultiDataset: Found {len(self.data_dirs)} dataset(s)")
        for d in self.data_dirs:
            print(f"  - {d.name}")
        
        # 计算总大小，决定加载方式
        total_samples, estimated_gb = self._estimate_size()
        print(f"\n  Total samples: {total_samples:,}")
        print(f"  Estimated memory: {estimated_gb:.1f} GB")
        
        # 自动决定加载方式
        if preload is None:
            available_gb = psutil.virtual_memory().available / (1024**3)
            preload = estimated_gb < min(max_memory_gb, available_gb * 0.5)
            print(f"  Available memory: {available_gb:.1f} GB")
            print(f"  Auto-select preload: {preload}")
        
        self.preload = preload
        
        # 加载所有数据集
        self.datasets = []
        self.offsets = [0]
        
        for data_dir in self.data_dirs:
            ds = DatasetReader(str(data_dir), preload=preload, device=device)
            self.datasets.append(ds)
            self.offsets.append(self.offsets[-1] + len(ds))
        
        self.total_samples = self.offsets[-1]
        print(f"\n  Final total samples: {self.total_samples:,}")
    
    def _discover_datasets(self, subdirs: Optional[List[str]]) -> List[Path]:
        """发现数据集目录"""
        if subdirs is not None:
            # 使用指定的子目录
            dirs = [self.root_dir / d for d in subdirs]
            return [d for d in dirs if d.exists() and (d / 'meta.json').exists()]
        else:
            # 遍历所有子目录
            dirs = []
            for path in self.root_dir.iterdir():
                if path.is_dir() and (path / 'meta.json').exists():
                    dirs.append(path)
            return sorted(dirs)
    
    def _estimate_size(self) -> tuple:
        """估算数据大小"""
        total_samples = 0
        total_bytes = 0
        
        for data_dir in self.data_dirs:
            with open(data_dir / 'meta.json', 'r') as f:
                meta = json.load(f)
            
            samples = meta.get('total_samples', 0)
            total_samples += samples
            
            # 估算: waveform (4096 * 4 bytes) + targets (2048 * 2 * 2 bytes)
            bytes_per_sample = 4096 * 4 + 2048 * 4
            total_bytes += samples * bytes_per_sample
        
        estimated_gb = total_bytes / (1024**3)
        return total_samples, estimated_gb
    
    def __len__(self):
        return self.total_samples
    
    def __getitem__(self, idx):
        """获取样本，自动路由到对应数据集"""
        if idx < 0 or idx >= self.total_samples:
            raise IndexError(f"Index {idx} out of range [0, {self.total_samples})")
        
        # 找到对应的数据集
        for i, offset in enumerate(self.offsets[:-1]):
            if idx < self.offsets[i + 1]:
                local_idx = idx - offset
                return self.datasets[i][local_idx]
        
        raise IndexError(f"Index {idx} out of range")
    
    def get_dataset_info(self) -> Dict[str, Any]:
        """获取数据集信息"""
        info = {
            'root_dir': str(self.root_dir),
            'num_datasets': len(self.datasets),
            'total_samples': self.total_samples,
            'preload': self.preload,
            'datasets': []
        }
        
        for i, ds in enumerate(self.datasets):
            info['datasets'].append({
                'name': ds.data_dir.name,
                'samples': len(ds),
                'offset': self.offsets[i]
            })
        
        return info


def load_dataset(data_dirs: Union[str, List[str]],
                 root_dir: Optional[str] = None,
                 preload: Optional[bool] = None,
                 max_memory_gb: float = 4.0,
                 device: str = 'cpu') -> torch.utils.data.Dataset:
    """
    便捷函数：加载数据集
    
    Args:
        data_dirs: 数据目录或目录列表，如果是相对路径则从 root_dir 解析
        root_dir: 数据根目录（当 data_dirs 为相对路径时使用）
        preload: 是否预加载到内存，None表示自动决定
        max_memory_gb: 超过此内存使用流式加载
        device: 预加载到的设备
    
    Returns:
        Dataset 对象
    
    Examples:
        # 单目录
        ds = load_dataset('/path/to/data')
        
        # 多目录
        ds = load_dataset(['/path/to/data1', '/path/to/data2'])
        
        # 从根目录加载指定子目录
        ds = load_dataset(['SingleSanity', 'NoiseDataset'], root_dir='TrainingData')
        
        # 从根目录加载所有子目录
        ds = load_dataset(None, root_dir='TrainingData')
    """
    # 处理 data_dirs 为 None 的情况（从 root_dir 加载所有）
    if data_dirs is None:
        if root_dir is None:
            raise ValueError("Either data_dirs or root_dir must be provided")
        return MultiDataset(root_dir, subdirs=None, preload=preload,
                          max_memory_gb=max_memory_gb, device=device)
    
    # 处理字符串（单目录）
    if isinstance(data_dirs, str):
        # 如果是相对路径，从 root_dir 解析
        if root_dir is not None and not Path(data_dirs).is_absolute():
            data_dirs = Path(root_dir) / data_dirs
        return DatasetReader(data_dirs, preload=preload or False, device=device)
    
    # 处理列表（多目录）
    if isinstance(data_dirs, list):
        # 如果有 root_dir，解析相对路径
        if root_dir is not None:
            resolved_dirs = []
            for d in data_dirs:
                path = Path(d)
                if not path.is_absolute():
                    path = Path(root_dir) / d
                resolved_dirs.append(path)
            data_dirs = resolved_dirs
        
        # 检查是否只有一个目录
        if len(data_dirs) == 1:
            return DatasetReader(data_dirs[0], preload=preload or False, device=device)
        
        # 多个目录使用 MultiDataset
        # 找到共同根目录（简化处理，使用第一个目录的父目录）
        common_root = Path(data_dirs[0]).parent
        subdirs = [Path(d).name for d in data_dirs]
        
        return MultiDataset(str(common_root), subdirs=subdirs, preload=preload,
                          max_memory_gb=max_memory_gb, device=device)
    
    raise ValueError(f"Invalid data_dirs type: {type(data_dirs)}")


__all__ = ['DatasetReader', 'MultiDataset', 'load_dataset']

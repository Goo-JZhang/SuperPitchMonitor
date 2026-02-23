#!/usr/bin/env python3
"""
数据集写入器 - 通用数据存储/分片管理工具

功能:
- 分片管理: 自动按大小分片
- 格式支持: .npy (不压缩), .npz (压缩), 可选 .indices.npy (调试)
- 内存映射友好: 波形数据保持 .npy 格式
- 真值压缩: confs + energies 合并为 .npz

使用示例:
    writer = DatasetWriter('output_dir', shard_size_gb=1.0)
    
    # 添加样本
    writer.add_sample(
        waveform=waveform,  # [4096] float32
        confs=confs,        # [2048] float16/float32
        energies=energies,  # [2048] float16/float32
        indices=None        # 可选，调试用的稀疏索引
    )
    
    # 完成写入
    writer.finalize(meta_dict={'samples_per_bin': 10})
"""

import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Optional, Any
import os


class DatasetWriter:
    """
    数据集写入器
    
    管理数据分片、存储格式和元数据
    """
    
    def __init__(self, 
                 output_dir: str,
                 shard_size_gb: float = 1.0,
                 save_indices: bool = False,
                 confs_dtype=np.float16,
                 energies_dtype=np.float16):
        """
        Args:
            output_dir: 输出目录
            shard_size_gb: 触发分片的波形数据大小(GB)
            save_indices: 是否保存稀疏索引（调试用）
            confs_dtype: confidence数据类型
            energies_dtype: energy数据类型
        """
        self.output_path = Path(output_dir)
        self.wave_dir = self.output_path / 'waveforms'
        self.target_dir = self.output_path / 'targets'
        
        # 创建目录
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.wave_dir.mkdir(exist_ok=True)
        self.target_dir.mkdir(exist_ok=True)
        
        self.shard_size_bytes = int(shard_size_gb * 1024 * 1024 * 1024)
        self.save_indices = save_indices
        self.confs_dtype = confs_dtype
        self.energies_dtype = energies_dtype
        
        # 分片状态
        self.current_shard = 0
        self.waveforms = []
        self.confs_list = []
        self.energies_list = []
        self.indices_list = []
        self.current_waveform_bytes = 0
        self.total_samples = 0
        
        # 统计
        self.total_wave_bytes = 0
        self.total_target_bytes = 0
    
    def add_sample(self,
                   waveform: np.ndarray,
                   confs: np.ndarray,
                   energies: np.ndarray,
                   indices: Optional[List[int]] = None):
        """
        添加单个样本到缓冲
        
        Args:
            waveform: [4096] float32 波形数据
            confs: [2048] confidence 频谱
            energies: [2048] energy 频谱
            indices: 可选，稀疏索引列表（调试用）
        """
        # 确保数据类型正确
        waveform = np.asarray(waveform, dtype=np.float32)
        confs = np.asarray(confs, dtype=self.confs_dtype)
        energies = np.asarray(energies, dtype=self.energies_dtype)
        
        self.waveforms.append(waveform)
        self.confs_list.append(confs)
        self.energies_list.append(energies)
        
        if self.save_indices and indices is not None:
            self.indices_list.append(indices)
        
        self.current_waveform_bytes += waveform.nbytes
        
        # 检查是否触发分片
        if self.current_waveform_bytes >= self.shard_size_bytes:
            self._save_current_shard()
    
    def add_batch(self,
                  waveforms: List[np.ndarray],
                  confs_list: List[np.ndarray],
                  energies_list: List[np.ndarray],
                  indices_list: Optional[List[List[int]]] = None):
        """
        批量添加样本
        
        Args:
            waveforms: 波形列表
            confs_list: confidence列表
            energies_list: energy列表
            indices_list: 可选，稀疏索引列表
        """
        for i in range(len(waveforms)):
            indices = indices_list[i] if indices_list else None
            self.add_sample(waveforms[i], confs_list[i], energies_list[i], indices)
    
    def _save_current_shard(self):
        """保存当前分片到磁盘"""
        if not self.waveforms:
            return
        
        n_samples = len(self.waveforms)
        
        # 1. 保存波形 [N, 4096] float32 (不压缩，支持内存映射)
        wave_path = self.wave_dir / f"shard_{self.current_shard:05d}.npy"
        wave_array = np.array(self.waveforms, dtype=np.float32)
        np.save(wave_path, wave_array)
        self.total_wave_bytes += wave_path.stat().st_size
        
        # 2. 保存真值 [N, 2048] float16 (压缩为npz)
        target_path = self.target_dir / f"shard_{self.current_shard:05d}.npz"
        confs_array = np.array(self.confs_list, dtype=self.confs_dtype)
        energies_array = np.array(self.energies_list, dtype=self.energies_dtype)
        
        np.savez_compressed(target_path, confs=confs_array, energies=energies_array)
        self.total_target_bytes += target_path.stat().st_size
        
        # 3. 可选：保存稀疏索引（调试用）
        if self.save_indices and self.indices_list:
            indices_path = self.target_dir / f"shard_{self.current_shard:05d}.indices.npy"
            max_len = max(len(idx) for idx in self.indices_list)
            indices_padded = []
            for idxs in self.indices_list:
                padded = idxs + [0] * (max_len - len(idxs))
                indices_padded.append(padded[:max_len])
            np.save(indices_path, np.array(indices_padded, dtype=np.int16))
        
        # 打印信息
        compression_ratio = target_path.stat().st_size / (confs_array.nbytes + energies_array.nbytes) * 100
        print(f"  Shard {self.current_shard}: {n_samples} samples "
              f"(wave: {wave_path.stat().st_size/1024/1024:.1f}MB, "
              f"target: {target_path.stat().st_size/1024:.0f}KB, "
              f"compression: {compression_ratio:.1f}%)")
        
        self.total_samples += n_samples
        self.current_shard += 1
        
        # 清空缓冲
        self._clear_buffer()
    
    def _clear_buffer(self):
        """清空缓冲"""
        self.waveforms = []
        self.confs_list = []
        self.energies_list = []
        self.indices_list = []
        self.current_waveform_bytes = 0
    
    def finalize(self, meta_dict: Optional[Dict[str, Any]] = None):
        """
        完成写入，保存元数据
        
        Args:
            meta_dict: 额外的元数据字典
        """
        self._save_current_shard()
        
        # 构建元数据
        meta = {
            'total_samples': self.total_samples,
            'num_shards': self.current_shard,
            'shard_size_gb': self.shard_size_bytes / (1024 ** 3),
            'format': 'dataset',
            'target_format': 'dense_compressed',
            'confs_dtype': str(self.confs_dtype),
            'energies_dtype': str(self.energies_dtype),
            'has_indices': self.save_indices,
        }
        
        if meta_dict:
            meta.update(meta_dict)
        
        # 保存元数据 (UTF-8 编码的 JSON)
        with open(self.output_path / 'meta.json', 'w', encoding='utf-8') as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
        
        # 打印统计
        print(f"\nDataset saved to: {self.output_path}")
        print(f"  Total samples: {self.total_samples}")
        print(f"  Num shards: {self.current_shard}")
        print(f"  Waveforms: {self.total_wave_bytes / 1024 / 1024:.1f} MB")
        print(f"  Targets (compressed): {self.total_target_bytes / 1024 / 1024:.2f} MB")
        print(f"  Total: {(self.total_wave_bytes + self.total_target_bytes) / 1024 / 1024:.1f} MB")
    
    def __enter__(self):
        """上下文管理器入口"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口，自动finalize"""
        if exc_type is None:
            self.finalize()
        return False


if __name__ == "__main__":
    # 测试 DatasetWriter
    print("=" * 60)
    print("DatasetWriter Test")
    print("=" * 60)
    
    # 从 generate_note 导入测试数据生成函数
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from generate_note import generate_single_note
    
    rng = np.random.RandomState(42)
    
    with DatasetWriter('test_output', shard_size_gb=0.01, save_indices=True) as writer:
        for i in range(100):
            sample = generate_single_note(bin_idx=i % 2048, rng=rng)
            writer.add_sample(
                waveform=sample['waveform'],
                confs=sample['confs'],
                energies=sample['energies'],
                indices=sample['indices']
            )
    
    print("\nTest completed!")

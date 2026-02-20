#!/usr/bin/env python3
"""
频域能量计算器
基于音色谐波结构直接计算能量，无需时域合成
"""

import numpy as np
from typing import List, Dict, Tuple
from dataclasses import dataclass

from timbre_loader import get_timbre_loader


@dataclass
class Note:
    """音符定义"""
    frequency: float      # Hz
    velocity: float       # 0.0-1.0
    timbre_id: int        # 音色ID
    onset_idx: int        # 相对窗口起点
    offset_idx: int       # 相对窗口起点
    phase: float = 0.0    # 起始相位
    
    @property
    def note_type(self) -> int:
        """判断四类音类型"""
        window_size = 4096
        if self.onset_idx <= 0 and self.offset_idx >= window_size:
            return 4  # 第四类：全程持续
        elif self.onset_idx > 0 and self.offset_idx >= window_size:
            return 2  # 第二类：窗口内开始，持续到结束
        elif self.onset_idx <= 0 and self.offset_idx < window_size:
            return 1  # 第一类：窗口前开始，窗口内结束
        else:
            return 3  # 第三类：窗口内开始并结束


class FrequencyDomainEnergyCalculator:
    """
    频域能量计算器
    
    核心公式:
    E = unit_energy * velocity^2 * adsr_factor * truncation_penalty
    
    其中:
    - unit_energy: 音色预计算的单位能量 (sum(|harmonic|^2))
    - adsr_factor: 窗口可见区域的平均ADS包络值
    - truncation_penalty: 第一/三类音的能量惩罚
    """
    
    def __init__(self, window_samples: int = 4096):
        self.window_samples = window_samples
        self.timbre_loader = get_timbre_loader()
        
        # 预计算对数频率到bin的映射参数
        self.log_min = np.log2(20.0)
        self.log_max = np.log2(5000.0)
        self.num_bins = 2048
    
    def freq_to_bin(self, frequency: float) -> float:
        """频率到bin索引（连续值，支持插值）"""
        log_freq = np.log2(frequency)
        bin_idx = (log_freq - self.log_min) / (self.log_max - self.log_min) * (self.num_bins - 1)
        return bin_idx
    
    def bin_to_freq(self, bin_idx: float) -> float:
        """bin索引到频率"""
        log_bin = self.log_min + (bin_idx / (self.num_bins - 1)) * (self.log_max - self.log_min)
        return 2 ** log_bin
    
    def compute_adsr_factor(self, 
                           note: Note, 
                           attack_samples: int = 441,
                           decay_samples: int = 4410) -> float:
        """
        计算ADSR在窗口可见区域的平均能量因子
        
        简化模型:
        - Attack: 线性上升 (0→1), 默认10ms @ 44.1kHz = 441 samples
        - Decay: 指数衰减 (1→sustain), 默认100ms
        - Sustain: 假设为0.3 (钢琴类) 或 1.0 (管风琴类)
        """
        # 确定可见范围
        visible_start = max(note.onset_idx, 0)
        visible_end = min(note.offset_idx, self.window_samples)
        
        if visible_end <= visible_start:
            return 0.0
        
        # 计算相对于音符onset的时间
        note_start = visible_start - note.onset_idx
        note_end = visible_end - note.onset_idx
        
        # 简化的ADSR包络计算（假设sustain=0.7）
        sustain_level = 0.7
        
        def envelope_at(t):
            if t < 0:
                return 0.0
            elif t < attack_samples:
                # Attack: 线性上升
                return t / attack_samples
            else:
                # Decay: 指数衰减到sustain
                t_decay = t - attack_samples
                decay_factor = np.exp(-t_decay / decay_samples)
                return sustain_level + (1.0 - sustain_level) * decay_factor
        
        # 使用Simpson法则近似积分
        e_start = envelope_at(note_start) ** 2
        e_end = envelope_at(note_end) ** 2
        e_mid = envelope_at((note_start + note_end) / 2) ** 2
        
        avg_envelope = (e_start + 4 * e_mid + e_end) / 6
        
        return avg_envelope
    
    def compute_note_energy(self, note: Note) -> float:
        """
        计算单个音符的能量
        
        Returns:
            energy: 该音符在窗口内的能量值
        """
        # 获取音色单位能量
        timbre = self.timbre_loader.get_timbre(note.timbre_id)
        unit_energy = timbre['unit_energy']
        
        # 基础能量 = unit_energy * velocity^2
        base_energy = unit_energy * (note.velocity ** 2)
        
        # ADSR因子
        adsr_factor = self.compute_adsr_factor(note)
        
        # 类型相关处理
        note_type = note.note_type
        
        if note_type in [2, 4]:
            # 第二/四类：完整能量
            energy = base_energy * adsr_factor
        else:
            # 第一/三类：能量打折
            visible_start = max(note.onset_idx, 0)
            visible_end = min(note.offset_idx, self.window_samples)
            visible_ratio = (visible_end - visible_start) / self.window_samples
            
            truncation_penalty = 0.5  # 额外50%惩罚
            energy = base_energy * adsr_factor * visible_ratio * truncation_penalty
        
        return energy
    
    def compute_energy_spectrum(self, notes: List[Note]) -> np.ndarray:
        """
        计算多音的能量频谱目标
        
        Returns:
            target_energy: [2048] 每个bin的能量占比
        """
        if not notes:
            return np.zeros(self.num_bins, dtype=np.float32)
        
        # 计算每个音的能量
        energies = []
        bin_indices = []
        
        for note in notes:
            energy = self.compute_note_energy(note)
            bin_idx = self.freq_to_bin(note.frequency)
            
            energies.append(energy)
            bin_indices.append(bin_idx)
        
        energies = np.array(energies)
        total_energy = np.sum(energies)
        
        if total_energy == 0:
            return np.zeros(self.num_bins, dtype=np.float32)
        
        # 归一化能量占比
        ratios = energies / total_energy
        
        # 生成能量频谱（高斯平滑）
        target_energy = np.zeros(self.num_bins)
        
        for bin_idx, ratio in zip(bin_indices, ratios):
            # 高斯分布到附近bins
            center = int(round(bin_idx))
            sigma = 1.5  # bins
            
            # 影响范围：±3sigma
            start = max(0, int(center - 3 * sigma))
            end = min(self.num_bins, int(center + 3 * sigma) + 1)
            
            for i in range(start, end):
                # 高斯权重
                weight = np.exp(-0.5 * ((i - bin_idx) / sigma) ** 2)
                target_energy[i] += weight * ratio
        
        return target_energy.astype(np.float32)
    
    def compute_confidence_spectrum(self, notes: List[Note]) -> np.ndarray:
        """
        计算置信度频谱目标（四类音策略）
        
        Returns:
            target_conf: [2048] 每个bin的置信度
        """
        target_conf = np.zeros(self.num_bins)
        
        for note in notes:
            note_type = note.note_type
            
            # 确定置信度权重
            if note_type in [2, 4]:
                weight = 1.0
            else:
                # 第一/三类：软标签
                visible_start = max(note.onset_idx, 0)
                visible_end = min(note.offset_idx, self.window_samples)
                visible_ratio = (visible_end - visible_start) / self.window_samples
                weight = min(visible_ratio * 0.5, 0.3)
            
            bin_idx = self.freq_to_bin(note.frequency)
            center = int(round(bin_idx))
            
            # 高斯平滑
            sigma = 1.5
            start = max(0, int(center - 3 * sigma))
            end = min(self.num_bins, int(center + 3 * sigma) + 1)
            
            for i in range(start, end):
                gaussian_weight = np.exp(-0.5 * ((i - bin_idx) / sigma) ** 2)
                target_conf[i] = max(target_conf[i], weight * gaussian_weight)
        
        # 裁剪到[0, 1]
        target_conf = np.clip(target_conf, 0, 1)
        
        return target_conf.astype(np.float32)


if __name__ == "__main__":
    # 测试
    calc = FrequencyDomainEnergyCalculator()
    
    # 测试第四类音（全程持续）
    note1 = Note(
        frequency=440.0,
        velocity=1.0,
        timbre_id=0,  # sine
        onset_idx=0,
        offset_idx=8192,
        phase=0.0
    )
    
    print(f"Note 1 (Type {note1.note_type}):")
    print(f"  Frequency: {note1.frequency} Hz -> Bin {calc.freq_to_bin(note1.frequency):.2f}")
    print(f"  Energy: {calc.compute_note_energy(note1):.4f}")
    
    # 测试第一类音（窗口内结束）
    note2 = Note(
        frequency=261.63,
        velocity=0.8,
        timbre_id=2,  # sawtooth
        onset_idx=-512,
        offset_idx=2048,
        phase=0.0
    )
    
    print(f"\nNote 2 (Type {note2.note_type}):")
    print(f"  Frequency: {note2.frequency} Hz -> Bin {calc.freq_to_bin(note2.frequency):.2f}")
    print(f"  Energy: {calc.compute_note_energy(note2):.4f}")
    
    # 测试多音能量谱
    notes = [note1, note2]
    energy_spec = calc.compute_energy_spectrum(notes)
    conf_spec = calc.compute_confidence_spectrum(notes)
    
    print(f"\nMulti-note spectrum:")
    print(f"  Energy spectrum max: {energy_spec.max():.4f}")
    print(f"  Energy spectrum sum: {energy_spec.sum():.4f}")
    print(f"  Confidence spectrum max: {conf_spec.max():.4f}")

#!/usr/bin/env python3
"""
频域能量计算器 (声压域版本 - 修正后)

核心物理模型:
音色表的谐波信息是声压域的傅里叶系数 Bn = nf * An
因此能量计算直接使用 Σ|Bn|²，无需额外频率因子

信号能量: E = velocity² × unit_energy × ADSR × truncation
其中 unit_energy = Σ|Bn|² (已预计算)

噪声能量: 同样基于声压域，需要重新计算能量系数
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

from timbre_loader import get_timbre_loader


class NoiseType(Enum):
    """噪声类型"""
    WHITE = "white"
    PINK = "pink"
    BROWN = "brown"
    BLUE = "blue"
    VIOLET = "violet"


@dataclass
class Note:
    """音符定义"""
    frequency: float      # Hz (基频，仅用于bin定位)
    velocity: float       # 0.0-1.0 (声压振幅)
    timbre_id: int        # 音色ID
    onset_idx: int        # 相对窗口起点
    offset_idx: int       # 相对窗口起点
    phase: float = 0.0    # 起始相位
    
    @property
    def note_type(self) -> int:
        """判断四类音类型"""
        window_size = 4096
        if self.onset_idx <= 0 and self.offset_idx >= window_size:
            return 4
        elif self.onset_idx > 0 and self.offset_idx >= window_size:
            return 2
        elif self.onset_idx <= 0 and self.offset_idx < window_size:
            return 1
        else:
            return 3


class FrequencyDomainEnergyCalculator:
    """
    频域能量计算器 (声压域)
    
    信号能量公式 (简化):
        E = velocity² × unit_energy × ADSR × truncation
        
    其中 unit_energy = Σ|Bn|² (声压域谐波能量和)
    """
    
    def __init__(self, window_samples: int = 4096, sample_rate: int = 44100):
        self.window_samples = window_samples
        self.sample_rate = sample_rate
        self.timbre_loader = get_timbre_loader()
        
        # 频谱参数
        self.log_min = np.log2(20.0)
        self.log_max = np.log2(5000.0)
        self.num_bins = 2048
        self.min_freq = 20.0
        self.max_freq = 5000.0
        
        # 预计算bin中心
        self._precompute_bins()
        
    def _precompute_bins(self):
        """预计算bin中心频率"""
        self.bin_centers = np.zeros(self.num_bins)
        for i in range(self.num_bins):
            log_frac = i / (self.num_bins - 1)
            log_freq = self.log_min + log_frac * (self.log_max - self.log_min)
            self.bin_centers[i] = 2 ** log_freq
    
    def freq_to_bin(self, frequency: float) -> float:
        """频率到bin索引（连续值）"""
        log_freq = np.log2(np.clip(frequency, self.min_freq, self.max_freq))
        bin_idx = (log_freq - self.log_min) / (self.log_max - self.log_min) * (self.num_bins - 1)
        return bin_idx
    
    def compute_adsr_factor(self, note: Note, attack_samples: int = 441, decay_samples: int = 4410) -> float:
        """计算ADSR在窗口可见区域的平均能量因子"""
        visible_start = max(note.onset_idx, 0)
        visible_end = min(note.offset_idx, self.window_samples)
        
        if visible_end <= visible_start:
            return 0.0
        
        note_start = visible_start - note.onset_idx
        note_end = visible_end - note.onset_idx
        sustain_level = 0.7
        
        def envelope_at(t):
            if t < 0:
                return 0.0
            elif t < attack_samples:
                return t / attack_samples
            else:
                t_decay = t - attack_samples
                decay_factor = np.exp(-t_decay / decay_samples)
                return sustain_level + (1.0 - sustain_level) * decay_factor
        
        e_start = envelope_at(note_start) ** 2
        e_end = envelope_at(note_end) ** 2
        e_mid = envelope_at((note_start + note_end) / 2) ** 2
        
        return (e_start + 4 * e_mid + e_end) / 6
    
    def compute_note_energy(self, note: Note) -> float:
        """
        计算单个音符的能量 (声压域)
        
        公式: E = velocity² × unit_energy × ADSR × truncation
        """
        velocity = note.velocity
        
        # 获取声压域 unit_energy
        timbre = self.timbre_loader.get_timbre(note.timbre_id)
        unit_energy = timbre.get('unit_energy', 1.0)
        
        # 基础能量: velocity² × unit_energy
        base_energy = (velocity ** 2) * unit_energy
        
        # ADSR因子
        adsr_factor = self.compute_adsr_factor(note)
        
        # 类型处理
        note_type = note.note_type
        
        if note_type in [2, 4]:
            energy = base_energy * adsr_factor
        else:
            visible_start = max(note.onset_idx, 0)
            visible_end = min(note.offset_idx, self.window_samples)
            visible_ratio = (visible_end - visible_start) / self.window_samples
            truncation_penalty = 0.5
            energy = base_energy * adsr_factor * visible_ratio * truncation_penalty
        
        return energy
    
    def compute_signal_energy_spectrum(self, notes: List[Note]) -> np.ndarray:
        """
        计算多音的能量频谱目标 (声压域)
        
        Returns:
            target_energy: [2048] 每个bin的能量值
        """
        target_energy = np.zeros(self.num_bins, dtype=np.float64)
        
        for note in notes:
            timbre = self.timbre_loader.get_timbre(note.timbre_id)
            spectrum = timbre.get('spectrum', [[1.0, 0.0]])
            
            # 计算各谐波能量 (声压域)
            for n, (real, imag) in enumerate(spectrum):
                harmonic_num = n + 1
                freq = note.frequency * harmonic_num
                if freq > self.max_freq:
                    break
                
                amp = np.sqrt(real**2 + imag**2)
                if amp < 1e-10:
                    continue
                
                # 声压域能量: velocity² × |Bn|²
                harmonic_energy = (note.velocity ** 2) * (amp ** 2)
                
                # 应用ADSR和truncation
                adsr = self.compute_adsr_factor(note)
                note_type = note.note_type
                
                if note_type not in [2, 4]:
                    visible_start = max(note.onset_idx, 0)
                    visible_end = min(note.offset_idx, self.window_samples)
                    visible_ratio = (visible_end - visible_start) / self.window_samples
                    harmonic_energy *= visible_ratio * 0.5
                else:
                    harmonic_energy *= adsr
                
                # 分配到频谱bin
                bin_idx = self.freq_to_bin(freq)
                center = int(round(bin_idx))
                sigma = 1.5
                
                start = max(0, int(center - 3 * sigma))
                end = min(self.num_bins, int(center + 3 * sigma) + 1)
                
                for i in range(start, end):
                    weight = np.exp(-0.5 * ((i - bin_idx) / sigma) ** 2)
                    target_energy[i] += weight * harmonic_energy
        
        return target_energy.astype(np.float32)
    
    def compute_confidence_spectrum(self, notes: List[Note]) -> np.ndarray:
        """计算置信度频谱目标（四类音策略）"""
        target_conf = np.zeros(self.num_bins)
        
        for note in notes:
            note_type = note.note_type
            
            if note_type in [2, 4]:
                weight = 1.0
            else:
                visible_start = max(note.onset_idx, 0)
                visible_end = min(note.offset_idx, self.window_samples)
                visible_ratio = (visible_end - visible_start) / self.window_samples
                weight = min(visible_ratio * 0.5, 0.3)
            
            bin_idx = self.freq_to_bin(note.frequency)
            center = int(round(bin_idx))
            
            sigma = 1.5
            start = max(0, int(center - 3 * sigma))
            end = min(self.num_bins, int(center + 3 * sigma) + 1)
            
            for i in range(start, end):
                gaussian_weight = np.exp(-0.5 * ((i - bin_idx) / sigma) ** 2)
                target_conf[i] = max(target_conf[i], weight * gaussian_weight)
        
        target_conf = np.clip(target_conf, 0, 1)
        return target_conf.astype(np.float32)
    
    # ========== 噪声能量计算 (声压域) ==========
    
    # 预计算的噪声能量系数 (amplitude=1.0)
    NOISE_ENERGY_COEF = {
        NoiseType.WHITE: 4108.20,
        NoiseType.PINK: 4112.24,
        NoiseType.BROWN: 4096.00,
        NoiseType.BLUE: 4100.00,   # 估计值
        NoiseType.VIOLET: 4100.00,  # 估计值
    }
    
    def compute_noise_energy(self, noise_type: NoiseType, snr_db: float, signal_energy: float) -> np.ndarray:
        """
        计算噪声能量 (声压域)
        
        使用预计算的噪声能量系数
        """
        # 由SNR计算目标噪声能量
        target_noise_energy = signal_energy / (10 ** (snr_db / 10))
        
        # 从预计算的系数生成噪声频谱
        # 这里简化处理：噪声频谱形状由其类型决定，总能量由SNR决定
        noise_spectrum = self._generate_noise_spectrum(noise_type, target_noise_energy)
        
        return noise_spectrum.astype(np.float32)
    
    def _generate_noise_spectrum(self, noise_type: NoiseType, target_energy: float) -> np.ndarray:
        """
        生成噪声频谱形状 (声压域)
        
        基于标准噪声类型的PSD形状
        """
        freqs = self.bin_centers
        
        # 基础PSD形状 (功率谱密度)
        if noise_type == NoiseType.WHITE:
            psd = np.ones(self.num_bins)
        elif noise_type == NoiseType.PINK:
            psd = 1.0 / freqs
        elif noise_type == NoiseType.BROWN:
            psd = 1.0 / (freqs ** 2)
        elif noise_type == NoiseType.BLUE:
            psd = freqs
        elif noise_type == NoiseType.VIOLET:
            psd = freqs ** 2
        else:
            psd = np.ones(self.num_bins)
        
        # 归一化PSD
        psd_sum = np.sum(psd)
        if psd_sum > 0:
            psd_norm = psd / psd_sum
        else:
            psd_norm = np.ones(self.num_bins) / self.num_bins
        
        # 频谱能量 = PSD形状 × 目标总能量
        noise_spectrum = psd_norm * target_energy
        
        return noise_spectrum
    
    def compute_noise_only_energy(self, noise_type: NoiseType, reference_energy: Optional[float] = None) -> np.ndarray:
        """计算纯噪声的能量分布 (负样本)"""
        if reference_energy is None:
            # 使用参考440Hz纯音的能量
            reference_energy = 1.0  # velocity=1, unit_energy=1
        
        return self._generate_noise_spectrum(noise_type, reference_energy)
    
    def compute_mixed_energy_spectrum(self,
                                     notes: List[Note],
                                     noise_type: Optional[NoiseType] = None,
                                     snr_db: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
        """计算混合信号+噪声的能量谱和置信度"""
        signal_energy = self.compute_signal_energy_spectrum(notes)
        target_conf = self.compute_confidence_spectrum(notes)
        
        if noise_type is not None and snr_db is not None:
            signal_total = np.sum(signal_energy)
            noise_energy = self.compute_noise_energy(noise_type, snr_db, signal_total)
            target_energy = signal_energy + noise_energy
        else:
            target_energy = signal_energy
        
        return target_energy.astype(np.float32), target_conf


if __name__ == "__main__":
    # 测试
    calc = FrequencyDomainEnergyCalculator()
    
    print("=" * 60)
    print("Energy Calculator Test (Pressure Domain)")
    print("=" * 60)
    
    # 测试纯音
    note_a4 = Note(
        frequency=440.0,
        velocity=1.0,
        timbre_id=0,  # sine
        onset_idx=0,
        offset_idx=8192,
        phase=0.0
    )
    
    energy_a4 = calc.compute_note_energy(note_a4)
    print(f"\nA4 (440Hz, v=1.0, sine): E = {energy_a4:.4f}")
    print(f"  Expected: v^2 * unit_energy = 1.0 * 1.0 = 1.0")
    
    # 测试锯齿波
    note_saw = Note(
        frequency=440.0,
        velocity=1.0,
        timbre_id=2,  # sawtooth
        onset_idx=0,
        offset_idx=8192,
        phase=0.0
    )
    energy_saw = calc.compute_note_energy(note_saw)
    print(f"\nSawtooth A4: E = {energy_saw:.4f}")
    print(f"  Expected: ~1.64 (sum of 1/n^2)")
    
    # 测试能量比例
    print(f"\nSaw/Sine ratio = {energy_saw/energy_a4:.4f}")
    
    # 测试噪声
    print("\n" + "-" * 60)
    print("Noise Energy Test")
    print("-" * 60)
    
    signal_e = calc.compute_note_energy(note_a4)
    for noise_type in [NoiseType.WHITE, NoiseType.PINK, NoiseType.BROWN]:
        noise_e = calc.compute_noise_energy(noise_type, snr_db=0, signal_energy=signal_e)
        print(f"{noise_type.value:8s}: noise_energy = {np.sum(noise_e):.4f} (signal={signal_e:.4f}, SNR=0dB)")
    
    print("\n" + "=" * 60)
    print("Test Complete")
    print("=" * 60)

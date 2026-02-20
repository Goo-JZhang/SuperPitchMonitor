#!/usr/bin/env python3
"""
样本生成器
实现通用的随机数据生成逻辑
"""

import numpy as np
import random
from typing import List, Dict, Tuple
from dataclasses import dataclass

from timbre_loader import get_timbre_loader
from energy_calculator import Note, FrequencyDomainEnergyCalculator


class WindowedSampleGenerator:
    """
    固定窗口样本生成器
    
    生成4096-sample的短样本，覆盖所有四类音场景
    """
    
    def __init__(self, 
                 window_samples: int = 4096,
                 sample_rate: int = 44100,
                 num_freq_bins: int = 2048):
        self.window_samples = window_samples
        self.sample_rate = sample_rate
        self.num_freq_bins = num_freq_bins
        
        self.timbre_loader = get_timbre_loader()
        self.energy_calc = FrequencyDomainEnergyCalculator(window_samples)
        
        # 频率范围（对数）
        self.freq_min = 20.0
        self.freq_max = 5000.0
    
    def generate_random_notes(self, 
                             polyphony_dist: List[int] = None) -> List[Note]:
        """
        随机生成音符列表
        
        Args:
            polyphony_dist: 多音数量分布，默认偏向少音
        
        Returns:
            notes: List[Note]
        """
        if polyphony_dist is None:
            # 默认偏向少音的分布
            polyphony_dist = [1, 1, 1, 2, 2, 2, 3, 3, 4, 5, 6, 7, 8, 10, 12, 16]
        
        # 随机选择音数量
        num_notes = random.choice(polyphony_dist)
        num_notes = min(num_notes, 16)  # 最多16个音
        
        notes = []
        for _ in range(num_notes):
            # 1. 随机频率 (20-5000 Hz, 对数分布)
            freq = np.exp(random.uniform(
                np.log(self.freq_min), 
                np.log(self.freq_max)
            ))
            
            # 2. 随机音色 (从100个音色库)
            timbre_id = random.randint(0, 99)
            
            # 3. 随机起始相位 (0-2π)
            phase = random.uniform(0, 2 * np.pi)
            
            # 4. 随机 velocity (0.0-1.0)
            velocity = random.uniform(0.0, 1.0)
            
            # 5. 随机 onset_idx
            # 范围: [-4095, 4095]，然后与0取max
            onset_raw = random.randint(-4095, 4095)
            onset_idx = max(onset_raw, 0)
            
            # 6. 随机 offset_idx
            # 范围: [onset_idx, 8192]
            offset_idx = random.randint(onset_idx, 8192)
            
            note = Note(
                frequency=freq,
                velocity=velocity,
                timbre_id=timbre_id,
                onset_idx=onset_idx,
                offset_idx=offset_idx,
                phase=phase
            )
            notes.append(note)
        
        return notes
    
    def synthesize_notes(self, notes: List[Note]) -> np.ndarray:
        """
        合成多音波形
        
        Args:
            notes: 音符列表
        
        Returns:
            waveform: [window_samples]
        """
        waveform = np.zeros(self.window_samples)
        
        for note in notes:
            # 确定该音在窗口内的可见范围
            visible_start = max(note.onset_idx, 0)
            visible_end = min(note.offset_idx, self.window_samples)
            
            if visible_end <= visible_start:
                continue  # 完全不在窗口内
            
            # 合成完整波形（需要覆盖窗口前和窗口后的部分）
            # 计算需要合成的长度
            total_duration = note.offset_idx - note.onset_idx
            
            if total_duration <= 0:
                continue
            
            # 合成波形（包括窗口外的部分，以便ADSR正确应用）
            synth_samples = total_duration + 1000  # 额外缓冲
            note_wave = self.timbre_loader.compute_harmonic_waveform(
                timbre_id=note.timbre_id,
                frequency=note.frequency,
                phase_offset=note.phase,
                num_samples=synth_samples,
                sample_rate=self.sample_rate
            )
            
            # 应用ADSR包络
            note_wave = self._apply_adsr(note_wave, note)
            
            # 提取窗口内的部分
            window_offset = -note.onset_idx if note.onset_idx < 0 else 0
            extract_start = visible_start - note.onset_idx
            extract_end = visible_end - note.onset_idx
            
            if extract_start < len(note_wave) and extract_end > 0:
                extract_start = max(0, extract_start)
                extract_end = min(len(note_wave), extract_end)
                
                visible_wave = note_wave[extract_start:extract_end]
                
                # 放置到窗口对应位置
                window_start = visible_start
                window_end = visible_end
                
                waveform[window_start:window_end] += visible_wave * note.velocity
        
        # 归一化
        max_amp = np.max(np.abs(waveform))
        if max_amp > 0:
            waveform = waveform / max_amp * 0.8  # 留一些headroom
        
        return waveform.astype(np.float32)
    
    def _apply_adsr(self, waveform: np.ndarray, note: Note) -> np.ndarray:
        """
        应用简化的ADSR包络
        
        Args:
            waveform: 原始波形
            note: 音符信息
        
        Returns:
            应用包络后的波形
        """
        num_samples = len(waveform)
        
        # 简化的ADSR参数
        attack_samples = int(0.01 * self.sample_rate)  # 10ms
        decay_samples = int(0.1 * self.sample_rate)    # 100ms
        sustain_level = 0.7
        
        # 生成包络
        envelope = np.ones(num_samples)
        
        for i in range(num_samples):
            if i < attack_samples:
                # Attack: 线性上升
                envelope[i] = i / attack_samples
            else:
                # Decay: 指数衰减到sustain
                t = i - attack_samples
                envelope[i] = sustain_level + (1.0 - sustain_level) * np.exp(-t / decay_samples)
        
        return waveform * envelope
    
    def generate_sample(self, 
                       polyphony_dist: List[int] = None) -> Dict[str, np.ndarray]:
        """
        生成单个训练样本
        
        Returns:
            {
                'waveform': float32[4096],
                'target_confidence': float32[2048],
                'target_energy': float32[2048]
            }
        """
        # 1. 生成随机音符
        notes = self.generate_random_notes(polyphony_dist)
        
        # 2. 合成时域波形
        waveform = self.synthesize_notes(notes)
        
        # 3. 计算真值
        target_conf = self.energy_calc.compute_confidence_spectrum(notes)
        target_energy = self.energy_calc.compute_energy_spectrum(notes)
        
        return {
            'waveform': waveform,
            'target_confidence': target_conf,
            'target_energy': target_energy,
            'notes': notes  # 用于调试
        }
    
    def generate_dataset(self, 
                        num_samples: int,
                        output_dir: str = None) -> List[Dict]:
        """
        生成数据集
        
        Args:
            num_samples: 样本数量
            output_dir: 输出目录（可选）
        
        Returns:
            samples: List[Dict]
        """
        samples = []
        
        for i in range(num_samples):
            if i % 1000 == 0:
                print(f"Generated {i}/{num_samples} samples...")
            
            sample = self.generate_sample()
            samples.append(sample)
        
        print(f"Dataset generation complete: {num_samples} samples")
        return samples


if __name__ == "__main__":
    # 测试
    print("Testing WindowedSampleGenerator...")
    
    generator = WindowedSampleGenerator()
    
    # 生成单个样本
    print("\nGenerating single sample...")
    sample = generator.generate_sample()
    
    print(f"  Waveform shape: {sample['waveform'].shape}")
    print(f"  Waveform range: [{sample['waveform'].min():.4f}, {sample['waveform'].max():.4f}]")
    print(f"  Confidence max: {sample['target_confidence'].max():.4f}")
    print(f"  Energy max: {sample['target_energy'].max():.4f}")
    print(f"  Num notes: {len(sample['notes'])}")
    
    # 显示音符信息
    print("\n  Notes:")
    for i, note in enumerate(sample['notes'][:3]):  # 只显示前3个
        print(f"    {i+1}. Freq={note.frequency:.2f}Hz, "
              f"Type={note.note_type}, "
              f"Onset={note.onset_idx}, "
              f"Offset={note.offset_idx}, "
              f"Vel={note.velocity:.2f}")

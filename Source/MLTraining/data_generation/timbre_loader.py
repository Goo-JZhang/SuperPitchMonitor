#!/usr/bin/env python3
"""
音色数据加载器
加载TrainingData/timbre_profiles.json中的音色频谱数据
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple


class TimbreLoader:
    """加载和管理音色数据"""
    
    def __init__(self, timbre_json_path: str = None):
        """
        Args:
            timbre_json_path: 音色JSON文件路径，默认为项目根目录下的TrainingData/timbre_profiles.json
        """
        if timbre_json_path is None:
            # 从当前文件位置推算项目根目录
            current_file = Path(__file__).resolve()
            project_root = current_file.parent.parent.parent.parent
            timbre_json_path = project_root / "TrainingData" / "timbre_profiles.json"
        
        self.timbre_json_path = Path(timbre_json_path)
        self.timbres = {}
        self.categories = {}
        self._load_timbres()
    
    def _load_timbres(self):
        """加载音色数据"""
        with open(self.timbre_json_path, 'r') as f:
            data = json.load(f)
        
        # 解析音色数据
        for timbre in data['timbres']:
            timbre_id = timbre['id']
            self.timbres[timbre_id] = {
                'id': timbre_id,
                'name': timbre['name'],
                'category': timbre['category'],
                'spectrum': np.array(timbre['spectrum']),  # [16, 2] 复数数组
                'unit_energy': timbre['unit_energy']
            }
        
        self.categories = data['categories']
        self.version = data['version']
        
        print(f"Loaded {len(self.timbres)} timbres from {self.timbre_json_path}")
    
    def get_timbre(self, timbre_id: int) -> Dict:
        """获取指定ID的音色"""
        if timbre_id not in self.timbres:
            raise ValueError(f"Timbre ID {timbre_id} not found")
        return self.timbres[timbre_id]
    
    def get_timbre_by_name(self, name: str) -> Dict:
        """通过名称获取音色"""
        for timbre in self.timbres.values():
            if timbre['name'] == name:
                return timbre
        raise ValueError(f"Timbre name '{name}' not found")
    
    def get_timbres_by_category(self, category: str) -> List[Dict]:
        """获取某个类别的所有音色"""
        if category not in self.categories:
            raise ValueError(f"Category '{category}' not found")
        
        timbre_ids = self.categories[category]
        return [self.timbres[tid] for tid in timbre_ids]
    
    def get_sine_timbre(self) -> Dict:
        """获取正弦波音色（ID=0）"""
        return self.timbres[0]
    
    def compute_harmonic_waveform(self, 
                                   timbre_id: int, 
                                   frequency: float,
                                   phase_offset: float,
                                   num_samples: int,
                                   sample_rate: int = 44100) -> np.ndarray:
        """
        基于音色谐波数据合成时域波形
        
        Args:
            timbre_id: 音色ID
            frequency: 基频(Hz)
            phase_offset: 起始相位
            num_samples: 样本数
            sample_rate: 采样率
        
        Returns:
            waveform: [num_samples]
        """
        timbre = self.get_timbre(timbre_id)
        spectrum = timbre['spectrum']  # [16, 2] - 基频+15次谐波
        
        t = np.arange(num_samples) / sample_rate
        waveform = np.zeros(num_samples)
        
        for harmonic_idx, (real, imag) in enumerate(spectrum):
            # 复数幅度
            amplitude = np.sqrt(real**2 + imag**2)
            
            # 谐波频率
            harmonic_freq = frequency * (harmonic_idx + 1)
            
            # 谐波相位（从复数计算）
            harmonic_phase = np.arctan2(imag, real) if amplitude > 0 else 0
            
            # 添加波形
            if amplitude > 0:
                waveform += amplitude * np.sin(
                    2 * np.pi * harmonic_freq * t + 
                    harmonic_phase + 
                    phase_offset * (harmonic_idx + 1)  # 相位随谐波累积
                )
        
        return waveform
    
    def get_random_timbre(self) -> Dict:
        """随机获取一个音色"""
        timbre_id = np.random.randint(0, len(self.timbres))
        return self.timbres[timbre_id]


# 全局音色加载器实例
_timbre_loader = None

def get_timbre_loader() -> TimbreLoader:
    """获取全局音色加载器（单例模式）"""
    global _timbre_loader
    if _timbre_loader is None:
        _timbre_loader = TimbreLoader()
    return _timbre_loader


if __name__ == "__main__":
    # 测试
    loader = TimbreLoader()
    
    # 打印前5个音色
    print("\nFirst 5 timbres:")
    for i in range(5):
        t = loader.get_timbre(i)
        print(f"  {i}: {t['name']} ({t['category']}), unit_energy={t['unit_energy']:.4f}")
    
    # 测试合成
    print("\nTesting waveform synthesis...")
    sine_wave = loader.compute_harmonic_waveform(
        timbre_id=0,  # sine
        frequency=440.0,
        phase_offset=0.0,
        num_samples=4096
    )
    print(f"  Sine wave shape: {sine_wave.shape}")
    print(f"  Sine wave RMS: {np.sqrt(np.mean(sine_wave**2)):.4f}")
    
    saw_wave = loader.compute_harmonic_waveform(
        timbre_id=2,  # sawtooth
        frequency=440.0,
        phase_offset=0.0,
        num_samples=4096
    )
    print(f"  Sawtooth wave shape: {saw_wave.shape}")
    print(f"  Sawtooth wave RMS: {np.sqrt(np.mean(saw_wave**2)):.4f}")

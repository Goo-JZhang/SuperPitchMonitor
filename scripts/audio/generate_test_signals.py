#!/usr/bin/env python3
"""
SuperPitchMonitor 测试音频生成脚本
生成各种测试信号用于调试
"""

import numpy as np
import wave
import struct
import os

SAMPLE_RATE = 44100
DURATION = 10  # seconds - 10s for 0.1Hz FFT resolution

def save_wav(filename, samples):
    """保存为 WAV 文件"""
    samples = (samples * 32767).astype(np.int16)
    
    with wave.open(filename, 'w') as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(SAMPLE_RATE)
        wav.writeframes(samples.tobytes())
    
    print(f"Generated: {filename}")

def generate_sine_wave(freq, duration=DURATION, fade=True):
    """生成正弦波"""
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration))
    samples = np.sin(2 * np.pi * freq * t) * 0.5
    
    if fade:
        # 添加淡入淡出
        fade_samples = int(0.01 * SAMPLE_RATE)  # 10ms
        fade_in = np.linspace(0, 1, fade_samples)
        fade_out = np.linspace(1, 0, fade_samples)
        
        samples[:fade_samples] *= fade_in
        samples[-fade_samples:] *= fade_out
    
    return samples

def generate_chord():
    """生成 C 大调和弦"""
    # C4, E4, G4
    freqs = [261.63, 329.63, 392.00]
    amps = [0.33, 0.33, 0.34]
    
    t = np.linspace(0, DURATION, int(SAMPLE_RATE * DURATION))
    samples = np.zeros_like(t)
    
    for freq, amp in zip(freqs, amps):
        samples += amp * np.sin(2 * np.pi * freq * t) * 0.5
    
    # 淡入淡出
    fade_samples = int(0.01 * SAMPLE_RATE)
    fade_in = np.linspace(0, 1, fade_samples)
    fade_out = np.linspace(1, 0, fade_samples)
    
    samples[:fade_samples] *= fade_in
    samples[-fade_samples:] *= fade_out
    
    return samples

def generate_sweep():
    """生成频率扫描"""
    t = np.linspace(0, DURATION, int(SAMPLE_RATE * DURATION))
    
    # 对数扫频 100Hz - 2000Hz
    start_freq = 100
    end_freq = 2000
    
    # 瞬时频率
    freq = start_freq * (end_freq / start_freq) ** (t / DURATION)
    
    # 相位积分
    phase = 2 * np.pi * np.cumsum(freq) / SAMPLE_RATE
    samples = np.sin(phase) * 0.5
    
    return samples

def generate_white_noise():
    """生成白噪声"""
    samples = np.random.uniform(-0.3, 0.3, int(SAMPLE_RATE * DURATION))
    
    # 淡入淡出
    fade_samples = int(0.1 * SAMPLE_RATE)
    fade_in = np.linspace(0, 1, fade_samples)
    fade_out = np.linspace(1, 0, fade_samples)
    
    samples[:fade_samples] *= fade_in
    samples[-fade_samples:] *= fade_out
    
    return samples

def generate_chromatic_scale():
    """生成半音音阶 (用于测试音高检测准确性)"""
    note_duration = 0.5  # 每个音符持续时间
    notes = 13  # C4 到 C5
    
    start_freq = 261.63  # C4
    samples = []
    
    for i in range(notes):
        freq = start_freq * (2 ** (i / 12))
        note_samples = generate_sine_wave(freq, note_duration, fade=True)
        samples.extend(note_samples)
    
    return np.array(samples)

def main():
    print("SuperPitchMonitor Test Signal Generator")
    print("=" * 50)
    
    # 确保目录存在
    output_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 1. A4 正弦波 (440Hz)
    samples = generate_sine_wave(440)
    save_wav(os.path.join(output_dir, "sine_440hz.wav"), samples)
    
    # 2. C 大调和弦
    samples = generate_chord()
    save_wav(os.path.join(output_dir, "c_major_chord.wav"), samples)
    
    # 3. 频率扫描
    samples = generate_sweep()
    save_wav(os.path.join(output_dir, "freq_sweep.wav"), samples)
    
    # 4. 白噪声
    samples = generate_white_noise()
    save_wav(os.path.join(output_dir, "white_noise.wav"), samples)
    
    # 5. 半音音阶
    samples = generate_chromatic_scale()
    save_wav(os.path.join(output_dir, "chromatic_scale.wav"), samples)
    
    print("=" * 50)
    print("All test signals generated successfully!")
    print("")
    print("Files generated:")
    print("  - sine_440hz.wav       : A4 标准音")
    print("  - c_major_chord.wav    : C大调和弦 (多音高测试)")
    print("  - freq_sweep.wav       : 频率扫描 (频谱显示测试)")
    print("  - white_noise.wav      : 白噪声")
    print("  - chromatic_scale.wav  : 半音音阶 (音高检测准确性测试)")

if __name__ == "__main__":
    main()

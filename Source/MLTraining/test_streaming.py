#!/usr/bin/env python3
"""
模拟实时streaming测试框架
读取WAV文件，使用4096采样窗口模拟SuperPitchMonitor的实时处理
"""

import numpy as np
import torch
import wave
import sys
from pathlib import Path
from collections import deque

sys.path.insert(0, str(Path(__file__).parent))
from model import PitchNetBaseline


class StreamingTester:
    """模拟实时流式处理测试器"""
    
    def __init__(self, model_path, window_size=4096, hop_size=512):
        """
        Args:
            model_path: 模型检查点路径
            window_size: 分析窗口大小 (默认4096)
            hop_size: 帧移 (默认512，与SuperPitchMonitor一致)
        """
        self.window_size = window_size
        self.hop_size = hop_size
        
        # 加载模型
        self.model = PitchNetBaseline()
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=True)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        # 缓冲区
        self.buffer = deque(maxlen=window_size)
        
        # 频率映射
        self.log_min = np.log2(20.0)
        self.log_max = np.log2(5000.0)
        
    def bin_to_freq(self, bin_idx):
        """bin索引转换为频率"""
        return 20 * (5000/20) ** (bin_idx / 2047)
    
    def process_frame(self, audio_chunk):
        """
        处理一帧音频 (模拟SuperPitchMonitor的processAudioBlock)
        
        Args:
            audio_chunk: 输入音频片段 (hop_size长度)
        
        Returns:
            dict: 检测结果
        """
        # 添加到缓冲区
        self.buffer.extend(audio_chunk)
        
        # 缓冲区填满后才处理
        if len(self.buffer) < self.window_size:
            return None
        
        # 转换为tensor [1, 1, 4096]
        window = np.array(self.buffer, dtype=np.float32)
        x = torch.from_numpy(window).unsqueeze(0).unsqueeze(0)
        
        # 模型推理
        with torch.no_grad():
            pred = self.model(x)
        
        conf = pred[0, :, 0].numpy()
        energy = pred[0, :, 1].numpy()
        
        # 找峰值 (置信度>0.1)
        peaks = []
        for i in range(1, 2047):
            if conf[i] > conf[i-1] and conf[i] > conf[i+1] and conf[i] > 0.1:
                peaks.append({
                    'bin': i,
                    'freq': self.bin_to_freq(i),
                    'conf': conf[i],
                    'energy': energy[i]
                })
        
        # 按置信度排序
        peaks.sort(key=lambda x: x['conf'], reverse=True)
        
        return {
            'peaks': peaks,
            'conf': conf,
            'energy': energy
        }
    
    def process_file(self, wav_path, target_freq=None):
        """
        处理整个WAV文件
        
        Args:
            wav_path: WAV文件路径
            target_freq: 目标频率 (用于计算误差)
        
        Returns:
            list: 每帧的检测结果
        """
        # 读取WAV文件
        with wave.open(str(wav_path), 'rb') as wf:
            n_channels = wf.getnchannels()
            sample_width = wf.getsampwidth()
            sample_rate = wf.getframerate()
            n_frames = wf.getnframes()
            
            print(f"\nProcessing: {wav_path.name}")
            print(f"  Sample rate: {sample_rate} Hz")
            print(f"  Channels: {n_channels}")
            print(f"  Frames: {n_frames}")
            print(f"  Duration: {n_frames/sample_rate:.3f}s")
            
            # 读取数据
            raw_data = wf.readframes(n_frames)
            if sample_width == 2:
                data = np.frombuffer(raw_data, dtype=np.int16)
            else:
                data = np.frombuffer(raw_data, dtype=np.int8)
            
            # 转单声道
            if n_channels == 2:
                data = data.reshape(-1, 2).mean(axis=1)
            
            # 归一化到 -1~1
            data = data.astype(np.float32) / float(2**(sample_width*8 - 1))
        
        # 清空缓冲区
        self.buffer.clear()
        
        # 模拟实时处理
        results = []
        num_chunks = (len(data) - self.window_size) // self.hop_size + 1
        
        for i in range(num_chunks):
            start = i * self.hop_size
            chunk = data[start:start + self.hop_size]
            
            result = self.process_frame(chunk)
            if result is not None:
                results.append({
                    'frame': i,
                    'time': (start + self.window_size) / sample_rate,
                    **result
                })
        
        # 分析结果
        self._analyze_results(results, target_freq)
        
        return results
    
    def _analyze_results(self, results, target_freq):
        """分析检测结果"""
        if not results:
            print("  No results!")
            return
        
        print(f"\n  Total frames: {len(results)}")
        
        # 统计检测到的peak数量
        peak_counts = [len(r['peaks']) for r in results]
        print(f"  Peaks per frame: min={min(peak_counts)}, max={max(peak_counts)}, "
              f"mean={np.mean(peak_counts):.1f}")
        
        # 统计最强检测的频率
        primary_freqs = []
        primary_confs = []
        for r in results:
            if r['peaks']:
                primary_freqs.append(r['peaks'][0]['freq'])
                primary_confs.append(r['peaks'][0]['conf'])
        
        if primary_freqs:
            print(f"\n  Primary detection (strongest peak):")
            print(f"    Freq range: [{min(primary_freqs):.1f}, {max(primary_freqs):.1f}] Hz")
            print(f"    Freq mean: {np.mean(primary_freqs):.1f} ± {np.std(primary_freqs):.1f} Hz")
            print(f"    Confidence: {np.mean(primary_confs):.3f} ± {np.std(primary_confs):.3f}")
            
            if target_freq:
                errors = [1200 * np.log2(f / target_freq) for f in primary_freqs]
                print(f"    Error: {np.mean(errors):+.1f} ± {np.std(errors):.1f} cents")
                print(f"    Max error: {max(np.abs(errors)):.1f} cents")
        
        # 打印前10帧的详细结果
        print(f"\n  First 10 frames detail:")
        for r in results[:10]:
            peaks_str = ", ".join([f"{p['freq']:.0f}Hz({p['conf']:.2f})" for p in r['peaks'][:3]])
            print(f"    Frame {r['frame']:3d} @ {r['time']:.3f}s: {peaks_str}")


def main():
    # 项目根目录
    project_root = Path(__file__).parent.parent.parent
    
    # 模型路径
    model_path = project_root / 'MLModel' / 'checkpoints' / 'best_model_live.pth'
    
    # 测试文件
    test_files = [
        (project_root / 'Resources' / 'TestAudio' / 'sine_220hz.wav', 220),
        (project_root / 'Resources' / 'TestAudio' / 'sine_440hz.wav', 440),
        (project_root / 'Resources' / 'TestAudio' / 'sine_880hz.wav', 880),
    ]
    
    print("="*70)
    print("Streaming Test Framework")
    print("="*70)
    print(f"Model: {model_path}")
    print(f"Window size: 4096 samples")
    print(f"Hop size: 512 samples")
    print(f"Frame rate: {44100/512:.1f} fps")
    
    # 创建测试器
    tester = StreamingTester(model_path)
    
    # 处理每个文件
    for wav_path, target_freq in test_files:
        if wav_path.exists():
            tester.process_file(wav_path, target_freq)
        else:
            print(f"\nFile not found: {wav_path}")
    
    print("\n" + "="*70)
    print("Test completed!")
    print("="*70)


if __name__ == '__main__':
    main()

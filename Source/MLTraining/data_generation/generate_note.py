#!/usr/bin/env python3
"""
音符生成器

提供各种音生成函数：
- 单音（正弦波）
- 噪声（白噪声、粉红噪声、布朗噪声等）
- 后续可添加：和弦、复合音等
"""

import numpy as np
from typing import Dict, List, Optional, Tuple


def generate_single_note(bin_idx: int, 
                         num_bins: int = 2048,
                         min_freq: float = 20.0,
                         max_freq: float = 5000.0,
                         sigma: float = 1.5,
                         rng: Optional[np.random.RandomState] = None) -> Dict[str, np.ndarray]:
    """
    生成单音样本
    
    Args:
        bin_idx: 目标bin索引
        num_bins: 总bin数
        min_freq: 最低频率
        max_freq: 最高频率
        sigma: 高斯平滑参数
        rng: 随机数生成器
    
    Returns:
        {
            'waveform': [4096] float32,
            'confs': [2048] float32,
            'energies': [2048] float32 (归一化，总和=1),
            'indices': [K] int (稀疏索引，调试用)
        }
    """
    if rng is None:
        rng = np.random.RandomState()
    
    # 计算该bin的频率范围
    log_min = np.log2(min_freq)
    log_max = np.log2(max_freq)
    
    # bin中心频率
    log_freq_center = log_min + (bin_idx / (num_bins - 1)) * (log_max - log_min)
    
    # bin边界
    if bin_idx == 0:
        log_freq_low = log_min
        log_freq_high = (log_min + (1 / (num_bins - 1)) * (log_max - log_min) + log_freq_center) / 2
    elif bin_idx == num_bins - 1:
        log_freq_low = (log_min + ((num_bins - 2) / (num_bins - 1)) * (log_max - log_min) + log_freq_center) / 2
        log_freq_high = log_max
    else:
        log_freq_low = (log_min + ((bin_idx - 1) / (num_bins - 1)) * (log_max - log_min) + log_freq_center) / 2
        log_freq_high = (log_min + ((bin_idx + 1) / (num_bins - 1)) * (log_max - log_min) + log_freq_center) / 2
    
    # 在bin内随机选择频率
    log_freq = rng.uniform(log_freq_low, log_freq_high)
    freq = 2 ** log_freq
    
    # 随机相位
    phase = rng.uniform(0, 2 * np.pi)
    
    # 生成波形 - 归一化到标准差为1.0（与噪声数据一致）
    sample_rate = 44100
    t = np.arange(4096) / sample_rate
    waveform = np.sin(2 * np.pi * freq * t + phase).astype(np.float32)
    # 正弦波 RMS = 1/sqrt(2) ≈ 0.707，归一化使标准差为1.0
    waveform = waveform / np.std(waveform)
    
    # 生成真值
    confs = np.zeros(num_bins, dtype=np.float32)
    energies = np.zeros(num_bins, dtype=np.float32)
    indices = []
    
    range_size = int(3 * sigma) + 1
    for i in range(max(0, bin_idx - range_size), min(num_bins, bin_idx + range_size + 1)):
        dist = abs(i - bin_idx)
        val = np.exp(-0.5 * (dist / sigma) ** 2)
        if val > 0.01:
            confs[i] = min(val, 1.0)
            energies[i] = val
            indices.append(i)
    
    # 归一化能量（使总和为1）
    energy_sum = energies.sum()
    if energy_sum > 0:
        energies = energies / energy_sum
    
    return {
        'waveform': waveform,
        'confs': confs,
        'energies': energies,
        'indices': indices
    }


def generate_silence(num_bins: int = 2048,
                     rng: Optional[np.random.RandomState] = None) -> Dict[str, np.ndarray]:
    """
    生成静音样本（用于负样本）
    
    Args:
        num_bins: 总bin数
        rng: 随机数生成器
    
    Returns:
        {
            'waveform': [4096] float32,
            'confs': [2048] float32,
            'energies': [2048] float32,
            'indices': []
        }
    """
    if rng is None:
        rng = np.random.RandomState()
    
    # 生成低幅值噪声
    waveform = (rng.randn(4096) * 0.01).astype(np.float32)
    
    # 全零置信度和能量
    confs = np.zeros(num_bins, dtype=np.float32)
    energies = np.zeros(num_bins, dtype=np.float32)
    
    return {
        'waveform': waveform,
        'confs': confs,
        'energies': energies,
        'indices': []
    }


def _generate_colored_noise(n_samples: int, 
                            color: str,
                            rng: np.random.RandomState) -> np.ndarray:
    """
    生成彩色噪声
    
    Args:
        n_samples: 样本数
        color: 噪声类型 ('white', 'pink', 'brown', 'blue', 'violet')
        rng: 随机数生成器
    
    Returns:
        [n_samples] float32 噪声波形
    """
    # 生成白噪声频谱
    white = rng.randn(n_samples)
    
    # FFT
    fft = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(n_samples)
    freqs[0] = 1e-10  # 避免除零
    
    # 根据颜色调整频谱
    if color == 'white':
        # 白噪声：频谱平坦
        scaling = np.ones_like(freqs)
    elif color == 'pink':
        # 粉红噪声：1/f，每倍频程能量相等
        scaling = 1 / np.sqrt(freqs)
    elif color == 'brown':
        # 布朗噪声：1/f²
        scaling = 1 / freqs
    elif color == 'blue':
        # 蓝噪声：f
        scaling = np.sqrt(freqs)
    elif color == 'violet':
        # 紫噪声：f²
        scaling = freqs
    else:
        raise ValueError(f"Unknown noise color: {color}")
    
    # 应用频谱整形
    fft = fft * scaling
    
    # 反FFT得到时域信号
    noise = np.fft.irfft(fft, n=n_samples)
    
    # 归一化到标准差为1.0（与单音数据一致）
    noise = noise / np.std(noise)
    
    return noise.astype(np.float32)


def _get_noise_energy_profile(color: str, num_bins: int) -> np.ndarray:
    """
    获取噪声的能量分布（概率分布形式，已归一化）
    
    注意：返回的是归一化的概率分布（总和为1），这样不同噪声类型可以在同一尺度上混合
    
    Args:
        color: 噪声类型
        num_bins: bin数量
    
    Returns:
        [num_bins] 归一化的能量分布（概率分布，总和为1）
    """
    # 计算每个bin的边界频率（对数刻度）
    min_freq = 20.0
    max_freq = 5000.0
    log_min = np.log2(min_freq)
    log_max = np.log2(max_freq)
    
    # bin 边界（共 num_bins+1 个边界）
    bin_edges = np.array([
        2 ** (log_min + (i / num_bins) * (log_max - log_min))
        for i in range(num_bins + 1)
    ])
    
    # 每个 bin 的宽度（delta_f）
    delta_f = bin_edges[1:] - bin_edges[:-1]
    
    # bin 中心频率
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # 根据噪声类型计算功率谱密度
    if color == 'white':
        psd = np.ones(num_bins)
    elif color == 'pink':
        psd = 1 / bin_centers
    elif color == 'brown':
        psd = 1 / (bin_centers ** 2)
    elif color == 'blue':
        psd = bin_centers
    elif color == 'violet':
        psd = bin_centers ** 2
    else:
        raise ValueError(f"Unknown noise color: {color}")
    
    # 每个bin的能量 = PSD × delta_f
    energies = psd * delta_f
    
    # 归一化为概率分布（总和为1），确保不同噪声类型在同一尺度上
    energy_sum = energies.sum()
    if energy_sum > 0:
        energies = energies / energy_sum
    
    return energies


def generate_noise(color: str = 'white',
                   num_bins: int = 2048,
                   rng: Optional[np.random.RandomState] = None) -> Dict[str, np.ndarray]:
    """
    生成噪声样本
    
    Args:
        color: 噪声类型 ('white', 'pink', 'brown', 'blue', 'violet')
        num_bins: 总bin数
        rng: 随机数生成器
    
    Returns:
        {
            'waveform': [4096] float32,
            'confs': [2048] float32 (全0),
            'energies': [2048] float32 (归一化，按噪声频谱分布),
            'indices': []
        }
    """
    if rng is None:
        rng = np.random.RandomState()
    
    # 生成波形
    waveform = _generate_colored_noise(4096, color, rng)
    
    # 置信度全为0（噪声没有音高）
    confs = np.zeros(num_bins, dtype=np.float32)
    
    # 能量按噪声频谱分布
    energies = _get_noise_energy_profile(color, num_bins)
    
    # 归一化能量
    energy_sum = energies.sum()
    if energy_sum > 0:
        energies = energies / energy_sum
    
    return {
        'waveform': waveform,
        'confs': confs,
        'energies': energies.astype(np.float32),
        'indices': []
    }


# 支持的噪声类型列表
NOISE_TYPES = ['white', 'pink', 'brown', 'blue', 'violet']


def generate_mixed_noise(colors: list,
                         num_bins: int = 2048,
                         weights: Optional[list] = None,
                         rng: Optional[np.random.RandomState] = None) -> Dict[str, np.ndarray]:
    """
    生成混合噪声（多种颜色噪声的加权组合）
    
    注意：每种噪声使用独立的随机种子，确保它们在频域上不相关，
    这样混合后的能量才是各能量按 weight^2 的叠加
    
    Args:
        colors: 噪声颜色列表，如 ['white', 'pink']
        num_bins: 总bin数
        weights: 各噪声的权重列表，None表示随机权重
        rng: 随机数生成器
    
    Returns:
        {
            'waveform': [4096] float32,
            'confs': [2048] float32 (全0),
            'energies': [2048] float32 (各噪声能量分布的加权组合),
            'indices': []
        }
    """
    if rng is None:
        rng = np.random.RandomState()
    
    # 生成随机权重（如果未提供）
    if weights is None:
        weights = rng.rand(len(colors))
    weights = np.array(weights)
    weights = weights / weights.sum()  # 归一化
    
    # 生成各噪声的波形并混合
    # 重要：每种噪声使用独立的 RNG，确保它们在频域上不相关
    mixed_waveform = np.zeros(4096, dtype=np.float32)
    for color, weight in zip(colors, weights):
        # 为每种噪声创建独立的 RNG
        noise_rng = np.random.RandomState(rng.randint(0, 2**31))
        wave = _generate_colored_noise(4096, color, noise_rng)
        mixed_waveform += weight * wave
    
    # 归一化波形到标准差为1.0（与单音数据一致）
    mixed_waveform = mixed_waveform / np.std(mixed_waveform)
    
    # 生成混合能量分布（平滑的理论 profile，已归一化）
    # _get_noise_energy_profile 返回的是归一化的概率分布
    # 混合：按 weight^2 加权（能量是平方关系）
    energies = np.zeros(num_bins, dtype=np.float32)
    for color, weight in zip(colors, weights):
        energy = _get_noise_energy_profile(color, num_bins)  # 已归一化
        energies += (weight ** 2) * energy
    
    # 再次归一化
    energy_sum = energies.sum()
    if energy_sum > 0:
        energies = energies / energy_sum
    
    return {
        'waveform': mixed_waveform,
        'confs': np.zeros(num_bins, dtype=np.float32),
        'energies': energies,
        'indices': [],
        'mix_info': {  # 记录混合信息用于调试
            'colors': colors,
            'weights': weights.tolist()
        }
    }


def get_all_noise_combinations():
    """
    获取所有噪声组合
    
    Returns:
        list: [(name, colors), ...]
    """
    from itertools import combinations
    
    all_combinations = []
    
    # 单一噪声（1重）
    for color in NOISE_TYPES:
        all_combinations.append((f'single_{color}', [color]))
    
    # 双重组合
    for combo in combinations(NOISE_TYPES, 2):
        name = f'mix_{"_".join(combo)}'
        all_combinations.append((name, list(combo)))
    
    # 三重组合
    for combo in combinations(NOISE_TYPES, 3):
        name = f'mix_{"_".join(combo)}'
        all_combinations.append((name, list(combo)))
    
    # 四重组合
    for combo in combinations(NOISE_TYPES, 4):
        name = f'mix_{"_".join(combo)}'
        all_combinations.append((name, list(combo)))
    
    # 五重组合
    all_combinations.append(('mix_all', NOISE_TYPES.copy()))
    
    return all_combinations


def generate_all_noise_types(samples_per_type: int = 100,
                             num_bins: int = 2048,
                             seed: int = 42) -> List[Tuple[str, Dict[str, np.ndarray]]]:
    """
    生成所有类型的噪声样本
    
    Args:
        samples_per_type: 每种噪声类型的样本数
        num_bins: bin数量
        seed: 随机种子
    
    Returns:
        [(noise_type, sample), ...] 列表
    """
    rng = np.random.RandomState(seed)
    samples = []
    
    for color in NOISE_TYPES:
        for _ in range(samples_per_type):
            sample = generate_noise(color, num_bins, rng)
            samples.append((color, sample))
    
    return samples

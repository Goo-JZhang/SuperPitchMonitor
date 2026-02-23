# SuperPitchMonitor ML 实现路线图

**策略**: 先集成，后训练 —— 使用随机权重模型验证端到端流程，再迭代优化模型

**当前状态**: ✅ Phase 3 完成 - 小数据训练成功，随机相位增强显著改善streaming性能

**最新成果**:
- ✅ 20K样本训练完成 (Val Loss: 0.209)
- ✅ 随机相位增强: 误差从±25 cents降至±3 cents
- ✅ NumPy Shards格式: 支持大规模训练(20-30GB)
- ✅ 跨平台训练脚本: macOS MPS + Windows CUDA

---

## Phase 1: 集成验证 ✅ (已完成)

**目标**: 建立C++到ONNX Runtime的完整推理管道，使用随机初始化模型验证数据流

### 环境准备 ✅

- [x] **0.1** 创建Conda虚拟环境
  ```bash
  conda env create -f MLModel/environment.yml
  conda activate spm_ml
  ```

### 1.1 创建随机权重模型 stub ✅

- [x] **1.1.1** 使用PyTorch创建网络架构 stub (`MLModel/export_stub_model.py`)
  ```python
  # 创建与最终架构一致的dummy模型
  class PitchNetStub(nn.Module):
      def __init__(self):
          super().__init__()
          # 输入: [1, 4096], 输出: [2048, 2]
          self.frontend = nn.Conv1d(1, 64, 512, stride=128, padding=256)
          self.temporal = nn.Sequential(...)  # 3层时序压缩
          self.conf_head = nn.Linear(64, 2048)  # Confidence头
          self.energy_head = nn.Linear(64, 2048)  # Energy头
      
      def forward(self, x):
          # 返回模拟音高峰值的输出，用于验证数据流
          # 默认输出 C Major 和弦峰值 (C4, E4, G4)
          return torch.stack([confidence, energy], dim=-1)  # [1, 2048, 2]
  ```

- [x] **1.1.2** 导出ONNX格式 (见 `export_test_model_with_peaks.py`)
  - [x] 使用 `torch.onnx.export` 导出为 ONNX (opset 11+)
  - [x] 验证输入输出维度: `[1, 4096] -> [1, 2048, 2]`
  - [x] 使用 `onnxruntime` Python包验证推理正常
  - [x] 模型文件: `pitchnet_stub_v1.onnx` (~8.5MB)
  - [x] 输出范围: confidence [0.047, 0.993], energy [0.3, 2.0]

- [x] **1.1.3** 模型版本管理
  - [x] 模型存放于 `MLModel/` 目录
  - [x] 导出脚本: `export_test_model_with_peaks.py` (带合成峰值)

### 1.2 C++推理引擎搭建 ✅

- [x] **1.2.1** 集成ONNX Runtime到CMake (支持源构建和预构建)
  - ✅ `ThirdParty/fetch_onnxruntime.cmake` 自动下载平台特定版本
  - ✅ `ThirdParty/build_onnxruntime.cmake` 支持从源码构建 (带CoreML)
  - ✅ 支持Windows **CUDA** (RTX 4080S优化) / DirectML (通用)
  - ✅ 支持Android (NNAPI), macOS/iOS (CoreML)
  - ✅ 库文件自动复制到输出目录

- [x] **1.2.2** 实现 `MLPitchDetector` 基础类 (`MLPitchDetector.h/cpp`)
  - ✅ GPU异步推理支持 (CoreML/DirectML/NNAPI/CUDA)
  - ✅ 独立推理线程 (ThreadPoolExecutor)
  - ✅ Lock-free队列 (音频输入 → 推理)
  - ✅ 双输出接口: `getLatestResults()` (高置信度) + `getFullSpectrum()` (完整频谱)

- [x] **1.2.3** 内存管理验证
  - [x] 确认输入Tensor内存布局: `[batch=1, channels=1, samples=4096]`
  - [x] 确认输出Tensor内存布局: `[batch=1, bins=2048, channels=2]`
  - [x] 输出格式: [confidence, energy] 交替存储
  - [ ] 测试连续推理1000次无内存泄漏 (待验证)

### 1.3 JUCE集成 ✅

- [x] **1.3.1** 集成到 `AudioEngine`
  - ✅ `MLPitchDetector` 作为 `AudioEngine` 成员
  - ✅ `setMLAnalysisEnabled()` / `isMLAnalysisEnabled()` 接口
  - ✅ `setMLGPUEnabled()` GPU/CPU 切换
  - ✅ `setMLModelPath()` 模型选择
  - ✅ Settings面板添加 "ML Analysis" 选项 (默认ON)
  - ✅ ML模式数据流: 音频 → ML推理 → SpectrumData (confidence/energy) + PitchVector

- [x] **1.3.2** 数据流管道
  - ✅ 音频输入 (4096 samples) → `submitAudio()`
  - ✅ 异步推理结果 → `getLatestResults()` (高置信度音高)
  - ✅ 异步推理结果 → `getFullSpectrum()` (所有bins，用于频谱显示)
  - ✅ SpectrumData 支持 ML 双通道: `mlConfidence[]`, `mlEnergy[]`
  - ✅ PitchCandidate 支持 ML 标记: `isMLEnergy`

- [x] **1.3.3** UI集成
  - ✅ SpectrumDisplay: ML模式双纵轴 (Confidence/energy)
  - ✅ PitchDisplay: 音名 + cents偏差 + energy显示
  - ✅ PitchWaterfall: 音高历史散点图
  - ✅ SettingsPanel: ML开关 + GPU/CPU切换 + 模式标签
  - ✅ 半透明模式标签 (MLMODE/FFTMODE)

### 1.4 集成测试验证点 ✅

- [x] **1.4.1** 数据流验证
  - [x] 生成ONNX测试模型 (`python export_test_model_with_peaks.py`)
  - [x] 运行验证: 合成C Major和弦输入，确认数据正确传入ONNX
  - [x] 输出值范围合理: confidence [0.047, 0.993], energy [0.3, 2.0]
  - [x] UI显示验证: SpectrumDisplay 显示3个峰值 (C4, E4, G4)
  - [x] UI显示验证: DetectedPitches 正确显示音名和energy值
  - [x] 处理时间 < 20ms (实测 ~3-5ms on Apple Silicon)

- [x] **1.4.2** UI/UX验证
  - [x] ML模式: 双纵轴显示 (Confidence 绿色, Energy 橙色)
  - [x] ML模式: 5000Hz外直接截断为0，无插值
  - [x] FFT模式: 传统dB频谱显示
  - [x] 模式标签: 半透明背景，左下角显示
  - [x] 音名计算: 正确处理负octave，显示C-1至B-1等
  - [x] 能量显示: ML模式显示"E: 1.23"，FFT模式显示"-45.2 dB"

- [ ] **1.4.3** 稳定性测试 (待进行)
  - [ ] 连续运行10分钟无崩溃
  - [ ] 内存占用稳定无增长
  - [ ] CPU/GPU占用在合理范围

- [ ] **1.4.4** 跨平台验证 (待进行)
  - [x] macOS (Apple Silicon) 编译运行通过 ✅
  - [ ] Windows x64 编译运行通过
  - [ ] iOS 模拟器运行通过
  - [ ] Android 编译通过

**Phase 1 完成标准** ✅: 
- 任意音频输入 → C++推理 → 看到输出数值（即使是随机的）
- macOS平台编译运行通过
- UI数据流验证通过 (SpectrumDisplay + DetectedPitches)

---

## 关键文件路径

| 组件 | 路径 | 说明 |
|------|------|------|
| **ML模型导出脚本** | `MLModel/export_test_model_with_peaks.py` | 生成带合成峰值的测试模型 |
| **ONNX模型文件** | `MLModel/pitchnet_stub_v1.onnx` | 当前测试模型 (~8.5MB) |
| **C++推理引擎** | `Source/MLAlgorithm/MLPitchDetector.h/cpp` | ONNX Runtime封装 |
| **AudioEngine集成** | `Source/Audio/AudioEngine.cpp` | ML/FFT模式切换逻辑 |
| **频谱显示** | `Source/UI/SpectrumDisplay.cpp` | ML模式双纵轴显示 |
| **音高卡片** | `Source/UI/PitchCard.cpp` | 音名/Energy显示 |
| **ONNX构建配置** | `ThirdParty/build_onnxruntime.cmake` | 支持CoreML的源构建 |
| **数据定义** | `Source/Audio/SpectrumData.h` | SpectrumData/PitchCandidate结构 |
| **CMake配置** | `CMakeLists.txt` | `BUILD_ONNXRUNTIME_FROM_SOURCE` 选项 |

---

## Phase 2: 训练数据结构设计 (Week 3-4) ⏳

**目标**: 设计并实现训练数据生成流水线

**核心设计原则**: 生成时完成所有计算，训练时零计算

### 2.1 数据格式设计

#### 2.1.1 格式选择: HDF5

**理由**: 二进制存储、O(1)随机访问、PyTorch生态成熟、支持压缩

**文件结构**:
```
training_data_v1.hdf5
├── attrs (全局属性)
│   ├── version: "1.0"
│   ├── sample_rate: 44100
│   ├── num_samples: 100000
│   └── generation_config: {...}
│
├── data/ (核心训练数据 - 3个必需tensor)
│   ├── waveform: float32[N, 4096]           # 音频波形
│   ├── target_confidence: float32[N, 2048]  # 置信度目标（已平滑）
│   └── target_energy: float32[N, 2048]      # 能量目标（已平滑）
│
└── meta/ (可选调试数据，支持16个音)
    ├── polyphony: uint8[N]              # 实际多音数量
    ├── snr_db: float32[N]               # 信噪比
    ├── note_freqs: float32[N, 16]       # 频率(Hz)
    ├── note_onsets: int16[N, 16]        # onset采样点索引（相对窗口起点）
    ├── note_offsets: int16[N, 16]       # offset采样点索引（相对窗口起点）
    ├── note_velocities: float32[N, 16]  # velocity值
    ├── note_timbre_ids: uint8[N, 16]    # 音色类型ID
    └── note_energy_ratios: float32[N, 16] # 实际能量占比（用于验证）
```

**时间表示**: 使用int16采样点索引（vs float32秒）
- 0 = 窗口起点
- 2048 = 窗口中间
- 负数 = 窗口前开始（如-1024）
- >4096 = 窗口后结束（如5120）
- 优势：存储减半、无浮点精度问题、可直接当array index

**能量计算**: 频域快速计算（无需时域合成）
1. 预计算音色谐波能量系数：`coeff = Σ(A_k²)`
2. 频域能量：`E = coeff × velocity² × ADSR_factor × (visible_penalty)`
3. 多音归一化：`ratio_i = E_i / ΣE`
4. 只标注基频bin（让网络学习泛音→基频映射）

**性能**: 频域计算比时域合成快~200倍（20次乘法 vs 4096次乘加）

#### 2.1.2 四类音策略（置信度与能量统一）

每个4096-sample窗口（93ms）中，音符与窗口的重叠方式同时决定**置信度**和**能量占比**：

| 类型 | 图示 | onset | offset | **置信度** | **能量处理** |
|------|------|-------|--------|-----------|-------------|
| **第一类** | `\|███\|` | ≤0 | <4096 | **Soft** (<0.3) | **打折**: × visible_ratio × 0.5 |
| **第二类** | `\|  ██████\|` | >0 | ≥4096 | **Hard=1.0** ✅ | **完整**: 不打折 |
| **第三类** | `\|  ██ \|` | >0 | <4096 | **Soft** (<0.3) | **打折**: × visible_ratio × 0.5 |
| **第四类** | `\|███████\|` | ≤0 | ≥4096 | **Hard=1.0** ✅ | **完整**: 不打折 |

**统一设计原则**:
- 第二/四类音：完整延伸到窗口边界 → 高confidence + 完整能量
- 第一/三类音：窗口内结束（不完整存在）→ 低confidence + 能量打折

**能量计算逻辑**（频域快速计算）:
```python
def compute_energy(note, timbre_profile):
    # 基础谐波能量（预计算系数 × velocity²）
    base_energy = timbre_profile.energy_coeff * (note.velocity ** 2)
    
    # ADSR因子（从可见区域计算）
    visible_start = max(note.onset_idx, 0)
    visible_end = min(note.offset_idx, 4096)
    adsr_factor = compute_adsr_average(timbre_profile, visible_start, visible_end)
    
    # 类型相关处理（与置信度策略统一）
    if is_type_2_or_4(note):
        # 完整存在，能量不打折
        energy = base_energy * adsr_factor
    else:
        # 第一/三类：窗口内结束，能量打折
        visible_ratio = (visible_end - visible_start) / 4096
        energy = base_energy * adsr_factor * visible_ratio * 0.5  # 额外50%惩罚
    
    return energy

# 多音能量占比
energies = [compute_energy(n, profile) for n in notes]
energy_ratios = energies / sum(energies)

# 生成target_energy（只标注基频，高斯平滑）
target_energy = np.zeros(2048)
for note, ratio in zip(notes, energy_ratios):
    bin_idx = freq_to_bin(note.frequency)
    target_energy += gaussian_1d(bin_idx, ratio, sigma=1.5)

# 生成target_confidence（同策略）
target_conf = np.zeros(2048)
for note in notes:
    weight = 1.0 if is_type_2_or_4(note) else min(visible_ratio * 0.5, 0.3)
    target_conf[freq_to_bin(note.frequency)] = weight
target_conf = gaussian_smooth_1d(target_conf, sigma=1.5)
```

**关键优势**:
- 置信度和能量策略一致：不完整的音 → 低confidence + 低energy
- 结束中的音不会"抢走"能量占比
- 持续音的能量估计更准确

### 2.2 数据生成流水线

#### 2.2.1 核心类设计

```python
# MLModel/data/generator.py

@dataclass
class Note:
    """生成时的音符定义"""
    frequency: float        # Hz
    onset_idx: int16        # 相对窗口起点的onset（sample index）
    offset_idx: int16       # 相对窗口起点的offset（sample index）
    velocity: float         # 0.0-1.0
    timbre: str             # 'piano'|'sine'|'saw'

class WindowedSampleGenerator:
    """生成固定窗口长度的训练样本"""
    
    def __init__(self, window_samples=4096, sample_rate=44100):
        self.window_samples = window_samples
        self.sample_rate = sample_rate
    
    def generate(self, notes: List[Note]) -> Dict[str, np.ndarray]:
        """
        生成单个训练样本
        
        Returns:
            waveform: float32[4096]
            target_confidence: float32[2048]
            target_energy: float32[2048]
            meta: dict (用于调试)
        """
        # 1. 合成音频（窗口长度）
        waveform = self.synthesize(notes)
        
        # 2. 计算目标（应用四类音策略）
        target_conf = self.compute_confidence(notes)
        target_energy = self.compute_energy(notes)
        
        # 3. 高斯平滑
        target_conf = gaussian_smooth(target_conf, sigma=1.5)
        target_energy = gaussian_smooth(target_energy, sigma=1.5)
        
        return {
            'waveform': waveform,
            'target_confidence': target_conf,
            'target_energy': target_energy
        }
```

#### 2.2.2 数据生成脚本

```bash
# 生成单音数据集
python MLModel/generate_dataset.py \
    --output data/train_single_v1.hdf5 \
    --num-samples 100000 \
    --polyphony-dist "[1.0, 0, 0, 0, 0]" \
    --f0-range "[20, 5000]"

# 生成和弦数据集  
python MLModel/generate_dataset.py \
    --output data/train_chord_v1.hdf5 \
    --num-samples 100000 \
    --polyphony-dist "[0.3, 0.3, 0.2, 0.15, 0.05]"
```

**配置参数**:
```yaml
# MLModel/configs/default.yaml
window_samples: 4096
sample_rate: 44100

# 四类音位置分布
onset_distribution:  # 模拟滑动窗口效果
  - -1024  # 窗口前开始（第四类）
  - 0      # 窗口起点（第四类）
  - 1024   # 窗口1/4处（第二类/第三类）
  - 2048   # 窗口中间（第二类/第三类）
  - 3072   # 窗口3/4处（第二类/第三类）

# 音符长度分布（相对于窗口）
duration_distribution:
  min: 0.02   # 20ms（约2个周期@100Hz）
  max: 0.15   # 150ms（超出窗口）

# 噪声
snr_range: [-10, 40]  # dB

# 输出
chunk_size: 1000      # HDF5 chunk大小
compression: gzip
compression_level: 4
```

### 2.3 PyTorch Dataset 实现

```python
# Source/MLTraining/training/dataset_shards.py

import numpy as np
import torch
from pathlib import Path
import pickle

class MemoryCachedShardDataset(torch.utils.data.Dataset):
    """完全载入内存的shards数据集 (推荐<5GB)"""
    
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        
        with open(self.data_dir / 'meta.pkl', 'rb') as f:
            self.meta = pickle.load(f)
        
        self.total_samples = self.meta['total_samples']
        self.num_shards = self.meta['num_shards']
        
        print(f"Loading dataset into memory: {self.data_dir}")
        print(f"  Total samples: {self.total_samples}")
        
        # 加载所有分片
        waveforms_list = []
        confs_list = []
        energies_list = []
        
        for i in range(self.num_shards):
            shard_path = self.data_dir / f"shard_{i:05d}.npz"
            shard = np.load(shard_path)
            waveforms_list.append(shard['waveforms'])
            confs_list.append(shard['confs'])
            energies_list.append(shard['energies'])
        
        # 合并
        self.waveforms = np.concatenate(waveforms_list, axis=0)
        self.confs = np.concatenate(confs_list, axis=0)
        self.energies = np.concatenate(energies_list, axis=0)
        
        mem_mb = (self.waveforms.nbytes + self.confs.nbytes + 
                 self.energies.nbytes) / 1024 / 1024
        print(f"  Memory usage: {mem_mb:.1f} MB")
    
    def __len__(self):
        return self.total_samples
    
    def __getitem__(self, idx):
        return {
            'waveform': torch.from_numpy(self.waveforms[idx]).unsqueeze(0),
            'target_confidence': torch.from_numpy(self.confs[idx]),
            'target_energy': torch.from_numpy(self.energies[idx])
        }


class ShardDataset(torch.utils.data.Dataset):
    """内存映射Shards (适合大数据集)"""
    
    def __init__(self, data_dir: str, cache_size: int = 100):
        self.data_dir = Path(data_dir)
        
        with open(self.data_dir / 'meta.pkl', 'rb') as f:
            self.meta = pickle.load(f)
        
        self.total_samples = self.meta['total_samples']
        self.num_shards = self.meta['num_shards']
        
        # 内存映射
        self.shards = []
        self.shard_sizes = []
        for i in range(self.num_shards):
            shard_path = self.data_dir / f"shard_{i:05d}.npz"
            shard = np.load(shard_path, mmap_mode='r')
            self.shards.append(shard)
            self.shard_sizes.append(len(shard['waveforms']))
        
        # 计算偏移
        self.shard_offsets = [0]
        for size in self.shard_sizes[:-1]:
            self.shard_offsets.append(self.shard_offsets[-1] + size)
        
        # LRU缓存
        self.cache = {}
        self.cache_keys = []
        self.cache_size = cache_size
    
    def __len__(self):
        return self.total_samples
    
    def __getitem__(self, idx):
        if idx in self.cache:
            return self.cache[idx]
        
        # 找到分片
        for shard_idx, offset in enumerate(self.shard_offsets):
            if idx < offset + self.shard_sizes[shard_idx]:
                local_idx = idx - offset
                shard = self.shards[shard_idx]
                
                sample = {
                    'waveform': np.array(shard['waveforms'][local_idx]),
                    'target_confidence': np.array(shard['confs'][local_idx]),
                    'target_energy': np.array(shard['energies'][local_idx]),
                }
                break
        
        # 更新缓存
        self.cache[idx] = sample
        self.cache_keys.append(idx)
        if len(self.cache_keys) > self.cache_size:
            del self.cache[self.cache_keys.pop(0)]
        
        return sample
```

### 2.4 存储格式选择

#### 2.4.1 当前格式 (推荐)

**架构设计**:
- **dataset_writer.py**: 抽象工具类，管理数据存储/分片/压缩
- **generate_*.py**: 具体生成脚本，只关注数据生成逻辑

**DatasetWriter 功能**:
- 自动分片管理 (按波形数据大小)
- 波形 .npy 不压缩 (支持内存映射)
- 真值 .npz 压缩 (confs + energies 合并)
- 上下文管理器支持 (`with` 语句)

**文件结构**:
```
TrainingData/SingleSanity/
├── meta.pkl
├── waveforms/            # .npy 格式，不压缩，支持 mmap
│   ├── shard_00000.npy   # [N, 4096] float32
│   └── ...
└── targets/              # .npz 格式，confs + energies 合并压缩
    ├── shard_00000.npz   # confs: [N, 2048], energies: [N, 2048]
    └── ...
```

**存储估算** (20K样本):

| 格式 | 大小 | 读取速度 | 适用场景 |
|------|-----|---------|---------|
| HDF5 | 33 MB | 5500 s/s | 小规模测试 |
| Shards V1 | 291 MB | 9000 s/s | 旧版兼容 |
| **当前格式** | **~330 MB** | **9000 s/s** | **大规模训练** ✅ |

**生成命令**:
```bash
# Sanity 数据集 (单音)
python3 generate_sanity.py \
    --output ../../../TrainingData/SingleSanity \
    --samples-per-bin 10 \
    --shard-size-gb 1.0

# 多音数据集
python3 generate_polyphony.py \
    --output ../../../TrainingData/Polyphony \
    --num-samples 10000
```

**自定义数据集**:
```python
from data_generation.dataset_writer import DatasetWriter, generate_single_note

with DatasetWriter('my_dataset', shard_size_gb=1.0) as writer:
    for bin_idx in range(2048):
        sample = generate_single_note(bin_idx=bin_idx)
        writer.add_sample(
            waveform=sample['waveform'],
            confs=sample['confs'],
            energies=sample['energies']
        )
```

#### 2.4.2 随机相位增强 (关键！)

**问题**: 模型对窗口位置敏感（非整数周期导致频谱泄漏）

**解决方案**:
```python
# 生成时随机相位
phase = rng.uniform(0, 2 * np.pi)
waveform = np.sin(2 * np.pi * freq * t + phase) * 0.8
```

**效果**:
- 220Hz: +13.9 ± 25.3 cents → **+1.3 ± 3.3 cents** (8x改进)
- 440Hz: +4.8 ± 12.6 cents → **+2.0 ± 2.3 cents** (5x改进)

### 2.5 Phase 2 完成标准

- [ ] **2.1** 实现 `WindowedSampleGenerator` 类（四类音策略）
- [ ] **2.2** 实现 HDF5 生成脚本 `generate_dataset.py`
- [ ] **2.3** 实现 PyTorch `PitchDetectionDataset`
- [ ] **2.4** 生成测试数据集（1K样本），验证target_confidence分布合理
- [ ] **2.5** 可视化工具：检查onset/offset分布和对应的置信度权重

---

## Phase 3: 小数据训练测试 (Week 5)

**目标**: 在完整训练前，用最简单的场景验证训练流程

**为什么需要**: 
- 快速验证数据→模型→训练→导出的完整链路
- 排除复杂数据带来的干扰，先确保基础功能正常
- 如果最简单的场景都训不出来，说明架构或实现有问题

### 3.1 极简数据集生成

**数据约束**（最简单的场景）:
```yaml
# TrainingData/configs/sanity_check.yaml
timbre: "sine"           # 只有正弦波（无谐波干扰）
note_type: 4             # 只有第四类音（全程持续，onset=0, offset>=4096）
polyphony: 1             # 单音
velocity: 1.0            # 固定最大音量
frequency_range:         # 覆盖所有2048个bin
  min: 20
  max: 5000
  distribution: "log_uniform_per_bin"

# 生成数量
samples_per_bin: 10      # 每个频率bin 10个样本
total_samples: 20480     # 2048 bins × 10
```

**生成命令** (使用NumPy Shards):
```bash
cd Source/MLTraining/data_generation

# 推荐：NumPy Shards格式 (随机相位增强)
python3 generate_sanity_shards.py \
    --output ../../../TrainingData/sanity_shards \
    --samples-per-bin 10 \
    --shard-size 2048

# 输出：
# - 20480 samples (2048 bins × 10 phases)
# - 10 shards (~29MB each)
# - Total: ~291MB
```

**关键：随机相位增强**:
```python
# 每个样本使用随机初始相位
phase = rng.uniform(0, 2 * np.pi)
waveform = np.sin(2 * np.pi * freq * t + phase) * 0.8
```

这解决了模型对窗口位置敏感的问题。

### 3.2 PitchNet-Baseline 模型实现

```python
# Source/MLTraining/training/model.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class PitchNetBaseline(nn.Module):
    """
    PitchNet Baseline 模型
    
    架构:
    - 前端: Conv1d 1->64, kernel=512, stride=128 (提取时频特征)
    - 主干: 3层 Conv 进行时序压缩
    - 池化: 全局平均池化
    - 双头: 独立输出 confidence 和 energy
    
    输出:
    - confidence: Sigmoid激活 (0-1，每个bin独立概率)
    - energy: Softmax激活 (总和=1，归一化分布)
    """
    
    def __init__(self, input_samples: int = 4096, num_bins: int = 2048):
        super().__init__()
        
        # 前端: 可学习滤波器组
        self.frontend = nn.Conv1d(
            in_channels=1, out_channels=64,
            kernel_size=512, stride=128, padding=256
        )  # [B, 64, 32]
        
        # 主干: 时序压缩
        self.backbone = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(128), nn.GELU(),
            nn.Conv1d(128, 256, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(256), nn.GELU(),
            nn.Conv1d(256, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(256), nn.GELU(),
        )  # [B, 256, 4]
        
        # 全局池化
        self.pool = nn.AdaptiveAvgPool1d(1)
        
        # Confidence 头 (Sigmoid激活)
        self.conf_head = nn.Sequential(
            nn.Linear(256, 1024), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(1024, num_bins), nn.Sigmoid()
        )
        
        # Energy 头 (Softmax激活)
        self.energy_head = nn.Sequential(
            nn.Linear(256, 1024), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(1024, num_bins),
            # Softmax在forward中应用
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None: nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args: x: [B, 1, 4096]
        Returns: [B, 2048, 2] - [confidence, energy]
        """
        x = self.frontend(x)      # [B, 64, 32]
        x = self.backbone(x)      # [B, 256, 4]
        x = self.pool(x).squeeze(-1)  # [B, 256]
        
        conf = self.conf_head(x)  # [B, 2048], Sigmoid
        energy = self.energy_head(x)  # [B, 2048]
        energy = F.softmax(energy, dim=-1)  # Softmax (总和=1)
        
        return torch.stack([conf, energy], dim=-1)  # [B, 2048, 2]
    
    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

# 参数量: ~2.2M
```

### 3.3 训练配置

```yaml
# Source/MLTraining/training/configs/sanity_check.yaml
model:
  name: "PitchNetBaseline"
  input_samples: 4096
  num_bins: 2048

training:
  epochs: 50
  batch_size: 64
  learning_rate: 0.001
  optimizer: "AdamW"
  weight_decay: 1e-4
  scheduler: "CosineAnnealingWarmRestarts"
  
  # 损失权重
  loss_conf_weight: 1.0
  loss_energy_weight: 0.5
  loss_sparsity_weight: 0.01
  
  # 激活函数
  # - confidence: Sigmoid (0-1, 独立概率)
  # - energy: Softmax (总和=1, 归一化分布)
  
  # 验证
  val_split: 0.1
  
  # 早停
  early_stop:
    patience: 10
    min_delta: 0.0001
  
data:
  train_path: "TrainingData/test_data/sanity_check_20480.hdf5"
  val_path: null  # 从训练集划分
```

### 3.4 验证标准

| 指标 | 最低要求 | 目标 |
|------|---------|------|
| **Confidence Loss** | < 0.1 | < 0.05 |
| **Energy Loss** | < 0.05 | < 0.02 |
| **Bin Accuracy** | > 80% | > 95% |
| **频率误差 (cents)** | < 50 | < 10 |

**Bin Accuracy定义**: 预测峰值bin与真值bin的匹配率

### 3.5 调试检查清单

如果训练失败（损失不下降/准确率不达标）：

- [ ] **数据检查**: 可视化波形和真值是否对齐
- [ ] **模型检查**: 确认输出维度正确 [batch, 2048, 2]
- [ ] **损失检查**: 确认能量损失使用Softmax输出
- [ ] **梯度检查**: 确认梯度在正常流动（不过大/过小）
- [ ] **学习率检查**: 尝试更大/更小的学习率

**Phase 3 完成标准** (✅ 已完成):
- [x] 能成功训练50个epoch，损失持续下降 (Val Loss: 0.209)
- [x] Bin Accuracy > 80% (实际 > 95%)
- [x] 能导出ONNX并在C++中加载 (pitchnet_phaseaug_*.onnx)
- [x] 对正弦波输入能预测正确频率（±3 cents）

**实际结果**:
- 数据集: sanity_shards (20480 samples, 随机相位)
- 训练时间: ~5分钟 (MPS GPU加速)
- 验证损失: 0.209 (vs 0.245 固定相位版本)
- Streaming误差: 220Hz ±3.3 cents, 440Hz ±2.3 cents, 880Hz ±2.3 cents

---

## Phase 4: 架构对比与优化 (Week 6)

**目标**: 基于小数据训练的经验，对比不同架构，确定最终模型

**为什么放在小数据训练之后**:
- 小数据训练验证了基础训练流程是通的
- 排除数据问题干扰，专注比较架构本身
- 快速迭代，不需要大数据集训练时间

### 4.1 候选架构

基于小数据训练的结果，设计以下候选架构：

#### 4.1.1 PitchNet-Baseline (参考架构)

```python
class PitchNetBaseline(nn.Module):
    def __init__(self):
        super().__init__()
        # 前端: 可学习滤波器组
        self.frontend = nn.Conv1d(1, 64, kernel_size=512, stride=128, padding=256)
        
        # 主干: 时序压缩
        self.backbone = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(128),
            nn.GELU(),
            
            nn.Conv1d(128, 256, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(256),
            nn.GELU(),
            
            nn.Conv1d(256, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(256),
            nn.GELU(),
        )
        
        # 全局池化
        self.pool = nn.AdaptiveAvgPool1d(1)
        
        # 双输出头
        self.conf_head = nn.Sequential(
            nn.Linear(256, 1024),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 2048),
            nn.Sigmoid()  # Confidence: 0-1
        )
        
        self.energy_head = nn.Sequential(
            nn.Linear(256, 1024),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 2048),
            nn.Softplus()  # Energy: >=0
        )
    
    def forward(self, x):
        # x: [B, 1, 4096]
        x = self.frontend(x)  # [B, 64, 32]
        x = self.backbone(x)  # [B, 256, 4]
        x = self.pool(x).squeeze(-1)  # [B, 256]
        
        conf = self.conf_head(x)  # [B, 2048]
        energy = self.energy_head(x)  # [B, 2048]
        
        return torch.stack([conf, energy], dim=-1)  # [B, 2048, 2]

# 参数量: ~2.2M
```

#### 4.1.2 PitchNet-Lite (轻量化)

```python
class PitchNetLite(nn.Module):
    """轻量化版本，前端和主干通道数减半"""
    def __init__(self):
        super().__init__()
        self.frontend = nn.Conv1d(1, 32, kernel_size=512, stride=128, padding=256)
        
        self.backbone = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(64),
            nn.GELU(),
            
            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(128),
            nn.GELU(),
            
            # 少一层
        )
        
        self.pool = nn.AdaptiveAvgPool1d(1)
        
        self.conf_head = nn.Sequential(
            nn.Linear(128, 512),
            nn.GELU(),
            nn.Linear(512, 2048),
            nn.Sigmoid()
        )
        
        self.energy_head = nn.Sequential(
            nn.Linear(128, 512),
            nn.GELU(),
            nn.Linear(512, 2048),
            nn.Softplus()
        )
    
    def forward(self, x):
        x = self.frontend(x)
        x = self.backbone(x)
        x = self.pool(x).squeeze(-1)
        conf = self.conf_head(x)
        energy = self.energy_head(x)
        return torch.stack([conf, energy], dim=-1)

# 参数量: ~580K
```

#### 4.1.3 PitchNet-Tiny (极简)

```python
class PitchNetTiny(nn.Module):
    """极简版本，用于资源极度受限场景"""
    def __init__(self):
        super().__init__()
        # 前端大幅简化
        self.frontend = nn.Conv1d(1, 16, kernel_size=256, stride=64, padding=128)
        
        # 简单主干
        self.backbone = nn.Sequential(
            nn.Conv1d(16, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        
        self.pool = nn.AdaptiveAvgPool1d(1)
        
        # 共享部分特征后分叉
        self.shared = nn.Linear(64, 256)
        self.conf_head = nn.Linear(256, 2048)
        self.energy_head = nn.Linear(256, 2048)
    
    def forward(self, x):
        x = self.frontend(x)
        x = self.backbone(x)
        x = self.pool(x).squeeze(-1)
        x = F.relu(self.shared(x))
        conf = torch.sigmoid(self.conf_head(x))
        energy = F.softplus(self.energy_head(x))
        return torch.stack([conf, energy], dim=-1)

# 参数量: ~180K
```

### 4.2 对比实验设计

**在小数据集上对比（sanity check dataset）**:

| 指标 | Baseline | Lite | Tiny |
|------|----------|------|------|
| **参数量** | ~2.2M | ~580K | ~180K |
| **推理延迟** (M1) | 目标 <10ms | 目标 <5ms | 目标 <3ms |
| **Bin Accuracy** | 目标 >95% | 目标 >92% | 目标 >85% |
| **Cent误差** | 目标 <10 | 目标 <15 | 目标 <25 |

**实验流程**:
1. 三个架构分别在小数据集上训练50 epoch
2. 记录训练时间、收敛速度
3. 测试推理延迟（100次平均）
4. 选择**精度/速度权衡最佳**的架构

### 4.3 输出格式确定（已确定）

基于Phase 1验证结果：

```python
# 输出格式: [B, 2048, 2]
# - dim 0: confidence (Sigmoid激活) -> [0, 1]
# - dim 1: energy (Softplus激活) -> [0, ∞)

output = model(waveform)  # [B, 2048, 2]
confidence = output[..., 0]   # [B, 2048]
energy = output[..., 1]       # [B, 2048]
```

**不需要实验对比的方案**:
- 两个独立头（confidence/energy分开输出）→ 已经确定用stack方式
- 其他激活函数 → 确定Sigmoid(confidence) + Softplus(energy)

### 4.4 模型导出

**最终选定架构后**:

- [ ] **4.4.1** 使用Xavier/He初始化权重
- [ ] **4.4.2** 导出为ONNX: `pitchnet_trainable_v1.onnx`
- [ ] **4.4.3** 验证C++能正确加载
- [ ] **4.4.4** 确认输出维度: 输入 `[1, 1, 4096]` -> 输出 `[1, 2048, 2]`

**Phase 4 完成标准**:
- [ ] 确定最终网络架构（Baseline/Lite/Tiny之一）
- [ ] 有初始化权重的可训练ONNX模型
- [ ] C++代码无需修改即可加载新模型
- [ ] 在小数据上验证精度 > 80%

---

## Phase 5: 大规模训练 (Week 7-8)

**目标**: 在通用数据集上训练最终模型

### 5.1 训练基础设施

**目录结构**:
```
Source/MLTraining/training/
├── train.py              # 主训练循环
├── model.py              # 网络定义（从Phase 4选定的架构）
├── dataset.py            # 数据加载（已实现）
├── loss.py               # 损失函数
├── config.yaml           # 训练配置
└── utils/
    ├── metrics.py        # 评估指标
    ├── checkpoint.py     # 模型保存/加载
    └── visualization.py  # 可视化工具
```

### 5.2 数据集准备

**训练集**: 100K-300K样本，通用分布
```bash
# 生成通用训练集
python Source/MLTraining/data_generation/hdf5_writer.py \
    --output TrainingData/train_general_100k.hdf5 \
    --num-samples 100000
```

**验证集**: 10K样本，固定种子
```bash
# 生成验证集（固定种子确保可复现）
python generate_validation_set.py \
    --output TrainingData/val_10k.hdf5 \
    --num-samples 10000 \
    --seed 42
```

### 5.3 训练配置

```yaml
# configs/train_full.yaml
model:
  name: "PitchNet-Baseline"  # 或 Lite/Tiny，从Phase 4选定
  pretrained: null           # 可从sanity check的权重初始化

data:
  train_path: "TrainingData/train_general_100k.hdf5"
  val_path: "TrainingData/val_10k.hdf5"
  batch_size: 64
  num_workers: 4

training:
  epochs: 200
  optimizer: "AdamW"
  lr: 0.001
  weight_decay: 1e-4
  scheduler: "CosineAnnealingWarmRestarts"
  
  # 损失权重（基于Phase 3经验调整）
  loss_conf_weight: 1.0
  loss_energy_weight: 0.5
  loss_sparsity_weight: 0.01  # 可选：稀疏性约束
  
  # 早停
  early_stop:
    patience: 20
    min_delta: 0.0001
  
  # 检查点
  checkpoint_dir: "checkpoints/"
  save_every: 10  # 每10个epoch保存

logging:
  use_tensorboard: true
  log_dir: "runs/"
  log_every: 100  # 每100个batch记录
```

### 5.4 损失函数

```python
class PitchDetectionLoss(nn.Module):
    """
    组合损失函数
    
    设计要点:
    - Energy Loss 使用 target_conf 作为软权重 (不推荐用 pred_conf，避免循环依赖)
    - 高 confidence bin 对 energy loss 贡献大，低 confidence bin 贡献小但仍有梯度
    """
    def __init__(self, conf_weight=1.0, energy_weight=0.5, sparsity_weight=0.01):
        super().__init__()
        self.conf_weight = conf_weight
        self.energy_weight = energy_weight
        self.sparsity_weight = sparsity_weight
        
        self.bce = nn.BCELoss()
    
    def forward(self, pred, target):
        """
        Args:
            pred: [B, 2048, 2] - model output
            target: dict with 'confidence' and 'energy'
        """
        pred_conf = pred[..., 0]      # [B, 2048]
        pred_energy = pred[..., 1]    # [B, 2048]
        
        target_conf = target['target_confidence']   # [B, 2048]
        target_energy = target['target_energy']     # [B, 2048]
        
        # 1. Confidence损失 (BCE) - 所有bin参与
        loss_conf = self.bce(pred_conf, target_conf)
        
        # 2. Energy损失 (MSE with soft weighting)
        # 使用 target_conf 作为软权重，而非二值mask
        # 原因: target_conf 是ground truth，可靠；pred_conf 会导致循环依赖
        weights = target_conf  # [B, 2048], 范围[0, 1]
        
        # 逐元素MSE，然后加权平均
        mse_per_bin = F.mse_loss(pred_energy, target_energy, reduction='none')
        loss_energy = (weights * mse_per_bin).sum() / (weights.sum() + 1e-8)
        
        # 3. 稀疏性约束 (抑制假阳性)
        loss_sparsity = pred_conf.mean()
        
        # 总损失
        total_loss = (self.conf_weight * loss_conf + 
                     self.energy_weight * loss_energy +
                     self.sparsity_weight * loss_sparsity)
        
        return {
            'total': total_loss,
            'confidence': loss_conf,
            'energy': loss_energy,
            'sparsity': loss_sparsity
        }
```

**设计说明**:

| 损失项 | 计算方式 | 权重 | 说明 |
|--------|---------|------|------|
| **Confidence** | BCE(所有bins) | 1.0 | 主任务，所有bin都要学 |
| **Energy** | MSE(soft加权) | 0.5 | 用target_conf加权，高conf bin贡献大 |
| **Sparsity** | mean(pred_conf) | 0.01 | 轻微抑制假阳性 |

**为什么不用 `pred_conf` 做mask**:
- ❌ 循环依赖：网络预测错误 → mask错误 → 无法学习修正
- ❌ 训练初期可能全预测0 → energy loss被mask → 梯度消失

**为什么用 `target_conf` 软权重而非硬mask**:
- ✅ 平滑学习：所有bin都有梯度，只是权重不同
- ✅ 高斯软标签自然反映bin重要性（峰值附近权重高，边缘权重低）
- ✅ 避免threshold选择问题

### 5.5 评估指标

**训练时监控**:
- [ ] **Loss曲线**: total/confidence/energy分别记录
- [ ] **Bin Accuracy**: Top-1 bin匹配率
- [ ] **Frequency Error**: 预测频率与真值的cents误差

**验证时评估**:
- [ ] **Precision/Recall**: 在不同confidence阈值下
- [ ] **F1 Score**: 综合指标
- [ ] **Multi-note Accuracy**: 多音检测准确率

### 5.6 训练执行

```bash
# 启动训练
cd Source/MLTraining/training
python train.py --config configs/train_full.yaml

# 监控
tensorboard --logdir runs/
```

### 5.7 模型导出与验证

- [ ] **5.7.1** 选择验证集上表现最佳的epoch
- [ ] **5.7.2** 导出ONNX: `MLModel/pitchnet_v1.onnx`
- [ ] **5.7.3** C++加载验证：确保输出合理（非随机）
- [ ] **5.7.4** 性能测试：确认延迟<20ms，内存占用合理

**Phase 5 完成标准**:
- [ ] 在100K+样本上训练完成
- [ ] 验证集Bin Accuracy > 85%
- [ ] 导出ONNX能在C++中正确推理
- [ ] 实时测试：输入乐器音频能得到合理音高

---

## Phase 6: 迭代优化 (Week 9+)

**目标**: 持续优化模型精度和效率

### 6.1 精度优化

- [ ] **6.1.1** 错误分析
  - [ ] 收集验证集上的失败案例
  - [ ] 分析: 低音乐器? 多音重叠? 噪声?

- [ ] **6.1.2** 数据增强调优
  - [ ] 根据错误分析增强特定场景数据
  - [ ] 增加困难样本 (强噪声、快速连音)

- [ ] **6.1.3** 架构调优
  - [ ] 实验不同的前端kernel size
  - [ ] 实验注意力机制变体

### 6.2 效率优化

- [ ] **6.2.1** 量化实验
  - [ ] INT8量化训练 (QAT)
  - [ ] 对比FP32和INT8的精度损失

- [ ] **6.2.2** 剪枝实验
  - [ ] 尝试结构化剪枝
  - [ ] 减小模型到Lite版本

### 6.3 部署优化

- [ ] **6.3.1** 多模型策略
  - [ ] 高质量模式 (大模型，高精度)
  - [ ] 省电模式 (小模型，低功耗)

- [ ] **6.3.2** A/B测试
  - [ ] 与原有FFT算法对比
  - [ ] 收集用户反馈

---

## 附录: 快速开始命令

### 环境搭建

```bash
# 创建conda环境（推荐）
cd MLModel/
conda env create -f environment.yml
conda activate spm_ml

# 或者手动创建
conda create -n spm_ml python=3.10
conda activate spm_ml
pip install torch torchvision torchaudio
pip install onnx onnxruntime
pip install numpy librosa soundfile h5py
pip install tensorboard

# C++环境 (macOS示例)
brew install cmake
```

### Phase 2: 数据生成

```bash
# 1. 生成sanity check数据集（正弦波，第四类音，单音）
cd TrainingData/test_data
./generate_test.sh

# 2. 生成通用训练数据
cd ../../Source/MLTraining/data_generation
python hdf5_writer.py \
    --output ../../../TrainingData/train_general_100k.hdf5 \
    --num-samples 100000
```

### Phase 3: 小数据训练测试

```bash
cd Source/MLTraining/training

# 使用sanity check数据集训练
python train.py --config configs/train_sanity.yaml

# 验证
python evaluate.py --model checkpoints/sanity_best.pth
```

### Phase 5: 大规模训练

```bash
cd Source/MLTraining/training

# 正式训练
python train.py --config configs/train_full.yaml

# 导出最佳模型
python export_model.py --checkpoint checkpoints/best.pth
```

---

## 关键检查点

| 里程碑 | 验收标准 | 预计时间 |
|--------|---------|---------|
| **M1** | C++加载ONNX模型并得到输出 | Week 1 ✅ |
| **M2** | JUCE集成完成，UI能看到数值变化 | Week 2 ✅ |
| **M3** | 训练数据结构确定，可生成10万样本 | Week 4 |
| **M4** | 小数据训练通过，验证训练流程 | Week 5 |
| **M5** | 最终架构确定，可训练模型导出 | Week 6 |
| **M6** | 大规模训练完成，能检测正弦波 | Week 8 |
| **M7** | 多音检测准确率达标 | Week 9 |

---

## 设计决策记录

### 决策1: 预生成窗口 vs 滑动窗口
**选择**: 预生成4096-sample固定窗口
**理由**: 生成时即知真值，避免训练时复杂的时间查询；通过多位置变体模拟滑动窗口效果

### 决策2: 四类音置信度策略
**选择**: 仅第二类和第四类标注为confidence=1.0
**理由**: 网络只应学习"确信存在"的音，避免对片段/结尾音的假阳性训练

### 决策3: int16采样点索引 vs float32时间
**选择**: int16(onset_idx, offset_idx)
**理由**: 存储减半、无浮点精度问题、可直接当array index

### 决策4: 生成时预计算标签
**选择**: HDF5中只存3个tensor（waveform, target_conf, target_energy）
**理由**: 训练代码极简（直接读取），标签逻辑集中在一处易于调试

---

**文档版本**: 3.0  
**最后更新**: 2026-02-19 
- 重构Phase顺序：小数据训练(3) → 架构对比(4) → 大规模训练(5)
- 更新Phase内容：基于最新设计决策
- 添加Phase 2代码实现细节

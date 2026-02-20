# 训练数据生成完整指南

**版本**: 2.0  
**日期**: 2026-02-19

---

## 1. 整体架构

### 1.1 核心设计原则

**预生成固定窗口**（vs 滑动窗口切分长音频）

```
原方案（滑动窗口）:
生成长音频(10s) ──切分──► [W0][W1][W2]... ──查询真值──► 标签
    ↓
  慢(磁盘I/O + 重复查询)

新方案（预生成窗口）:
直接生成 4096-sample 短样本 + 预计算标签 ──保存──► NumPy Shards
    ↓
  快(生成时即知真值，训练时直接读取)
```

**优势**:
- 生成时即完成所有计算，训练时零计算
- 避免复杂的时间重叠查询
- 支持内存映射，适合大规模训练(20-30GB)

### 1.2 窗口设计

- **长度**: 4096 samples @ 44.1kHz = **93ms**
- **覆盖**: 低频20Hz约2个完整周期（50ms×2=100ms>93ms，足够）
- **相位增强**: 每个样本随机初始相位，提高模型鲁棒性

---

## 2. 存储格式对比

### 2.1 三种格式概览

| 格式 | 适用场景 | 压缩率 | 读取速度 | 文件数量 |
|------|---------|--------|---------|---------|
| **HDF5** | 小规模测试(<1GB) | 高(zip) | 中等 | 单文件 |
| **NumPy Shards** | 大规模训练(推荐) | 中 | 快 | 多文件(10-50) |
| **Sparse Shards** | 超大规模(实验性) | 极高 | 快 | 分离存储 |

### 2.2 NumPy Shards (推荐)

**文件结构**:

```
TrainingData/sanity_shards/
├── meta.pkl              # 元数据
├── shard_00000.npz       # 分片0 (~30MB)
├── shard_00001.npz       # 分片1
├── ...
└── shard_00009.npz       # 分片9
```

**分片内容** (每个.npz):
- `waveforms`: [N, 4096] float32 - 音频波形
- `confs`: [N, 2048] float32 - 目标置信度
- `energies`: [N, 2048] float32 - 目标能量

**优势**:
- **分片存储**: 每个shard 30MB，避免单文件过大
- **内存映射**: `mmap_mode='r'`，按需加载
- **随机相位**: 每个bin生成多个相位版本，提高模型鲁棒性

**生成命令**:

```bash
cd Source/MLTraining/data_generation
python3 generate_sanity_shards.py \
    --output ../../../TrainingData/sanity_shards \
    --samples-per-bin 10 \
    --shard-size 2048
```

**使用方式**:

```python
from training.dataset_shards import MemoryCachedShardDataset

# 完全载入内存 (适合<5GB)
dataset = MemoryCachedShardDataset('TrainingData/sanity_shards')

# 或使用内存映射 (适合大数据集)
from training.dataset_shards import ShardDataset
dataset = ShardDataset('TrainingData/sanity_shards', cache_size=100)
```

### 2.3 Sparse Shards (分离存储)

**核心洞察**: 频谱数据非常稀疏（单音只有3-7个bin非零）

**文件结构**:

```
TrainingData/sanity_separate/
├── meta.pkl
├── waveforms/            # 波形分片 (不压缩)
│   ├── shard_00000.npy   # [N, 4096] float32
│   ├── shard_00001.npy
│   └── ...
└── spectra/              # 稀疏频谱 (紧凑存储)
    ├── indices_00000.npy   # [N, 7] int16 - 非零bin索引
    ├── confs_00000.npy     # [N, 7] float16 - confidence值
    ├── energies_00000.npy  # [N, 7] float16 - energy值
    └── ...
```

**存储效率对比** (20480样本):

| 数据类型 | 密集格式 | 稀疏格式 | 节省 |
|---------|---------|---------|-----|
| 波形 | 320 MB | 320 MB | - |
| 频谱 | 320 MB | **0.8 MB** | **99.7%** |
| **总计** | 640 MB | **321 MB** | **50%** |

**注意**: 实际项目中使用np.savez_compressed后，HDF5和Shards格式已经压缩到~33MB，稀疏格式优势不明显。建议仅在**原始数据>10GB**时考虑。

### 2.4 HDF5 (已废弃)

**警告**: HDF5格式不再推荐用于大规模训练

**问题**:
- 单文件过大容易损坏
- 随机读取性能差
- 压缩/解压开销大

**仅用于**小规模测试(<1000样本):

```python
from training.dataset import PitchDataset
dataset = PitchDataset('test_data/sanity_check_1000.hdf5')
```

---

## 3. 四类音策略（核心设计）

### 3.1 分类定义

每个4096-sample窗口中，音符按与窗口的重叠方式分为四类：

| 类型 | 图示 | onset | offset | 物理含义 | 置信度 | 能量处理 |
|------|------|-------|--------|---------|--------|----------|
| **第一类** | `\|███\|` | ≤0 | <4096 | 延续音，窗口内结束 | Soft(<0.3) | 打折(×0.5) |
| **第二类** | `\|  ██████\|` | >0 | ≥4096 | 新开始，持续到窗口后 | Hard=1.0 ✅ | 完整 |
| **第三类** | `\|  ██ \|` | >0 | <4096 | 短音，窗口内开始并结束 | Soft(<0.3) | 打折(×0.5) |
| **第四类** | `\|███████\|` | ≤0 | ≥4096 | 全程持续 | Hard=1.0 ✅ | 完整 |

### 3.2 统一处理策略

**核心原则**: 信息完整的音（第二/四类）→ 高confidence + 完整能量

**第一/三类音惩罚原因**:
- 音实际已结束或太短，信息量不足
- 不应"抢走"能量占比
- 网络应学会：不完整的音 = 低confidence + 低energy

```python
def classify_note(onset_idx, offset_idx, window_size=4096):
    if onset_idx <= 0 and offset_idx >= window_size:
        return 4  # 第四类
    elif onset_idx > 0 and offset_idx >= window_size:
        return 2  # 第二类
    elif onset_idx <= 0 and offset_idx < window_size:
        return 1  # 第一类
    else:
        return 3  # 第三类

def compute_confidence_weight(note_type, visible_ratio):
    if note_type in [2, 4]:
        return 1.0
    else:
        return min(visible_ratio * 0.5, 0.3)
```

---

## 4. 能量计算

### 4.1 频域快速计算

**核心洞察**: 已知音色（谐波结构）可直接频域计算能量，无需时域合成

**公式**:
```
E = coeff × velocity² × ADSR_factor × (visible_penalty)

coeff = Σ(A_k²)  # 预计算音色系数
        sine: 1.00
        sawtooth: 1.64 (π²/6)
        piano: ~2.0
```

**vs 时域合成加速**: ~200x（20次乘法 vs 4096次乘加）

### 4.2 统一能量策略

```python
def compute_energy(note, timbre_profile):
    # 基础能量
    base = timbre_profile.energy_coeff * (note.velocity ** 2)
    
    # 可见范围
    visible_start = max(note.onset_idx, 0)
    visible_end = min(note.offset_idx, 4096)
    
    # ADSR因子（区间平均）
    adsr = compute_adsr_average(timbre_profile, visible_start, visible_end)
    
    # 类型处理（与置信度统一）
    note_type = classify_note(note.onset_idx, note.offset_idx)
    
    if note_type in [2, 4]:
        # 完整存在，不打折
        return base * adsr
    else:
        # 第一/三类，能量打折
        visible_ratio = (visible_end - visible_start) / 4096
        return base * adsr * visible_ratio * 0.5  # 额外50%惩罚
```

### 4.3 多音能量占比

```python
# 计算所有音能量
energies = [compute_energy(n, profile) for n in notes]

# 归一化占比
ratios = energies / sum(energies)

# 生成target_energy（只标注基频，高斯平滑）
target_energy = np.zeros(2048)
for note, ratio in zip(notes, ratios):
    bin_idx = freq_to_bin(note.frequency)
    target_energy += gaussian_1d(bin_idx, ratio, sigma=1.5)
```

**关键**: 只标注基频bin，让网络学习从泛音识别基频

---

## 5. 随机相位增强

### 5.1 问题背景

**原始问题**: 模型对窗口起始位置敏感

| 窗口位置 | 预测频率 | 置信度 | 误差 |
|---------|---------|--------|-----|
| 文件开始(0) | 220.0 Hz | 0.866 | ✅ |
| +512 samples | 226.6 Hz | 0.312 | ❌ |
| 中间位置 | 217.1 Hz | 0.695 | ⚠️ |

**原因**: 非整数周期导致频谱泄漏，窗口滑动改变泄漏程度

### 5.2 解决方案

**数据增强**: 每个样本使用随机初始相位

```python
# 生成时随机相位
phase = rng.uniform(0, 2 * np.pi)
waveform = np.sin(2 * np.pi * freq * t + phase) * 0.8
```

**效果**:

| 频率 | 相位增强前 | 相位增强后 | 改进 |
|------|-----------|-----------|-----|
| 220Hz | +13.9 ± 25.3 cents | **+1.3 ± 3.3 cents** | 8x |
| 440Hz | +4.8 ± 12.6 cents | **+2.0 ± 2.3 cents** | 5x |
| 880Hz | 0.0 ± 5.2 cents | **-1.6 ± 2.3 cents** | 更稳定 |

---

## 6. 通用数据生成算法

### 6.1 随机采样策略

**目标**: 通过随机采样覆盖所有四类音场景

**采样流程**:

```python
def generate_random_sample():
    """
    通用随机样本生成算法
    覆盖所有四类音场景
    """
    # 1. 随机音数量 (1-16个)
    # 使用偏向少音的分布
    polyphony = random.choice([1, 1, 1, 2, 2, 2, 3, 3, 4, 5, 6, 7, 8, 10, 12, 16])
    
    notes = []
    for i in range(polyphony):
        # 2. 随机频率 (20-5000 Hz, 对数分布)
        freq = np.exp(random.uniform(np.log(20), np.log(5000)))
        
        # 3. 随机音色 (从100个音色库)
        timbre_id = random.randint(0, 99)
        
        # 4. 随机起始相位 (0-2π) - 关键！
        phase = random.uniform(0, 2 * np.pi)
        
        # 5. 随机 velocity (0.0-1.0)
        velocity = random.uniform(0.0, 1.0)
        
        # 6. 随机 onset_idx
        onset_raw = random.randint(-4095, 4095)
        onset_idx = max(onset_raw, 0)
        
        # 7. 随机 offset_idx
        offset_raw = random.randint(onset_idx, 8192)
        offset_idx = min(offset_raw, 8192)
    
        note = Note(
            frequency=freq,
            velocity=velocity,
            timbre=timbre_id,
            onset_idx=onset_idx,
            offset_idx=offset_idx,
            phase=phase  # 随机相位
        )
        notes.append(note)
    
    return notes
```

**四类音覆盖分析**:

| 条件组合 | 概率 | 类型 |
|---------|------|------|
| `onset_idx=0` AND `offset_idx>=4096` | ~25% | 第四类 (全程持续) |
| `onset_idx>0` AND `offset_idx>=4096` | ~25% | 第二类 (窗口内开始) |
| `onset_idx=0` AND `offset_idx<4096` | ~25% | 第一类 (窗口内结束) |
| `onset_idx>0` AND `offset_idx<4096` | ~25% | 第三类 (短音) |

---

## 7. 实现架构

### 7.1 核心类设计

```python
# Source/MLTraining/data_generation/

@dataclass
class Note:
    frequency: float      # Hz
    velocity: float       # 0.0-1.0
    timbre: int           # 0-99音色ID
    onset_idx: int        # 相对窗口起点
    offset_idx: int       # 相对窗口起点
    phase: float          # 初始相位(关键！)

def generate_sanity_shards(output_dir, samples_per_bin=10, shard_size=2048):
    """生成sanity check数据集"""
    for bin_idx in range(2048):
        freq = bin_to_freq(bin_idx)
        for _ in range(samples_per_bin):
            # 随机相位 - 关键增强
            phase = rng.uniform(0, 2 * np.pi)
            t = np.arange(4096) / 44100.0
            waveform = np.sin(2 * np.pi * freq * t + phase) * 0.8
            
            # 生成target...
```

### 7.2 PyTorch Dataset

```python
class MemoryCachedShardDataset(Dataset):
    """完全载入内存的shards数据集"""
    
    def __init__(self, data_dir):
        # 加载所有分片到内存
        for i in range(num_shards):
            shard = np.load(shard_path)
            waveforms.append(shard['waveforms'])
            confs.append(shard['confs'])
            energies.append(shard['energies'])
        
        self.waveforms = np.concatenate(waveforms)
        self.confs = np.concatenate(confs)
        self.energies = np.concatenate(energies)
    
    def __getitem__(self, idx):
        return {
            'waveform': torch.from_numpy(self.waveforms[idx]).unsqueeze(0),
            'target_confidence': torch.from_numpy(self.confs[idx]),
            'target_energy': torch.from_numpy(self.energies[idx])
        }
```

### 7.3 数据生成脚本

```bash
# 生成sanity check数据集（推荐）
cd Source/MLTraining/data_generation
python3 generate_sanity_shards.py \
    --output ../../../TrainingData/sanity_shards \
    --samples-per-bin 10 \
    --shard-size 2048

# 输出：
# - 20480 samples (2048 bins × 10 phases)
# - 10 shards (~30MB each)
# - Total: ~300MB
```

---

## 8. 性能对比

### 8.1 读取速度

| 格式 | 随机读取 | 顺序读取 | 内存占用 |
|------|---------|---------|---------|
| HDF5 | 5500 s/s | 8000 s/s | 高(缓存) |
| Shards (mmap) | 9000 s/s | 15000 s/s | 低(按需) |
| Shards (内存) | 50000 s/s | 50000 s/s | 640MB |

### 8.2 训练时间对比 (20K样本, 50 epochs)

| 配置 | 每epoch时间 | 总时间 | 瓶颈 |
|------|------------|--------|-----|
| HDF5 + CPU | ~90s | ~75min | IO读取 |
| HDF5 + MPS | ~7s | ~6min | 无 |
| **Shards + MPS** | **~4s** | **~3.5min** | **无** ✅ |

---

## 9. 设计决策总结

| 决策 | 选择 | 理由 |
|------|------|------|
| **存储格式** | NumPy Shards | 分片存储，内存映射，适合大数据集 |
| **窗口生成** | 预生成固定窗口 | 避免滑动窗口的I/O和查询开销 |
| **相位处理** | 随机初始相位 | 解决窗口位置敏感问题，提高鲁棒性 |
| **多音上限** | 16个音 | 复杂和弦可能超过8个音 |
| **置信度策略** | 四类音分类 | 信息完整的音才给高confidence |
| **能量策略** | 与置信度统一 | 不完整音能量打折，避免"抢占比" |
| **能量计算** | 频域直接计算 | 比时域合成快~200倍 |
| **标注位置** | 只标基频bin | 让网络学习泛音→基频映射 |

---

## 10. 快速参考

### 10.1 生成数据集

```bash
# Sanity Check (推荐用于实验)
python3 generate_sanity_shards.py \
    --output ../../../TrainingData/sanity_shards \
    --samples-per-bin 10

# 大规模数据集 (20-30GB)
python3 generate_dataset.py \
    --output ../../../TrainingData/train_shards \
    --num-samples 1000000 \
    --shard-size 10000
```

### 10.2 使用数据集

```python
from training.dataset_shards import MemoryCachedShardDataset
from torch.utils.data import DataLoader

# 加载数据集
dataset = MemoryCachedShardDataset('TrainingData/sanity_shards')

# 创建DataLoader
loader = DataLoader(dataset, batch_size=64, shuffle=True)

# 训练
for batch in loader:
    waveform = batch['waveform']  # [B, 1, 4096]
    target_conf = batch['target_confidence']  # [B, 2048]
    target_energy = batch['target_energy']  # [B, 2048]
```

### 10.3 文件命名规范

```
TrainingData/
├── sanity_shards/          # Sanity check (20K samples)
│   ├── meta.pkl
│   ├── shard_00000.npz
│   └── ...
├── train_shards/           # 训练集 (1M+ samples)
│   ├── meta.pkl
│   ├── shard_00000.npz
│   └── ...
└── test_data/              # 小规模测试
    └── sanity_check_1000.hdf5
```

---

**相关文档**:
- `Technical_Solution_Report_v1.0.md` - 整体技术方案
- `TODO_Implementation_Plan.md` - 实现路线图与任务清单

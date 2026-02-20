# 数据集格式说明

## 概述

SuperPitchMonitor训练支持两种数据集格式：

1. **NumPy Shards** (推荐) - 用于大规模训练
2. **HDF5** (已废弃) - 仅用于小规模测试

## NumPy Shards 格式 (推荐)

### 特点

- **分片存储**: 数据分为多个 `.npz` 文件，每个约 100-300MB
- **内存映射**: 支持 `mmap_mode`，按需加载，不占用大量RAM
- **顺序读取优化**: 适合大规模训练 (20-30GB+)
- **随机相位**: 每个样本有不同初始相位，提高模型鲁棒性

### 文件结构

```
TrainingData/sanity_shards/
├── meta.pkl              # 元数据 (样本数、分片数等)
├── shard_00000.npz       # 分片0
├── shard_00001.npz       # 分片1
├── ...
└── shard_00009.npz       # 分片9
```

### 分片内容

每个 `.npz` 文件包含：

- `waveforms`: [N, 4096] float32 - 音频波形
- `confs`: [N, 2048] float32 - 目标置信度
- `energies`: [N, 2048] float32 - 目标能量
- `freqs`: [N] float32 - 频率标签
- `bins`: [N] uint16 - bin索引

### 生成数据集

```bash
cd Source/MLTraining/data_generation
python3 generate_sanity_shards.py \
    --output ../../../TrainingData/sanity_shards \
    --samples-per-bin 10 \
    --shard-size 2048
```

参数:
- `--output`: 输出目录
- `--samples-per-bin`: 每个bin的样本数 (不同相位)
- `--shard-size`: 每个分片多少样本

### 使用数据集

```python
from training.dataset_shards import MemoryCachedShardDataset

# 完全载入内存 (适合<5GB数据集)
dataset = MemoryCachedShardDataset('TrainingData/sanity_shards')

# 或者使用内存映射 (适合大数据集)
from training.dataset_shards import ShardDataset
dataset = ShardDataset('TrainingData/sanity_shards', cache_size=100)
```

## HDF5 格式 (已废弃)

### 警告

**HDF5格式不再推荐用于大规模训练**，原因：
- 单文件过大 (>1GB) 容易损坏
- 随机读取性能差
- 压缩/解压开销大

### 仅用于小规模测试

```python
# 仅用于 <1000 样本的快速测试
from training.dataset import PitchDataset
dataset = PitchDataset('test_data/sanity_check_1000.hdf5')
```

## 性能对比

| 格式 | 文件大小 | 读取速度 | 适用场景 |
|------|---------|---------|---------|
| NumPy Shards | 291MB | ~9000 samples/sec | 20-30GB训练 |
| HDF5 | 33MB | ~5500 samples/sec | 小规模测试 |

## 训练脚本

### 使用Shards训练 (推荐)

```bash
cd Source/MLTraining/training
python3 train_live.py
```

脚本会自动使用 `TrainingData/sanity_shards`。

### 配置文件

```python
config = {
    'data_path': 'TrainingData/sanity_shards',  # shards目录
    'batch_size': 64,      # Mac M4
    # 'batch_size': 128,   # RTX 4080S
    'epochs': 50,
    'lr': 0.001,
    'val_split': 0.05,
}
```

## 数据增强

### 随机相位

生成时每个样本有随机初始相位：

```python
phase = rng.uniform(0, 2 * np.pi)
waveform = np.sin(2 * np.pi * freq * t + phase) * 0.8
```

这让模型学习对相位不敏感，适应streaming场景。

### 批量大小的选择

- **Mac M4 (16GB)**: batch_size=64
- **RTX 4080S (16GB)**: batch_size=128 或更高
- **CPU训练**: batch_size=32 (避免内存不足)

## 故障排除

### 内存不足

如果使用 `MemoryCachedShardDataset` 出现OOM：

```python
# 改用内存映射版本
from training.dataset_shards import ShardDataset
dataset = ShardDataset('TrainingData/sanity_shards', cache_size=50)
```

### 文件损坏

如果shard文件损坏，重新生成：

```bash
rm -rf TrainingData/sanity_shards
python3 generate_sanity_shards.py --output ../../../TrainingData/sanity_shards
```

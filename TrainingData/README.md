# TrainingData

此目录用于存储机器学习训练所需的数据集。

## 说明

本目录下的数据文件**不应提交到 Git 仓库**，因为它们通常很大且可以通过脚本重新生成。

## 生成训练数据

使用以下脚本生成训练数据：

```bash
cd Source/MLTraining/data_generation

# 生成 Sanity Check 数据集（单音正弦波）
python generate_sanity.py \
    --output ../../../TrainingData/SingleSanity \
    --samples-per-bin 10 \
    --shard-size-gb 1.0

# 生成多音数据集
python generate_polyphony.py \
    --output ../../../TrainingData/Polyphony \
    --num-samples 10000 \
    --shard-size-gb 1.0
```

## 目录结构

生成的数据集结构如下：

```
TrainingData/
├── SingleSanity/           # Sanity check 数据集
│   ├── meta.json           # 元数据
│   ├── waveforms/          # 波形数据 (.npy)
│   └── targets/            # 真值数据 (.npz)
├── Polyphony/              # 多音数据集
│   ├── meta.json
│   ├── waveforms/
│   └── targets/
└── ...                     # 其他数据集
```

## 数据格式

- **波形**: `.npy` 格式，float32，shape `[N, 4096]`
- **真值**: `.npz` 格式，包含 `confs` 和 `energies`，float16
- **元数据**: `.json` 格式，UTF-8 编码

## 读取数据

```python
from dataset_writer import DatasetReader

# 加载数据集
dataset = DatasetReader('TrainingData/SingleSanity', preload=False)

# 获取样本
sample = dataset[0]
waveform = sample['waveform']           # [1, 4096]
confs = sample['target_confidence']     # [2048]
energies = sample['target_energy']      # [2048]
```

## 注意事项

1. 首次克隆仓库后，需要运行生成脚本创建训练数据
2. 确保有足够的磁盘空间（单个数据集可能需要数GB）
3. 生成的数据可以根据需要删除和重新生成

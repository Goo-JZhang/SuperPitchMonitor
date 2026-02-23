#!/usr/bin/env python3
import sys
from pathlib import Path

# 添加路径
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / 'Model'))

try:
    print("Testing imports...")
    from PitchNetBaseline import PitchNetBaseline
    print("  PitchNetBaseline OK")
    
    from dataset import DatasetReader
    print("  DatasetReader OK")
    
    from torch.utils.data import random_split, ConcatDataset
    print("  PyTorch data utils OK")
    
    print("\nTesting data loading...")
    data_root = Path('../../../TrainingData')
    data_dirs = sorted([p for p in data_root.iterdir() if p.is_dir() and (p / 'meta.json').exists()])
    print(f"  Found {len(data_dirs)} datasets")
    
    val_split = 0.02
    train_datasets = []
    val_datasets = []
    
    for data_dir in data_dirs:
        print(f"  Loading {data_dir.name}...")
        ds = DatasetReader(str(data_dir), preload=False, device='cpu')
        n_val = max(1, int(len(ds) * val_split))
        n_train = len(ds) - n_val
        ds_train, ds_val = random_split(ds, [n_train, n_val])
        train_datasets.append(ds_train)
        val_datasets.append(ds_val)
        print(f"    {data_dir.name}: {len(ds)} -> Train: {n_train}, Val: {n_val}")
    
    train_dataset = ConcatDataset(train_datasets)
    val_dataset = ConcatDataset(val_datasets)
    
    print(f"\n  Total Train: {len(train_dataset)}")
    print(f"  Total Val: {len(val_dataset)}")
    
    # 测试取一个样本
    sample = train_dataset[0]
    print(f"\nSample keys: {sample.keys()}")
    print(f"Waveform shape: {sample['waveform'].shape}")
    
    print("\nAll tests passed!")
    
except Exception as e:
    import traceback
    print(f"\nError: {e}")
    traceback.print_exc()

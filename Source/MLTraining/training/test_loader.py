#!/usr/bin/env python3
import torch
from torch.utils.data import DataLoader, ConcatDataset
from dataset import DatasetReader
from pathlib import Path

data_root = Path('../../../TrainingData')
data_dirs = [data_root / 'SingleSanity', data_root / 'NoiseDataset']

print('Loading with preload=True...')
train_datasets = []
for d in data_dirs:
    ds = DatasetReader(str(d), preload=True, device='cpu')
    train_datasets.append(ds)
    print(f'  {d.name}: {len(ds)} samples')

print('Concatenating...')
train_ds = ConcatDataset(train_datasets)
print(f'Total: {len(train_ds)}')

print('Creating DataLoader...')
loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=0)

print('Testing first batch...')
for batch in loader:
    wf = batch['waveform']
    print(f'Batch shape: {wf.shape}')
    break

print('OK!')

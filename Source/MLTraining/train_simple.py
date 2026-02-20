#!/usr/bin/env python3
"""
简化训练脚本 - 无实时可视化，保存日志供后续绘图
"""

import os
import sys
import json
import time
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

sys.path.insert(0, str(Path(__file__).parent))
from model import PitchNetBaseline


class PitchDataset(torch.utils.data.Dataset):
    """HDF5数据集"""
    
    def __init__(self, hdf5_path):
        import h5py
        self.hdf5_path = hdf5_path
        with h5py.File(hdf5_path, 'r') as f:
            self.num_samples = f.attrs['num_samples']
        self._h5_file = None
    
    def __len__(self):
        return self.num_samples
    
    def _get_h5(self):
        if self._h5_file is None:
            import h5py
            self._h5_file = h5py.File(self.hdf5_path, 'r')
        return self._h5_file
    
    def __getitem__(self, idx):
        h5 = self._get_h5()
        waveform = torch.from_numpy(h5['data/waveform'][idx].copy()).float()
        target_conf = torch.from_numpy(h5['data/target_confidence'][idx].copy()).float()
        target_energy = torch.from_numpy(h5['data/target_energy'][idx].copy()).float()
        waveform = waveform.unsqueeze(0)
        return {
            'waveform': waveform,
            'target_confidence': target_conf,
            'target_energy': target_energy
        }
    
    def close(self):
        if self._h5_file is not None:
            self._h5_file.close()
            self._h5_file = None


def train_epoch(model, dataloader, optimizer, device):
    model.train()
    metrics_sum = {'total': 0, 'confidence': 0, 'energy': 0, 'sparsity': 0}
    count = 0
    
    for batch in dataloader:
        waveform = batch['waveform'].to(device)
        target_conf = batch['target_confidence'].to(device)
        target_energy = batch['target_energy'].to(device)
        
        optimizer.zero_grad()
        pred = model(waveform)
        
        pred_conf = pred[..., 0]
        pred_energy = pred[..., 1]
        
        loss_conf = F.binary_cross_entropy(pred_conf, target_conf)
        weights = target_conf
        mse_per_bin = F.mse_loss(pred_energy, target_energy, reduction='none')
        loss_energy = (weights * mse_per_bin).sum() / (weights.sum() + 1e-8)
        loss_sparsity = pred_conf.mean()
        
        loss = loss_conf + 0.5 * loss_energy + 0.01 * loss_sparsity
        
        loss.backward()
        optimizer.step()
        
        metrics_sum['total'] += loss.item()
        metrics_sum['confidence'] += loss_conf.item()
        metrics_sum['energy'] += loss_energy.item()
        metrics_sum['sparsity'] += loss_sparsity.item()
        count += 1
    
    return {k: v / count for k, v in metrics_sum.items()}


def validate(model, dataloader, device):
    model.eval()
    metrics_sum = {'total': 0, 'confidence': 0, 'energy': 0, 'sparsity': 0}
    count = 0
    
    with torch.no_grad():
        for batch in dataloader:
            waveform = batch['waveform'].to(device)
            target_conf = batch['target_confidence'].to(device)
            target_energy = batch['target_energy'].to(device)
            
            pred = model(waveform)
            pred_conf = pred[..., 0]
            pred_energy = pred[..., 1]
            
            loss_conf = F.binary_cross_entropy(pred_conf, target_conf)
            weights = target_conf
            mse_per_bin = F.mse_loss(pred_energy, target_energy, reduction='none')
            loss_energy = (weights * mse_per_bin).sum() / (weights.sum() + 1e-8)
            loss_sparsity = pred_conf.mean()
            
            loss = loss_conf + 0.5 * loss_energy + 0.01 * loss_sparsity
            
            metrics_sum['total'] += loss.item()
            metrics_sum['confidence'] += loss_conf.item()
            metrics_sum['energy'] += loss_energy.item()
            metrics_sum['sparsity'] += loss_sparsity.item()
            count += 1
    
    return {k: v / count for k, v in metrics_sum.items()}


def main():
    # 配置
    data_path = '../../TrainingData/test_data/sanity_check_1000.hdf5'
    batch_size = 32
    epochs = 50
    lr = 0.001
    val_split = 0.1
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Device: {device}")
    print(f"Data: {data_path}")
    print(f"Epochs: {epochs}, Batch: {batch_size}, LR: {lr}")
    
    # 数据集
    dataset = PitchDataset(data_path)
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    # 模型
    model = PitchNetBaseline().to(device)
    print(f"Parameters: {model.count_parameters()/1e6:.2f}M")
    
    # 优化器
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    
    # 训练记录
    history = {
        'train_total': [], 'val_total': [],
        'train_conf': [], 'val_conf': [],
        'train_energy': [], 'val_energy': [],
        'lr': []
    }
    
    best_val_loss = float('inf')
    Path('checkpoints').mkdir(exist_ok=True)
    
    print("\n" + "="*60)
    print("Training started!")
    print("="*60)
    
    for epoch in range(1, epochs + 1):
        start_time = time.time()
        
        train_metrics = train_epoch(model, train_loader, optimizer, device)
        val_metrics = validate(model, val_loader, device)
        
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step()
        
        # 记录
        history['train_total'].append(train_metrics['total'])
        history['val_total'].append(val_metrics['total'])
        history['train_conf'].append(train_metrics['confidence'])
        history['val_conf'].append(val_metrics['confidence'])
        history['train_energy'].append(train_metrics['energy'])
        history['val_energy'].append(val_metrics['energy'])
        history['lr'].append(current_lr)
        
        # 保存最佳模型
        if val_metrics['total'] < best_val_loss:
            best_val_loss = val_metrics['total']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss,
            }, 'checkpoints/best_model.pth')
        
        elapsed = time.time() - start_time
        print(f"Epoch {epoch:2d}/{epochs} | {elapsed:.1f}s | "
              f"Train: {train_metrics['total']:.4f} | "
              f"Val: {val_metrics['total']:.4f} | "
              f"Conf: {val_metrics['confidence']:.4f} | "
              f"Energy: {val_metrics['energy']:.4f} | "
              f"LR: {current_lr:.6f}")
    
    # 保存最终模型
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, 'checkpoints/final_model.pth')
    
    # 保存训练历史
    with open('checkpoints/training_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    print("\n" + "="*60)
    print("Training completed!")
    print(f"Best val loss: {best_val_loss:.4f}")
    print("="*60)
    
    dataset.close()


if __name__ == '__main__':
    main()

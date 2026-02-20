#!/usr/bin/env python3
"""
训练脚本 - 带实时可视化
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
# 跨平台matplotlib后端设置
import matplotlib
import platform

system = platform.system()
if system == 'Darwin':  # macOS
    matplotlib.use('TkAgg')
elif system == 'Windows':
    try:
        matplotlib.use('Qt5Agg')
    except:
        matplotlib.use('TkAgg')

import matplotlib.pyplot as plt

# 添加当前目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / 'Model'))

from PitchNetBaseline import PitchNetBaseline
from training_util import export_to_onnx_with_metadata, format_training_info
from dataset import MemoryCachedDataset


class LivePlot:
    """实时训练曲线可视化"""
    
    def __init__(self):
        self.fig, self.axes = plt.subplots(2, 2, figsize=(12, 8))
        self.fig.suptitle('Training Progress', fontsize=14)
        
        # 子图
        self.ax_loss = self.axes[0, 0]
        self.ax_conf = self.axes[0, 1]
        self.ax_energy = self.axes[1, 0]
        self.ax_lr = self.axes[1, 1]
        
        # 数据
        self.train_losses = []
        self.val_losses = []
        self.train_conf_losses = []
        self.val_conf_losses = []
        self.train_energy_losses = []
        self.val_energy_losses = []
        self.lrs = []
        self.epochs = []
        
        self._setup_axes()
        plt.tight_layout()
        plt.ion()  # 交互模式
        plt.show(block=False)
    
    def _setup_axes(self):
        self.ax_loss.set_xlabel('Epoch')
        self.ax_loss.set_ylabel('Total Loss')
        self.ax_loss.set_title('Total Loss')
        self.ax_loss.grid(True, alpha=0.3)
        
        self.ax_conf.set_xlabel('Epoch')
        self.ax_conf.set_ylabel('Confidence Loss')
        self.ax_conf.set_title('Confidence Loss')
        self.ax_conf.grid(True, alpha=0.3)
        
        self.ax_energy.set_xlabel('Epoch')
        self.ax_energy.set_ylabel('Energy Loss')
        self.ax_energy.set_title('Energy Loss')
        self.ax_energy.grid(True, alpha=0.3)
        
        self.ax_lr.set_xlabel('Epoch')
        self.ax_lr.set_ylabel('Learning Rate')
        self.ax_lr.set_title('Learning Rate')
        self.ax_lr.grid(True, alpha=0.3)
    
    def update(self, epoch, train_metrics, val_metrics, lr):
        """更新图表"""
        self.epochs.append(epoch)
        self.train_losses.append(train_metrics['total'])
        self.val_losses.append(val_metrics['total'])
        self.train_conf_losses.append(train_metrics['confidence'])
        self.val_conf_losses.append(val_metrics['confidence'])
        self.train_energy_losses.append(train_metrics['energy'])
        self.val_energy_losses.append(val_metrics['energy'])
        self.lrs.append(lr)
        
        # 清空并重绘
        self.ax_loss.clear()
        self.ax_conf.clear()
        self.ax_energy.clear()
        self.ax_lr.clear()
        self._setup_axes()
        
        self.ax_loss.plot(self.epochs, self.train_losses, 'b-', label='Train', linewidth=2)
        self.ax_loss.plot(self.epochs, self.val_losses, 'r-', label='Val', linewidth=2)
        self.ax_loss.legend()
        
        self.ax_conf.plot(self.epochs, self.train_conf_losses, 'b-', label='Train', linewidth=2)
        self.ax_conf.plot(self.epochs, self.val_conf_losses, 'r-', label='Val', linewidth=2)
        self.ax_conf.legend()
        
        self.ax_energy.plot(self.epochs, self.train_energy_losses, 'b-', label='Train', linewidth=2)
        self.ax_energy.plot(self.epochs, self.val_energy_losses, 'r-', label='Val', linewidth=2)
        self.ax_energy.legend()
        
        self.ax_lr.plot(self.epochs, self.lrs, 'g-', linewidth=2)
        
        self.fig.suptitle(f'Training Progress - Epoch {epoch}', fontsize=14)
        plt.tight_layout()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
    
    def close(self):
        plt.ioff()
        plt.close()


class PitchDetectionLoss(nn.Module):
    """音高检测损失函数"""
    
    def __init__(self, conf_weight=1.0, energy_weight=0.5, sparsity_weight=0.01):
        super().__init__()
        self.conf_weight = conf_weight
        self.energy_weight = energy_weight
        self.sparsity_weight = sparsity_weight
    
    def forward(self, pred, target_conf, target_energy):
        """
        Args:
            pred: [B, 2048, 2] - model output
            target_conf: [B, 2048]
            target_energy: [B, 2048]
        """
        pred_conf = pred[..., 0]
        pred_energy = pred[..., 1]
        
        # Confidence loss
        loss_conf = F.binary_cross_entropy(pred_conf, target_conf)
        
        # Energy loss (soft weighted)
        weights = target_conf
        mse_per_bin = F.mse_loss(pred_energy, target_energy, reduction='none')
        loss_energy = (weights * mse_per_bin).sum() / (weights.sum() + 1e-8)
        
        # Sparsity
        loss_sparsity = pred_conf.mean()
        
        total = (self.conf_weight * loss_conf + 
                self.energy_weight * loss_energy +
                self.sparsity_weight * loss_sparsity)
        
        return {
            'total': total.item(),
            'confidence': loss_conf.item(),
            'energy': loss_energy.item(),
            'sparsity': loss_sparsity.item()
        }


class PitchDataset(torch.utils.data.Dataset):
    """HDF5数据集加载器"""
    
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
        
        waveform = waveform.unsqueeze(0)  # [1, 4096]
        
        return {
            'waveform': waveform,
            'target_confidence': target_conf,
            'target_energy': target_energy
        }
    
    def close(self):
        if self._h5_file is not None:
            self._h5_file.close()
            self._h5_file = None


def train_epoch(model, dataloader, optimizer, criterion, device):
    """训练一个epoch"""
    model.train()
    
    metrics_sum = {'total': 0, 'confidence': 0, 'energy': 0, 'sparsity': 0}
    count = 0
    
    for batch in dataloader:
        waveform = batch['waveform'].to(device)
        target_conf = batch['target_confidence'].to(device)
        target_energy = batch['target_energy'].to(device)
        
        optimizer.zero_grad()
        
        pred = model(waveform)
        
        # 计算损失
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
    """验证"""
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
    # 配置 (跨平台路径)
    script_dir = Path(__file__).parent.resolve()
    project_root = script_dir.parent.parent.parent  # 到项目根目录
    
    # 默认使用20K数据集,可切换为1000样本快速测试
    data_path = str(project_root / 'TrainingData' / 'sanity_check_20k.hdf5')
    # data_path = str(project_root / 'TrainingData' / 'test_data' / 'sanity_check_1000.hdf5')
    batch_size = 32
    epochs = 50
    lr = 0.001
    val_split = 0.1
    # 自动选择设备: CUDA > MPS (Apple Silicon) > CPU
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Device: CUDA ({torch.cuda.get_device_name(0)})")
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        print(f"Device: MPS (Apple Silicon GPU)")
    else:
        device = torch.device('cpu')
        print(f"Device: CPU")
    
    # 平台特定优化
    if device.type == 'mps':
        print("  Tip: First epoch may be slower due to MPS shader compilation")
    elif device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
        print(f"  CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        print(f"  cuDNN benchmark: enabled")
    
    print(f"Data: {data_path}")
    
    # 数据集 - 使用内存缓存避免IO瓶颈
    full_dataset = MemoryCachedDataset(data_path)
    val_size = int(len(full_dataset) * val_split)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    # 内存数据集不需要多进程 workers
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=0,  # 内存数据不需要多进程
        pin_memory=(device.type == 'cuda')
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size,
        num_workers=0,
        pin_memory=(device.type == 'cuda')
    )
    
    # 模型
    model = PitchNetBaseline().to(device)
    print(f"Model parameters: {model.count_parameters()/1e6:.2f}M")
    
    # GPU预热 (MPS需要编译shader, CUDA需要warmup)
    if device.type in ['mps', 'cuda']:
        print("Warming up GPU...")
        with torch.no_grad():
            _ = model(torch.randn(4, 1, 4096).to(device))
        if device.type == 'cuda':
            torch.cuda.synchronize()
        print("Ready!")
    
    # 优化器
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    
    # 可视化
    live_plot = LivePlot()
    
    # 训练循环
    best_val_loss = float('inf')
    
    try:
        for epoch in range(1, epochs + 1):
            start_time = time.time()
            
            train_metrics = train_epoch(model, train_loader, optimizer, None, device)
            
            # MPS同步以确保准确计时
            if device.type == 'mps':
                torch.mps.synchronize()
            
            val_metrics = validate(model, val_loader, device)
            
            if device.type == 'mps':
                torch.mps.synchronize()
            
            current_lr = optimizer.param_groups[0]['lr']
            scheduler.step()
            
            # 更新可视化
            live_plot.update(epoch, train_metrics, val_metrics, current_lr)
            
            # 保存最佳模型
            if val_metrics['total'] < best_val_loss:
                best_val_loss = val_metrics['total']
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': best_val_loss,
                }, '../../../MLModel/checkpoints/best_model.pth')
            
            elapsed = time.time() - start_time
            print(f"Epoch {epoch}/{epochs} - {elapsed:.1f}s - "
                  f"Train: {train_metrics['total']:.4f}, "
                  f"Val: {val_metrics['total']:.4f}, "
                  f"LR: {current_lr:.6f}")
    
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        interrupted = True
    
    else:
        interrupted = False
    
    finally:
        live_plot.close()
        dataset.close()
        
        # 保存最终模型
        final_path = '../../../MLModel/checkpoints/final_model.pth'
        torch.save({
            'epoch': epoch if not interrupted else epoch - 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, final_path)
        
        # 如果不是中断的，导出ONNX
        if not interrupted:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            try:
                # 构建训练信息
                training_info = format_training_info(
                    config={'data_path': data_path, 'epochs': epochs, 
                            'batch_size': batch_size, 'lr': lr},
                    best_val_loss=best_val_loss,
                    device=device
                )
                
                onnx_path = export_to_onnx_with_metadata(
                    model, '../../../MLModel', timestamp, training_info
                )
                onnx_msg = f"ONNX exported: {onnx_path.name}"
            except Exception as e:
                onnx_msg = f"ONNX export failed: {e}"
            
            # 明显的完成提示
            print("\n" + "="*70)
            print("🎉  TRAINING COMPLETED SUCCESSFULLY!  🎉")
            print("="*70)
            print(f"✓ Best model:    ../../../MLModel/checkpoints/best_model.pth")
            print(f"✓ Final model:   {final_path}")
            print(f"✓ {onnx_msg}")
            print(f"✓ Total epochs:  {epochs}")
            print(f"✓ Best val loss: {best_val_loss:.6f}")
            print("="*70)
        else:
            print("\n" + "="*70)
            print("⚠️  TRAINING INTERRUPTED")
            print("="*70)
            print(f"Model saved at epoch {epoch}")
            print("="*70)


if __name__ == '__main__':
    # 创建checkpoint目录
    Path('../../../MLModel/checkpoints').mkdir(parents=True, exist_ok=True)
    main()

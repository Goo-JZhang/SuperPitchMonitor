#!/usr/bin/env python3
"""
实时可视化训练脚本 (Mac兼容版)
使用matplotlib的interactive mode显示实时训练曲线
"""

import os
import sys
import json
import time
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

# 必须在导入pyplot前设置后端 (跨平台)
import matplotlib
import platform

# 根据平台选择最佳后端
system = platform.system()
if system == 'Darwin':  # macOS
    matplotlib.use('TkAgg')
elif system == 'Windows':
    try:
        matplotlib.use('Qt5Agg')
    except:
        matplotlib.use('TkAgg')  # fallback
# Linux使用默认即可

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / 'Model'))
from PitchNetBaseline import PitchNetBaseline
from dataset import MemoryCachedDataset  # 使用统一数据集接口
from training_util import export_to_onnx_with_metadata, format_training_info


class LiveTrainingPlot:
    """实时训练图表"""
    
    def __init__(self):
        plt.ion()  # 开启交互模式
        self.fig, self.axes = plt.subplots(2, 2, figsize=(12, 8))
        self.fig.suptitle('Training Progress - Live', fontsize=14)
        
        self.ax_loss = self.axes[0, 0]
        self.ax_conf = self.axes[0, 1]
        self.ax_energy = self.axes[1, 0]
        self.ax_lr = self.axes[1, 1]
        
        self.train_loss_line, = self.ax_loss.plot([], [], 'b-', label='Train', linewidth=2)
        self.val_loss_line, = self.ax_loss.plot([], [], 'r-', label='Val', linewidth=2)
        
        self.train_conf_line, = self.ax_conf.plot([], [], 'b-', label='Train', linewidth=2)
        self.val_conf_line, = self.ax_conf.plot([], [], 'r-', label='Val', linewidth=2)
        
        self.train_energy_line, = self.ax_energy.plot([], [], 'b-', label='Train', linewidth=2)
        self.val_energy_line, = self.ax_energy.plot([], [], 'r-', label='Val', linewidth=2)
        
        self.lr_line, = self.ax_lr.plot([], [], 'g-', linewidth=2)
        
        for ax in [self.ax_loss, self.ax_conf, self.ax_energy]:
            ax.legend()
            ax.grid(True, alpha=0.3)
        self.ax_lr.grid(True, alpha=0.3)
        
        self.ax_loss.set_title('Total Loss')
        self.ax_conf.set_title('Confidence Loss')
        self.ax_energy.set_title('Energy Loss')
        self.ax_lr.set_title('Learning Rate')
        
        self.data = {
            'epochs': [],
            'train_total': [], 'val_total': [],
            'train_conf': [], 'val_conf': [],
            'train_energy': [], 'val_energy': [],
            'lr': []
        }
        
        plt.tight_layout()
        plt.show(block=False)
        print("Visualization window opened. If not visible, check your taskbar.")
    
    def update(self, epoch, train_metrics, val_metrics, lr):
        """更新图表"""
        self.data['epochs'].append(epoch)
        self.data['train_total'].append(train_metrics['total'])
        self.data['val_total'].append(val_metrics['total'])
        self.data['train_conf'].append(train_metrics['confidence'])
        self.data['val_conf'].append(val_metrics['confidence'])
        self.data['train_energy'].append(train_metrics['energy'])
        self.data['val_energy'].append(val_metrics['energy'])
        self.data['lr'].append(lr)
        
        # 更新数据
        self.train_loss_line.set_data(self.data['epochs'], self.data['train_total'])
        self.val_loss_line.set_data(self.data['epochs'], self.data['val_total'])
        
        self.train_conf_line.set_data(self.data['epochs'], self.data['train_conf'])
        self.val_conf_line.set_data(self.data['epochs'], self.data['val_conf'])
        
        self.train_energy_line.set_data(self.data['epochs'], self.data['train_energy'])
        self.val_energy_line.set_data(self.data['epochs'], self.data['val_energy'])
        
        self.lr_line.set_data(self.data['epochs'], self.data['lr'])
        
        # 自动调整坐标轴
        for ax in [self.ax_loss, self.ax_conf, self.ax_energy, self.ax_lr]:
            ax.relim()
            ax.autoscale_view()
        
        self.fig.suptitle(f'Training Progress - Epoch {epoch}', fontsize=14)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
    
    def close(self):
        plt.ioff()
        plt.close()


class PitchDataset(torch.utils.data.Dataset):
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
    metrics = {'total': 0, 'confidence': 0, 'energy': 0, 'sparsity': 0}
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
        
        metrics['total'] += loss.item()
        metrics['confidence'] += loss_conf.item()
        metrics['energy'] += loss_energy.item()
        metrics['sparsity'] += loss_sparsity.item()
        count += 1
    
    return {k: v / count for k, v in metrics.items()}


def validate(model, dataloader, device):
    model.eval()
    metrics = {'total': 0, 'confidence': 0, 'energy': 0, 'sparsity': 0}
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
            
            metrics['total'] += loss.item()
            metrics['confidence'] += loss_conf.item()
            metrics['energy'] += loss_energy.item()
            metrics['sparsity'] += loss_sparsity.item()
            count += 1
    
    return {k: v / count for k, v in metrics.items()}


def main():
    # 配置 (跨平台路径)
    # 获取脚本所在目录的绝对路径
    script_dir = Path(__file__).parent.resolve()
    project_root = script_dir.parent.parent.parent  # 到项目根目录
    
    # Mac用较大batch减少HDF5读取次数，Windows RTX4080S可用更大batch
    system = platform.system()
    default_batch = 64 if system == 'Darwin' else 128  # RTX 4080S 16GB显存可用大batch
    
    config = {
        # 使用新的SingleSanity数据集 (102,400样本, 50 samples/bin)
        'data_path': str(project_root / 'TrainingData' / 'SingleSanity'),
        'batch_size': default_batch,
        'epochs': 50,
        'lr': 0.001,
        'val_split': 0.02,  # 2%验证集 (约2000样本)
    }
    
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
        # CUDA优化: cuDNN benchmark自动寻找最快卷积算法
        torch.backends.cudnn.benchmark = True
        print(f"  CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        print(f"  cuDNN benchmark: enabled")
    
    print(f"Dataset: {config['data_path']}")
    print(f"Epochs: {config['epochs']}, Batch: {config['batch_size']}")
    
    # 数据集 - 使用内存缓存shards避免IO瓶颈
    full_dataset = MemoryCachedDataset(config['data_path'])
    val_size = int(len(full_dataset) * config['val_split'])
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    # 内存数据集不需要多进程 workers
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['batch_size'], 
        shuffle=True,
        num_workers=0,  # 内存数据不需要多进程
        pin_memory=(device.type == 'cuda')
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config['batch_size'],
        num_workers=0,
        pin_memory=(device.type == 'cuda')
    )
    
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    # 模型
    model = PitchNetBaseline().to(device)
    print(f"Parameters: {model.count_parameters()/1e6:.2f}M")
    
    # GPU预热 (MPS需要编译shader)
    if device.type == 'mps':
        print("Warming up GPU...")
        with torch.no_grad():
            _ = model(torch.randn(4, 1, 4096).to(device))
        print("Ready!")
    
    # 优化器
    optimizer = AdamW(model.parameters(), lr=config['lr'], weight_decay=1e-4)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    
    # 实时可视化
    print("\nOpening visualization window...")
    live_plot = LiveTrainingPlot()
    
    # 训练
    best_val_loss = float('inf')
    checkpoint_dir = project_root / 'MLModel' / 'checkpoints'
    mlmodel_dir = project_root / 'MLModel'
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    print(f"Checkpoints: {checkpoint_dir}")
    
    print("\nTraining started! Press Ctrl+C to stop early.")
    print("="*60)
    
    try:
        for epoch in range(1, config['epochs'] + 1):
            start = time.time()
            
            train_metrics = train_epoch(model, train_loader, optimizer, device)
            
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
                    'val_loss': best_val_loss,
                }, str(checkpoint_dir / 'best_model_live.pth'))
            
            elapsed = time.time() - start
            print(f"Epoch {epoch:2d}/{config['epochs']} | {elapsed:.1f}s | "
                  f"Train: {train_metrics['total']:.4f} | Val: {val_metrics['total']:.4f} | "
                  f"LR: {current_lr:.6f}")
            
            # MPS内存清理
            if device.type == 'mps' and epoch % 10 == 0:
                torch.mps.empty_cache()
    
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
        interrupted = True
    
    else:
        interrupted = False
    
    finally:
        current_epoch = epoch if not interrupted else epoch - 1
        
        # 保存最终模型
        final_path = checkpoint_dir / 'final_model_live.pth'
        torch.save({
            'epoch': current_epoch,
            'model_state_dict': model.state_dict(),
        }, str(final_path))
        
        # 保存历史
        with open(str(checkpoint_dir / 'history_live.json'), 'w') as f:
            json.dump(live_plot.data, f)
        
        # 如果不是中断的，导出ONNX并创建元数据
        if not interrupted:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            try:
                # 使用工具函数构建训练信息
                training_info = format_training_info(
                    config=config,
                    best_val_loss=best_val_loss,
                    device=device
                )
                
                onnx_path = export_to_onnx_with_metadata(
                    model, str(mlmodel_dir), timestamp, training_info
                )
                onnx_msg = f"ONNX exported: {onnx_path.name}"
            except Exception as e:
                onnx_msg = f"ONNX export failed: {e}"
            
            print("\n" + "="*70)
            print("🎉  TRAINING COMPLETED SUCCESSFULLY!  🎉")
            print("="*70)
            print(f"✓ Best model:    {checkpoint_dir / 'best_model_live.pth'}")
            print(f"✓ Final model:   {final_path}")
            print(f"✓ {onnx_msg}")
            print(f"✓ Metadata:      {onnx_path.with_suffix('.txt')}")
            print(f"✓ Total epochs:  {config['epochs']}")
            print(f"✓ Best val loss: {best_val_loss:.6f}")
            print("="*70)
            print("\nClose the plot window to exit.")
        else:
            print("\n" + "="*70)
            print("⚠️  TRAINING INTERRUPTED")
            print("="*70)
            print(f"Model saved at epoch {current_epoch}")
            print("="*70)
        
        # 保持窗口显示
        plt.ioff()
        plt.show()
        
        # MemoryCachedShardDataset不需要显式关闭


if __name__ == '__main__':
    main()

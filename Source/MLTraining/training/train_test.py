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
from torch.utils.data import DataLoader, random_split, ConcatDataset, ConcatDataset
from torch.nn import Module
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
from loss import PitchDetectionLoss
from modeloutput_utils import export_model_with_metadata, format_training_info
from dataset import DatasetReader


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
    
    def save(self, filepath):
        """保存图表到文件"""
        self.fig.savefig(filepath, dpi=150, bbox_inches='tight')
    
    def close(self):
        plt.ioff()
        plt.close()


def train_epoch(model, dataloader, optimizer, device, criterion):
    """训练一个epoch"""
    model.train()
    
    metrics_sum = {'total': 0, 'confidence': 0, 'energy': 0, 'sparsity': 0}
    count = 0
    
    for batch in dataloader:
        waveform = batch['waveform'].to(device)
        target_conf = batch['target_confidence'].to(device).float()
        target_energy = batch['target_energy'].to(device).float()
        
        optimizer.zero_grad()
        
        pred = model(waveform)
        
        # 计算损失
        pred_conf = pred[..., 0].float()
        pred_energy_raw = pred[..., 1].float()
        
        losses = criterion(pred_conf, pred_energy_raw, target_conf, target_energy)
        
        losses['total'].backward()
        optimizer.step()
        
        metrics_sum['total'] += losses['total'].item()
        metrics_sum['confidence'] += losses['confidence'].item()
        metrics_sum['energy'] += losses['energy'].item()
        metrics_sum['sparsity'] += losses['sparsity'].item()
        count += 1
    
    return {k: v / count for k, v in metrics_sum.items()}


def validate(model, dataloader, device, criterion):
    """验证"""
    model.eval()
    
    metrics_sum = {'total': 0, 'confidence': 0, 'energy': 0, 'sparsity': 0}
    count = 0
    
    with torch.no_grad():
        for batch in dataloader:
            waveform = batch['waveform'].to(device)
            target_conf = batch['target_confidence'].to(device).float()
            target_energy = batch['target_energy'].to(device).float()
            
            pred = model(waveform)
            
            pred_conf = pred[..., 0].float()
            pred_energy_raw = pred[..., 1].float()
            
            losses = criterion(pred_conf, pred_energy_raw, target_conf, target_energy)
            
            metrics_sum['total'] += losses['total'].item()
            metrics_sum['confidence'] += losses['confidence'].item()
            metrics_sum['energy'] += losses['energy'].item()
            metrics_sum['sparsity'] += losses['sparsity'].item()
            count += 1
    
    return {k: v / count for k, v in metrics_sum.items()}


def main():
    # 配置 (跨平台路径)
    script_dir = Path(__file__).parent.resolve()
    project_root = script_dir.parent.parent.parent  # 到项目根目录
    
    # 训练参数
    batch_size = 128  # RTX 4080S 可用更大batch
    epochs = 50
    lr = 0.001
    val_split = 0.02  # 2%验证集，每类数据单独拆分
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
    
    # 硬编码加载指定数据集（后续添加新数据集需手动修改此处）
    data_root = project_root / 'TrainingData'
    data_subdirs = ['SingleSanity', 'NoiseDatasetV2']  # 使用V2版本噪声数据（31种×1000=31000样本）
    
    print(f"Loading {len(data_subdirs)} dataset type(s): {data_subdirs}")
    train_datasets = []
    val_datasets = []

    for subdir in data_subdirs:
        data_dir = data_root / subdir
        if not (data_dir / 'meta.json').exists():
            print(f"  Warning: {subdir} not found or invalid, skipping...")
            continue
        ds = DatasetReader(str(data_dir), preload=True, device=str(device))
        n_val = max(1, int(len(ds) * val_split))
        n_train = len(ds) - n_val
        ds_train, ds_val = random_split(ds, [n_train, n_val])
        train_datasets.append(ds_train)
        val_datasets.append(ds_val)
        print(f"  {subdir:20s}: {len(ds):6d} samples -> Train: {n_train:6d}, Val: {n_val:4d}")

    # 合并所有类型的训练和验证集
    train_dataset = ConcatDataset(train_datasets)
    val_dataset = ConcatDataset(val_datasets)
    
    # 内存数据集不需要多进程 workers
    # DataLoader (pin_memory 禁用，因为数据可能已在 GPU)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=0,
        pin_memory=False
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size,
        num_workers=0,
        pin_memory=False
    )
    
    # 模型
    model = PitchNetBaseline().to(device)
    print(f"Model parameters: {model.count_parameters()/1e6:.2f}M")
    
    # GPU预热 (MPS需要编译shader, CUDA需要warmup + cuDNN算法搜索)
    if device.type in ['mps', 'cuda']:
        print("Warming up GPU...")
        with torch.no_grad():
            # 使用实际batch size预热，触发cuDNN算法搜索
            warmup_batch = batch_size
            dummy_input = torch.randn(warmup_batch, 1, 4096).to(device)
            _ = model(dummy_input)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        print("Ready!")
    
    # 优化器和损失函数
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    criterion = PitchDetectionLoss(
        conf_weight=1.0,
        energy_weight=0.3,
        sparsity_weight=0.01,
        energy_loss_type='kl'
    ).to(device)
    
    # 可视化
    live_plot = LivePlot()
    
    # 训练循环
    best_val_loss = float('inf')
    
    interrupted = False
    try:
        for epoch in range(1, epochs + 1):
            start_time = time.time()
            
            train_metrics = train_epoch(model, train_loader, optimizer, device, criterion)
            
            # 同步以确保准确计时
            if device.type == 'mps':
                torch.mps.synchronize()
            elif device.type == 'cuda':
                torch.cuda.synchronize()
            
            val_metrics = validate(model, val_loader, device, criterion)
            
            if device.type == 'mps':
                torch.mps.synchronize()
            elif device.type == 'cuda':
                torch.cuda.synchronize()
            
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
            print(f"Epoch {epoch:2d}/{epochs} | {elapsed:.1f}s | "
                  f"Train: {train_metrics['total']:.4f} (c:{train_metrics['confidence']:.3f} e:{train_metrics['energy']:.3f}) | "
                  f"Val: {val_metrics['total']:.4f} (c:{val_metrics['confidence']:.3f} e:{val_metrics['energy']:.3f}) | "
                  f"LR: {current_lr:.6f}")
    
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        interrupted = True
    
    else:
        interrupted = False
    
    finally:
        live_plot.close()
        
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
                # 准备模型导出（eval模式，移到CPU）
                model.eval()
                model.cpu()
                
                # 构建训练信息
                training_info = format_training_info(
                    config={'data_subdirs': data_subdirs, 'epochs': epochs, 
                            'batch_size': batch_size, 'lr': lr},
                    best_val_loss=best_val_loss,
                    device=device
                )
                
                # 使用绝对路径
                mlmodel_dir = Path(project_root) / 'MLModel'
                mlmodel_dir.mkdir(parents=True, exist_ok=True)
                
                print(f"\nExporting ONNX to: {mlmodel_dir}")
                onnx_path = export_model_with_metadata(
                    model, str(mlmodel_dir), timestamp, training_info
                )
                onnx_msg = f"ONNX exported: {onnx_path}"
                print(f"Successfully exported: {onnx_path}")
            except Exception as e:
                import traceback
                onnx_msg = f"ONNX export failed: {e}"
                print(f"\nONNX Export Error: {e}")
                traceback.print_exc()
            
            # 明显的完成提示
            print("\n" + "="*70)
            print("TRAINING COMPLETED SUCCESSFULLY!")
            print("="*70)
            print(f"Best model:    ../../../MLModel/checkpoints/best_model.pth")
            print(f"Final model:   {final_path}")
            print(f"{onnx_msg}")
            print(f"Total epochs:  {epochs}")
            print(f"Best val loss: {best_val_loss:.6f}")
            print("="*70)
        else:
            print("\n" + "="*70)
            print("TRAINING INTERRUPTED")
            print("="*70)
            print(f"Model saved at epoch {epoch}")
            print("="*70)


if __name__ == '__main__':
    # 创建checkpoint目录
    Path('../../../MLModel/checkpoints').mkdir(parents=True, exist_ok=True)
    main()

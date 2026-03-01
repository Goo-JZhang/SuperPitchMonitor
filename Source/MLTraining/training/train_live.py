#!/usr/bin/env python3
"""
实时可视化训练脚本

使用方式:
    # 默认训练 (PitchNetBaseline, TrainingData下所有数据)
    python train_live.py
    
    # 指定配置文件
    python train_live.py --config trainconfig.txt
    
    # 命令行参数覆盖
    python train_live.py --epochs 100 --batch-size 64 --lr 0.0005
    
    # 指定模型
    python train_live.py --model PitchNetEnhanced
    
    # 指定数据子目录
    python train_live.py --data SingleSanity,NoiseDataset
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split, ConcatDataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

# 必须在导入pyplot前设置后端 (跨平台)
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

# 添加路径
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / 'Model'))

# 导入配置工具
from train_config_utils import (
    add_common_training_args, get_training_config,
    create_model, list_available_models,
    get_default_data_root, get_project_root
)
from dataset import DatasetReader
from loss import PitchDetectionLoss, get_loss_config
from modeloutput_utils import export_model_with_metadata, format_training_info


class LiveTrainingPlot:
    """实时训练图表"""
    
    def __init__(self):
        plt.ion()
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
        self.data['epochs'].append(epoch)
        self.data['train_total'].append(train_metrics['total'])
        self.data['val_total'].append(val_metrics['total'])
        self.data['train_conf'].append(train_metrics['confidence'])
        self.data['val_conf'].append(val_metrics['confidence'])
        self.data['train_energy'].append(train_metrics['energy'])
        self.data['val_energy'].append(val_metrics['energy'])
        self.data['lr'].append(lr)
        
        self.train_loss_line.set_data(self.data['epochs'], self.data['train_total'])
        self.val_loss_line.set_data(self.data['epochs'], self.data['val_total'])
        
        self.train_conf_line.set_data(self.data['epochs'], self.data['train_conf'])
        self.val_conf_line.set_data(self.data['epochs'], self.data['val_conf'])
        
        self.train_energy_line.set_data(self.data['epochs'], self.data['train_energy'])
        self.val_energy_line.set_data(self.data['epochs'], self.data['val_energy'])
        
        self.lr_line.set_data(self.data['epochs'], self.data['lr'])
        
        for ax in [self.ax_loss, self.ax_conf, self.ax_energy, self.ax_lr]:
            ax.relim()
            ax.autoscale_view()
        
        self.fig.suptitle(f'Training Progress - Epoch {epoch}', fontsize=14)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
    
    def close(self):
        plt.ioff()
        plt.close()


def train_epoch(model, dataloader, optimizer, device, criterion):
    """训练一个epoch"""
    model.train()
    metrics = {'total': 0, 'confidence': 0, 'focal': 0, 'tversky': 0, 'sharpness': 0, 'energy': 0, 'sparsity': 0}
    count = 0
    
    for batch in dataloader:
        waveform = batch['waveform'].to(device)
        target_conf = batch['target_confidence'].to(device).float()
        target_energy = batch['target_energy'].to(device).float()
        
        optimizer.zero_grad()
        pred = model(waveform)
        
        pred_conf = pred[..., 0].float()
        pred_energy_raw = pred[..., 1].float()
        
        losses = criterion(pred_conf, pred_energy_raw, target_conf, target_energy)
        
        losses['total'].backward()
        optimizer.step()
        
        metrics['total'] += losses['total'].item()
        metrics['confidence'] += losses['confidence'].item()
        metrics['focal'] += losses.get('focal', torch.tensor(0.0)).item()
        metrics['tversky'] += losses.get('tversky', torch.tensor(0.0)).item()
        metrics['sharpness'] += losses.get('sharpness', torch.tensor(0.0)).item()
        metrics['energy'] += losses['energy'].item()
        metrics['sparsity'] += losses['sparsity'].item()
        count += 1
    
    return {k: v / count for k, v in metrics.items()}


def validate(model, dataloader, device, criterion):
    """验证"""
    model.eval()
    metrics = {'total': 0, 'confidence': 0, 'focal': 0, 'tversky': 0, 'sharpness': 0, 'energy': 0, 'sparsity': 0}
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
            
            metrics['total'] += losses['total'].item()
            metrics['confidence'] += losses['confidence'].item()
            metrics['focal'] += losses.get('focal', torch.tensor(0.0)).item()
            metrics['tversky'] += losses.get('tversky', torch.tensor(0.0)).item()
            metrics['sharpness'] += losses.get('sharpness', torch.tensor(0.0)).item()
            metrics['energy'] += losses['energy'].item()
            metrics['sparsity'] += losses['sparsity'].item()
            count += 1
    
    return {k: v / count for k, v in metrics.items()}


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Training script with live visualization')
    return add_common_training_args(parser).parse_args()


def main():
    args = parse_args()
    
    # 获取完整配置
    config = get_training_config(args)
    
    # 项目路径
    project_root = get_project_root()
    
    # 提取配置参数
    data_root = config.get('data_root') or config.get('root_dir') or str(get_default_data_root())
    data_subdirs = config.get('data_subdirs') or config.get('subdirs')
    model_name = config.get('model_name', 'PitchNetBaseline')
    pretrained_path = config.get('pretrained') or config.get('path')
    epochs = config.get('epochs', 50)
    batch_size = config.get('batch_size', 64)
    lr = config.get('lr', 0.001)
    val_split = config.get('val_split', 0.02)
    seed = config.get('seed', 42)
    preload = config.get('preload', True)
    use_viz = not config.get('no_viz', False)
    
    print("=" * 70)
    print("Training Configuration")
    print("=" * 70)
    print(f"Model: {model_name}")
    print(f"Data root: {data_root}")
    print(f"Data subdirs: {data_subdirs or 'Auto-discover all'}")
    print(f"Preload: {preload}")
    print(f"Epochs: {epochs}, Batch: {batch_size}, LR: {lr}")
    print(f"Val split: {val_split}")
    print("=" * 70)
    
    # 设置随机种子
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # 设备
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"\nDevice: CUDA ({torch.cuda.get_device_name(0)})")
        torch.backends.cudnn.benchmark = True
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        print(f"\nDevice: MPS")
    else:
        device = torch.device('cpu')
        print(f"\nDevice: CPU")
    
    # 加载数据集
    print(f"\nLoading datasets...")
    data_root_path = Path(data_root)
    
    if data_subdirs:
        data_dirs = [data_root_path / d for d in data_subdirs]
    else:
        data_dirs = sorted([p for p in data_root_path.iterdir() 
                           if p.is_dir() and (p / 'meta.json').exists()])
    
    print(f"Found {len(data_dirs)} dataset type(s):")
    train_datasets = []
    val_datasets = []
    
    for data_dir in data_dirs:
        ds = DatasetReader(str(data_dir), preload=preload, device=str(device))
        n_val = max(1, int(len(ds) * val_split))
        n_train = len(ds) - n_val
        
        ds_train, ds_val = random_split(ds, [n_train, n_val])
        train_datasets.append(ds_train)
        val_datasets.append(ds_val)
        
        print(f"  {data_dir.name:20s}: {len(ds):6d} samples -> Train: {n_train:6d}, Val: {n_val:4d}")
    
    # 合并所有类型的训练和验证集
    train_dataset = ConcatDataset(train_datasets)
    val_dataset = ConcatDataset(val_datasets)
    
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
    
    print(f"\nTrain: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    # 模型创建（使用工厂函数）
    print(f"\nInitializing model: {model_name}")
    try:
        model = create_model(model_name)
    except ValueError as e:
        print(f"Error: {e}")
        print(f"Available models: {list_available_models()}")
        return
    
    # 加载预训练权重
    if pretrained_path:
        print(f"Loading pretrained weights from: {pretrained_path}")
        checkpoint = torch.load(pretrained_path, map_location='cpu')
        model.load_state_dict(checkpoint.get('model_state_dict', checkpoint))
    
    model = model.to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    
    # 优化器
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    # 使用推荐配置，可选择 'default', 'position_focused', 'peak_focused', 'balanced', 'simple'
    loss_config = get_loss_config('default')
    criterion = PitchDetectionLoss(**loss_config).to(device)
    
    # GPU预热
    if device.type in ['mps', 'cuda']:
        print("\nWarming up GPU...")
        with torch.no_grad():
            dummy = torch.randn(batch_size, 1, 4096).to(device)
            _ = model(dummy)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        print("Ready!")
    
    # 可视化
    live_plot = None
    if use_viz:
        print("\nOpening visualization window...")
        live_plot = LiveTrainingPlot()
    else:
        print("\nVisualization disabled.")
    
    # 训练
    checkpoint_dir = Path(project_root) / 'MLModel' / 'checkpoints'
    mlmodel_dir = Path(project_root) / 'MLModel'
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Checkpoints: {checkpoint_dir}")
    print("\nTraining started! Press Ctrl+C to stop early.")
    print("=" * 70)
    
    best_val_loss = float('inf')
    interrupted = False
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    try:
        for epoch in range(1, epochs + 1):
            start = time.time()
            
            train_metrics = train_epoch(model, train_loader, optimizer, device, criterion)
            
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
            
            if live_plot:
                live_plot.update(epoch, train_metrics, val_metrics, current_lr)
            
            if val_metrics['total'] < best_val_loss:
                best_val_loss = val_metrics['total']
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'val_loss': best_val_loss,
                }, str(checkpoint_dir / 'best_model_live.pth'))
            
            elapsed = time.time() - start
            # 构建详细的损失日志
            train_conf = train_metrics.get('focal', train_metrics.get('confidence', 0))
            train_tversky = train_metrics.get('tversky', 0)
            train_sharp = train_metrics.get('sharpness', 0)
            train_energy = train_metrics['energy']
            
            val_conf = val_metrics.get('focal', val_metrics.get('confidence', 0))
            val_tversky = val_metrics.get('tversky', 0)
            val_sharp = val_metrics.get('sharpness', 0)
            val_energy = val_metrics['energy']
            
            print(f"Epoch {epoch:2d}/{epochs} | {elapsed:.1f}s | "
                  f"Train: {train_metrics['total']:.4f} (f:{train_conf:.3f} t:{train_tversky:.3f} s:{train_sharp:.3f} e:{train_energy:.3f}) | "
                  f"Val: {val_metrics['total']:.4f} (f:{val_conf:.3f} t:{val_tversky:.3f} s:{val_sharp:.3f} e:{val_energy:.3f}) | "
                  f"LR: {current_lr:.6f}")
            
            # GPU内存清理
            if epoch % 10 == 0:
                if device.type == 'mps':
                    torch.mps.empty_cache()
                elif device.type == 'cuda':
                    torch.cuda.empty_cache()
    
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
        interrupted = True
    
    finally:
        current_epoch = epoch if not interrupted else epoch - 1
        
        # 保存最终模型
        final_path = checkpoint_dir / 'final_model_live.pth'
        torch.save({
            'epoch': current_epoch,
            'model_state_dict': model.state_dict(),
        }, str(final_path))
        
        # 保存历史
        if live_plot:
            with open(str(checkpoint_dir / 'history_live.json'), 'w') as f:
                json.dump(live_plot.data, f)
        
        if not interrupted:
            # 导出ONNX
            onnx_path = None
            try:
                print("\n[DEBUG] Starting ONNX export process...")
                model.eval()
                model.cpu()
                
                training_info = format_training_info(
                    config={'data_root': data_root, 'subdirs': data_subdirs, 
                            'epochs': epochs, 'batch_size': batch_size, 'lr': lr,
                            'model': model_name},
                    best_val_loss=best_val_loss,
                    device=device
                )
                
                print(f"\nExporting ONNX to: {mlmodel_dir}")
                onnx_path = export_model_with_metadata(model, str(mlmodel_dir), timestamp, training_info)
                onnx_msg = f"ONNX exported: {onnx_path}"
                print(f"Successfully exported: {onnx_path}")
            except Exception as e:
                import traceback
                onnx_msg = f"ONNX export failed: {e}"
                print(f"\nONNX Export Error: {e}")
                traceback.print_exc()
            
            print("\n" + "=" * 70)
            print("TRAINING COMPLETED SUCCESSFULLY!")
            print("=" * 70)
            print(f"Model:         {model_name}")
            print(f"Best model:    {checkpoint_dir / 'best_model_live.pth'}")
            print(f"Final model:   {final_path}")
            print(f"{onnx_msg}")
            if onnx_path:
                print(f"Metadata:      {onnx_path.with_suffix('.txt')}")
            print(f"Total epochs:  {epochs}")
            print(f"Best val loss: {best_val_loss:.6f}")
            print("=" * 70)
        else:
            print("\n" + "=" * 70)
            print("TRAINING INTERRUPTED")
            print("=" * 70)
            print(f"Model saved at epoch {current_epoch}")
            print("=" * 70)
        
        # 保存图表
        if live_plot:
            plot_path = checkpoint_dir / f'training_history_{timestamp}.png'
            live_plot.fig.savefig(plot_path, dpi=150, bbox_inches='tight')
            print(f"\nTraining plot saved: {plot_path}")
            live_plot.close()
        print("Done!")


if __name__ == '__main__':
    main()

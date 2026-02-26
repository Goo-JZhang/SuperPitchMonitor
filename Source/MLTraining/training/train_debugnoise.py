#!/usr/bin/env python3
"""
调试脚本：分别显示 Noise 和 Sanity 验证集的 Loss

使用方式:
    python train_debugnoise.py --config trainconfig.txt
    python train_debugnoise.py --model PitchNetEnhanced --epochs 100 --batch-size 64 --lr 0.0005
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
from loss import PitchDetectionLoss
from modeloutput_utils import export_model_with_metadata, format_training_info


class DebugTrainingPlot:
    """调试训练图表 - 分别显示 Noise 和 Sanity 验证集"""
    
    def __init__(self):
        plt.ion()
        # 2x3 布局
        self.fig, self.axes = plt.subplots(3, 3, figsize=(15, 10))
        self.fig.suptitle('Training Progress - Separate Validation Sets', fontsize=14)
        
        # 第一列：训练指标
        self.ax_train_total = self.axes[0, 0]
        self.ax_train_conf = self.axes[1, 0]
        self.ax_train_energy = self.axes[2, 0]
        
        # 第二列：Sanity 验证集
        self.ax_sanity_total = self.axes[0, 1]
        self.ax_sanity_conf = self.axes[1, 1]
        self.ax_sanity_energy = self.axes[2, 1]
        
        # 第三列：Noise 验证集
        self.ax_noise_total = self.axes[0, 2]
        self.ax_noise_conf = self.axes[1, 2]
        self.ax_noise_energy = self.axes[2, 2]
        
        # 训练曲线
        self.train_total_line, = self.ax_train_total.plot([], [], 'b-', label='Train', linewidth=2)
        self.train_conf_line, = self.ax_train_conf.plot([], [], 'b-', label='Train', linewidth=2)
        self.train_energy_line, = self.ax_train_energy.plot([], [], 'b-', label='Train', linewidth=2)
        
        # Sanity 验证曲线
        self.sanity_total_line, = self.ax_sanity_total.plot([], [], 'g-', label='Sanity Val', linewidth=2)
        self.sanity_conf_line, = self.ax_sanity_conf.plot([], [], 'g-', label='Sanity Val', linewidth=2)
        self.sanity_energy_line, = self.ax_sanity_energy.plot([], [], 'g-', label='Sanity Val', linewidth=2)
        
        # Noise 验证曲线
        self.noise_total_line, = self.ax_noise_total.plot([], [], 'r-', label='Noise Val', linewidth=2)
        self.noise_conf_line, = self.ax_noise_conf.plot([], [], 'r-', label='Noise Val', linewidth=2)
        self.noise_energy_line, = self.ax_noise_energy.plot([], [], 'r-', label='Noise Val', linewidth=2)
        
        # 设置标题
        self.ax_train_total.set_title('Total Loss (Train)')
        self.ax_train_conf.set_title('Confidence Loss (Train)')
        self.ax_train_energy.set_title('Energy Loss (Train)')
        
        self.ax_sanity_total.set_title('Total Loss (Sanity Val)')
        self.ax_sanity_conf.set_title('Confidence Loss (Sanity Val)')
        self.ax_sanity_energy.set_title('Energy Loss (Sanity Val)')
        
        self.ax_noise_total.set_title('Total Loss (Noise Val)')
        self.ax_noise_conf.set_title('Confidence Loss (Noise Val)')
        self.ax_noise_energy.set_title('Energy Loss (Noise Val)')
        
        # 启用网格
        for ax in self.axes.flat:
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # 数据存储
        self.data = {
            'epochs': [],
            'train_total': [], 'train_conf': [], 'train_energy': [],
            'sanity_total': [], 'sanity_conf': [], 'sanity_energy': [],
            'noise_total': [], 'noise_conf': [], 'noise_energy': [],
        }
        
        plt.tight_layout()
        plt.show(block=False)
        print("Visualization window opened. If not visible, check your taskbar.")
    
    def update(self, epoch, train_metrics, sanity_metrics, noise_metrics):
        self.data['epochs'].append(epoch)
        
        # 训练指标
        self.data['train_total'].append(train_metrics['total'])
        self.data['train_conf'].append(train_metrics['confidence'])
        self.data['train_energy'].append(train_metrics['energy'])
        
        # Sanity 验证指标
        self.data['sanity_total'].append(sanity_metrics['total'])
        self.data['sanity_conf'].append(sanity_metrics['confidence'])
        self.data['sanity_energy'].append(sanity_metrics['energy'])
        
        # Noise 验证指标
        self.data['noise_total'].append(noise_metrics['total'])
        self.data['noise_conf'].append(noise_metrics['confidence'])
        self.data['noise_energy'].append(noise_metrics['energy'])
        
        # 更新训练曲线
        self.train_total_line.set_data(self.data['epochs'], self.data['train_total'])
        self.train_conf_line.set_data(self.data['epochs'], self.data['train_conf'])
        self.train_energy_line.set_data(self.data['epochs'], self.data['train_energy'])
        
        # 更新 Sanity 曲线
        self.sanity_total_line.set_data(self.data['epochs'], self.data['sanity_total'])
        self.sanity_conf_line.set_data(self.data['epochs'], self.data['sanity_conf'])
        self.sanity_energy_line.set_data(self.data['epochs'], self.data['sanity_energy'])
        
        # 更新 Noise 曲线
        self.noise_total_line.set_data(self.data['epochs'], self.data['noise_total'])
        self.noise_conf_line.set_data(self.data['epochs'], self.data['noise_conf'])
        self.noise_energy_line.set_data(self.data['epochs'], self.data['noise_energy'])
        
        # 自动缩放
        for ax in self.axes.flat:
            ax.relim()
            ax.autoscale_view()
        
        self.fig.suptitle(f'Training Progress - Epoch {epoch}', fontsize=14)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
    
    def close(self):
        plt.ioff()
        plt.close()


def train_epoch(model, dataloader, optimizer, device, criterion):
    """训练一个 epoch"""
    model.train()
    metrics = {'total': 0, 'confidence': 0, 'energy': 0, 'sparsity': 0}
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
        metrics['energy'] += losses['energy'].item()
        metrics['sparsity'] += losses['sparsity'].item()
        count += 1
    
    return {k: v / count for k, v in metrics.items()}


def validate(model, dataloader, device, criterion):
    """验证函数"""
    model.eval()
    metrics = {'total': 0, 'confidence': 0, 'energy': 0, 'sparsity': 0}
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
            metrics['energy'] += losses['energy'].item()
            metrics['sparsity'] += losses['sparsity'].item()
            count += 1
    
    return {k: v / count for k, v in metrics.items()}


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Debug training with separate validation sets')
    return add_common_training_args(parser).parse_args()


def main():
    args = parse_args()
    
    # 获取完整配置
    config = get_training_config(args)
    
    # 项目路径
    project_root = get_project_root()
    
    # 提取配置参数
    data_root = config.get('data_root') or config.get('root_dir') or str(get_default_data_root())
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
    print("Debug Training - Separate Validation Sets")
    print("=" * 70)
    print(f"Model: {model_name}")
    print(f"Data root: {data_root}")
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
    
    # 加载数据集 - 分别加载 Sanity 和 Noise
    print(f"\nLoading datasets...")
    data_root_path = Path(data_root)
    
    # 只加载这两个数据集
    sanity_dir = data_root_path / 'SingleSanity'
    noise_dir = data_root_path / 'NoiseDataset'
    
    if not sanity_dir.exists():
        raise FileNotFoundError(f"SingleSanity not found: {sanity_dir}")
    if not noise_dir.exists():
        raise FileNotFoundError(f"NoiseDataset not found: {noise_dir}")
    
    # 加载 Sanity
    print("Loading SingleSanity...")
    sanity_ds = DatasetReader(str(sanity_dir), preload=preload, device=str(device))
    sanity_n_val = max(1, int(len(sanity_ds) * val_split))
    sanity_n_train = len(sanity_ds) - sanity_n_val
    sanity_train, sanity_val = random_split(sanity_ds, [sanity_n_train, sanity_n_val])
    print(f"  SingleSanity: {len(sanity_ds):6d} samples -> Train: {sanity_n_train:6d}, Val: {sanity_n_val:4d}")
    
    # 加载 Noise
    print("Loading NoiseDataset...")
    noise_ds = DatasetReader(str(noise_dir), preload=preload, device=str(device))
    noise_n_val = max(1, int(len(noise_ds) * val_split))
    noise_n_train = len(noise_ds) - noise_n_val
    noise_train, noise_val = random_split(noise_ds, [noise_n_train, noise_n_val])
    print(f"  NoiseDataset: {len(noise_ds):6d} samples -> Train: {noise_n_train:6d}, Val: {noise_n_val:4d}")
    
    # 合并训练集
    train_dataset = ConcatDataset([sanity_train, noise_train])
    
    # 验证集保持分开
    sanity_val_loader = DataLoader(sanity_val, batch_size=batch_size, num_workers=0, pin_memory=False)
    noise_val_loader = DataLoader(noise_val, batch_size=batch_size, num_workers=0, pin_memory=False)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False
    )
    
    print(f"\nTrain: {len(train_dataset)}, Sanity Val: {len(sanity_val)}, Noise Val: {len(noise_val)}")
    
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
    criterion = PitchDetectionLoss(
        conf_weight=1.0,
        energy_weight=0.3,
        sparsity_weight=0.01,
        energy_loss_type='kl'
    ).to(device)
    
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
        live_plot = DebugTrainingPlot()
    else:
        print("\nVisualization disabled.")
    
    # 训练
    checkpoint_dir = Path(project_root) / 'MLModel' / 'checkpoints'
    mlmodel_dir = Path(project_root) / 'MLModel'
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Checkpoints: {checkpoint_dir}")
    print("\nTraining started! Press Ctrl+C to stop early.")
    print("=" * 90)
    print(f"{'Epoch':>6} | {'Time':>5} | {'Train Loss':>10} | {'Sanity Val':>10} | {'Noise Val':>10} | {'LR':>10}")
    print("=" * 90)
    
    best_val_loss = float('inf')
    interrupted = False
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    try:
        for epoch in range(1, epochs + 1):
            start = time.time()
            
            # 训练
            train_metrics = train_epoch(model, train_loader, optimizer, device, criterion)
            
            if device.type == 'mps':
                torch.mps.synchronize()
            elif device.type == 'cuda':
                torch.cuda.synchronize()
            
            # 分别验证 Sanity 和 Noise
            sanity_metrics = validate(model, sanity_val_loader, device, criterion)
            noise_metrics = validate(model, noise_val_loader, device, criterion)
            
            if device.type == 'mps':
                torch.mps.synchronize()
            elif device.type == 'cuda':
                torch.cuda.synchronize()
            
            current_lr = optimizer.param_groups[0]['lr']
            scheduler.step()
            
            # 更新图表
            if live_plot:
                live_plot.update(epoch, train_metrics, sanity_metrics, noise_metrics)
            
            # 保存最佳模型（基于平均验证损失）
            avg_val_loss = (sanity_metrics['total'] + noise_metrics['total']) / 2
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'val_loss': best_val_loss,
                }, str(checkpoint_dir / 'best_model_debug.pth'))
            
            elapsed = time.time() - start
            
            # 打印详细结果
            print(f"{epoch:6d} | {elapsed:5.1f}s | "
                  f"{train_metrics['total']:10.4f} | "
                  f"{sanity_metrics['total']:10.4f} | "
                  f"{noise_metrics['total']:10.4f} | "
                  f"{current_lr:10.6f}")
            print(f"         |       | c:{train_metrics['confidence']:8.3f} | "
                  f"c:{sanity_metrics['confidence']:8.3f} | "
                  f"c:{noise_metrics['confidence']:8.3f} |")
            print(f"         |       | e:{train_metrics['energy']:8.3f} | "
                  f"e:{sanity_metrics['energy']:8.3f} | "
                  f"e:{noise_metrics['energy']:8.3f} |")
            print("-" * 90)
            
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
        final_path = checkpoint_dir / 'final_model_debug.pth'
        torch.save({
            'epoch': current_epoch,
            'model_state_dict': model.state_dict(),
        }, str(final_path))
        
        # 保存历史
        if live_plot:
            with open(str(checkpoint_dir / 'history_debug.json'), 'w') as f:
                json.dump(live_plot.data, f)
        
        if not interrupted:
            # 导出ONNX
            onnx_path = None
            try:
                print("\n[DEBUG] Starting ONNX export process...")
                model.eval()
                model.cpu()
                
                training_info = format_training_info(
                    config={'data_root': data_root, 'subdirs': ['SingleSanity', 'NoiseDataset'], 
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
            print(f"Best model:    {checkpoint_dir / 'best_model_debug.pth'}")
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
            plot_path = checkpoint_dir / f'training_history_debug_{timestamp}.png'
            live_plot.fig.savefig(plot_path, dpi=150, bbox_inches='tight')
            print(f"\nTraining plot saved: {plot_path}")
            live_plot.close()
        print("Done!")


if __name__ == '__main__':
    main()

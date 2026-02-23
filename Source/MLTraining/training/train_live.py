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
    
    # 指定数据子目录
    python train_live.py --data SingleSanity,NoiseDataset
"""

import os
import sys
import json
import time
import argparse
import io
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

from PitchNetBaseline import PitchNetBaseline
from dataset import DatasetReader
from loss import EnergyLoss, ConfidenceLoss, PitchDetectionLoss
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
    parser = argparse.ArgumentParser(description='Training script with live visualization')
    
    # 配置文件
    parser.add_argument('--config', type=str, default=None,
                       help='Path to config file (.txt, .json, or .yaml)')
    
    # 数据配置
    parser.add_argument('--data-root', type=str, default=None,
                       help='Root directory for training data')
    parser.add_argument('--data', type=str, default=None,
                       help='Comma-separated list of data subdirectories (e.g., "SingleSanity,NoiseDataset")')
    parser.add_argument('--preload', action='store_true', default=None,
                       help='Preload data to memory')
    parser.add_argument('--streaming', action='store_true',
                       help='Use streaming mode (no preload)')
    parser.add_argument('--max-memory', type=float, default=4.0,
                       help='Max memory for auto preload decision (GB)')
    
    # 模型配置
    parser.add_argument('--model', type=str, default='PitchNetBaseline',
                       help='Model name (PitchNetBaseline)')
    parser.add_argument('--pretrained', type=str, default=None,
                       help='Path to pretrained weights')
    
    # 训练配置
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=None,
                       help='Batch size (default: 64 Mac, 128 others)')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--val-split', type=float, default=0.02,
                       help='Validation split ratio')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--no-viz', action='store_true',
                       help='Disable visualization window')
    
    return parser.parse_args()


def load_config_from_file(config_path: str):
    """从文件加载配置"""
    path = Path(config_path)
    
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    suffix = path.suffix.lower()
    
    with open(path, 'r', encoding='utf-8') as f:
        if suffix == '.json':
            import json
            return json.load(f)
        elif suffix in ['.yaml', '.yml']:
            import yaml
            return yaml.safe_load(f)
        else:
            # 简单文本格式 key=value 或 [section]
            return parse_txt_config(f.read())


def parse_txt_config(content: str):
    """解析简单文本配置"""
    config = {}
    current_section = None
    
    for line in content.strip().split('\n'):
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        
        if line.startswith('[') and line.endswith(']'):
            current_section = line[1:-1].lower()
            config[current_section] = {}
            continue
        
        if '=' in line:
            key, value = line.split('=', 1)
            key = key.strip()
            value = parse_value(value.strip())
            
            if current_section:
                config[current_section][key] = value
            else:
                config[key] = value
    
    return config


def parse_value(value: str):
    """解析配置值"""
    if value.lower() in ['true', 'yes']:
        return True
    if value.lower() in ['false', 'no']:
        return False
    if value.lower() in ['null', 'none']:
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    if ',' in value:
        return [parse_value(v.strip()) for v in value.split(',')]
    return value


def get_default_data_root():
    """获取默认数据根目录"""
    script_dir = Path(__file__).parent.resolve()
    return script_dir.parent.parent.parent / 'TrainingData'


def main():
    args = parse_args()
    
    # 加载配置文件（如果提供）
    config = {}
    if args.config:
        print(f"Loading config from: {args.config}")
        file_config = load_config_from_file(args.config)
        # 扁平化配置
        for section, values in file_config.items():
            if isinstance(values, dict):
                config.update(values)
            else:
                config[section] = values
    
    # 命令行参数覆盖配置
    script_dir = Path(__file__).parent.resolve()
    project_root = script_dir.parent.parent.parent
    
    # 数据根目录
    data_root = args.data_root or config.get('root_dir') or str(get_default_data_root())
    
    # 数据子目录
    if args.data:
        data_subdirs = [d.strip() for d in args.data.split(',')]
    elif config.get('subdirs'):
        data_subdirs = config.get('subdirs')
        if isinstance(data_subdirs, str):
            data_subdirs = [d.strip() for d in data_subdirs.split(',')]
    else:
        data_subdirs = None  # 自动发现所有
    
    # 加载方式 (Windows上mmap+ConcatDataset有问题，默认预加载)
    if args.streaming:
        preload = False
    elif args.preload:
        preload = True
    else:
        preload = config.get('preload', True)  # 默认True，避免Windows mmap问题
    
    max_memory = args.max_memory or config.get('max_memory_gb', 4.0)
    
    # 训练参数
    system = platform.system()
    default_batch = 64 if system == 'Darwin' else 128
    
    epochs = args.epochs or config.get('epochs', 50)
    batch_size = args.batch_size or config.get('batch_size') or default_batch
    lr = args.lr or config.get('lr', 0.001)
    val_split = args.val_split or config.get('val_split', 0.02)
    seed = args.seed or config.get('seed', 42)
    
    print("=" * 70)
    print("Training Configuration")
    print("=" * 70)
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
    
    # 加载数据集（按类型分别加载并拆分，确保验证集包含所有类型）
    print(f"\nLoading datasets...")
    from dataset import DatasetReader
    from torch.utils.data import ConcatDataset
    
    # 发现数据子目录
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
        n_val = max(1, int(len(ds) * val_split))  # 至少1个验证样本
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
    
    # 模型
    print(f"\nInitializing model: {args.model}")
    if args.model == 'PitchNetBaseline':
        model = PitchNetBaseline()
    else:
        raise ValueError(f"Unknown model: {args.model}")
    
    # 加载预训练权重
    if args.pretrained or config.get('path'):
        pretrained_path = args.pretrained or config.get('path')
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
    
    # 可视化 (可选，添加 --no-viz 参数可禁用)
    use_viz = not getattr(args, 'no_viz', False)
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
            print(f"Epoch {epoch:2d}/{epochs} | {elapsed:.1f}s | "
                  f"Train: {train_metrics['total']:.4f} (c:{train_metrics['confidence']:.3f} e:{train_metrics['energy']:.3f}) | "
                  f"Val: {val_metrics['total']:.4f} (c:{val_metrics['confidence']:.3f} e:{val_metrics['energy']:.3f}) | "
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
                # 准备模型导出（eval模式，移到CPU）
                model.eval()
                print("[DEBUG] Model set to eval mode")
                model.cpu()
                print("[DEBUG] Model moved to CPU")
                
                training_info = format_training_info(
                    config={'data_root': data_root, 'subdirs': data_subdirs, 'epochs': epochs, 'batch_size': batch_size, 'lr': lr},
                    best_val_loss=best_val_loss,
                    device=device
                )
                print(f"[DEBUG] Training info prepared")
                
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

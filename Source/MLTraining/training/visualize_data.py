#!/usr/bin/env python3
"""
训练数据可视化工具 + 模型推理

功能:
- 交互式查看数据集样本
- 加载 PyTorch 模型进行实时推理
- 对比真实标签和模型预测
"""

import sys
import json
import numpy as np
import tkinter as tk
from tkinter import ttk
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from pathlib import Path

# 添加路径以加载模型
sys.path.insert(0, str(Path(__file__).parent.parent / 'Model'))


class DataVisualizer:
    """数据可视化器 - 支持模型推理"""
    
    def __init__(self, data_root='../../../TrainingData', model_root='../../../MLModel'):
        self.data_root = Path(data_root)
        self.model_root = Path(model_root)
        self.current_idx = 0
        
        # 发现数据集
        self.datasets = self._discover_datasets()
        if not self.datasets:
            print(f"No valid datasets found in {data_root}")
            sys.exit(1)
        
        print(f"Found datasets: {self.datasets}")
        
        # 扫描所有数据集的分片
        self.dataset_shards = {}
        self.dataset_samples = {}
        for ds in self.datasets:
            self._scan_dataset(ds)
        
        # 发现模型
        self.models = self._discover_models()
        print(f"Found models: {[m['name'] for m in self.models]}")
        
        # 当前选择
        self.current_dataset = self.datasets[0]
        self.current_shards = self.dataset_shards[self.current_dataset]
        self.current_shard_idx = 0
        self.current_model = None  # 当前加载的模型
        
        # 加载第一个分片
        self._load_shard(self.current_dataset, 0)
        
        # 创建界面
        self._setup_ui()
        
    def _discover_datasets(self):
        """发现有效数据集"""
        datasets = []
        if not self.data_root.exists():
            return datasets
        
        for path in sorted(self.data_root.iterdir()):
            if path.is_dir() and (path / 'meta.json').exists():
                datasets.append(path.name)
        return datasets
    
    def _discover_models(self):
        """发现 MLModel 文件夹下的 PyTorch 模型"""
        models = [{'name': 'None', 'path': None, 'info': 'No model loaded'}]
        
        if not self.model_root.exists():
            return models
        
        for pth_file in sorted(self.model_root.glob('*.pth')):
            # 尝试读取元数据文件（同名 .txt）
            meta_file = pth_file.with_suffix('.txt')
            info = 'No metadata'
            if meta_file.exists():
                try:
                    with open(meta_file, 'r') as f:
                        lines = f.readlines()
                        # 提取关键信息
                        for line in lines:
                            if 'Total Parameters' in line:
                                info = line.strip().replace('# ', '')
                                break
                except:
                    pass
            
            models.append({
                'name': pth_file.stem,
                'path': str(pth_file),
                'info': info
            })
        
        return models
    
    def _scan_dataset(self, dataset_name):
        """扫描数据集的分片信息"""
        dataset_path = self.data_root / dataset_name
        wave_dir = dataset_path / 'waveforms'
        
        if not wave_dir.exists():
            self.dataset_shards[dataset_name] = []
            self.dataset_samples[dataset_name] = 0
            return
        
        wave_files = sorted(wave_dir.glob('shard_*.npy'))
        self.dataset_shards[dataset_name] = [f.name for f in wave_files]
        
        try:
            with open(dataset_path / 'meta.json', 'r') as f:
                meta = json.load(f)
            self.dataset_samples[dataset_name] = meta.get('total_samples', 0)
        except:
            self.dataset_samples[dataset_name] = 0
    
    def _load_shard(self, dataset_name, shard_idx):
        """加载指定分片"""
        if shard_idx >= len(self.dataset_shards[dataset_name]):
            return False
        
        dataset_path = self.data_root / dataset_name
        shard_name = self.dataset_shards[dataset_name][shard_idx]
        
        wave_path = dataset_path / 'waveforms' / shard_name
        target_path = dataset_path / 'targets' / shard_name.replace('.npy', '.npz')
        
        try:
            self.waveforms = np.load(wave_path)
            target = np.load(target_path)
            self.confs = target['confs']
            self.energies = target['energies']
            self.current_dataset = dataset_name
            self.current_shard_idx = shard_idx
            self.n_samples = len(self.waveforms)
            return True
        except Exception as e:
            print(f"Error loading {dataset_name}/{shard_name}: {e}")
            return False
    
    def _load_model(self, model_info):
        """加载 PyTorch 模型"""
        if model_info['path'] is None:
            self.current_model = None
            print("Model unloaded")
            return True
        
        try:
            import torch
            from PitchNetBaseline import PitchNetBaseline
            
            print(f"Loading model: {model_info['name']}...")
            model = PitchNetBaseline()
            checkpoint = torch.load(model_info['path'], map_location='cpu')
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            
            self.current_model = model
            print(f"Model loaded successfully!")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            self.current_model = None
            return False
    
    def _preprocess_waveform(self, waveform):
        """
        预处理波形 - 与 C++ 推理代码保持一致
        1. 去直流偏移 (mean = 0)
        2. 峰值归一化到 [-1, 1]
        """
        # 去直流偏移
        waveform = waveform - np.mean(waveform)
        
        # 峰值归一化
        max_amp = np.max(np.abs(waveform))
        if max_amp > 1e-8:
            waveform = waveform / max_amp
        
        return waveform
    
    def _run_inference(self, waveform):
        """运行模型推理"""
        if self.current_model is None:
            return None, None
        
        try:
            import torch
            
            # 预处理
            waveform = self._preprocess_waveform(waveform)
            
            # 转换为 tensor [1, 1, 4096]
            x = torch.from_numpy(waveform).float().unsqueeze(0).unsqueeze(0)
            
            # 推理
            with torch.no_grad():
                output = self.current_model(x)
            
            # 提取预测结果
            pred_conf = output[0, :, 0].numpy()
            pred_energy = output[0, :, 1].numpy()
            
            return pred_conf, pred_energy
        except Exception as e:
            print(f"Inference error: {e}")
            return None, None
    
    def _setup_ui(self):
        """设置 Tkinter 界面"""
        # 创建主窗口
        self.root = tk.Tk()
        self.root.title("Training Data Visualizer with Model Inference")
        self.root.geometry("1400x1000")
        
        # 控制面板
        control_frame = ttk.Frame(self.root, padding="10")
        control_frame.pack(fill=tk.X)
        
        # 数据集选择
        ttk.Label(control_frame, text="Dataset:").pack(side=tk.LEFT, padx=(0, 5))
        self.dataset_var = tk.StringVar(value=self.current_dataset)
        self.dataset_combo = ttk.Combobox(control_frame, textvariable=self.dataset_var,
                                         values=self.datasets, state="readonly", width=20)
        self.dataset_combo.pack(side=tk.LEFT, padx=(0, 20))
        self.dataset_combo.bind("<<ComboboxSelected>>", self._on_dataset_change)
        
        # 分片选择
        ttk.Label(control_frame, text="Shard:").pack(side=tk.LEFT, padx=(0, 5))
        self.shard_var = tk.StringVar()
        self.shard_combo = ttk.Combobox(control_frame, textvariable=self.shard_var,
                                       values=self.current_shards, state="readonly", width=25)
        self.shard_combo.pack(side=tk.LEFT, padx=(0, 20))
        self.shard_combo.bind("<<ComboboxSelected>>", self._on_shard_change)
        if self.current_shards:
            self.shard_combo.set(self.current_shards[0])
        
        # 索引输入
        ttk.Label(control_frame, text="Index:").pack(side=tk.LEFT, padx=(0, 5))
        self.index_var = tk.StringVar(value="0")
        self.index_entry = ttk.Entry(control_frame, textvariable=self.index_var, width=8)
        self.index_entry.pack(side=tk.LEFT, padx=(0, 5))
        self.index_entry.bind("<Return>", self._on_index_submit)
        self.index_entry.bind("<FocusOut>", self._on_index_submit)
        
        # 最大索引标签
        self.max_label = ttk.Label(control_frame, text=f"/ {self.n_samples-1}")
        self.max_label.pack(side=tk.LEFT, padx=(0, 20))
        
        # 导航按钮
        ttk.Button(control_frame, text="<< Prev", command=self._on_prev).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(control_frame, text="Next >>", command=self._on_next).pack(side=tk.LEFT, padx=(0, 20))
        
        # 模型选择
        ttk.Label(control_frame, text="Model:").pack(side=tk.LEFT, padx=(0, 5))
        self.model_var = tk.StringVar(value="None")
        model_names = [m['name'] for m in self.models]
        self.model_combo = ttk.Combobox(control_frame, textvariable=self.model_var,
                                       values=model_names, state="readonly", width=30)
        self.model_combo.pack(side=tk.LEFT, padx=(0, 10))
        self.model_combo.bind("<<ComboboxSelected>>", self._on_model_change)
        
        # 模型信息标签
        self.model_info_label = ttk.Label(control_frame, text="No model loaded")
        self.model_info_label.pack(side=tk.LEFT, padx=(0, 0))
        
        # Matplotlib 图形 - 3x2 布局
        self.fig = plt.figure(figsize=(14, 12))
        
        # 嵌入到 Tkinter
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # 创建子图
        # 第1行：波形和预测对比
        self.ax_wave = self.fig.add_subplot(3, 2, 1)
        self.ax_pred_conf = self.fig.add_subplot(3, 2, 2)
        
        # 第2行：真实标签
        self.ax_true_conf = self.fig.add_subplot(3, 2, 3)
        self.ax_true_energy = self.fig.add_subplot(3, 2, 4)
        
        # 第3行：预测标签
        self.ax_pred_energy = self.fig.add_subplot(3, 2, 5)
        self.ax_diff = self.fig.add_subplot(3, 2, 6)  # 差异图
        
        self.fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        self.fig.suptitle('Training Data Visualizer with Model Inference', fontsize=14, fontweight='bold')
        
        # 显示第一个样本
        self._display_sample(0)
        
        # 键盘绑定
        self.root.bind("<Left>", lambda e: self._on_prev())
        self.root.bind("<Right>", lambda e: self._on_next())
        
        # 窗口关闭处理
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        
    def _on_close(self):
        """关闭窗口"""
        print("Closing visualizer...")
        self.root.destroy()
        sys.exit(0)
        
    def _on_model_change(self, event):
        """模型改变"""
        model_name = self.model_var.get()
        
        # 查找模型信息
        model_info = None
        for m in self.models:
            if m['name'] == model_name:
                model_info = m
                break
        
        if model_info:
            self._load_model(model_info)
            self.model_info_label.config(text=model_info['info'])
            # 刷新当前样本的显示
            self._display_sample(self.current_idx)
    
    def _on_dataset_change(self, event):
        """数据集改变"""
        dataset_name = self.dataset_var.get()
        if dataset_name == self.current_dataset:
            return
        
        self.current_dataset = dataset_name
        self.current_shards = self.dataset_shards[dataset_name]
        
        # 更新分片下拉框
        self.shard_combo['values'] = self.current_shards
        if self.current_shards:
            self.shard_combo.set(self.current_shards[0])
            self._load_and_display(0)
    
    def _on_shard_change(self, event):
        """分片改变"""
        shard_name = self.shard_var.get()
        if not shard_name:
            return
        
        try:
            shard_idx = self.current_shards.index(shard_name)
            if shard_idx != self.current_shard_idx:
                self._load_and_display(shard_idx)
        except ValueError:
            pass
    
    def _load_and_display(self, shard_idx):
        """加载并显示"""
        if self._load_shard(self.current_dataset, shard_idx):
            self.current_idx = 0
            self.index_var.set("0")
            self.max_label.config(text=f"/ {self.n_samples-1}")
            self._display_sample(0)
    
    def _on_index_submit(self, event=None):
        """索引提交"""
        try:
            idx = int(self.index_var.get())
            idx = max(0, min(idx, self.n_samples - 1))
            if idx != self.current_idx:
                self.current_idx = idx
                self._display_sample(idx)
            self.index_var.set(str(self.current_idx))
        except ValueError:
            self.index_var.set(str(self.current_idx))
    
    def _on_prev(self):
        """上一个"""
        if self.current_idx > 0:
            self.current_idx -= 1
            self.index_var.set(str(self.current_idx))
            self._display_sample(self.current_idx)
    
    def _on_next(self):
        """下一个"""
        if self.current_idx < self.n_samples - 1:
            self.current_idx += 1
            self.index_var.set(str(self.current_idx))
            self._display_sample(self.current_idx)
    
    def _calc_entropy(self, energy):
        """计算能量分布的熵"""
        e = energy[energy > 0]
        if len(e) == 0:
            return 0.0
        e = e / e.sum()
        return -np.sum(e * np.log(e + 1e-10))
    
    def _display_sample(self, idx):
        """显示样本"""
        if not hasattr(self, 'waveforms'):
            return
        
        waveform = self.waveforms[idx].copy()
        true_conf = self.confs[idx]
        true_energy = self.energies[idx]
        
        # 运行推理
        pred_conf, pred_energy = self._run_inference(waveform)
        
        # 清除所有子图
        for ax in [self.ax_wave, self.ax_pred_conf, self.ax_true_conf, 
                   self.ax_true_energy, self.ax_pred_energy, self.ax_diff]:
            ax.clear()
        
        # 预处理后的波形（与输入模型的一致）
        waveform_processed = self._preprocess_waveform(waveform)
        
        sample_rate = 44100
        t = np.arange(len(waveform_processed)) / sample_rate * 1000
        
        # === 第1行：波形和预测置信度 ===
        # 波形（预处理后）
        self.ax_wave.plot(t, waveform_processed, 'b-', linewidth=0.5)
        self.ax_wave.set_xlabel('Time (ms)')
        self.ax_wave.set_ylabel('Amplitude')
        self.ax_wave.set_title(f'Waveform (Preprocessed) - Sample {idx}')
        self.ax_wave.grid(True, alpha=0.3)
        self.ax_wave.set_ylim(-1.2, 1.2)
        
        # 预测置信度（如果有）
        bins = np.arange(len(true_conf))
        if pred_conf is not None:
            self.ax_pred_conf.plot(bins, pred_conf, 'b-', linewidth=1, alpha=0.8, label='Predicted')
            if pred_conf.max() > 0.1:
                peak_idx = np.argmax(pred_conf)
                self.ax_pred_conf.plot(peak_idx, pred_conf[peak_idx], 'r*', markersize=10, 
                                      label=f'Peak@{peak_idx} ({pred_conf[peak_idx]:.3f})')
            self.ax_pred_conf.set_title(f'Predicted Confidence - max={pred_conf.max():.3f}, {(pred_conf > 0.5).sum()} > 0.5')
        else:
            self.ax_pred_conf.text(0.5, 0.5, 'No model loaded', ha='center', va='center', fontsize=12)
            self.ax_pred_conf.set_title('Predicted Confidence')
        self.ax_pred_conf.set_xlabel('Bin')
        self.ax_pred_conf.set_ylabel('Confidence')
        self.ax_pred_conf.set_xlim(0, len(true_conf))
        self.ax_pred_conf.set_ylim(0, 1.1)
        self.ax_pred_conf.grid(True, alpha=0.3)
        if pred_conf is not None:
            self.ax_pred_conf.legend()
        
        # === 第2行：真实标签 ===
        # 真实置信度
        self.ax_true_conf.plot(bins, true_conf, 'g-', linewidth=1, alpha=0.8, label='Ground Truth')
        if true_conf.max() > 0:
            peak_idx = np.argmax(true_conf)
            self.ax_true_conf.plot(peak_idx, true_conf[peak_idx], 'r*', markersize=10, 
                                  label=f'Peak@{peak_idx} ({true_conf[peak_idx]:.3f})')
        self.ax_true_conf.set_xlabel('Bin')
        self.ax_true_conf.set_ylabel('Confidence')
        self.ax_true_conf.set_title(f'True Confidence - {(true_conf > 0).sum()} non-zero, max={true_conf.max():.3f}')
        self.ax_true_conf.set_xlim(0, len(true_conf))
        self.ax_true_conf.set_ylim(0, 1.1)
        self.ax_true_conf.grid(True, alpha=0.3)
        self.ax_true_conf.legend()
        
        # 真实能量
        self.ax_true_energy.plot(bins, true_energy, color='orange', linewidth=1, alpha=0.8, label='Ground Truth')
        peak_e_idx = np.argmax(true_energy)
        self.ax_true_energy.plot(peak_e_idx, true_energy[peak_e_idx], 'r*', markersize=10, 
                                label=f'Peak@{peak_e_idx} ({true_energy[peak_e_idx]:.4f})')
        self.ax_true_energy.set_xlabel('Bin')
        self.ax_true_energy.set_ylabel('Energy')
        self.ax_true_energy.set_title(f'True Energy - sum={true_energy.sum():.4f}, entropy={self._calc_entropy(true_energy):.3f}')
        self.ax_true_energy.set_xlim(0, len(true_energy))
        self.ax_true_energy.grid(True, alpha=0.3)
        self.ax_true_energy.legend(loc='upper right')
        
        # === 第3行：预测能量和差异 ===
        # 预测能量（如果有）
        if pred_energy is not None:
            self.ax_pred_energy.plot(bins, pred_energy, color='purple', linewidth=1, alpha=0.8, label='Predicted')
            peak_p_idx = np.argmax(pred_energy)
            self.ax_pred_energy.plot(peak_p_idx, pred_energy[peak_p_idx], 'r*', markersize=10,
                                    label=f'Peak@{peak_p_idx} ({pred_energy[peak_p_idx]:.4f})')
            self.ax_pred_energy.set_title(f'Predicted Energy - sum={pred_energy.sum():.4f}, entropy={self._calc_entropy(pred_energy):.3f}')
        else:
            self.ax_pred_energy.text(0.5, 0.5, 'No model loaded', ha='center', va='center', fontsize=12)
            self.ax_pred_energy.set_title('Predicted Energy')
        self.ax_pred_energy.set_xlabel('Bin')
        self.ax_pred_energy.set_ylabel('Energy')
        self.ax_pred_energy.set_xlim(0, len(true_energy))
        self.ax_pred_energy.grid(True, alpha=0.3)
        if pred_energy is not None:
            self.ax_pred_energy.legend(loc='upper right')
        
        # 差异图（真实 vs 预测）
        if pred_conf is not None and pred_energy is not None:
            conf_diff = np.abs(true_conf - pred_conf)
            energy_diff = np.abs(true_energy - pred_energy)
            self.ax_diff.plot(bins, conf_diff, 'b-', linewidth=1, alpha=0.7, label='|Conf Diff|')
            self.ax_diff.plot(bins, energy_diff * 10, 'r-', linewidth=1, alpha=0.7, label='|Energy Diff| x10')
            self.ax_diff.set_xlabel('Bin')
            self.ax_diff.set_ylabel('Absolute Difference')
            self.ax_diff.set_title(f'Prediction Error - Conf MAE: {conf_diff.mean():.4f}, Energy MAE: {energy_diff.mean():.4f}')
            self.ax_diff.legend()
        else:
            self.ax_diff.text(0.5, 0.5, 'Load model to see prediction error', ha='center', va='center', fontsize=10)
            self.ax_diff.set_title('Prediction Error')
        self.ax_diff.set_xlim(0, len(true_conf))
        self.ax_diff.grid(True, alpha=0.3)
        
        self.fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        self.canvas.draw()
    
    def run(self):
        """运行"""
        print("\nControls:")
        print("  Dataset/Shard: Select data source")
        print("  Index: Type sample index (Enter to confirm)")
        print("  << Prev / Next >>: Navigate samples")
        print("  Left/Right arrow keys: Navigate samples")
        print("  Model: Select trained model for inference")
        self.root.mainloop()


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Training Data Visualizer with Model Inference')
    parser.add_argument('--data-root', type=str, default='../../../TrainingData')
    parser.add_argument('--model-root', type=str, default='../../../MLModel')
    args = parser.parse_args()
    
    visualizer = DataVisualizer(args.data_root, args.model_root)
    visualizer.run()


if __name__ == '__main__':
    main()

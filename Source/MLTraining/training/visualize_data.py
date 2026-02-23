#!/usr/bin/env python3
"""
训练数据可视化工具 + 模型推理

3x3 布局:
- Row 1: 原始波形 | 预处理后波形 | FFT频谱
- Row 2: Conf GT | Conf Pred | Conf Error
- Row 3: Energy GT | Energy Pred | Energy Error
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
        self.current_model = None
        
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
            meta_file = pth_file.with_suffix('.txt')
            info = 'No metadata'
            if meta_file.exists():
                try:
                    with open(meta_file, 'r') as f:
                        lines = f.readlines()
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
        """预处理波形 - Z-score 归一化，与 C++ 推理代码保持一致"""
        waveform = waveform - np.mean(waveform)
        std = np.std(waveform)
        if std > 1e-8:
            waveform = waveform / std
        return waveform
    
    def _run_inference(self, waveform):
        """运行模型推理"""
        if self.current_model is None:
            return None, None
        
        try:
            import torch
            
            waveform = self._preprocess_waveform(waveform)
            x = torch.from_numpy(waveform).float().unsqueeze(0).unsqueeze(0)
            
            with torch.no_grad():
                output = self.current_model(x)
            
            pred_conf = output[0, :, 0].numpy()
            pred_energy = output[0, :, 1].numpy()
            
            return pred_conf, pred_energy
        except Exception as e:
            print(f"Inference error: {e}")
            return None, None
    
    def _compute_fft_spectrum(self, waveform):
        """
        计算FFT频谱并映射到2048 bins（与GT一致）
        返回: bin_energy[2048] - 每个bin的能量（PSD × delta_f，然后归一化）
        """
        n_fft = 4096
        sample_rate = 44100
        
        # FFT
        fft = np.fft.rfft(waveform, n=n_fft)
        power = np.square(np.abs(fft))  # PSD（功率谱密度）
        
        # 频率轴
        freqs = np.fft.rfftfreq(n_fft, d=1/sample_rate)
        
        # 2048 bins的对数频率边界（与数据生成时一致）
        min_freq, max_freq = 20.0, 5000.0
        log_min, log_max = np.log2(min_freq), np.log2(max_freq)
        
        # bin 边界（共 2049 个）
        bin_edges = np.array([
            2 ** (log_min + (i / 2048) * (log_max - log_min))
            for i in range(2049)
        ])
        
        # 计算每个bin的能量 = PSD × delta_f（积分）
        bin_energy = np.zeros(2048)
        for i in range(2048):
            f_low, f_high = bin_edges[i], bin_edges[i+1]
            delta_f = f_high - f_low
            
            # 找到该bin内的所有FFT点
            mask = (freqs >= f_low) & (freqs < f_high)
            if mask.any():
                # bin能量 = 平均PSD × delta_f
                bin_energy[i] = np.mean(power[mask]) * delta_f
            else:
                # 使用 numpy 插值
                f_center = (f_low + f_high) / 2
                idx = np.searchsorted(freqs, f_center)
                if idx >= len(power):
                    idx = len(power) - 1
                if idx == 0:
                    idx = 1
                # 线性插值
                f0, f1 = freqs[idx-1], freqs[idx]
                p0, p1 = power[idx-1], power[idx]
                if f1 > f0:
                    t = (f_center - f0) / (f1 - f0)
                    psd_center = p0 * (1-t) + p1 * t
                else:
                    psd_center = power[idx]
                bin_energy[i] = psd_center * delta_f
        
        # 归一化为概率分布
        energy_sum = bin_energy.sum()
        if energy_sum > 0:
            bin_energy = bin_energy / energy_sum
        
        return bin_energy
    
    def _setup_ui(self):
        """设置 Tkinter 界面"""
        self.root = tk.Tk()
        self.root.title("Training Data Visualizer with Model Inference")
        self.root.geometry("1500x1100")
        
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
        
        # Matplotlib 图形 - 3x3 布局
        self.fig = plt.figure(figsize=(15, 12))
        
        # 嵌入到 Tkinter
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # 创建 3x3 子图
        # Row 1: Waveforms and FFT
        self.ax_wave_raw = self.fig.add_subplot(3, 3, 1)
        self.ax_wave_proc = self.fig.add_subplot(3, 3, 2)
        self.ax_fft = self.fig.add_subplot(3, 3, 3)
        
        # Row 2: Confidence (GT, Pred, Error)
        self.ax_conf_gt = self.fig.add_subplot(3, 3, 4)
        self.ax_conf_pred = self.fig.add_subplot(3, 3, 5)
        self.ax_conf_err = self.fig.add_subplot(3, 3, 6)
        
        # Row 3: Energy (GT, Pred, Error)
        self.ax_energy_gt = self.fig.add_subplot(3, 3, 7)
        self.ax_energy_pred = self.fig.add_subplot(3, 3, 8)
        self.ax_energy_err = self.fig.add_subplot(3, 3, 9)
        
        self.fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        self.fig.suptitle('Training Data Visualizer (3x3 Layout)', fontsize=14, fontweight='bold')
        
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
        model_info = None
        for m in self.models:
            if m['name'] == model_name:
                model_info = m
                break
        
        if model_info:
            self._load_model(model_info)
            self.model_info_label.config(text=model_info['info'])
            self._display_sample(self.current_idx)
    
    def _on_dataset_change(self, event):
        """数据集改变"""
        dataset_name = self.dataset_var.get()
        if dataset_name == self.current_dataset:
            return
        
        self.current_dataset = dataset_name
        self.current_shards = self.dataset_shards[dataset_name]
        
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
    
    def _calc_kl_divergence(self, p, q):
        """
        计算 KL 散度 D_KL(P || Q)
        训练使用的是 KL 散度损失
        """
        # 确保概率分布非零
        p = np.clip(p, 1e-10, 1.0)
        q = np.clip(q, 1e-10, 1.0)
        # 归一化
        p = p / p.sum()
        q = q / q.sum()
        # KL(P || Q) = sum(p * log(p/q))
        return np.sum(p * np.log(p / q))
    
    def _calc_bce_loss(self, y_true, y_pred):
        """
        计算二元交叉熵损失 BCE
        训练时 confidence 使用 BCE 损失
        """
        epsilon = 1e-7
        # 裁剪避免 log(0)
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        y_true = np.clip(y_true, epsilon, 1 - epsilon)
        # BCE = -[y*log(p) + (1-y)*log(1-p)]
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    
    def _display_sample(self, idx):
        """显示样本 - 3x3 布局"""
        if not hasattr(self, 'waveforms'):
            return
        
        waveform_raw = self.waveforms[idx].copy()
        true_conf = self.confs[idx]
        true_energy = self.energies[idx]
        
        # 运行推理
        pred_conf, pred_energy = self._run_inference(waveform_raw)
        
        # 预处理波形
        waveform_proc = self._preprocess_waveform(waveform_raw)
        
        # 清除所有子图
        for ax in [self.ax_wave_raw, self.ax_wave_proc, self.ax_fft,
                   self.ax_conf_gt, self.ax_conf_pred, self.ax_conf_err,
                   self.ax_energy_gt, self.ax_energy_pred, self.ax_energy_err]:
            ax.clear()
        
        sample_rate = 44100
        t = np.arange(len(waveform_raw)) / sample_rate * 1000
        bins = np.arange(len(true_conf))
        
        # ========== Row 1: Waveforms and FFT ==========
        # (1,1) 原始波形
        self.ax_wave_raw.plot(t, waveform_raw, 'b-', linewidth=0.5)
        self.ax_wave_raw.set_xlabel('Time (ms)')
        self.ax_wave_raw.set_ylabel('Amplitude')
        self.ax_wave_raw.set_title(f'Raw Waveform - Sample {idx}')
        self.ax_wave_raw.grid(True, alpha=0.3)
        
        # (1,2) 预处理后波形
        self.ax_wave_proc.plot(t, waveform_proc, 'g-', linewidth=0.5)
        self.ax_wave_proc.set_xlabel('Time (ms)')
        self.ax_wave_proc.set_ylabel('Amplitude')
        self.ax_wave_proc.set_title('Preprocessed Waveform (mean=0, std=1)')
        self.ax_wave_proc.grid(True, alpha=0.3)
        self.ax_wave_proc.set_ylim(waveform_proc.min(), waveform_proc.max())
        
        # (1,3) FFT频谱（映射到2048 bins，与conf/energy对齐）
        try:
            fft_spectrum = self._compute_fft_spectrum(waveform_proc)
            self.ax_fft.plot(bins, fft_spectrum, 'purple', linewidth=0.8)
            self.ax_fft.set_xlabel('Bin')
            self.ax_fft.set_ylabel('Magnitude(Normalize to 1)')
            self.ax_fft.set_title('FFT Spectrum (2048 bins, log-freq)')
            self.ax_fft.set_xlim(0, len(bins))
            self.ax_fft.grid(True, alpha=0.3)
        except Exception as e:
            self.ax_fft.text(0.5, 0.5, f'FFT Error: {e}', ha='center', va='center', fontsize=9)
            self.ax_fft.set_title('FFT Spectrum')
        
        # ========== Row 2: Confidence ==========
        # (2,1) Conf GT
        self.ax_conf_gt.plot(bins, true_conf, 'g-', linewidth=1, alpha=0.8)
        if true_conf.max() > 0:
            peak_idx = np.argmax(true_conf)
            self.ax_conf_gt.plot(peak_idx, true_conf[peak_idx], 'r*', markersize=10,
                                label=f'Peak@{peak_idx} ({true_conf[peak_idx]:.3f})')
            self.ax_conf_gt.legend(fontsize=8)
        self.ax_conf_gt.set_ylabel('Confidence')
        nz_count = (true_conf > 0).sum()
        if pred_conf is not None:
            bce_loss = self._calc_bce_loss(true_conf, pred_conf)
            self.ax_conf_gt.set_title(f'GT Conf - {nz_count} nz, max={true_conf.max():.3f}, BCE={bce_loss:.4f}')
        else:
            self.ax_conf_gt.set_title(f'GT Conf - {nz_count} nz, max={true_conf.max():.3f}')
        self.ax_conf_gt.set_xlim(0, len(true_conf))
        self.ax_conf_gt.set_ylim(0, 1.1)
        self.ax_conf_gt.grid(True, alpha=0.3)
        
        # (2,2) Conf Pred
        if pred_conf is not None:
            self.ax_conf_pred.plot(bins, pred_conf, 'b-', linewidth=1, alpha=0.8)
            if pred_conf.max() > 0.1:
                peak_idx = np.argmax(pred_conf)
                self.ax_conf_pred.plot(peak_idx, pred_conf[peak_idx], 'r*', markersize=10,
                                      label=f'Peak@{peak_idx} ({pred_conf[peak_idx]:.3f})')
                self.ax_conf_pred.legend(fontsize=8)
            gt_05_count = (pred_conf > 0.5).sum()
            bce_loss = self._calc_bce_loss(pred_conf, true_conf)
            self.ax_conf_pred.set_title(f'Pred Conf - {gt_05_count} > 0.5, max={pred_conf.max():.3f}, BCE={bce_loss:.4f}')
        else:
            self.ax_conf_pred.text(0.5, 0.5, 'No model', ha='center', va='center', fontsize=10)
            self.ax_conf_pred.set_title('Pred Conf')
        self.ax_conf_pred.set_xlabel('Bin')
        self.ax_conf_pred.set_ylabel('Confidence')
        self.ax_conf_pred.set_xlim(0, len(true_conf))
        self.ax_conf_pred.set_ylim(0, 1.1)
        self.ax_conf_pred.grid(True, alpha=0.3)
        
        # (2,3) Conf Error
        if pred_conf is not None:
            conf_err = true_conf - pred_conf
            # 用不同颜色区分正负
            pos_mask = conf_err >= 0
            neg_mask = conf_err < 0
            self.ax_conf_err.fill_between(bins, 0, conf_err, where=pos_mask, 
                                         color='green', alpha=0.5, label='GT > Pred')
            self.ax_conf_err.fill_between(bins, 0, conf_err, where=neg_mask,
                                         color='red', alpha=0.5, label='GT < Pred')
            self.ax_conf_err.axhline(y=0, color='k', linestyle='--', linewidth=0.8)
            self.ax_conf_err.set_xlabel('Bin')
            self.ax_conf_err.set_ylabel('Error (GT - Pred)')
            self.ax_conf_err.set_title(f'Conf Error - μ={conf_err.mean():.4f}, σ={conf_err.std():.4f}')
            self.ax_conf_err.legend(fontsize=8)
        else:
            self.ax_conf_err.text(0.5, 0.5, 'Load model to see error', ha='center', va='center', fontsize=9)
            self.ax_conf_err.set_title('Conf Error')
        self.ax_conf_err.set_xlim(0, len(true_conf))
        self.ax_conf_err.grid(True, alpha=0.3)
        
        # ========== Row 3: Energy ==========
        # (3,1) Energy GT
        self.ax_energy_gt.plot(bins, true_energy, color='orange', linewidth=1, alpha=0.8)
        peak_e_idx = np.argmax(true_energy)
        self.ax_energy_gt.plot(peak_e_idx, true_energy[peak_e_idx], 'r*', markersize=10,
                              label=f'Peak@{peak_e_idx} ({true_energy[peak_e_idx]:.4f})')
        self.ax_energy_gt.set_ylabel('Energy')
        gt_entropy = self._calc_entropy(true_energy)
        if pred_energy is not None:
            gt_kl = self._calc_kl_divergence(true_energy, pred_energy)
            self.ax_energy_gt.set_title(f'GT Energy - H={gt_entropy:.3f}, KL(gt||pred)={gt_kl:.4f}')
        else:
            self.ax_energy_gt.set_title(f'GT Energy - H={gt_entropy:.3f}')
        self.ax_energy_gt.set_xlim(0, len(true_energy))
        self.ax_energy_gt.grid(True, alpha=0.3)
        self.ax_energy_gt.legend(fontsize=8)
        
        # (3,2) Energy Pred
        if pred_energy is not None:
            self.ax_energy_pred.plot(bins, pred_energy, color='purple', linewidth=1, alpha=0.8)
            peak_p_idx = np.argmax(pred_energy)
            self.ax_energy_pred.plot(peak_p_idx, pred_energy[peak_p_idx], 'r*', markersize=10,
                                    label=f'Peak@{peak_p_idx} ({pred_energy[peak_p_idx]:.4f})')
            pred_entropy = self._calc_entropy(pred_energy)
            pred_kl = self._calc_kl_divergence(pred_energy, true_energy)
            self.ax_energy_pred.set_title(f'Pred Energy - H={pred_entropy:.3f}, KL(pred||gt)={pred_kl:.4f}')
            self.ax_energy_pred.legend(fontsize=8)
        else:
            self.ax_energy_pred.text(0.5, 0.5, 'No model', ha='center', va='center', fontsize=10)
            self.ax_energy_pred.set_title('Pred Energy')
        self.ax_energy_pred.set_xlabel('Bin')
        self.ax_energy_pred.set_ylabel('Energy')
        self.ax_energy_pred.set_xlim(0, len(true_energy))
        self.ax_energy_pred.grid(True, alpha=0.3)
        
        # (3,3) Energy Error
        if pred_energy is not None:
            energy_err = true_energy - pred_energy
            pos_mask = energy_err >= 0
            neg_mask = energy_err < 0
            self.ax_energy_err.fill_between(bins, 0, energy_err, where=pos_mask,
                                           color='green', alpha=0.5, label='GT > Pred')
            self.ax_energy_err.fill_between(bins, 0, energy_err, where=neg_mask,
                                           color='red', alpha=0.5, label='GT < Pred')
            self.ax_energy_err.axhline(y=0, color='k', linestyle='--', linewidth=0.8)
            self.ax_energy_err.set_xlabel('Bin')
            self.ax_energy_err.set_ylabel('Error (GT - Pred)')
            # 计算 JS 散度（对称的，更能代表整体差异）
            m = (true_energy + pred_energy) / 2
            js_div = (self._calc_kl_divergence(true_energy, m) + 
                     self._calc_kl_divergence(pred_energy, m)) / 2
            self.ax_energy_err.set_title(f'Energy Error - JS={js_div:.4f}, μ={energy_err.mean():.4f}')
            self.ax_energy_err.legend(fontsize=8)
        else:
            self.ax_energy_err.text(0.5, 0.5, 'Load model to see error', ha='center', va='center', fontsize=9)
            self.ax_energy_err.set_title('Energy Error')
        self.ax_energy_err.set_xlim(0, len(true_energy))
        self.ax_energy_err.grid(True, alpha=0.3)
        
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
        print("\nLayout (3x3):")
        print("  Row 1: Raw Wave | Preprocessed Wave | FFT Spectrum")
        print("  Row 2: Conf GT | Conf Pred | Conf Error")
        print("  Row 3: Energy GT | Energy Pred | Energy Error")
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

#!/usr/bin/env python3
"""
训练工具函数库

提供训练脚本共用的功能:
- ONNX导出与元数据生成
- 模型信息统计
- 其他通用工具
"""

import torch
import platform
from pathlib import Path
from datetime import datetime


def export_model_with_metadata(model, output_dir, timestamp, training_info=None):
    """
    导出模型（ONNX + PyTorch）并创建元数据文件
    
    文件名格式: 
    - ONNX: {ModelName}_{timestamp}.onnx
    - PyTorch: {ModelName}_{timestamp}.pth
    元数据文件: {ModelName}_{timestamp}.txt
    
    Args:
        model: PyTorch模型
        output_dir: 输出目录 (如 'MLModel')
        timestamp: 时间戳字符串 (如 '20260219_143052')
        training_info: 训练信息字典，包含:
            - Dataset: 数据集路径
            - Total Epochs: 训练轮数
            - Batch Size: 批大小
            - Learning Rate: 学习率
            - Final Val Loss: 最终验证损失
            - Device: 训练设备
            - Platform: 操作系统平台
            - 其他自定义信息
    
    Returns:
        onnx_path: 导出的ONNX文件路径 (Path对象)
    
    Example:
        >>> model = PitchNetBaseline()
        >>> training_info = {
        ...     'Dataset': 'TrainingData/SingleSanity',
        ...     'Total Epochs': 50,
        ...     'Final Val Loss': 0.208,
        ... }
        >>> onnx_path = export_to_onnx_with_metadata(
        ...     model, 'MLModel', '20260219_143052', training_info
        ... )
        >>> print(onnx_path)
        MLModel/PitchNetBaseline_20260219_143052.onnx
        >>> meta_path = onnx_path.with_suffix('.txt')
        >>> print(meta_path.exists())
        True
    """
    model.eval()
    
    # 模型名称 (从类名获取)
    model_name = model.__class__.__name__
    
    # 构建文件名: {模型名}_{时间戳}.onnx
    onnx_filename = f"{model_name}_{timestamp}.onnx"
    onnx_path = Path(output_dir) / onnx_filename
    
    # 确保输出目录存在
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 导出ONNX
    print(f"[DEBUG] Preparing dummy input...")
    device = next(model.parameters()).device
    dummy_input = torch.randn(1, 1, 4096, device=device)
    print(f"[DEBUG] Dummy input shape: {dummy_input.shape}, device: {device}")
    
    # 使用传统导出器 (更稳定)
    print(f"[DEBUG] Starting ONNX export to: {onnx_path}")
    try:
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
            opset_version=17,
            do_constant_folding=True,
        )
        print(f"[DEBUG] ONNX export completed successfully")
    except Exception as e:
        print(f"[DEBUG] ONNX export failed with error: {e}")
        raise
    
    # 导出 PyTorch 模型 (.pth)
    pth_filename = f"{model_name}_{timestamp}.pth"
    pth_path = Path(output_dir) / pth_filename
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_name': model_name,
        'timestamp': timestamp,
        'training_info': training_info,
    }, pth_path)
    print(f"[DEBUG] PyTorch model saved: {pth_path}")
    
    # 创建元数据文件 (同名.txt)
    meta_filename = f"{model_name}_{timestamp}.txt"
    meta_path = Path(output_dir) / meta_filename
    
    # 收集元信息
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # 构建元数据内容
    training_info_str = "\n".join(
        f"{k}: {v}" for k, v in (training_info or {}).items()
    ) if training_info else "N/A"
    
    meta_content = f"""# Model Metadata
Model Name: {model_name}
Export Time: {timestamp}
ONNX File: {onnx_filename}
PyTorch File: {pth_filename}

# Model Architecture
Total Parameters: {param_count:,} ({param_count/1e6:.2f}M)
Input Shape: [batch, 1, 4096]
Output Shape: [batch, 2048, 2]  # [confidence, energy]

# Training Information
{training_info_str}

# File Location
- ONNX: {onnx_path.absolute()}
- PyTorch: {pth_path.absolute()}
- Meta: {meta_path.absolute()}

# Usage Example (ONNX)
import onnxruntime as ort
import numpy as np

session = ort.InferenceSession("{onnx_filename}")
input_data = np.random.randn(1, 1, 4096).astype(np.float32)
output = session.run(None, {{'input': input_data}})
# output[0].shape = [1, 2048, 2]

# Usage Example (PyTorch)
import torch
from PitchNetBaseline import PitchNetBaseline

model = PitchNetBaseline()
checkpoint = torch.load("{pth_filename}")
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
"""
    
    with open(meta_path, 'w', encoding='utf-8') as f:
        f.write(meta_content)
    
    return onnx_path


def get_model_info(model):
    """
    获取模型信息统计
    
    Args:
        model: PyTorch模型
    
    Returns:
        dict: 包含模型信息的字典
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # 计算模型大小 (假设float32)
    model_size_mb = total_params * 4 / 1024 / 1024
    
    return {
        'model_name': model.__class__.__name__,
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'model_size_mb': model_size_mb,
    }


def format_training_info(config, best_val_loss, device, epochs_completed=None):
    """
    格式化训练信息
    
    Args:
        config: 训练配置字典
        best_val_loss: 最佳验证损失
        device: 训练设备
        epochs_completed: 实际完成的epochs (用于中断训练)
    
    Returns:
        dict: 训练信息字典
    """
    info = {
        'Dataset': Path(config.get('data_path', 'N/A')).name,
        'Total Epochs': epochs_completed or config.get('epochs', 'N/A'),
        'Batch Size': config.get('batch_size', 'N/A'),
        'Learning Rate': config.get('lr', 'N/A'),
        'Best Val Loss': f"{best_val_loss:.6f}" if best_val_loss else 'N/A',
        'Device': str(device),
        'Platform': platform.system(),
    }
    return info


# 导出函数
__all__ = ['export_model_with_metadata', 'format_training_info', 'get_model_info']

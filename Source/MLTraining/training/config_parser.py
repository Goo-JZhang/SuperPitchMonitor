#!/usr/bin/env python3
"""
训练配置解析器

支持从配置文件读取训练参数
"""

import json
import yaml
from pathlib import Path
from typing import Dict, List, Any, Optional


class TrainConfig:
    """训练配置类"""
    
    # 默认值
    DEFAULTS = {
        'model': {
            'name': 'PitchNetBaseline',
            'path': None,  # 如果指定，从该路径加载预训练权重
        },
        'data': {
            'root_dir': '../../../TrainingData',
            'subdirs': None,  # None表示遍历所有子目录，或指定列表 ['SingleSanity', 'NoiseDataset']
            'preload': True,  # True=内存加载，False=流式加载
            'max_memory_gb': 4.0,  # 超过此内存使用流式加载
        },
        'training': {
            'epochs': 50,
            'batch_size': 128,
            'lr': 0.001,
            'val_split': 0.02,
            'seed': 42,
        },
        'output': {
            'checkpoint_dir': '../../../MLModel/checkpoints',
            'model_dir': '../../../MLModel',
            'save_interval': 10,  # 每N个epoch保存一次
        }
    }
    
    def __init__(self, config_dict: Optional[Dict] = None):
        """初始化配置"""
        self.config = self._merge_config(config_dict or {})
        
    def _merge_config(self, user_config: Dict) -> Dict:
        """合并用户配置和默认配置"""
        config = {}
        for key, default_value in self.DEFAULTS.items():
            if isinstance(default_value, dict):
                config[key] = {**default_value, **user_config.get(key, {})}
            else:
                config[key] = user_config.get(key, default_value)
        return config
    
    @classmethod
    def from_file(cls, filepath: str) -> 'TrainConfig':
        """从文件加载配置"""
        path = Path(filepath)
        
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {filepath}")
        
        # 根据后缀解析
        suffix = path.suffix.lower()
        
        with open(path, 'r', encoding='utf-8') as f:
            if suffix == '.json':
                config_dict = json.load(f)
            elif suffix in ['.yaml', '.yml']:
                config_dict = yaml.safe_load(f)
            elif suffix == '.txt':
                # 简单格式: key=value
                config_dict = cls._parse_txt(f.read())
            else:
                raise ValueError(f"Unsupported config format: {suffix}")
        
        return cls(config_dict)
    
    @classmethod
    def _parse_txt(cls, content: str) -> Dict:
        """解析简单文本格式配置"""
        config = {}
        current_section = None
        
        for line in content.strip().split('\n'):
            line = line.strip()
            
            # 跳过空行和注释
            if not line or line.startswith('#'):
                continue
            
            # 段落标记 [section]
            if line.startswith('[') and line.endswith(']'):
                current_section = line[1:-1].lower()
                config[current_section] = {}
                continue
            
            # key = value
            if '=' in line:
                key, value = line.split('=', 1)
                key = key.strip()
                value = value.strip()
                
                # 尝试解析类型
                value = cls._parse_value(value)
                
                if current_section:
                    config[current_section][key] = value
                else:
                    config[key] = value
        
        return config
    
    @classmethod
    def _parse_value(cls, value: str) -> Any:
        """解析配置值类型"""
        # 布尔值
        if value.lower() in ['true', 'yes', 'on']:
            return True
        if value.lower() in ['false', 'no', 'off']:
            return False
        
        # None
        if value.lower() in ['null', 'none']:
            return None
        
        # 整数
        try:
            return int(value)
        except ValueError:
            pass
        
        # 浮点数
        try:
            return float(value)
        except ValueError:
            pass
        
        # 列表 (逗号分隔)
        if ',' in value:
            return [cls._parse_value(v.strip()) for v in value.split(',')]
        
        # 字符串
        return value
    
    def get(self, *keys, default=None):
        """获取嵌套配置值"""
        value = self.config
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        return value
    
    def __getitem__(self, key):
        return self.config[key]
    
    def to_dict(self) -> Dict:
        """导出为字典"""
        return self.config.copy()
    
    def __repr__(self):
        return f"TrainConfig({json.dumps(self.config, indent=2)})"


def create_default_config(output_path: str = 'trainconfig.txt'):
    """创建默认配置文件示例"""
    config_content = """# 训练配置文件
# 格式: key = value 或 [section] 下配置

[model]
name = PitchNetBaseline
# path = ../../../MLModel/pretrained.pth  # 可选：预训练权重路径

[data]
root_dir = ../../../TrainingData
# subdirs = SingleSanity, NoiseDataset  # 指定子目录，注释掉则遍历所有
preload = true
max_memory_gb = 4.0

[training]
epochs = 50
batch_size = 128
lr = 0.001
val_split = 0.02
seed = 42

[output]
checkpoint_dir = ../../../MLModel/checkpoints
model_dir = ../../../MLModel
save_interval = 10
"""
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(config_content)
    
    print(f"Default config created: {output_path}")
    return output_path


if __name__ == '__main__':
    # 测试
    import tempfile
    import os
    
    # 创建测试配置
    test_config = """
[model]
name = PitchNetBaseline

[data]
root_dir = ./data
subdirs = SingleSanity, NoiseDataset
preload = false

[training]
epochs = 100
batch_size = 64
"""
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(test_config)
        temp_path = f.name
    
    try:
        config = TrainConfig.from_file(temp_path)
        print("Parsed config:")
        print(f"  Model: {config.get('model', 'name')}")
        print(f"  Data root: {config.get('data', 'root_dir')}")
        print(f"  Subdirs: {config.get('data', 'subdirs')}")
        print(f"  Preload: {config.get('data', 'preload')}")
        print(f"  Epochs: {config.get('training', 'epochs')}")
        print(f"  Batch: {config.get('training', 'batch_size')}")
        print(f"  Default LR: {config.get('training', 'lr')}")  # 使用默认值
    finally:
        os.unlink(temp_path)

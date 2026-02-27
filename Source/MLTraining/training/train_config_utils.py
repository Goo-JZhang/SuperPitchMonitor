#!/usr/bin/env python3
"""
训练配置工具模块

提供训练脚本共用的配置解析、模型创建等功能。
避免在多个训练脚本中重复实现相同的逻辑。
"""

import argparse
from pathlib import Path
from typing import Dict, Any, Optional, List, Union


def parse_value(value: str) -> Union[str, int, float, bool, None, List]:
    """
    解析配置值为适当的Python类型
    
    支持的类型:
    - bool: 'true', 'yes' -> True; 'false', 'no' -> False
    - None: 'null', 'none' -> None
    - int: 整数字符串 -> int
    - float: 浮点数字符串 -> float
    - list: 逗号分隔值 -> list
    - str: 其他 -> str
    
    Args:
        value: 配置值字符串
    Returns:
        解析后的Python对象
    """
    value_lower = value.lower()
    
    if value_lower in ['true', 'yes']:
        return True
    if value_lower in ['false', 'no']:
        return False
    if value_lower in ['null', 'none']:
        return None
    
    # 尝试整数
    try:
        return int(value)
    except ValueError:
        pass
    
    # 尝试浮点数
    try:
        return float(value)
    except ValueError:
        pass
    
    # 尝试列表 (逗号分隔)
    if ',' in value:
        return [parse_value(v.strip()) for v in value.split(',')]
    
    # 默认为字符串
    return value


def parse_txt_config(content: str) -> Dict[str, Any]:
    """
    解析简单文本配置文件
    
    格式示例:
        # 这是注释
        name = PitchNetEnhanced
        epochs = 100
        
        [data]
        root_dir = TrainingData
        subdirs = SingleSanity, NoiseDataset
        
        [training]
        batch_size = 64
        lr = 0.001
    
    Args:
        content: 配置文件内容
    Returns:
        解析后的配置字典
    """
    config = {}
    current_section = None
    
    for line in content.strip().split('\n'):
        line = line.strip()
        
        # 跳过空行和注释
        if not line or line.startswith('#') or line.startswith(';'):
            continue
        
        # 节标题 [section]
        if line.startswith('[') and line.endswith(']'):
            current_section = line[1:-1].lower()
            config[current_section] = {}
            continue
        
        # 键值对 key = value
        if '=' in line:
            key, value = line.split('=', 1)
            key = key.strip()
            value = parse_value(value.strip())
            
            if current_section:
                config[current_section][key] = value
            else:
                config[key] = value
    
    return config


def load_config_from_file(config_path: str) -> Dict[str, Any]:
    """
    从文件加载配置
    
    支持格式:
    - .json: JSON格式
    - .yaml/.yml: YAML格式
    - 其他: 简单文本格式 (key=value)
    
    Args:
        config_path: 配置文件路径
    Returns:
        配置字典
    Raises:
        FileNotFoundError: 文件不存在
        ValueError: 格式错误
    """
    path = Path(config_path)
    
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    suffix = path.suffix.lower()
    
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    if suffix == '.json':
        import json
        return json.loads(content)
    
    elif suffix in ['.yaml', '.yml']:
        try:
            import yaml
            return yaml.safe_load(content)
        except ImportError:
            raise ImportError("PyYAML is required for YAML config files. Install with: pip install pyyaml")
    
    else:
        # 默认使用文本格式解析
        return parse_txt_config(content)


def merge_config(args: argparse.Namespace, file_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    合并命令行参数和配置文件
    
    优先级: 命令行参数 > 配置文件 > 默认值
    
    Args:
        args: argparse解析的命令行参数
        file_config: 配置文件解析的字典
    Returns:
        合并后的配置字典
    """
    # 扁平化文件配置（处理section）
    flat_config = {}
    for key, value in file_config.items():
        if isinstance(value, dict):
            # 处理section嵌套
            for sub_key, sub_value in value.items():
                flat_config[sub_key] = sub_value
        else:
            flat_config[key] = value
    
    # 合并配置：命令行参数覆盖配置文件
    merged = flat_config.copy()
    
    # 将命令行参数添加到合并配置
    for key, value in vars(args).items():
        # 只覆盖非None的命令行参数
        if value is not None:
            merged[key] = value
    
    return merged


def get_default_data_root() -> Path:
    """获取默认数据根目录"""
    script_dir = Path(__file__).parent.resolve()
    return script_dir.parent.parent.parent / 'TrainingData'


def get_project_root() -> Path:
    """获取项目根目录"""
    script_dir = Path(__file__).parent.resolve()
    return script_dir.parent.parent.parent


# ============== 模型创建工厂 ==============

# 支持的模型注册表
_MODEL_REGISTRY = {}


def register_model(name: str):
    """模型注册装饰器"""
    def decorator(cls):
        _MODEL_REGISTRY[name.lower()] = cls
        return cls
    return decorator


def create_model(model_name: str, **kwargs) -> Any:
    """
    工厂函数：根据名称创建模型实例
    
    支持的模型:
    - PitchNetBaseline: 基础模型
    - PitchNetEnhanced: 增强版模型 (分离Backbone + Attention)
    - PitchNetEnhancedV2: 添加门控融合
    - PitchNetEnhancedV3: 大容量版本 (256通道 + Transformer Block)
    - PitchNetEnhancedV4: 跨平台版本 (移动端友好算子，无LayerNorm/GELU)
    
    Args:
        model_name: 模型名称
        **kwargs: 传递给模型构造函数的参数 (如 input_samples, num_bins)
    Returns:
        模型实例
    Raises:
        ValueError: 未知模型名称
    """
    # 延迟导入模型（避免循环导入）
    if not _MODEL_REGISTRY:
        _register_builtin_models()
    
    name_lower = model_name.lower()
    
    if name_lower not in _MODEL_REGISTRY:
        available = ', '.join(_MODEL_REGISTRY.keys())
        raise ValueError(f"Unknown model: '{model_name}'. Available models: {available}")
    
    model_class = _MODEL_REGISTRY[name_lower]
    return model_class(**kwargs)


def _register_builtin_models():
    """注册内置模型"""
    import sys
    
    # 添加Model目录到路径
    model_dir = Path(__file__).parent.parent / 'Model'
    if str(model_dir) not in sys.path:
        sys.path.insert(0, str(model_dir))
    
    try:
        from PitchNetBaseline import PitchNetBaseline
        _MODEL_REGISTRY['pitchnetbaseline'] = PitchNetBaseline
        _MODEL_REGISTRY['baseline'] = PitchNetBaseline
    except ImportError as e:
        print(f"Warning: Failed to import PitchNetBaseline: {e}")
    
    try:
        from PitchNetEnhanced import PitchNetEnhanced
        _MODEL_REGISTRY['pitchnetenhanced'] = PitchNetEnhanced
        _MODEL_REGISTRY['enhanced'] = PitchNetEnhanced
    except ImportError as e:
        print(f"Warning: Failed to import PitchNetEnhanced: {e}")
    
    try:
        from PitchNetEnhancedV2 import PitchNetEnhancedV2
        _MODEL_REGISTRY['pitchnetenhancedv2'] = PitchNetEnhancedV2
        _MODEL_REGISTRY['enhancedv2'] = PitchNetEnhancedV2
        _MODEL_REGISTRY['v2'] = PitchNetEnhancedV2
    except ImportError as e:
        print(f"Warning: Failed to import PitchNetEnhancedV2: {e}")
    
    try:
        from PitchNetEnhancedV3 import PitchNetEnhancedV3
        _MODEL_REGISTRY['pitchnetenhancedv3'] = PitchNetEnhancedV3
        _MODEL_REGISTRY['enhancedv3'] = PitchNetEnhancedV3
        _MODEL_REGISTRY['v3'] = PitchNetEnhancedV3
    except ImportError as e:
        print(f"Warning: Failed to import PitchNetEnhancedV3: {e}")
    
    try:
        from PitchNetEnhancedV4 import PitchNetEnhancedV4
        _MODEL_REGISTRY['pitchnetenhancedv4'] = PitchNetEnhancedV4
        _MODEL_REGISTRY['enhancedv4'] = PitchNetEnhancedV4
        _MODEL_REGISTRY['v4'] = PitchNetEnhancedV4
    except ImportError as e:
        print(f"Warning: Failed to import PitchNetEnhancedV4: {e}")


def list_available_models() -> List[str]:
    """列出所有可用模型名称"""
    if not _MODEL_REGISTRY:
        _register_builtin_models()
    return list(_MODEL_REGISTRY.keys())


# ============== 常用训练参数解析 ==============

def add_common_training_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """
    添加常用的训练参数到ArgumentParser
    
    Args:
        parser: ArgumentParser实例
    Returns:
        配置好的ArgumentParser
    """
    # 配置
    parser.add_argument('--config', type=str, default=None,
                       help='Path to config file (.txt, .json, or .yaml)')
    
    # 数据配置
    parser.add_argument('--data-root', type=str, default=None,
                       help='Root directory for training data')
    parser.add_argument('--data', type=str, default=None,
                       help='Comma-separated list of data subdirectories')
    parser.add_argument('--preload', action='store_true', default=None,
                       help='Preload data to memory')
    parser.add_argument('--streaming', action='store_true',
                       help='Use streaming mode (no preload)')
    parser.add_argument('--max-memory', type=float, default=None,
                       help='Max memory for auto preload decision (GB)')
    
    # 模型配置
    parser.add_argument('--model', type=str, default=None,
                       help='Model name (PitchNetBaseline, PitchNetEnhanced, etc.)')
    parser.add_argument('--pretrained', type=str, default=None,
                       help='Path to pretrained weights')
    
    # 训练配置
    parser.add_argument('--epochs', type=int, default=None,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=None,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=None,
                       help='Learning rate')
    parser.add_argument('--val-split', type=float, default=None,
                       help='Validation split ratio')
    parser.add_argument('--seed', type=int, default=None,
                       help='Random seed')
    parser.add_argument('--no-viz', action='store_true',
                       help='Disable visualization window')
    
    return parser


def get_training_config(args: argparse.Namespace) -> Dict[str, Any]:
    """
    从命令行参数获取完整的训练配置
    
    Args:
        args: argparse解析的参数
    Returns:
        完整的配置字典
    """
    config = {}
    
    # 加载配置文件（如果提供）
    if args.config:
        file_config = load_config_from_file(args.config)
        config = merge_config(args, file_config)
    else:
        # 直接使用命令行参数
        config = {k: v for k, v in vars(args).items() if v is not None}
    
    # 处理特殊参数
    # data 参数转换为列表
    if 'data' in config and isinstance(config['data'], str):
        config['data_subdirs'] = [d.strip() for d in config['data'].split(',')]
    
    # 模型名称规范化（支持 'model' 或 'name' 作为模型名称键）
    if 'model' in config and config['model'] is not None:
        config['model_name'] = config['model']
    elif 'name' in config and config['name'] is not None:
        config['model_name'] = config['name']
    
    # 设置默认值
    defaults = {
        'epochs': 50,
        'lr': 0.001,
        'val_split': 0.02,
        'seed': 42,
        'preload': True,
        'max_memory': 4.0,
        'model_name': 'PitchNetBaseline',
    }
    
    for key, value in defaults.items():
        if key not in config or config[key] is None:
            config[key] = value
    
    return config


# ============== 测试代码 ==============

if __name__ == "__main__":
    print("=" * 60)
    print("Train Config Utils - Test")
    print("=" * 60)
    
    # 测试配置解析
    print("\n--- Test parse_value ---")
    test_cases = [
        "true", "False", "123", "3.14", "hello", "1, 2, 3", "none"
    ]
    for case in test_cases:
        result = parse_value(case)
        print(f"  {case:15s} -> {result} ({type(result).__name__})")
    
    # 测试文本配置解析
    print("\n--- Test parse_txt_config ---")
    sample_config = """
# 训练配置
name = PitchNetEnhanced
epochs = 100

[data]
root_dir = TrainingData
subdirs = SingleSanity, NoiseDataset

[training]
batch_size = 64
lr = 0.001
use_attention = true
"""
    parsed = parse_txt_config(sample_config)
    for key, value in parsed.items():
        if isinstance(value, dict):
            print(f"  [{key}]")
            for k, v in value.items():
                print(f"    {k} = {v}")
        else:
            print(f"  {key} = {value}")
    
    # 测试模型列表
    print("\n--- Test available models ---")
    try:
        models = list_available_models()
        print(f"  Available models: {models}")
    except Exception as e:
        print(f"  Error loading models: {e}")
    
    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)

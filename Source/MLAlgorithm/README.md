# MLAlgorithm - Quick Validation

C++集成验证模块，用于测试ONNX Runtime与SuperPitchMonitor的集成。

## 目录结构

```
Source/MLAlgorithm/
├── MLPitchDetector.h/cpp   # 主要检测器类
├── RingBuffer.h            # 线程安全环形缓冲区
├── test_mlpitch.cpp        # 独立测试程序
├── CMakeLists.txt          # 构建配置
└── README.md               # 本文档

MLModel/
├── export_stub_model.py    # 导出ONNX模型脚本
├── pitchnet_stub_v1.onnx   # 生成的模型文件
└── ...
```

## 快速开始

### 0. 创建Conda环境（推荐）

```bash
# 创建专用虚拟环境
cd MLModel/
conda env create -f environment.yml

# 激活环境
conda activate spm_ml

# 后续所有Python操作都在此环境中进行
```

如需GPU支持，修改 `environment.yml` 中的pytorch行：
```yaml
- pytorch::pytorch>=2.0.0
- pytorch::pytorch-cuda=11.8  # 添加CUDA支持
```

### 1. 生成ONNX模型（需要PyTorch）

```bash
# 确保在conda环境中
conda activate spm_ml

cd MLModel/
python export_stub_model.py
```

这会生成 `pitchnet_stub_v1.onnx` 模型文件。

### 2. 下载ONNX Runtime

从 [GitHub Releases](https://github.com/microsoft/onnxruntime/releases) 下载对应平台的预编译库：

- macOS: `onnxruntime-osx-universal2-1.x.x.tgz`
- Windows: `onnxruntime-win-x64-1.x.x.zip`
- Linux: `onnxruntime-linux-x64-1.x.x.tgz`

解压到合适位置，例如：
```bash
# macOS/Linux
sudo mkdir -p /usr/local/onnxruntime
sudo tar -xzf onnxruntime-osx-universal2-1.16.3.tgz -C /usr/local/onnxruntime --strip-components=1

# Windows
# 解压到 C:\Program Files\onnxruntime
```

### 3. 构建测试程序

```bash
cd Source/MLAlgorithm/
mkdir build && cd build

# 如果ONNX Runtime在标准位置
cmake ..

# 或者指定路径
cmake .. -DONNXRUNTIME_ROOT=/path/to/onnxruntime

make -j
```

### 4. 运行测试

```bash
./test_mlpitch ../../../MLModel/pitchnet_stub_v1.onnx
```

预期输出：
```
==============================================
MLPitchDetector Quick Validation Test
==============================================

Model path: ../../../MLModel/pitchnet_stub_v1.onnx

Initializing detector...
✓ Detector initialized
Model info:
Input: [1, 1, 4096]
Output: [1, 2048, 2]
Last inference: X.XX ms

==============================================
Test 1: Single sine wave (440 Hz)
==============================================
Running inference...
✓ Inference completed in X.XX ms

Top 5 detections (random weights, values are meaningless):
  1. Bin XXX (XXX Hz): conf=X.XX, energy=X.XX
  ...

==============================================
All tests passed!
==============================================
```

## 集成到主项目

在SuperPitchMonitor的 `CMakeLists.txt` 中添加：

```cmake
# 添加子目录
add_subdirectory(Source/MLAlgorithm)

# 链接到主目标
target_link_libraries(SuperPitchMonitor PRIVATE mlalgorithm)
```

## 下一步

1. **验证通过后**，开始训练真实模型（见 `Docs/11_MachineLearningSolution/TODO_Implementation_Plan.md` Phase 2-4）
2. **替换模型文件**：将训练好的模型导出为ONNX，替换stub模型
3. **无需修改C++代码**：只要输入输出维度一致，直接替换 `.onnx` 文件即可

## 故障排除

### ONNX Runtime not found
```
CMake Error: ONNX Runtime not found
```
确保设置了正确的路径：
```bash
cmake .. -DONNXRUNTIME_ROOT=/path/to/onnxruntime
# 或
export ONNXRUNTIME_ROOT=/path/to/onnxruntime
```

### 模型加载失败
```
ERROR: Failed to initialize detector
```
检查模型文件路径是否正确，以及模型是否成功导出。

### 推理失败
检查ONNX Runtime版本与导出时的opset版本是否兼容（当前使用opset 17）。

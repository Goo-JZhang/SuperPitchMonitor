# macOS 构建指南

## 前提条件

1. **Xcode Command Line Tools**
   ```bash
   xcode-select --install
   ```

2. **CMake** (3.15+)
   ```bash
   brew install cmake
   ```

3. **Conda环境** (用于生成ONNX模型)
   ```bash
   conda activate spm_ml
   ```

## 构建步骤

### 1. 生成ONNX Stub模型

```bash
cd /Users/zhangjinlin/Project/SuperPitchMonitor/MLModel/
conda activate spm_ml
python export_stub_model.py
```

预期输出：
```
模型参数量: XXXXXX
输入形状: torch.Size([1, 1, 4096])
输出形状: torch.Size([1, 2048, 2])
模型已导出: .../pitchnet_stub_v1.onnx
ONNX模型验证通过
```

### 2. 创建构建目录

```bash
cd /Users/zhangjinlin/Project/SuperPitchMonitor/
mkdir -p build-macos
cd build-macos
```

### 3. 运行CMake

```bash
cmake .. -DCMAKE_BUILD_TYPE=Release
```

首次运行会自动下载ONNX Runtime（约20MB）：
```
-- Downloading ONNX Runtime 1.16.3...
-- URL: https://github.com/microsoft/onnxruntime/releases/download/v1.16.3/onnxruntime-osx-arm64-1.16.3.tgz
-- ONNX Runtime extracted to .../ThirdParty/onnxruntime-bin/1.16.3/
-- Downloading CoreML provider factory header...
-- ONNX Runtime configured successfully
-- Configuring done
-- Generating done
```

### 4. 构建

```bash
make -j$(sysctl -n hw.ncpu)
```

或指定线程数：
```bash
make -j8
```

### 5. 运行

```bash
# 从构建目录运行
cd /Users/zhangjinlin/Project/SuperPitchMonitor/
./SuperPitchMonitor.app/Contents/MacOS/SuperPitchMonitor

# 或双击运行
open SuperPitchMonitor.app
```

## 常见问题

### 问题1: ONNX Runtime下载失败

**现象**：
```
CMake Error: Failed to download ONNX Runtime
```

**解决**：
```bash
# 手动下载并放置
mkdir -p ThirdParty/onnxruntime-bin/1.16.3
curl -L -o onnxruntime.tgz \
  "https://github.com/microsoft/onnxruntime/releases/download/v1.16.3/onnxruntime-osx-arm64-1.16.3.tgz"
tar -xzf onnxruntime.tgz -C ThirdParty/onnxruntime-bin/1.16.3/ --strip-components=1
rm onnxruntime.tgz

# 重新cmake
cd build-macos && cmake ..
```

### 问题2: 模型文件未找到

**现象**：
```
ML: Error - Model file not found
```

**解决**：
```bash
# 确保模型文件存在
ls MLModel/pitchnet_stub_v1.onnx

# 如果不存在，重新生成
cd MLModel && python export_stub_model.py
```

### 问题3: 编译错误 (JUCE相关)

**现象**：
```
error: 'setScrollOnDragEnabled' is deprecated
```

**解决**：已修复，如果仍有警告可忽略。

### 问题4: 找不到ONNX Runtime库

**现象**：
```
dyld: Library not loaded: @rpath/libonnxruntime.dylib
```

**解决**：
```bash
# 检查库文件是否存在
ls SuperPitchMonitor.app/Contents/MacOS/libonnxruntime.dylib

# 如果不存在，手动复制
cp ThirdParty/onnxruntime-bin/1.16.3/onnxruntime-osx-arm64-1.16.3/lib/libonnxruntime.dylib \
   SuperPitchMonitor.app/Contents/MacOS/
```

## 清理重建

如果遇到奇怪的问题，清理后重建：

```bash
cd /Users/zhangjinlin/Project/SuperPitchMonitor/
rm -rf build-macos
mkdir build-macos && cd build-macos
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j8
```

## 验证构建成功

运行后查看日志输出：
```
ML: Available execution providers:
  - CPUExecutionProvider
ML: Using CPU execution provider
ML: MLPitchDetector initialized (CPU)
```

看到类似输出表示ML模块加载成功。

## 生产构建

对于发布版本：
```bash
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j8

# 签名（如果需要分发）
codesign --force --deep --sign - SuperPitchMonitor.app
```

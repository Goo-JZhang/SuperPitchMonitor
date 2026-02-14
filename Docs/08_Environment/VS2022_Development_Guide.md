# Visual Studio 2022 Community 开发调试指南

## 🚀 快速开始

### 方式 1: 直接打开现有解决方案（推荐）

```powershell
# 打开 Visual Studio 2022
cd C:\SuperPitchMonitor\build-windows
start SuperPitchMonitor.sln
```

### 方式 2: 作为 CMake 项目打开（更灵活）

```powershell
# 在项目根目录打开 VS2022
cd C:\SuperPitchMonitor
# 打开 Visual Studio，选择 "打开本地文件夹"
```

---

## 📁 项目结构

### 解决方案文件位置
```
C:\SuperPitchMonitor\build-windows\SuperPitchMonitor.sln
```

### 输出目录
```
Debug:   C:\SuperPitchMonitor\build-windows\SuperPitchMonitor_artefacts\Debug
Release: C:\SuperPitchMonitor\build-windows\SuperPitchMonitor_artefacts\Release
```

---

## ⚙️ 调试配置

### 已配置的启动目标

在 `.vs\launch.vs.json` 中已配置：

| 配置名称 | 目标 | 工作目录 |
|---------|------|---------|
| SuperPitchMonitor (Debug) | Debug 版本可执行文件 | 项目根目录 |
| SuperPitchMonitor (Release) | Release 版本可执行文件 | 项目根目录 |

### 设置启动项目

**方式 1 - 解决方案方式:**
1. 在解决方案资源管理器中右键点击 `SuperPitchMonitor` 项目
2. 选择 "设为启动项目"

**方式 2 - CMake 方式:**
1. 选择菜单 "CMake" → "更改 CMake 设置"
2. 在 `CMakeSettings.json` 中选择 `x64-Debug`
3. 从下拉框选择启动目标

---

## 🔨 构建配置

### 预定义配置

| 配置 | 平台 | 用途 |
|------|------|------|
| Debug | x64 | 开发调试（推荐） |
| Release | x64 | 性能测试 |

### 切换配置

```
工具栏: Debug → Release
或
生成 → 配置管理器
```

---

## 🐛 调试技巧

### 1. 设置断点

在以下关键位置设置断点：

```cpp
// MainComponent.cpp - 启动流程
MainComponent::MainComponent()          // 构造函数
setupAudio()                           // 音频初始化
handlePermissionDenied()               // 权限处理

// AudioEngine.cpp - 音频处理
AudioEngine::processAudioBlock()       // 音频处理线程

// SpectrumAnalyzer.cpp - 频谱分析
SpectrumAnalyzer::process()            // FFT 处理

// PolyphonicDetector.cpp - 音高检测
PolyphonicDetector::detect()           // 检测算法
```

### 2. 条件断点

在音频处理循环中，可以设置条件断点避免频繁中断：

```cpp
// 只在检测到音高时中断
if (!pitches.empty()) {  // 设置条件断点
    pitchCallback_(pitches);
}
```

### 3. 调试输出

使用 Visual Studio 的 "输出" 窗口查看 DBG 输出：

```cpp
DBG("[DEBUG] Current frequency: " << freq);  // 输出到 VS 输出窗口
```

### 4. 性能分析

使用 Visual Studio 性能分析器：

```
调试 → 性能分析器 (Alt+F2)
选择: CPU 使用率 / 内存使用率
```

---

## 📝 常用操作

### 重新生成项目

```powershell
# 如果需要重新生成 CMake 项目
rd /s /q build-windows
mkdir build-windows
cd build-windows
cmake .. -G "Visual Studio 17 2022" -A x64
```

### 增量构建

在 Visual Studio 中：

```
生成 → 生成解决方案 (Ctrl+Shift+B)
```

### 清理项目

```
生成 → 清理解决方案
```

---

## 🎯 开发工作流

### 日常开发循环

1. **修改代码**
   ```cpp
   // 编辑 Source/ 下的文件
   ```

2. **构建项目**
   ```
   Ctrl+Shift+B 或 F5(自动构建并调试)
   ```

3. **启动调试**
   ```
   F5 - 启动调试
   Ctrl+F5 - 启动不调试
   ```

4. **测试功能**
   - 点击 "Start" 按钮
   - 观察频谱显示
   - 验证音高检测

### 调试特定功能

#### 测试音频模拟器
```cpp
// 设置断点
AudioSimulator::generateTestSignal()   // 生成测试信号
AudioSimulator::processAudioBlock()    // 处理音频块
```

#### 调试频谱分析
```cpp
// 设置断点
SpectrumAnalyzer::performFFT()         // FFT 变换
SpectrumAnalyzer::extractMagnitudes()  // 提取幅度
```

#### 调试音高检测
```cpp
// 设置断点
PolyphonicDetector::findPeaks()        // 峰值检测
PolyphonicDetector::analyzeHarmonics() // 谐波分析
```

---

## 🔧 高级配置

### 修改调试命令参数

编辑 `.vs\launch.vs.json`:

```json
{
  "configurations": [
    {
      "name": "SuperPitchMonitor (Custom)",
      "args": ["--debug-mode", "--sample-rate", "48000"],
      "currentDir": "${projectDir}",
      "env": {
        "MY_CUSTOM_VAR": "value"
      }
    }
  ]
}
```

### 附加到进程

如果应用已经运行，可以附加调试器：

```
调试 → 附加到进程 (Ctrl+Alt+P)
选择: SuperPitchMonitor.exe
```

---

## 🐛 常见问题

### 问题 1: "无法找到 SuperPitchMonitor.exe"

**解决**: 确保先构建项目
```
生成 → 生成解决方案 (Ctrl+Shift+B)
```

### 问题 2: 断点不生效

**可能原因**:
- 代码已修改但未重新构建
- 优化导致代码被内联（Release 模式）

**解决**:
- 确保在 Debug 模式下调试
- 重新构建项目

### 问题 3: CMake 项目加载失败

**解决**:
1. 删除 `CMakeCache.txt`
2. 删除 `.vs/` 目录
3. 重新打开项目

```powershell
cd C:\SuperPitchMonitor
rm CMakeCache.txt
rm -r .vs
```

### 问题 4: 找不到 JUCE 模块

**解决**: 确保 JUCE 子模块已初始化

```powershell
git submodule update --init --recursive
```

---

## 📊 性能优化建议

### Debug 模式
- 禁用优化，方便调试
- 启用所有断言和检查
- 包含完整的调试符号

### Release 模式测试
- 测试实际性能表现
- 验证 Release 版本没有崩溃
- 对比 Debug 和 Release 的差异

---

## 🔗 相关文件

| 文件 | 用途 |
|------|------|
| `build-windows/SuperPitchMonitor.sln` | 解决方案文件 |
| `CMakeSettings.json` | CMake 项目配置 |
| `.vs/launch.vs.json` | 调试启动配置 |
| `build-windows/SuperPitchMonitor_artefacts/Debug/` | Debug 输出 |
| `build-windows/SuperPitchMonitor_artefacts/Release/` | Release 输出 |

---

## ✅ 验证清单

首次配置后，验证以下功能：

- [ ] 解决方案正常打开
- [ ] 项目成功编译 (Debug)
- [ ] 项目成功编译 (Release)
- [ ] F5 启动调试正常
- [ ] 断点可以命中
- [ ] 调试输出窗口可见
- [ ] 应用程序正常显示
- [ ] Debug 按钮可见（Debug 模式）
- [ ] 音频模拟器可用

---

**现在可以使用 Visual Studio 2022 进行高效的 Windows 桌面端开发了！**

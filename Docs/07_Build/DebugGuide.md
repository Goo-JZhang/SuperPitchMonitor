# SuperPitchMonitor 调试指南

## 概述

为了方便开发和测试，项目内置了 **音频模拟器 (AudioSimulator)** 功能。这使得你可以在没有真实音频设备或麦克风权限的情况下测试 UI 和算法。

## 调试功能

### 内置测试信号

1. **Sine Wave (440Hz A4)** - 标准正弦波，用于基础测试
2. **C Major Chord (C4-E4-G4)** - C大调和弦，用于多音高检测测试
3. **Frequency Sweep** - 频率扫描，用于测试频谱显示
4. **White Noise** - 白噪声，用于测试降噪和滤波
5. **Pink Noise** - 粉噪声，更接近真实音频特性

### 调试控制面板

在 DEBUG 模式下，界面上会显示 **Debug** 按钮，点击打开调试控制面板：

- **模式选择**: 切换真实设备 / 音频文件 / 测试信号
- **播放控制**: Play / Pause / Stop / Loop
- **进度条**: 查看和跳转播放位置
- **播放速度**: 0.25x - 2.0x 调节
- **文件选择**: 加载本地音频文件测试

## 如何使用

### 桌面端开发 (Windows/macOS/Linux)

桌面端默认自动启用调试模式：

```bash
# 创建 Debug 构建
mkdir build-debug
cd build-debug
cmake .. -DCMAKE_BUILD_TYPE=Debug
cmake --build . --parallel

# 运行
./SuperPitchMonitor
```

运行后：
1. 界面会自动加载 **C大调和弦** 测试信号
2. 点击 **Start** 按钮开始播放
3. 频谱显示器会显示三个峰值 (C4, E4, G4)
4. 音高显示器会显示检测到的三个音

### Android 模拟器

在 Android Studio 模拟器中：

1. 构建 Debug APK
2. 启动应用
3. 如果没有麦克风权限，会自动切换到调试模式
4. 点击 **Debug** 按钮打开控制面板
5. 选择测试信号并开始播放

### 切换模式

在代码中切换工作模式：

```cpp
// 切换到真实设备模式
audioEngine->setMode(AudioEngine::Mode::RealDevice);

// 切换到模拟模式
audioEngine->setMode(AudioEngine::Mode::Simulated);
```

### 加载自定义音频

```cpp
// 加载本地音频文件
File audioFile("/path/to/test.wav");
audioSimulator.loadAudioFile(audioFile);
audioSimulator.start();
```

支持的格式: WAV, MP3, AIFF, FLAC

## 调试技巧

### 1. 测试频谱显示

使用 **Frequency Sweep** 信号，观察频谱显示器是否平滑跟随频率变化。

### 2. 测试多音高检测

使用 **C Major Chord** 信号，验证是否能正确识别三个音高。

### 3. 测试调音器

使用 **Sine Wave** 信号，验证：
- 440Hz 显示 "In Tune"
- 445Hz 显示 "Sharp" (+20 cents)
- 435Hz 显示 "Flat" (-20 cents)

### 4. 测试响应速度

调节播放速度到 2.0x，观察 UI 是否能平滑跟随快速变化的音频。

### 5. 性能分析

在 Debug 构建中，代码会自动输出性能分析信息：

```
[PROFILE] AudioProcessing: 2.35 ms
[PROFILE] SpectrumAnalysis: 1.12 ms
[PROFILE] PitchDetection: 0.89 ms
```

## Release 构建

Release 构建会自动禁用调试功能：

```bash
mkdir build-release
cd build-release
cmake .. -DCMAKE_BUILD_TYPE=Release -DENABLE_DEBUG_FEATURES=OFF
cmake --build . --parallel
```

Release 版本中：
- Debug 按钮不会显示
- 调试控制面板不可用
- 只能使用真实音频输入
- 性能优化开启

## 常见问题

### Q: 为什么 Release 版本没有 Debug 按钮？
A: Release 构建默认禁用调试功能，确保生产环境安全性。

### Q: 可以在真机上使用调试功能吗？
A: 可以，使用 Debug APK 即可。但建议仅在开发阶段使用，发布前切换到 Release 模式。

### Q: 调试模式的音频延迟是多少？
A: 约 23ms (60fps 定时器)，与真实音频设备相当。

### Q: 如何添加更多测试信号？
A: 在 `AudioSimulator::generateTestSignal()` 中添加新的信号生成函数。

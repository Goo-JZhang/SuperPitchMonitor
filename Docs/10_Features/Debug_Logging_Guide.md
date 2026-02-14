# 调试日志指南

## 添加的日志点

### MainComponent.cpp

| 位置 | 日志内容 | 说明 |
|------|---------|------|
| 构造函数 | `Constructor start/complete` | 组件生命周期 |
| setupAudio | `setupAudio start/complete` | 音频设置流程 |
| setupAudio | `Registering callbacks` | 回调注册 |
| setupAudio | `Audio init error: xxx` | 初始化错误 |
| setupAudio | `Audio initialized successfully` | 初始化成功 |
| 权限回调 | `Permission callback: granted=true/false` | 权限结果 |
| buttonClicked | `Starting/Stopping audio engine` | 按钮点击 |
| onSpectrumData | `onSpectrumData: X bins` | 频谱数据 |
| onPitchDetected | `onPitchDetected: X pitches` | 音高检测 |

### AudioEngine.cpp

| 位置 | 日志内容 | 说明 |
|------|---------|------|
| start() | `Starting in mode=Simulated/RealDevice` | 启动模式 |
| start() | `ERROR: Simulator mode but simulator is null!` | 错误：模拟器为空 |
| start() | `Real device started` | 真实设备启动 |
| start() | `Simulator started` | 模拟器启动 |
| processAudioBlock | `Processing audio block #X` | 处理音频块 |
| processAudioBlock | `Spectrum data: X bins` | 频谱分析结果 |
| processAudioBlock | `Detected X pitches` | 音高检测数量 |
| processAudioBlock | `Spectrum/Pitch callback triggered` | 回调触发 |

### DebugControlPanel.cpp

| 位置 | 日志内容 | 说明 |
|------|---------|------|
| 音频回调 | `Audio callback: X samples` | 音频回调数据 |
| 音频回调 | `ERROR: AudioEngine is null in callback!` | 错误：引擎为空 |
| 电平回调 | `Level callback: X` | 电平值 |

### AudioSimulator.cpp

| 位置 | 日志内容 | 说明 |
|------|---------|------|
| start() | `Starting playback, buffer=X samples` | 开始播放 |
| start() | `Cannot start: audio buffer is empty!` | 错误：缓冲区为空 |
| timerCallback | `Timer callback #X` | 定时器回调 |
| processAudioBlock | `Processed X samples, RMS=X` | 处理样本 |
| processAudioBlock | `Audio/Level callback triggered` | 回调触发 |
| processAudioBlock | `WARNING: No audio callback set!` | 警告：未设置回调 |

## 查看日志

在 Visual Studio 中，日志输出到 **输出窗口** (Output Window)：

```
视图 → 输出  (Ctrl+Alt+O)
```

或在 **即时窗口** (Immediate Window) 中查看：

```
视图 → 即时  (Ctrl+Alt+I)
```

## 日志过滤

在输出窗口中，可以搜索特定关键字：
- `[MainComponent]` - 主组件日志
- `[AudioEngine]` - 音频引擎日志
- `[AudioSimulator]` - 模拟器日志
- `[DebugControlPanel]` - 调试面板日志
- `ERROR` / `WARNING` - 错误和警告

## 常见问题诊断

### 问题：点击 Start 后显示 "Waiting for audio"

**查看日志：**
1. 检查 `[MainComponent] Starting audio engine` 是否出现
2. 检查 `[AudioEngine] Starting in mode=Simulated` 是否出现
3. 检查 `[AudioSimulator] Starting playback` 是否出现
4. 检查 `[AudioEngine] Processing audio block` 是否出现

**可能的原因：**
- 如果缺少 `[AudioSimulator] Starting playback` → 模拟器缓冲区为空
- 如果缺少 `[AudioEngine] Processing audio block` → 回调未正确设置
- 如果出现 `ERROR: AudioEngine is null in callback!` → 引擎指针为空

### 问题：没有频谱显示

**查看日志：**
- 检查 `[AudioEngine] Spectrum callback triggered` 是否出现
- 检查 `[MainComponent] onSpectrumData` 是否出现

### 问题：没有音高检测

**查看日志：**
- 检查 `[AudioEngine] Detected X pitches` - X 应该 > 0
- 检查 `[AudioEngine] Pitch callback triggered` 是否出现

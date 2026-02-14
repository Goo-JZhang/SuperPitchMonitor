# 重新构建说明

## 问题
程序正在运行，导致无法写入可执行文件。

## 解决步骤

### 1. 关闭正在运行的程序

确保 `SuperPitchMonitor.exe` 没有在运行：
- 关闭 Visual Studio 中的调试会话 (Shift+F5)
- 或关闭独立运行的程序窗口

### 2. 重新构建

在 Visual Studio 中：
```
生成 → 重新生成解决方案 (Ctrl+Alt+F7)
```

或在命令行：
```powershell
cd C:\SuperPitchMonitor\build-windows
cmake --build . --config Debug
```

## 新添加的功能

### 调试日志
在关键路径添加了日志，帮助诊断 "Waiting for audio" 问题：

1. **MainComponent** - 生命周期和按钮点击
2. **AudioEngine** - 启动和处理流程
3. **AudioSimulator** - 模拟器状态
4. **DebugControlPanel** - 回调触发

### 关键修复
- 将 `AudioEngine::processAudioBlock` 声明为 public，以便 `DebugControlPanel` 可以调用
- 在 `DebugControlPanel` 中实际调用了 `audioEngine_->processAudioBlock(buffer)`（之前是 TODO）

## 测试步骤

1. 重新构建项目
2. 按 F5 启动调试
3. 打开 **输出窗口** (Ctrl+Alt+O)
4. 点击 **Start** 按钮
5. 观察日志输出：
   - `[MainComponent] Starting audio engine`
   - `[AudioEngine] Starting in mode=Simulated`
   - `[AudioSimulator] Starting playback`
   - `[AudioEngine] Processing audio block`
   - `[MainComponent] onSpectrumData`

## 如果仍然有问题

根据日志判断：
- 如果没有 `[AudioSimulator] Starting playback` → 检查模拟器缓冲区
- 如果没有 `[AudioEngine] Processing audio block` → 检查回调设置
- 如果出现 `ERROR: AudioEngine is null` → 检查引擎指针

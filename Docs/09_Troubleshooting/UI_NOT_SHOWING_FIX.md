# UI 和频谱不显示问题修复

## 问题描述
重新编译后：
1. UI 界面和原来一样（没有显示新的输入源选择器）
2. 点击 Start 后仍然没有显示频谱

## 根本原因

### 1. InputSourceSelector 未添加到 MainComponent
创建了 `InputSourceSelector` 组件，但没有在 `MainComponent` 中实例化和显示。

### 2. 音频输入源未正确设置
`MainComponent` 没有使用新的 `AudioInputSource` 架构，而是仍然依赖旧的调试模拟器。

### 3. FileAudioInput 缺少 TestSignal 枚举
`FileAudioInput` 类声明了 `generateTestSignal()` 方法，但没有定义 `TestSignal` 枚举和实现。

## 修复内容

### 修复 1: 在 MainComponent 中添加 InputSourceSelector

**文件**: `Source/MainComponent.h`

```cpp
// 添加包含
#include "Audio/FileAudioInput.h"
#include "UI/InputSourceSelector.h"

// 添加成员
std::unique_ptr<InputSourceSelector> inputSelector_;
std::shared_ptr<AudioInputSource> currentInputSource_;
```

### 修复 2: 修改 setupAudio() 使用新的音频输入源

**文件**: `Source/MainComponent.cpp`

```cpp
void MainComponent::setupAudio()
{
    audioEngine_ = std::make_unique<AudioEngine>();
    
    // 设置回调...
    
    // 创建默认文件音频输入
    auto fileInput = std::make_shared<FileAudioInput>();
    
    // 尝试加载测试文件，如果不存在则生成默认信号
    auto testFiles = FileAudioInput::getAvailableTestFiles();
    if (!testFiles.isEmpty())
    {
        fileInput->loadTestFile(testFiles[0]);
    }
    else
    {
        fileInput->generateTestSignal(FileAudioInput::TestSignal::Chord, 5.0);
    }
    
    // 设置到音频引擎
    currentInputSource_ = fileInput;
    audioEngine_->setInputSource(currentInputSource_);
}
```

### 修复 3: 在 setupUI() 中添加输入源选择器

```cpp
void MainComponent::setupUI()
{
    // ... 其他 UI 组件
    
    // 添加输入源选择器
    inputSelector_ = std::make_unique<InputSourceSelector>();
    addAndMakeVisible(inputSelector_.get());
}
```

### 修复 4: 在 resized() 中添加输入源选择器布局

```cpp
void MainComponent::resized()
{
    // 输入源选择器（底部左侧）
    auto inputSelectorArea = bounds.removeFromBottom(120);
    inputSelector_->setBounds(inputSelectorArea.removeFromLeft(400).reduced(5));
    
    // ... 其他布局
}
```

### 修复 5: 添加 TestSignal 枚举和实现

**文件**: `Source/Audio/FileAudioInput.h`

```cpp
// 添加枚举
enum class TestSignal {
    SineWave,
    Chord,
    Sweep,
    WhiteNoise,
    PinkNoise
};

// 添加方法声明
bool generateTestSignal(TestSignal type, double durationSeconds = 5.0);

// 添加私有生成方法
void generateSineWave(double duration, float frequency);
void generateChord(double duration);
void generateSweep(double duration);
void generateWhiteNoise(double duration);
void generatePinkNoise(double duration);
```

**文件**: `Source/Audio/FileAudioInput.cpp`

实现了所有测试信号生成方法。

## 验证步骤

1. **重新构建项目**
   ```powershell
   cd C:\SuperPitchMonitor\build-windows
   cmake --build . --config Debug
   ```

2. **运行程序**
   ```powershell
   .\SuperPitchMonitor_artefacts\Debug\SuperPitchMonitor.exe
   ```

3. **验证界面**
   - 应该看到左下角出现输入源选择器
   - 状态栏应该显示 "Loaded: xxx" 或 "Using default test signal"

4. **点击 Start**
   - 应该听到/看到音频正在播放
   - 频谱显示应该开始显示频谱
   - 音高检测应该显示检测到的音高

## 预期行为

### 如果没有测试文件
- 自动生成 C Major 和弦测试信号
- 状态栏显示: "No test files - Using default signal - Press Start"

### 如果有测试文件
- 自动加载第一个测试文件
- 状态栏显示: "Loaded: xxx.wav - Press Start"

### 点击 Start 后
- 按钮变为 "Stop"（红色）
- 状态栏显示: "Running (Simulated)"
- 频谱显示开始更新
- 音高检测显示检测到的音高

## 调试

如果仍然有问题，查看输出窗口的日志：
```
[MainComponent] setupAudio start
[MainComponent] Creating default file audio input
[MainComponent] Loaded test file: xxx.wav
[MainComponent] setupAudio complete
[MainComponent] Starting audio engine
[AudioEngine] Starting in mode=Simulated
[AudioEngine] Input source started: File: xxx.wav
[AudioEngine] Processing audio block #1
```

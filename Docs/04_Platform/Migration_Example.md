# 迁移示例：使用 PlatformUtils

## 当前代码 → 改进代码

### 1. MainComponent.cpp 权限请求

**当前代码:**
```cpp
// MainComponent.cpp:14-35
#if JUCE_ANDROID
    juce::RuntimePermissions::request(
        juce::RuntimePermissions::recordAudio,
        [this](bool granted)
        {
            if (granted)
            {
                setupAudio();
                statusLabel_.setText("Ready", juce::dontSendNotification);
            }
            else
            {
                statusLabel_.setText("Microphone permission denied", juce::dontSendNotification);
               #if defined(DEBUG) || defined(_DEBUG)
                audioEngine_->setMode(AudioEngine::Mode::Simulated);
                debugButton_.setButtonText("Debug [ON]");
                statusLabel_.setText("Debug mode - using simulated input", juce::dontSendNotification);
               #endif
            }
        }
    );
#endif
```

**改进代码:**
```cpp
// MainComponent.cpp
#include "Utils/PlatformUtils.h"

// 在构造函数中
Platform::requestPermission(Platform::Permission::AudioInput,
    [this](bool granted)
    {
        if (granted)
        {
            setupAudio();
            statusLabel_.setText("Ready", juce::dontSendNotification);
        }
        else
        {
            statusLabel_.setText("Microphone permission denied", juce::dontSendNotification);
            
            if (Platform::isSimulatorAllowed())
            {
                audioEngine_->setMode(AudioEngine::Mode::Simulated);
                debugButton_.setButtonText("Debug [ON]");
                statusLabel_.setText("Debug mode - using simulated input", juce::dontSendNotification);
            }
        }
    }
);
```

**优势:**
- Windows 版本也能模拟权限拒绝流程 (用于测试)
- 代码更简洁，无需条件编译
- 集中管理权限逻辑

---

### 2. Main.cpp 窗口全屏

**当前代码:**
```cpp
// Main.cpp:84-91
#if JUCE_ANDROID
    setFullScreen(true);
#else
    centreWithSize(800, 1200);
    setResizable(true, true);
#endif
```

**改进代码:**
```cpp
// Main.cpp
#include "Utils/PlatformUtils.h"

// 在 MainWindow 构造函数中
Platform::setFullscreen(true);  // Android 自动全屏，桌面可配置
```

**或者保留当前方式** (这是合理的平台差异)

---

### 3. Debug 模式判断

**当前代码:**
```cpp
// MainComponent.cpp:39-43
#if !JUCE_ANDROID && (defined(DEBUG) || defined(_DEBUG))
   audioEngine_->setMode(AudioEngine::Mode::Simulated);
   debugButton_.setButtonText("Debug [ON]");
   statusLabel_.setText("Debug mode - Press Start to play test signal", juce::dontSendNotification);
#endif
```

**改进代码:**
```cpp
// MainComponent.cpp
if (Platform::isSimulatorAllowed())
{
    audioEngine_->setMode(AudioEngine::Mode::Simulated);
    debugButton_.setButtonText("Debug [ON]");
    statusLabel_.setText("Debug mode - Press Start to play test signal", juce::dontSendNotification);
}
```

**优势:**
- 逻辑更清晰："如果允许模拟器，就使用模拟器"
- 集中管理 Debug/Release 和平台差异

---

### 4. 添加文件存储功能 (未来功能示例)

**不推荐的方式:**
```cpp
// ❌ 不好：平台代码分散
void saveConfig()
{
#if JUCE_WINDOWS
    juce::File configFile("C:\\Users\\...\\config.json");
#elif JUCE_ANDROID
    juce::File configFile("/sdcard/.../config.json");
#endif
    // ...
}
```

**推荐的方式:**
```cpp
// ✅ 正确：使用 PlatformUtils
void saveConfig()
{
    juce::File configDir = Platform::getAppDataDirectory();
    juce::File configFile = configDir.getChildFile("config.json");
    
    // 写入配置...
}
```

---

## 完整的 MainComponent 修改示例

```cpp
// MainComponent.h
#pragma once

#include "Utils/PlatformUtils.h"  // 添加

namespace spm {

class MainComponent : public juce::Component,
                      public juce::Button::Listener
{
    // ... 原有代码 ...
};

}
```

```cpp
// MainComponent.cpp - 构造函数简化
MainComponent::MainComponent()
{
    setupUI();
    setupAudio();
    
    setSize(800, 1200);
    
    // 统一的权限请求 (Windows 直接回调 granted=true)
    Platform::requestPermission(Platform::Permission::AudioInput,
        [this](bool granted)
        {
            if (granted)
            {
                setupAudio();
                statusLabel_.setText("Ready", juce::dontSendNotification);
            }
            else
            {
                handlePermissionDenied();
            }
        }
    );
    
    // 统一的 Debug 模式设置
    if (Platform::isSimulatorAllowed())
    {
        audioEngine_->setMode(AudioEngine::Mode::Simulated);
        debugButton_.setButtonText("Debug [ON]");
        statusLabel_.setText("Debug mode - Press Start to play test signal", juce::dontSendNotification);
    }
}

void MainComponent::handlePermissionDenied()
{
    statusLabel_.setText("Microphone permission denied", juce::dontSendNotification);
    
    if (Platform::isSimulatorAllowed())
    {
        audioEngine_->setMode(AudioEngine::Mode::Simulated);
        debugButton_.setButtonText("Debug [ON]");
        statusLabel_.setText("Debug mode - using simulated input", juce::dontSendNotification);
    }
}
```

---

## 总结

**修改量很小:**
- 只需修改 3 个地方
- 删除 3 个 `#if` 条件编译
- 添加对 `Platform` 命名空间的调用

**收益很大:**
- 代码更易读
- 更容易测试
- 为未来功能提供统一的跨平台接口

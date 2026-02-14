# SuperPitchMonitor 跨平台开发策略

## 🎯 核心原则："Write Once, Run Everywhere"

基于 JUCE 框架的优势，我们的目标是**大部分代码完全跨平台**，只在必要时使用条件编译。

---

## 📊 当前兼容性风险评估

### 现状分析（良好 ✅）

| 组件 | 跨平台状态 | 风险等级 |
|------|-----------|---------|
| AudioEngine | ✅ 使用 JUCE Thread | 低 |
| SpectrumAnalyzer | ✅ 纯 JUCE DSP | 低 |
| PolyphonicDetector | ✅ 纯算法代码 | 低 |
| UI Components | ✅ JUCE GUI | 低 |
| 权限处理 | ⚠️ 有条件编译 | 中 |
| 文件路径 | ✅ 未使用 | 低 |

### 当前平台相关代码位置

```cpp
// Source/Main.cpp:84
#if JUCE_ANDROID
    setFullScreen(true);
#else
    centreWithSize(800, 1200);
#endif

// Source/MainComponent.cpp:14
#if JUCE_ANDROID
    juce::RuntimePermissions::request(...)
#endif

// Source/MainComponent.cpp:39
#if !JUCE_ANDROID && (defined(DEBUG) || defined(_DEBUG))
    audioEngine_->setMode(AudioEngine::Mode::Simulated);
#endif
```

**评估**: 现有平台代码都是合理的抽象，风险很低。

---

## 🏗️ 架构设计策略

### 1. 平台抽象层 (PAL)

```cpp
// Utils/PlatformUtils.h
#pragma once

namespace spm {
namespace Platform {

/**
 * 平台抽象接口
 * 所有平台相关功能都通过这里暴露
 */

// 应用生命周期
void initializePlatform();
void shutdownPlatform();

// 权限管理 (Android 需要运行时权限)
enum class Permission {
    AudioInput,
    Storage,
    Camera
};

void requestPermission(Permission permission, 
                       std::function<void(bool granted)> callback);
bool hasPermission(Permission permission);

// 文件系统路径
juce::File getAppDataDirectory();
juce::File getCacheDirectory();
juce::File getDocumentsDirectory();

// 显示设置
void setFullscreen(bool fullscreen);
bool isFullscreen();
float getDisplayScale();  // DPI scale

// 性能模式
void setLowPowerMode(bool enabled);
bool isLowPowerModeEnabled();

// 调试功能
bool isDebugBuild();
bool isSimulatorAllowed();  // 桌面端允许，Release 移动端不允许

} // namespace Platform
} // namespace spm
```

```cpp
// Utils/PlatformUtils.cpp - 通用实现
#include "PlatformUtils.h"

namespace spm {
namespace Platform {

bool isDebugBuild() {
#if defined(DEBUG) || defined(_DEBUG)
    return true;
#else
    return false;
#endif
}

bool isSimulatorAllowed() {
#if defined(DEBUG) || defined(_DEBUG)
    return true;  // Debug build 允许
#else
    return false;  // Release build 不允许
#endif
}

} // namespace Platform
} // namespace spm
```

```cpp
// Utils/PlatformUtils_Android.cpp - Android 特定实现
#if JUCE_ANDROID

#include "PlatformUtils.h"

namespace spm {
namespace Platform {

void requestPermission(Permission permission, 
                       std::function<void(bool granted)> callback) {
    juce::RuntimePermissions::request(
        juce::RuntimePermissions::recordAudio,
        [callback](bool granted) { callback(granted); }
    );
}

juce::File getAppDataDirectory() {
    return juce::File::getSpecialLocation(
        juce::File::userApplicationDataDirectory
    );
}

void setFullscreen(bool fullscreen) {
    // Android 全屏处理
    if (auto* desktop = juce::Desktop::getInstance()) {
        if (auto* window = desktop->getComponent(0)) {
            // 使用 JUCE 的 Android 全屏 API
        }
    }
}

} // namespace Platform
} // namespace spm

#endif // JUCE_ANDROID
```

```cpp
// Utils/PlatformUtils_Windows.cpp - Windows 特定实现
#if JUCE_WINDOWS

#include "PlatformUtils.h"

namespace spm {
namespace Platform {

void requestPermission(Permission permission, 
                       std::function<void(bool granted)> callback) {
    // Windows 不需要运行时权限，直接回调 true
    callback(true);
}

juce::File getAppDataDirectory() {
    return juce::File::getSpecialLocation(
        juce::File::userApplicationDataDirectory
    );
}

void setFullscreen(bool fullscreen) {
    // Windows 全屏处理
}

} // namespace Platform
} // namespace spm

#endif // JUCE_WINDOWS
```

---

## 📝 编码规范

### DO ✅ (推荐)

```cpp
// 1. 使用 JUCE 的跨平台抽象
juce::File configFile = juce::File::getSpecialLocation(
    juce::File::userApplicationDataDirectory
).getChildFile("config.json");

// 2. 使用 JUCE 的线程
class Worker : private juce::Thread {
    // 而不是 std::thread
};

// 3. 使用 JUCE 的同步原语
juce::CriticalSection dataLock;
juce::WaitableEvent signal;

// 4. 使用平台抽象层
Platform::requestPermission(Platform::Permission::AudioInput, 
    [](bool granted) { /* ... */ });

// 5. 条件编译集中在平台抽象层
#if JUCE_ANDROID
    // Android specific
#elif JUCE_WINDOWS
    // Windows specific
#else
    // Default
#endif

// 6. 路径使用 JUCE 的 File 类
juce::File path = baseDir.getChildFile("subdir")
                          .getChildFile("file.txt");
// 而不是字符串拼接
```

### DON'T ❌ (避免)

```cpp
// 1. 不要使用原生平台 API
#ifdef _WIN32
    CreateFileW(...);  // ❌ Windows only
#elif __ANDROID__
    open(...);         // ❌ POSIX only
#endif

// 2. 不要硬编码路径分隔符
juce::String path = dir + "\\file.txt";  // ❌ Windows only
juce::String path = dir + "/file.txt";   // ❌ POSIX only

// 3. 避免使用 std::filesystem (C++17 但在某些平台不完整)
std::filesystem::path p;  // ⚠️ 可能有问题

// 4. 不要分散平台条件编译
// ❌ 不好：到处都有 #if JUCE_ANDROID
void someFunction() {
    doSomething();
#if JUCE_ANDROID
    androidSpecific();
#endif
    doMore();
}

// 5. 不要假设文件系统结构
juce::File f("C:\\Users\\...");  // ❌ Windows only
juce::File f("/sdcard/...");     // ❌ Android only
```

---

## 🔍 兼容性检查工具

### 1. 静态检查脚本

```powershell
# scripts/check_cross_platform.ps1

$issues = @()

# 检查危险的系统调用
$dangerousPatterns = @(
    @{ Pattern = 'CreateFile|CreateDirectoryW|RegOpenKey'; Desc = 'Windows API' },
    @{ Pattern = 'fopen|fread|fwrite|__android_log_print'; Desc = 'C/POSIX API' },
    @{ Pattern = 'std::filesystem|std::thread|std::mutex'; Desc = 'C++17 features (check compatibility)' },
    @{ Pattern = '\\".*\\"|''.*\\\\.*'''; Desc = 'Hardcoded backslash paths' }
)

Get-ChildItem -Path "Source" -Filter "*.cpp" -Recurse | ForEach-Object {
    $content = Get-Content $_.FullName -Raw
    foreach ($pattern in $dangerousPatterns) {
        if ($content -match $pattern.Pattern) {
            $issues += "$($_.Name): $($pattern.Desc)"
        }
    }
}

if ($issues) {
    Write-Host "Potential cross-platform issues found:" -ForegroundColor Yellow
    $issues | ForEach-Object { Write-Host "  ⚠️ $_" }
} else {
    Write-Host "✅ No obvious cross-platform issues found!" -ForegroundColor Green
}
```

### 2. 头文件保护检查

```cpp
// 每个 .h 文件应该有:
#pragma once
// 或
#ifndef HEADER_NAME_H
#define HEADER_NAME_H
// ...
#endif
```

### 3. CMake 跨平台配置

```cmake
# 平台检测
if(CMAKE_SYSTEM_NAME STREQUAL "Android")
    target_compile_definitions(SuperPitchMonitor PRIVATE JUCE_ANDROID=1)
elseif(CMAKE_SYSTEM_NAME STREQUAL "Windows")
    target_compile_definitions(SuperPitchMonitor PRIVATE JUCE_WINDOWS=1)
elseif(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
    target_compile_definitions(SuperPitchMonitor PRIVATE JUCE_MAC=1)
elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    target_compile_definitions(SuperPitchMonitor PRIVATE JUCE_LINUX=1)
endif()

# 条件编译源文件
target_sources(SuperPitchMonitor PRIVATE
    ${CMAKE_SOURCE_DIR}/Source/Utils/PlatformUtils.cpp
    
    $<$<BOOL:${JUCE_ANDROID}>:
        ${CMAKE_SOURCE_DIR}/Source/Utils/PlatformUtils_Android.cpp
    >
    
    $<$<BOOL:${JUCE_WINDOWS}>:
        ${CMAKE_SOURCE_DIR}/Source/Utils/PlatformUtils_Windows.cpp
    >
)
```

---

## 🚀 分阶段开发流程

### 阶段 1: 核心算法开发 (Windows 桌面版)

```
时长: 60% 开发时间
平台: Windows Desktop
重点: DSP 算法、UI 设计、音频处理
```

**优势**:
- 编译快 (秒级 vs 分钟级)
- 调试方便 (Visual Studio 调试器)
- 文件访问方便
- 模拟器内置 (无需设备)

**交付物**:
- ✅ 频谱分析算法
- ✅ 音高检测算法
- ✅ UI 布局和交互
- ✅ 调试工具

### 阶段 2: 跨平台适配层实现

```
时长: 15% 开发时间
平台: Windows + Android
重点: 平台抽象层、权限、文件系统
```

**任务**:
1. 实现 `PlatformUtils` 各平台版本
2. 迁移现有平台相关代码到 PAL
3. 在 Windows 上测试 PAL (模拟 Android 行为)

### 阶段 3: Android 移植测试

```
时长: 20% 开发时间
平台: Android (真机优先)
重点: 集成测试、性能优化
```

**测试重点**:
- 权限流程
- 音频输入
- 性能 (帧率、延迟)
- 内存使用

### 阶段 4: 并行维护

```
时长: 5% 开发时间 (持续)
平台: All
```

**规则**:
- 新功能先在 Windows 实现
- 每次提交前在 Android 验证
- CI/CD 自动化双平台构建

---

## 🛡️ 预防兼容性问题的实践

### 1. 每日构建验证

```yaml
# .github/workflows/build.yml (示例)
name: Cross Platform Build

on: [push, pull_request]

jobs:
  build-windows:
    runs-on: windows-latest
    steps:
      - uses: actions/checkout@v2
      - name: Build Windows
        run: scripts/build_windows.bat

  build-android:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Build Android
        run: scripts/build_android.sh
```

### 2. 代码审查清单

```markdown
## PR Review Checklist

- [ ] 代码中不包含平台特定的 API 调用
- [ ] 文件路径使用 juce::File 而不是字符串拼接
- [ ] 线程使用 juce::Thread 而不是 std::thread
- [ ] 新增的平台代码已放入 PlatformUtils
- [ ] 在 Windows 和 Android 上都能编译通过
```

### 3. 模拟 Android 行为 (Windows 调试)

```cpp
// 在 Windows 上模拟 Android 的权限请求流程
void MainComponent::setupAudio() {
#if JUCE_WINDOWS && SIMULATE_ANDROID_BEHAVIOR
    // 模拟延迟权限授予
    Timer::callAfterDelay(1000, [this]() {
        onPermissionResult(false);  // 测试拒绝场景
    });
#else
    // 正常流程
    Platform::requestPermission(Permission::AudioInput, 
        [this](bool granted) { onPermissionResult(granted); });
#endif
}
```

---

## 📋 总结

### 你的担心是合理的，但可控 ✅

**好消息**:
1. JUCE 框架已经处理了 95% 的平台差异
2. 当前代码架构良好，没有明显的兼容性问题
3. 核心算法 (DSP) 是纯 C++，天然跨平台

**建议策略**:
1. **80% 时间** Windows 桌面版开发 (算法 + UI)
2. **15% 时间** 每 1-2 周在 Android 上验证一次
3. **5% 时间** 修复发现的兼容性问题

**关键措施**:
- 建立 PlatformUtils 抽象层
- 每次 PR 前双平台编译检查
- 避免引入平台特定 API

这样既能享受 Windows 桌面开发的高效，又能控制 Android 兼容风险。

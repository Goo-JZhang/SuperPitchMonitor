# Visual Studio 2022 代码片段

## 常用代码模板

### 1. JUCE 组件类模板

```cpp
#pragma once

#include <juce_gui_basics/juce_gui_basics.h>

namespace spm {

class $ComponentName$ : public juce::Component
{
public:
    $ComponentName$();
    ~$ComponentName$() override;

    void paint(juce::Graphics& g) override;
    void resized() override;

private:
    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR($ComponentName$)
};

} // namespace spm
```

### 2. JUCE 线程类模板

```cpp
#pragma once

#include <juce_core/juce_core.h>

namespace spm {

class $ThreadName$ : private juce::Thread
{
public:
    $ThreadName$()
        : Thread("$ThreadName$")
    {
    }

    ~$ThreadName$() override
    {
        stopThread(2000);
    }

    void start()
    {
        startThread(juce::Thread::Priority::normal);
    }

    void stop()
    {
        signalThreadShouldExit();
    }

private:
    void run() override
    {
        while (!threadShouldExit())
        {
            // Do work here
            juce::Thread::sleep(10);
        }
    }

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR($ThreadName$)
};

} // namespace spm
```

### 3. 平台工具函数模板

```cpp
void $FunctionName$()
{
    // Windows implementation
    #if JUCE_WINDOWS
    
    #endif
    
    // Android implementation
    #if JUCE_ANDROID
    
    #endif
}
```

### 4. 调试日志宏

```cpp
#if defined(DEBUG) || defined(_DEBUG)
    #define SPM_LOG(msg) DBG("[SuperPitchMonitor] " << msg)
#else
    #define SPM_LOG(msg)
#endif
```

### 5. 性能分析代码块

```cpp
{
    SPM_PROFILE_SCOPE("$FunctionName$");
    // Code to profile
}
```

---

## 快捷键

| 快捷键 | 功能 |
|-------|------|
| F5 | 启动调试 |
| Ctrl+F5 | 启动不调试 |
| F9 | 设置/取消断点 |
| F10 | 逐过程执行 |
| F11 | 逐语句执行 |
| Shift+F5 | 停止调试 |
| Ctrl+Shift+B | 生成解决方案 |
| Ctrl+Shift+S | 保存全部 |
| Ctrl+K, Ctrl+C | 注释选中行 |
| Ctrl+K, Ctrl+U | 取消注释 |

---

## 调试窗口

| 窗口 | 快捷键 | 用途 |
|------|--------|------|
| 输出 | Ctrl+Alt+O | 查看 DBG 输出 |
| 断点 | Ctrl+Alt+B | 管理断点 |
| 监视 | Ctrl+Alt+W, 1 | 监视变量 |
| 自动窗口 | Ctrl+Alt+V, A | 自动变量 |
| 局部变量 | Ctrl+Alt+V, L | 局部变量 |
| 调用堆栈 | Ctrl+Alt+C | 查看调用栈 |
| 线程 | Ctrl+Alt+H | 查看线程 |
| 模块 | Ctrl+Alt+U | 查看加载模块 |

---

## 有用的调试命令

### 即时窗口 (Immediate Window)

```
? variableName    // 查看变量值
? this            // 查看当前对象
? variableName = 5    // 修改变量值
```

### 条件断点

右键断点 → 条件：
```cpp
// 只在特定条件触发
frequency > 440.0
pitches.size() > 0
i == 10
```

### 命中计数

右键断点 → 命中次数：
```
等于 100    // 只在第100次触发
```

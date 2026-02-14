# 崩溃日志系统

## 概述

增强的日志系统能够捕获：
- 正常日志信息
- 断言失败 (jassert/jcheckf)
- 崩溃和异常 (SEH, SIGSEGV, SIGABRT 等)
- 完整的堆栈跟踪

## 日志文件位置

运行时日志（应用执行时生成）：
```
Saved/Logs/                               # 运行时日志目录（项目根目录）
├── app_YYYY-MM-DD_HH-MM-SS.log          # 主日志文件
├── app_YYYY-MM-DD_HH-MM-SS_rotated.log  # 轮转后的日志 (超过10MB)
├── crash_YYYYMMDD_HHMMSS.txt            # 崩溃时生成的独立崩溃文件
└── crash_assertion.txt                  # 最新断言失败的快速查看文件
```

构建日志（编译时生成）：
```
build-windows/logs/                       # Windows 构建日志
build-macos/logs/                         # macOS 构建日志
build-linux/logs/                         # Linux 构建日志
```

## 崩溃日志格式

```
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!! EXCEPTION DETECTED (SEH)
!!! Code: 0xC0000005
!!! Description: Access violation
!!! Address: 0x00007FF612345678
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

Exception Stack Trace:
  [0] juce::AudioBuffer::setSample (AudioBuffer.h:657)
  [1] spm::SpectrumAnalyzer::process (SpectrumAnalyzer.cpp:61)
  [2] spm::AudioEngine::processAudioBlock (AudioEngine.cpp:369)
  ...

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
```

## 确保堆栈信息完整

### 1. 生成 PDB 文件

在 Visual Studio 中：
- 项目属性 → C/C++ → 常规 → 调试信息格式: **程序数据库 (/Zi)**
- 项目属性 → 链接器 → 调试 → 生成调试信息: **是 (/DEBUG)**

### 2. 保留 PDB 文件

PDB 文件必须与可执行文件在同一目录或指定路径：
```
SuperPitchMonitor_artefacts/
├── Debug/
│   ├── SuperPitchMonitor.exe
│   ├── SuperPitchMonitor.pdb  ← 必须存在
│   └── Saved/Logs/
```

### 3. 发布构建时的注意事项

发布构建的堆栈可能只显示地址而非函数名，因为：
- 优化可能导致内联函数
- 缺少 PDB 文件

## 常见崩溃代码

| 代码 | 描述 | 常见原因 |
|------|------|----------|
| 0xC0000005 | Access violation | 空指针解引用、数组越界 |
| 0xC0000094 | Integer divide by zero | 除零错误 |
| 0xC00000FD | Stack overflow | 无限递归、栈溢出 |
| 0x80000003 | Breakpoint | __debugbreak() 被调用 |
| 0xE06D7363 | C++ exception | 未捕获的 C++ 异常 |

## 手动触发测试

在代码中添加以下代码测试崩溃日志：

```cpp
// 测试断言
jassertfalse;  // 触发 JUCE 断言

// 测试空指针解引用 (将触发 0xC0000005)
int* p = nullptr;
*p = 42;

// 测试除零
int x = 1 / 0;
```

## 调试技巧

### 1. 使用 WinDbg 分析崩溃

```
> windbg -z crash.dmp
> .sympath C:\SuperPitchMonitor\build-windows\SuperPitchMonitor_artefacts\Debug
> .reload
> !analyze -v
```

### 2. 查看最近的崩溃

崩溃信息会同时写入：
1. 主日志文件 (带时间戳)
2. 独立的 crash_*.txt 文件
3. crash_assertion.txt (仅最新断言)

### 3. 在开发时捕获崩溃

Visual Studio 调试器会先于崩溃处理程序捕获异常。要测试崩溃日志：
- 在 VS 中：调试 → 异常 → 取消勾选 Win32 异常
- 或直接运行可执行文件 (Ctrl+F5)

## 故障排除

### 崩溃文件未生成

1. 检查日志目录权限
2. 确保 `FileLogger::initialize()` 在崩溃前被调用
3. 检查控制台输出是否有错误

### 堆栈缺少函数名

1. 确认 PDB 文件存在
2. 检查是否启用优化 (优化会内联函数)
3. 确认 DbgHelp.dll 在系统路径中

### 递归崩溃

如果崩溃处理程序本身崩溃：
- 系统会终止程序
- 可能生成 Windows 事件日志
- 检查 `Saved/Logs/crash_*.txt` 的部分内容

# SuperPitchMonitor 构建指南

## 快速验证流程

### 第 1 步：环境检查

```powershell
# 在项目根目录运行
powershell -ExecutionPolicy Bypass -File scripts/verify_environment.ps1
```

此脚本会检查：
- ✅ CMake 安装
- ✅ Visual Studio / 编译器
- ✅ Android SDK / NDK
- ✅ JUCE 库
- ✅ 项目文件完整性

**日志输出位置**: `build-windows/logs/environment_check_YYYYMMDD_HHMMSS.log` (Windows) 或 `build-macos/logs/` (macOS)

---

## 桌面端构建 (Windows)

### 前置要求
- Visual Studio 2022 (Community 版即可)
- CMake 3.22+

### 构建命令

```powershell
# 方式 1: 使用脚本 (推荐，带详细日志)
scripts\build_windows.bat

# 方式 2: 手动构建
mkdir build-windows
cd build-windows
cmake .. -G "Visual Studio 17 2022" -A x64
cmake --build . --config Debug --parallel
```

**日志输出位置**: `build-windows/logs/build_YYYYMMDD_HHMMSS.log`

### 常见问题

#### 错误: "CMake 未找到"
```
[ERROR] CMake 未找到！
```
**解决**: 下载安装 CMake https://cmake.org/download/，安装时勾选 "Add to PATH"

#### 错误: "Visual Studio 未找到"
```
[WARNING] Visual Studio 环境找到
```
**解决**: 安装 Visual Studio 2022，选择 "Desktop development with C++" 工作负载

---

## Android 构建

### 前置要求

根据你的配置 (Android Studio 16.0, Build-Tools 36.1.0)，需要额外安装：

1. **NDK r25+** (必须)
2. **CMake 3.22+** (必须)
3. **Command-line Tools** (必须)

### 自动配置 SDK

```powershell
# 运行配置助手
powershell -ExecutionPolicy Bypass -File scripts/setup_android_sdk.ps1
```

此脚本会：
- 检测 SDK 位置
- 安装缺失的 NDK
- 安装 CMake 工具
- 设置环境变量

### 手动配置 SDK

#### 1. 通过 Android Studio 安装

1. 打开 Android Studio
2. **Tools** → **SDK Manager**
3. 切换到 **SDK Tools** 标签
4. 勾选以下项目并安装：
   - ☑️ NDK (Side by side) → 选择 `25.2.9519653` 或更新
   - ☑️ CMake → 选择 `3.22.1`
   - ☑️ Android SDK Command-line Tools (latest)

#### 2. 命令行安装

```bash
# 找到 sdkmanager 的位置
# 通常在: %LOCALAPPDATA%\Android\Sdk\cmdline-tools\latest\bin\

# 安装 NDK
sdkmanager.bat "ndk;25.2.9519653"

# 安装 CMake
sdkmanager.bat "cmake;3.22.1"

# 安装平台 (如果缺少)
sdkmanager.bat "platforms;android-26"
```

### 构建 Android APK

```powershell
# 方式 1: 使用脚本 (推荐，带详细日志)
scripts\build_android.bat

# 方式 2: 手动构建
mkdir build-android
cd build-android

cmake .. ^
  -DCMAKE_SYSTEM_NAME=Android ^
  -DCMAKE_ANDROID_NDK=%ANDROID_SDK_ROOT%\ndk\25.2.9519653 ^
  -DCMAKE_ANDROID_ARCH_ABI=arm64-v8a ^
  -DCMAKE_ANDROID_PLATFORM=android-26 ^
  -DCMAKE_BUILD_TYPE=Debug

cmake --build . --parallel
```

**日志输出位置**: `Builds/Android/app/build/outputs/logs/` (Android Studio) 或 `build-android/logs/` (命令行)

---

## 日志文件说明

所有构建日志保存在 `build-*/logs/` 目录：

| 日志文件 | 内容 |
|---------|------|
| `environment_check.log` | 环境验证结果 |
| `build_windows_*.log` | Windows 桌面端构建日志 |
| `build_android_*.log` | Android 构建日志 |
| `cmake_configure_*.log` | CMake 配置详细日志 |

### 日志格式

```
[2026-02-13 10:30:15] [INFO] 信息消息
[2026-02-13 10:30:16] [SUCCESS] 成功消息
[2026-02-13 10:30:17] [WARNING] 警告消息
[2026-02-13 10:30:18] [ERROR] 错误消息
```

---

## 常见构建错误

### CMake 错误

#### "Could not find a package configuration file provided by JUCE"
```
解决方案:
  1. 确保 JUCE 目录存在: git clone https://github.com/juce-framework/JUCE.git
  2. 或在 cmake 命令中指定 -DJUCE_DIR=/path/to/JUCE
```

#### "No CMAKE_C_COMPILER could be found"
```
解决方案:
  1. 安装 Visual Studio 2022
  2. 选择 "Desktop development with C++" 工作负载
  3. 或安装 MinGW-w64 并添加到 PATH
```

### Android 错误

#### "CMAKE_ANDROID_NDK not set"
```
解决方案:
  1. 设置环境变量: set ANDROID_SDK_ROOT=C:\Users\<用户名>\AppData\Local\Android\Sdk
  2. 或在 cmake 中指定: -DCMAKE_ANDROID_NDK=/path/to/ndk
```

#### "NDK version may be outdated"
```
警告: NDK 版本较旧
解决方案:
  1. 运行 scripts/setup_android_sdk.ps1 自动升级
  2. 或在 Android Studio 中升级 NDK
```

#### "No toolchains found in the NDK"
```
解决方案:
  1. NDK 可能不完整，重新安装 NDK
  2. 检查 NDK 路径是否正确
```

---

## 验证构建成功

### Windows 桌面端
```powershell
# 检查可执行文件
ls build-windows\Debug\SuperPitchMonitor.exe

# 运行
.\build-windows\Debug\SuperPitchMonitor.exe
```

### Android
```bash
# 检查 APK
ls build-android\SuperPitchMonitor.apk

# 安装到设备
adb install build-android\SuperPitchMonitor.apk

# 查看日志
adb logcat -c
adb logcat | findstr SuperPitchMonitor
```

---

## 调试构建问题

### 启用详细日志

```powershell
# CMake 详细输出
cmake .. -DCMAKE_VERBOSE_MAKEFILE=ON

# 构建详细输出
cmake --build . --verbose
```

### 检查编译器

```powershell
# 检查 MSVC
cl.exe

# 检查 GCC (MinGW)
gcc --version

# 检查 Clang
clang --version
```

### 检查 Android 工具

```powershell
# 检查 adb
adb version

# 检查 NDK
ls $env:ANDROID_SDK_ROOT\ndk\

# 检查 CMake
ls $env:ANDROID_SDK_ROOT\cmake\
```

---

## 需要帮助？

如果构建仍然失败：

1. **运行环境检查**: `scripts\verify_environment.ps1`
2. **查看完整日志**: `build-windows\logs\` 目录下的最新日志
3. **提供日志文件**: 将 `build-windows\logs\*.log` 发送给开发团队

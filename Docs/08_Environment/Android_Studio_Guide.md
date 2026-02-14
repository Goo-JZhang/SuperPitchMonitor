# Android Studio 自动配置和构建指南

## 快速开始 (3步完成)

### 第 1 步：打开项目

1. 启动 **Android Studio**
2. 点击 **Open** (不要选 Import)
3. 选择目录：`C:\SuperPitchMonitor\Builds\Android`
4. 点击 **OK**

![Open Project](https://developer.android.com/static/studio/images/project/open-project.png)

### 第 2 步：等待自动配置

Android Studio 会自动完成以下操作：
- ✅ 下载 Gradle (如果需要)
- ✅ 同步项目配置
- ✅ 配置 CMake
- ✅ 下载依赖库

**首次配置需要 5-10 分钟，请耐心等待。**

### 第 3 步：构建和运行

1. 点击菜单 **Build** → **Make Project** (或按 `Ctrl+F9`)
2. 等待构建完成
3. 连接 Android 设备或启动模拟器
4. 点击 **Run** 按钮 (绿色三角形) 或按 `Shift+F10`

![Run Button](https://developer.android.com/static/studio/images/run/run-app.png)

---

## 文件结构

```
Builds/Android/
├── app/
│   ├── build.gradle              # App构建配置
│   └── src/main/
│       ├── AndroidManifest.xml   # 权限声明
│       └── ...
├── build.gradle                  # 项目构建配置
├── settings.gradle               # 项目设置
└── QUICK_START.md               # 详细指南

# 实际C++源代码位置
Source/
├── Main.cpp
├── MainComponent.cpp
├── Audio/
├── UI/
└── ...
```

---

## 常见问题

### 问题 1："CMake not found"

**解决：**
1. 打开 SDK Manager (Tools → SDK Manager)
2. 切换到 **SDK Tools** 标签
3. 勾选 **CMake 3.22.1** 或更新版本
4. 点击 **Apply** 安装

### 问题 2："NDK not found"

**解决：**
1. 打开 SDK Manager
2. 切换到 **SDK Tools** 标签
3. 勾选 **NDK (Side by side)** 版本 25.2.9519653
4. 点击 **Apply** 安装

### 问题 3：同步失败

**解决：**
1. 检查网络连接
2. 点击 **File** → **Invalidate Caches** → **Invalidate and Restart**
3. 重启后等待重新同步

### 问题 4：构建失败

**解决：**
1. 确保 Windows 版本已成功构建（生成 juceaide 工具）
2. 检查 Build 窗口的错误信息
3. 尝试 **Build** → **Clean Project**，然后重新构建

---

## 输出文件位置

构建成功后，APK 文件位置：

| 类型 | 路径 |
|-----|------|
| Debug | `Builds/Android/app/build/outputs/apk/debug/app-debug.apk` |
| Release | `Builds/Android/app/build/outputs/apk/release/app-release.apk` |

---

## 命令行构建 (备选)

如果 Android Studio GUI 无法工作，使用命令行：

```bash
# 打开 Android Studio 终端或 Windows CMD
cd C:\SuperPitchMonitor\Builds\Android

# 构建 Debug APK
.\gradlew assembleDebug

# 构建 Release APK
.\gradlew assembleRelease
```

APK 输出位置相同。

---

## 技术说明

### 为什么能自动配置？

1. **settings.gradle** - 告诉 Gradle 这是一个 Android 应用项目
2. **build.gradle** - 配置了：
   - 编译 SDK 版本 (34)
   - 最低 SDK 版本 (24 - Android 7.0)
   - NDK 和 CMake 路径
   - 指向主项目的 CMakeLists.txt
3. **AndroidManifest.xml** - 声明了录音权限和其他配置

### Windows 构建先行的重要性

JUCE 需要 `juceaide.exe` 工具来生成资源代码：
1. Windows 构建会生成这个工具
2. Android 构建复用该工具
3. 所以必须先完成 Windows 构建

---

## 下一步

1. 打开 Android Studio
2. 按照上述 3 步操作
3. 如果遇到问题，查看 Build 窗口的具体错误信息
4. 将错误信息复制给我，我会帮你解决

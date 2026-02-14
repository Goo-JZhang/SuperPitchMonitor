# Android Studio JSON 解析错误修复指南

## 错误信息
```
:app:debug:arm64-v8a failed to configure C/C++
Use JsonReader.setLenient(true) to accept malformed JSON at line 1 column 1 path $
```

## 错误原因
CMake 生成的 JSON 配置文件损坏或为空，通常发生在：
1. 电脑突然重启（我们之前的问题）导致文件写入中断
2. CMake 配置过程中断
3. 缓存文件版本不兼容

## ✅ 已自动完成的清理
以下目录已被清理：
- ✅ `build-android\.cxx` 
- ✅ `build-android\app\intermediates\cmake`
- ✅ `build-android\app\intermediates\cxx`
- ✅ `build-android\app\externalNativeBuild`

## 🔧 你需要手动完成的步骤

### Step 1: 关闭 Android Studio
确保 Android Studio 完全关闭（包括后台进程）

### Step 2: 清理 Android Studio 缓存
```
1. 打开 Android Studio
2. 点击菜单: File → Invalidate Caches...
3. 勾选以下选项:
   ☑️ Invalidate and Restart
   ☑️ Clear file system cache and Local History
   ☑️ Clear VCS Log caches and indexes
4. 点击 "Invalidate and Restart"
```

### Step 3: 等待 Gradle Sync
Android Studio 重启后会自动进行 Gradle Sync，等待完成。

### Step 4: 重新构建 Native 代码
```
Build → Make Project (Ctrl+F9)
```

或者使用命令行:
```bash
cd C:\SuperPitchMonitor\build-android
.\gradlew clean
.\gradlew assembleDebug
```

---

## 🛡️ 预防措施

### 避免再次损坏的方法:

1. **不要强制关机**
   - 构建过程中不要强制关闭 Android Studio
   - 不要强制关机或重启

2. **定期清理缓存**
   ```bash
   # 每周运行一次
   cd C:\SuperPitchMonitor\build-android
   .\gradlew clean
   ```

3. **使用桌面版进行日常开发**
   - Windows 桌面版 (`build-windows\Debug\SuperPitchMonitor.exe`)
   - 只在必要时使用 Android Studio

4. **备份重要文件**
   - 定期备份项目代码
   - CMakeLists.txt 等关键文件使用版本控制

---

## 🆘 如果仍然失败

如果上述步骤后仍然出现同样的错误，尝试完全重建：

### 核选项：完全重建

```powershell
# 1. 关闭 Android Studio

# 2. 删除整个 build-android 目录
Remove-Item -Recurse -Force C:\SuperPitchMonitor\build-android

# 3. 重新创建构建目录
mkdir C:\SuperPitchMonitor\build-android
cd C:\SuperPitchMonitor\build-android

# 4. 重新运行 CMake
cmake .. `
  -DCMAKE_SYSTEM_NAME=Android `
  -DCMAKE_ANDROID_NDK=$env:ANDROID_SDK_ROOT\ndk\25.2.9519653 `
  -DCMAKE_ANDROID_ARCH_ABI=arm64-v8a `
  -DCMAKE_ANDROID_PLATFORM=android-26 `
  -DCMAKE_BUILD_TYPE=Debug

# 5. 或者在 Android Studio 中重新导入项目
```

---

## 📋 快速检查清单

| 检查项 | 状态 |
|-------|------|
| Android Studio 已关闭再重新打开 | ☐ |
| Invalidate Caches 已执行 | ☐ |
| Gradle Sync 成功完成 | ☐ |
| Make Project 成功 | ☐ |

---

## 💡 替代方案

如果 Android Studio 仍然有问题，可以使用命令行构建:

```powershell
cd C:\SuperPitchMonitor\build-android

# 清理
.\gradlew clean

# 构建 Debug APK
.\gradlew assembleDebug

# 安装到连接的设备
.\gradlew installDebug
```

APK 将生成在:
```
C:\SuperPitchMonitor\build-android\app\build\outputs\apk\debug\app-debug.apk
```

---

**完成以上步骤后，JSON 解析错误应该会被解决。**

# Android 模拟器调试导致电脑重启 - 排查与解决方案

## 问题现象
使用 Android Studio 模拟器调试 SuperPitchMonitor 时，电脑偶尔自动重启。

## 可能原因（按概率排序）

### 1. 🔥 过热保护 (最可能)
- Android Studio 编译 + 模拟器运行 = CPU/GPU 高负载
- i7-14700KF 在满载时功耗可达 253W+
- 散热器不够强或硅脂干了会导致温度超过 100°C 触发保护

### 2. 💾 内存耗尽
- Android Studio: ~2-4GB
- Android 模拟器 (默认配置): ~2-4GB
- 编译过程中 Gradle 守护进程: ~2GB+
- 总共可能需要 8-12GB 内存

### 3. ⚡ 电源供电不足
- i7-14700KF + 高性能显卡在高负载时功耗很高
- 如果电源功率不足或老化，可能导致系统不稳定

### 4. 🎮 显卡驱动问题
- 模拟器使用 GPU 加速 (OpenGL/Vulkan)
- 驱动 Bug 可能导致 TDR (超时检测和恢复) 失败，引发重启

### 5. 🔧 Android Emulator 配置过高
- 默认分配的 RAM/CPU 核心过多

---

## 🔍 排查步骤

### Step 1: 监控温度 (关键！)

下载 HWiNFO64 或 Core Temp：
```
https://www.hwinfo.com/download/
https://www.alcpu.com/CoreTemp/
```

在运行 Android Studio 和模拟器时观察：
- CPU 温度是否超过 90°C
- 是否触发 thermal throttling

**如果温度过高 (>95°C)：**
- 检查 CPU 散热器是否安装正确
- 更换硅脂（如果超过 1 年）
- 改善机箱风道
- 临时解决方案：降低模拟器 CPU 核心数

### Step 2: 检查内存使用

打开任务管理器 (Ctrl+Shift+Esc)，观察：
- 内存使用是否接近 100%
- 提交大小 (Commit Size) 是否超过物理内存 + 页面文件

**解决方案：**
- 关闭不必要的程序
- 增加虚拟内存大小
- 降低模拟器内存分配

### Step 3: 检查电源

查看电源额定功率是否足够：
```
i7-14700KF 最大功耗: 253W
RTX 4070 最大功耗: ~200W
其他组件: ~100W
总计需要: 550W+

建议电源: 650W-750W 80Plus 金牌以上
```

### Step 4: 更新驱动

1. **显卡驱动** (关键)
   - 更新到最新稳定版
   - 如果是 NVIDIA，尝试 Studio Driver 而非 Game Ready Driver

2. **芯片组驱动**
   - 安装 Intel Chipset Driver

---

## 🛠️ 解决方案

### 方案 1: 降低模拟器资源分配 (推荐先尝试)

```
Android Studio > Device Manager > 你的设备 > Edit (笔图标)
```

**推荐配置：**
- **RAM**: 2048 MB (不要太高)
- **VM heap**: 576 MB
- **多核 CPU**: 选 2 核或 4 核 (不要选全部核心)
- **图形**: 如果之前是 Hardware，尝试改为 Software (GLES 2.0)

### 方案 2: 使用真机调试 (最稳定)

如果有 Android 手机，真机调试更稳定且资源占用少：

1. 手机开启开发者选项和 USB 调试
2. 连接 USB 线
3. Android Studio 会自动识别
4. 直接运行到真机

### 方案 3: 使用桌面端调试 (推荐日常开发)

SuperPitchMonitor 支持 Windows 桌面端调试，无需模拟器：

```powershell
cd C:\SuperPitchMonitor
scripts\build_windows.bat
```

桌面端调试优势：
- 启动快
- 调试功能完整 (Debug 按钮、音频模拟器)
- 不依赖 Android 环境

### 方案 4: 调整 Gradle/编译内存

编辑 `gradle.properties` (如果项目有)：
```properties
org.gradle.jvmargs=-Xmx2048m -XX:MaxMetaspaceSize=512m
org.gradle.parallel=true
org.gradle.configureondemand=true
```

降低 Gradle 内存使用。

### 方案 5: 禁用模拟器硬件加速 (如果怀疑 GPU 问题)

```
Android Studio > Settings > Emulator > 取消勾选 "Enable Emulator in Tool Window"
```

或者在模拟器启动参数中添加：
```
-gpu swiftshader_indirect
```

使用软件渲染替代 GPU 渲染。

---

## 📝 临时应急方案

如果你需要继续开发，但重启问题还没解决：

### 使用 Windows 桌面版调试 (强烈推荐)

```powershell
cd C:\SuperPitchMonitor
mkdir build-windows
cd build-windows
cmake .. -G "Visual Studio 17 2022" -A x64
cmake --build . --config Debug --parallel

# 运行
.\Debug\SuperPitchMonitor.exe
```

桌面版拥有完整的调试功能：
- Debug 按钮
- 音频模拟器 (C大调和弦等测试信号)
- 无需麦克风权限

### 降低编译负载

```powershell
# 使用单线程编译，降低 CPU 负载
cmake --build . --config Debug

# 或者限制并行任务数
cmake --build . --config Debug --parallel 4
```

---

## ⚡ 快速检查清单

| 检查项 | 方法 | 预期结果 |
|-------|------|---------|
| CPU 温度 | HWiNFO64 | < 85°C |
| 内存使用 | 任务管理器 | < 85% |
| 电源功率 | 查看电源标签 | > 600W |
| 显卡驱动 | 设备管理器 | 最新版本 |
| 模拟器配置 | AVD Manager | RAM ≤ 2GB |

---

## 🎯 推荐解决顺序

1. **立即**: 使用 Windows 桌面版继续开发（避免重启影响工作）
2. **并行**: 检查 CPU 温度（运行模拟器时监控）
3. **然后**: 降低模拟器配置到 2GB RAM + 2 核 CPU
4. **长期**: 考虑改善散热或升级电源（如果确认是硬件问题）

---

## 🆘 如果以上都无效

可能是更底层的硬件问题：
- 内存条故障（运行 MemTest86）
- 电源老化（更换电源测试）
- 主板问题

建议暂时避免使用模拟器，改用真机或桌面端开发。

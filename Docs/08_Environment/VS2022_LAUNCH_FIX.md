# 解决 "无法启动 ALL_BUILD" 问题

## 问题原因
`ALL_BUILD` 是一个 CMake 生成的元项目（Meta-project），用于构建所有其他项目，但本身不是可执行文件。

## 解决方法

### 方法 1: 设置启动项目（推荐）

1. 在 **解决方案资源管理器** 中找到 `SuperPitchMonitor` 项目
2. **右键点击** `SuperPitchMonitor`
3. 选择 **"设为启动项目"** (Set as Startup Project)
4. 按 **F5** 运行

### 方法 2: 直接运行可执行文件

```powershell
# 手动运行 Debug 版本
C:\SuperPitchMonitor\build-windows\SuperPitchMonitor_artefacts\Debug\SuperPitchMonitor.exe

# 或 Release 版本
C:\SuperPitchMonitor\build-windows\SuperPitchMonitor_artefacts\Release\SuperPitchMonitor.exe
```

### 方法 3: 使用 CMake 目标选择器

1. 在 Visual Studio 工具栏中找到 **"Select Startup Item"** 下拉框
2. 从 `ALL_BUILD` 改为 `SuperPitchMonitor.exe`
3. 按 **F5** 运行

## 验证成功

设置正确后，工具栏应该显示：
```
[SuperPitchMonitor] [Debug] [x64] [本地 Windows 调试器]
```

而不是：
```
[ALL_BUILD] [Debug] [x64] [本地 Windows 调试器]
```

## 快捷操作

| 操作 | 步骤 |
|------|------|
| 设置启动项目 | 右键 SuperPitchMonitor → 设为启动项目 |
| 直接运行 | Ctrl+F5 (开始执行不调试) |
| 调试运行 | F5 (开始调试) |

---

**设置启动项目后即可正常调试！**

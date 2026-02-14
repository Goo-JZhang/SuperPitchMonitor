# Visual Studio 2022 快速参考

## 🚀 打开项目

```powershell
# 方式 1: 使用脚本（推荐）
scripts\open_vs2022.bat

# 方式 2: 手动打开
cd build-windows
start SuperPitchMonitor.sln
```

## 📝 常用快捷键

| 操作 | 快捷键 |
|------|--------|
| 启动调试 | `F5` |
| 启动不调试 | `Ctrl+F5` |
| 生成解决方案 | `Ctrl+Shift+B` |
| 设置断点 | `F9` |
| 逐过程 | `F10` |
| 逐语句 | `F11` |
| 停止调试 | `Shift+F5` |

## 🐛 调试技巧

### 设置条件断点
```cpp
// 在 PolyphonicDetector::detect() 中设置断点
// 条件: !pitches.empty()
```

### 查看音频数据
```cpp
// 在 AudioEngine.cpp 中
// 监视: buffer.getReadPointer(0)[0]
```

### 输出调试信息
```cpp
DBG("Frequency: " << frequency);  // 出现在 VS 输出窗口
```

## 📁 重要路径

| 内容 | 路径 |
|------|------|
| 解决方案 | `build-windows\SuperPitchMonitor.sln` |
| Debug 可执行文件 | `build-windows\SuperPitchMonitor_artefacts\Debug\` |
| Release 可执行文件 | `build-windows\SuperPitchMonitor_artefacts\Release\` |
| 源代码 | `Source\` |

## ⚙️ 切换配置

```
工具栏: [Debug/Release] → [x64]
```

## 🔧 重新生成

```powershell
# 如果需要重新生成
rd /s /q build-windows
scripts\build_windows.bat
```

## 📊 性能分析

```
调试 → 性能分析器 (Alt+F2)
选择: CPU 使用率
```

## 🆘 常见问题

| 问题 | 解决 |
|------|------|
| 找不到 exe | 先执行 `Ctrl+Shift+B` 构建 |
| 断点不生效 | 确保在 Debug 模式下 |
| 项目加载失败 | 删除 `.vs/` 和 `CMakeCache.txt` |

---

**祝开发愉快！**

# Visual Studio 2022 开发环境配置完成

## ✅ 已创建的配置文件

### 1. Visual Studio 启动配置
**文件**: `.vs\launch.vs.json`
- 配置 Debug 和 Release 启动目标
- 自动设置工作目录和 PATH 环境变量

### 2. CMake 项目配置
**文件**: `CMakeSettings.json`
- 配置 x64-Debug 和 x64-Release
- 使用 Visual Studio 17 2022 生成器

### 3. 快速启动脚本
**文件**: `scripts\open_vs2022.bat`
- 一键打开 Visual Studio 2022
- 自动检查解决方案是否存在

### 4. 文档
**文件**: `Docs\VS2022_Development_Guide.md`
- 完整的开发调试指南
- 常见问题解决方案

**文件**: `Docs\VS2022_CodeSnippets.md`
- 代码模板
- 快捷键参考

---

## 🚀 快速开始

### 方式 1: 使用批处理脚本（最简单）

```powershell
cd C:\SuperPitchMonitor
scripts\open_vs2022.bat
```

### 方式 2: 直接打开解决方案

```powershell
cd C:\SuperPitchMonitor\build-windows
start SuperPitchMonitor.sln
```

### 方式 3: 作为 CMake 项目打开

1. 打开 Visual Studio 2022
2. 选择 "打开本地文件夹"
3. 选择 `C:\SuperPitchMonitor`
4. VS 会自动识别 CMakeLists.txt

---

## ⚙️ 配置说明

### 启动配置 (`launch.vs.json`)

```json
{
  "configurations": [
    {
      "name": "SuperPitchMonitor (Debug)",
      "projectTarget": "SuperPitchMonitor.exe",
      "args": [],
      "currentDir": "${projectDir}"
    }
  ]
}
```

### CMake 配置 (`CMakeSettings.json`)

```json
{
  "configurations": [
    {
      "name": "x64-Debug",
      "generator": "Visual Studio 17 2022 Win64",
      "configurationType": "Debug",
      "buildRoot": "${projectDir}\\build-windows"
    }
  ]
}
```

---

## 📝 使用流程

### 首次使用

1. **确保项目已构建**
   ```powershell
   scripts\build_windows.bat
   ```

2. **打开 Visual Studio**
   ```powershell
   scripts\open_vs2022.bat
   ```

3. **设置启动项目**
   - 在解决方案资源管理器中
   - 右键 `SuperPitchMonitor` → "设为启动项目"

4. **选择配置**
   - 工具栏选择 `Debug` 或 `Release`
   - 平台选择 `x64`

### 日常开发

1. **修改代码** - 编辑 `Source/` 下的文件
2. **构建项目** - `Ctrl+Shift+B`
3. **启动调试** - `F5`
4. **测试功能** - 验证修改效果

---

## 🐛 调试配置

### 已配置的功能

- ✅ Debug 和 Release 启动目标
- ✅ 自动设置工作目录
- ✅ 包含调试符号 (.pdb)
- ✅ 源代码级调试

### 关键断点位置

| 文件 | 函数 | 用途 |
|------|------|------|
| MainComponent.cpp | `MainComponent()` | 启动流程 |
| AudioEngine.cpp | `processAudioBlock()` | 音频处理 |
| SpectrumAnalyzer.cpp | `process()` | 频谱分析 |
| PolyphonicDetector.cpp | `detect()` | 音高检测 |

---

## 📁 输出文件位置

| 配置 | 可执行文件 | 符号文件 |
|------|-----------|---------|
| Debug | `build-windows\SuperPitchMonitor_artefacts\Debug\SuperPitchMonitor.exe` | `.pdb` |
| Release | `build-windows\SuperPitchMonitor_artefacts\Release\SuperPitchMonitor.exe` | `.pdb` |

---

## 🔧 高级选项

### 修改调试参数

编辑 `.vs\launch.vs.json`:

```json
{
  "configurations": [
    {
      "name": "SuperPitchMonitor (Debug)",
      "args": ["--custom-arg"],
      "env": {
        "CUSTOM_VAR": "value"
      }
    }
  ]
}
```

### 添加新的构建配置

编辑 `CMakeSettings.json`:

```json
{
  "configurations": [
    {
      "name": "x64-RelWithDebInfo",
      "configurationType": "RelWithDebInfo"
    }
  ]
}
```

---

## 📚 参考文档

- [完整开发指南](Docs/VS2022_Development_Guide.md)
- [代码片段](Docs/VS2022_CodeSnippets.md)

---

## ✅ 验证步骤

确保以下功能正常工作：

- [ ] `scripts\open_vs2022.bat` 能打开 VS2022
- [ ] 解决方案加载成功
- [ ] 可以编译 Debug 版本
- [ ] 可以编译 Release 版本
- [ ] F5 能启动调试
- [ ] 断点能命中
- [ ] 应用程序正常运行

---

**Visual Studio 2022 开发环境配置完成！**

现在你可以：
- ✅ 使用 F5 快速启动调试
- ✅ 在断点处检查变量
- ✅ 单步执行代码
- ✅ 查看调用堆栈
- ✅ 分析性能问题

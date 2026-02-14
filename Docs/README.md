# SuperPitchMonitor 文档导航

本文档库包含 SuperPitchMonitor 项目的完整技术文档。

---

## 文档结构

### 01. 技术分析 (TechnicalAnalysis)
项目前期的技术可行性分析。

- [技术可行性分析报告](01_TechnicalAnalysis/技术可行性分析报告.md)

### 02. 系统架构 (Architecture)
系统整体架构设计。

- [系统架构设计](02_Architecture/系统架构设计.md)

### 03. 算法设计 (Algorithms)
核心算法的详细设计文档。

**频谱分析算法**
- [频谱分析算法设计](03_Algorithms/SpectrumAnalysis/频谱分析算法设计.md)

**音高检测算法**
- [多音高检测算法设计](03_Algorithms/PitchDetection/多音高检测算法设计.md)

### 04. 平台适配 (Platform)
跨平台开发和平台特定适配。

- [Android 平台适配与性能优化](04_Platform/Android平台适配与性能优化.md)
- [跨平台开发策略](04_Platform/CrossPlatform_Development_Strategy.md)
- [跨平台迁移总结](04_Platform/CrossPlatform_Setup_Summary.md)
- [迁移完成报告](04_Platform/Migration_Complete.md)
- [迁移示例代码](04_Platform/Migration_Example.md)

### 05. UI 设计 (UI)
用户界面设计规范。

- [UI 设计与交互规范](05_UI/UI设计与交互规范.md)

### 06. 开发规划 (Development)
项目开发路线图。

- [开发路线图](06_Development/开发路线图.md)

### 07. 构建指南 (Build)
项目构建、编译和调试指南。

- [构建指南](07_Build/BuildGuide.md) - 完整的构建说明
- [调试指南](07_Build/DebugGuide.md) - 调试功能使用说明
- [CMake 修复总结](07_Build/CMAKE_FIX_SUMMARY.md)
- [构建修复总结](07_Build/BUILD_FIX_SUMMARY.md)
- [最终构建修复](07_Build/FINAL_BUILD_FIX.md)
- [重新构建说明](07_Build/REBUILD_INSTRUCTIONS.md)
- [快速验证指南](07_Build/QUICK_VERIFY.md)

### 08. 开发环境 (Environment)
开发工具和环境配置。

**Visual Studio 2022**
- [VS2022 开发指南](08_Environment/VS2022_Development_Guide.md)
- [VS2022 代码片段](08_Environment/VS2022_CodeSnippets.md)
- [VS2022 启动修复](08_Environment/VS2022_LAUNCH_FIX.md)
- [VS2022 快速参考](08_Environment/VS2022_QUICK_REFERENCE.md)
- [VS2022 配置完成](08_Environment/VS2022_SETUP_COMPLETE.md)

**Android Studio**
- [Android Studio 指南](08_Environment/Android_Studio_Guide.md)

### 09. 问题排查 (Troubleshooting)
常见问题及解决方案。

- [Android JSON 错误修复](09_Troubleshooting/Android_JSON_Error_Fix.md)
- [Android 模拟器调试重启问题排查指南](09_Troubleshooting/Android模拟器调试重启问题排查指南.md)

### 10. 功能特性 (Features)
功能实现说明和使用指南。

- [音频输入功能说明](10_Features/Audio_Input_Features.md)
- [功能完成总结](10_Features/Feature_Complete_Summary.md)
- [新功能总结](10_Features/New_Features_Summary.md)
- [调试日志指南](10_Features/Debug_Logging_Guide.md)

---

## 快速链接

### 新开发者必读
1. [构建指南](07_Build/BuildGuide.md) - 开始构建项目
2. [调试指南](07_Build/DebugGuide.md) - 了解调试功能
3. [系统架构设计](02_Architecture/系统架构设计.md) - 了解项目架构

### 跨平台开发
1. [跨平台开发策略](04_Platform/CrossPlatform_Development_Strategy.md)
2. [VS2022 开发指南](08_Environment/VS2022_Development_Guide.md)

### 遇到问题
1. [问题排查目录](09_Troubleshooting/)
2. [CMake 修复总结](07_Build/CMAKE_FIX_SUMMARY.md)

---

## 文档规范

- 技术文档使用 Markdown 格式
- 文档名使用英文（中文内容使用中文文件名）
- 图片资源放在对应目录的 `images/` 子目录中
- 代码示例需使用 ```cpp 标记

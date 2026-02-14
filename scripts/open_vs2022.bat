@echo off
chcp 65001 >nul
:: Open SuperPitchMonitor in Visual Studio 2022

echo ========================================
echo  SuperPitchMonitor - Visual Studio 2022
echo ========================================
echo.

set "BUILD_DIR=%~dp0..\build-windows"
set "SLN_FILE=%BUILD_DIR%\SuperPitchMonitor.sln"

:: Check if solution exists
if not exist "%SLN_FILE%" (
    echo ❌ 解决方案文件不存在: %SLN_FILE%
    echo.
    echo 请先构建项目:
    echo   scripts\build_windows.bat
    echo.
    pause
    exit /b 1
)

echo ✅ 找到解决方案: %SLN_FILE%
echo.
echo 正在启动 Visual Studio 2022...
echo.

:: Open Visual Studio
start "" "%SLN_FILE%"

if %errorlevel% neq 0 (
    echo ❌ 启动 Visual Studio 失败
    echo.
    echo 请确保已安装 Visual Studio 2022
echo.
    pause
    exit /b 1
)

echo ✅ Visual Studio 2022 已启动
echo.
echo 提示:
echo   - 确保在工具栏选择 "Debug" 配置
echo   - 按 F5 开始调试
echo   - 按 Ctrl+F5 开始不调试
echo.

:: Wait a moment then exit
timeout /t 3 >nul

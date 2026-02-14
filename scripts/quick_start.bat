@echo off
chcp 65001 >nul
setlocal EnableDelayedExpansion

REM SuperPitchMonitor Quick Start Script
REM Auto-detect environment and provide build options

title SuperPitchMonitor Quick Start

cd /d "%~dp0.."
set PROJECT_ROOT=%CD%

echo.
echo  ============================================
echo    SuperPitchMonitor Quick Start
echo  ============================================
echo.
echo  Project path: %PROJECT_ROOT%
echo.

REM Check log directory
if not exist "%PROJECT_ROOT%\build_logs" mkdir "%PROJECT_ROOT%\build_logs"

:menu
echo.
echo  [1] Check environment
set /p has_env="     Have you run environment check before? (Y/n): "
if "%has_env%"=="" set has_env=Y

if /i "%has_env%"=="n" (
    echo.
    echo  Running environment check...
    echo.
    powershell -ExecutionPolicy Bypass -File "%PROJECT_ROOT%\scripts\verify_environment.ps1"
    if errorlevel 1 (
        echo.
        echo  [!] Environment check found issues
        echo.
    )
)

echo.
echo  [2] Select build target
echo.
echo      [1] Windows Desktop (Debug)
echo      [2] Windows Desktop (Release)
echo      [3] Android (Debug)
echo      [4] Android (Release)
echo      [5] Configure Android SDK (Install NDK/CMake)
echo      [6] Open build logs directory
echo      [0] Exit
echo.

set /p choice="  Enter option (0-6): "

if "%choice%"=="1" goto :build_windows_debug
if "%choice%"=="2" goto :build_windows_release
if "%choice%"=="3" goto :build_android_debug
if "%choice%"=="4" goto :build_android_release
if "%choice%"=="5" goto :setup_android
if "%choice%"=="6" goto :open_logs
if "%choice%"=="0" goto :exit

echo  [!] Invalid option
goto :menu

:build_windows_debug
echo.
echo  [*] Starting Windows Desktop (Debug) build...
echo.
call "%PROJECT_ROOT%\scripts\build_windows.bat"
pause
goto :menu

:build_windows_release
echo.
echo  [*] Starting Windows Desktop (Release) build...
echo.
if not exist "%PROJECT_ROOT%\build-windows-release" mkdir "%PROJECT_ROOT%\build-windows-release"
cd "%PROJECT_ROOT%\build-windows-release"
cmake .. -G "Visual Studio 17 2022" -A x64 -DCMAKE_BUILD_TYPE=Release -DJUCE_DIR="%PROJECT_ROOT%\JUCE"
if errorlevel 1 (
    echo  [!] CMake configuration failed
    pause
    goto :menu
)
cmake --build . --config Release --parallel
if errorlevel 1 (
    echo  [!] Build failed
) else (
    echo  [OK] Build successful!
    echo  Output: %PROJECT_ROOT%\build-windows-release\Release\SuperPitchMonitor.exe
)
pause
goto :menu

:build_android_debug
echo.
echo  [*] Starting Android (Debug) build...
echo.
call "%PROJECT_ROOT%\scripts\build_android.bat"
pause
goto :menu

:build_android_release
echo.
echo  [*] Starting Android (Release) build...
echo.
set /p ndk_path="  Enter NDK path (or press Enter for auto-detect): "
if "%ndk_path%"=="" (
    for /d %%D in ("%LOCALAPPDATA%\Android\Sdk\ndk\*") do set ndk_path=%%~D
)
if "%ndk_path%"=="" (
    echo  [!] NDK not found, please configure SDK first
    pause
    goto :menu
)

echo  Using NDK: %ndk_path%
if not exist "%PROJECT_ROOT%\build-android-release" mkdir "%PROJECT_ROOT%\build-android-release"
cd "%PROJECT_ROOT%\build-android-release"

cmake .. ^
    -DCMAKE_SYSTEM_NAME=Android ^
    -DCMAKE_ANDROID_NDK="%ndk_path%" ^
    -DCMAKE_ANDROID_ARCH_ABI=arm64-v8a ^
    -DCMAKE_ANDROID_PLATFORM=android-26 ^
    -DCMAKE_BUILD_TYPE=Release ^
    -DJUCE_DIR="%PROJECT_ROOT%\JUCE"

if errorlevel 1 (
    echo  [!] CMake configuration failed
    pause
    goto :menu
)

cmake --build . --parallel
if errorlevel 1 (
    echo  [!] Build failed
) else (
    echo  [OK] Build successful!
)
pause
goto :menu

:setup_android
echo.
echo  [*] Configuring Android SDK...
echo.
powershell -ExecutionPolicy Bypass -File "%PROJECT_ROOT%\scripts\setup_android_sdk.ps1"
pause
goto :menu

:open_logs
echo.
echo  [*] Opening build logs directory...
echo.
start explorer "%PROJECT_ROOT%\build_logs"
goto :menu

:exit
echo.
echo  Thank you for using SuperPitchMonitor build system!
echo.
pause
exit /b 0


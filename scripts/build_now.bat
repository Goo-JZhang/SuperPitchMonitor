@echo off
chcp 65001 >nul
setlocal EnableDelayedExpansion

echo ========================================
echo SuperPitchMonitor Quick Build
echo ========================================
echo.

set PROJECT_ROOT=%~dp0..
cd /d %PROJECT_ROOT%

if not exist "build_logs" mkdir "build_logs"

REM Run environment check first
echo [1/3] Checking environment...
powershell -ExecutionPolicy Bypass -File scripts\verify_environment.ps1 > build_logs\env_check.log 2>&1
if errorlevel 1 (
    echo [!] Environment check failed. See build_logs\env_check.log
    pause
    exit /b 1
)

REM Build Windows
echo.
echo [2/3] Building Windows Desktop (Debug)...
echo This may take 5-15 minutes on first build...
echo.

if not exist "build-windows" mkdir "build-windows"
cd "build-windows"

echo Running CMake configuration...
cmake .. -G "Visual Studio 17 2022" -A x64 -DJUCE_DIR="%PROJECT_ROOT%\JUCE" > ..\build_logs\cmake_config.log 2>&1
if errorlevel 1 (
    echo [!] CMake configuration failed!
    echo.
    echo Common fixes:
    echo 1. Ensure Visual Studio 2022 is installed with "Desktop development with C++"
    echo 2. Ensure JUCE is cloned: git clone https://github.com/juce-framework/JUCE.git
    echo.
    echo See build_logs\cmake_config.log for details
    cd ..
    pause
    exit /b 1
)

echo Building project...
cmake --build . --config Debug --parallel > ..\build_logs\build_output.log 2>&1
if errorlevel 1 (
    echo [!] Build failed!
    echo See build_logs\build_output.log for details
    cd ..
    pause
    exit /b 1
)

cd "%PROJECT_ROOT%"

REM Check output - JUCE uses different output directories
set EXE_PATH=
if exist "build-windows\Debug\SuperPitchMonitor.exe" (
    set EXE_PATH=build-windows\Debug\SuperPitchMonitor.exe
) else if exist "build-windows\SuperPitchMonitor_artefacts\Debug\SuperPitchMonitor.exe" (
    set EXE_PATH=build-windows\SuperPitchMonitor_artefacts\Debug\SuperPitchMonitor.exe
) else if exist "build-windows\SuperPitchMonitor_artefacts\Release\SuperPitchMonitor.exe" (
    set EXE_PATH=build-windows\SuperPitchMonitor_artefacts\Release\SuperPitchMonitor.exe
)

if defined EXE_PATH (
    echo.
    echo ========================================
    echo SUCCESS! Build completed!
    echo ========================================
    echo.
    echo Executable: %EXE_PATH%
    for %%F in (%EXE_PATH%) do echo Size: %%~zF bytes
    echo.
    echo Run with:
    echo   %EXE_PATH%
    echo.
    choice /C YN /M "Run now"
    if errorlevel 1 if not errorlevel 2 (
        start %EXE_PATH%
    )
) else (
    echo [!] Build may have succeeded but executable not found
    echo Check build_logs\build_output.log
    echo.
    echo Searching for exe files...
    dir /s /b build-windows\*.exe 2>nul | findstr /i superpitch
)

echo.
pause
exit /b 0

@echo off
chcp 65001 >nul
setlocal EnableDelayedExpansion

REM SuperPitchMonitor Windows Build Script

set PROJECT_ROOT=%~dp0..
cd /d %PROJECT_ROOT%

set LOG_DIR=%PROJECT_ROOT%\build_logs
set LOG_FILE=%LOG_DIR%\build_windows_%date:~0,4%%date:~5,2%%date:~8,2%_%time:~0,2%%time:~3,2%%time:~6,2%.log
set LOG_FILE=%LOG_FILE: =0%

if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"

echo ========================================
echo SuperPitchMonitor Windows Builder
echo ========================================
echo Log: %LOG_FILE%
echo.

echo [%date% %time%] Build started > "%LOG_FILE%"

call :log "INFO" "========================================"
call :log "INFO" "Starting build"
call :log "INFO" "========================================"

REM 1. Check CMake
call :log "INFO" ""
call :log "INFO" "[1/6] Checking CMake..."
where cmake >nul 2>&1
if %errorlevel% neq 0 (
    call :log "ERROR" "CMake not found!"
    call :log "INFO" "Install from: https://cmake.org/download/"
    goto :error
)
call :log "SUCCESS" "CMake found"
cmake --version >> "%LOG_FILE%" 2>&1

REM 2. Check JUCE
call :log "INFO" ""
call :log "INFO" "[2/6] Checking JUCE..."
if not exist "JUCE\CMakeLists.txt" (
    call :log "ERROR" "JUCE not found!"
    call :log "INFO" "Run: git clone https://github.com/juce-framework/JUCE.git"
    goto :error
)
call :log "SUCCESS" "JUCE found"

REM 3. Check Visual Studio
call :log "INFO" ""
call :log "INFO" "[3/6] Checking Visual Studio..."
set VSWHERE="%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
if exist %VSWHERE% (
    call :log "SUCCESS" "Visual Studio found"
) else (
    call :log "WARNING" "vswhere.exe not found"
)

REM 4. Configure CMake
call :log "INFO" ""
call :log "INFO" "[4/6] Configuring CMake..."

if not exist "build-windows" mkdir "build-windows"
cd "build-windows"

echo. >> "%LOG_FILE%" 2>&1
echo CMake command: >> "%LOG_FILE%" 2>&1
echo cmake .. -G "Visual Studio 17 2022" -A x64 -DJUCE_DIR="%PROJECT_ROOT%\JUCE" >> "%LOG_FILE%" 2>&1
echo. >> "%LOG_FILE%" 2>&1

cmake .. -G "Visual Studio 17 2022" -A x64 -DJUCE_DIR="%PROJECT_ROOT%\JUCE" >> "%LOG_FILE%" 2>&1
if %errorlevel% neq 0 (
    call :log "ERROR" "CMake configuration failed!"
    goto :error
)
call :log "SUCCESS" "CMake configured"

REM 5. Build
call :log "INFO" ""
call :log "INFO" "[5/6] Building..."
call :log "INFO" "This may take a few minutes..."

echo [%date% %time%] Build started >> "%LOG_FILE%" 2>&1
cmake --build . --config Debug --parallel >> "%LOG_FILE%" 2>&1
if %errorlevel% neq 0 (
    call :log "ERROR" "Build failed!"
    goto :error
)
call :log "SUCCESS" "Build successful"

REM 6. Check output
call :log "INFO" ""
call :log "INFO" "[6/6] Checking output..."

set EXE_PATH=Debug\SuperPitchMonitor.exe
if exist "%EXE_PATH%" (
    call :log "SUCCESS" "Executable generated"
    call :log "INFO" "Path: %CD%\%EXE_PATH%"
) else (
    call :log "WARNING" "Output file not found"
)

cd "%PROJECT_ROOT%"

call :log "INFO" ""
call :log "INFO" "========================================"
call :log "SUCCESS" "Build complete!"
call :log "INFO" "========================================"
call :log "INFO" "Log: %LOG_FILE%"
call :log "INFO" ""
call :log "INFO" "Run:"
call :log "INFO" "  cd build-windows"
call :log "INFO" "  Debug\SuperPitchMonitor.exe"
call :log "INFO" ""

echo.
echo ========================================
echo Build successful!
echo ========================================
echo.
echo Log: %LOG_FILE%
echo.

pause
exit /b 0

:error
call :log "ERROR" ""
call :log "ERROR" "========================================"
call :log "ERROR" "Build failed"
call :log "ERROR" "========================================"
call :log "ERROR" "Log: %LOG_FILE%"
call :log "ERROR" ""

echo.
echo ========================================
echo Build failed - Check log for details
echo ========================================
echo.
echo Log: %LOG_FILE%
echo.

pause
exit /b 1

:log
set TIMESTAMP=%date:~0,4%-%date:~5,2%-%date:~8,2% %time:~0,2%:%time:~3,2%:%time:~6,2%
set LEVEL=%~1
set MESSAGE=%~2
echo [%TIMESTAMP%] [%LEVEL%] %MESSAGE%
echo [%TIMESTAMP%] [%LEVEL%] %MESSAGE% >> "%LOG_FILE%"
exit /b 0

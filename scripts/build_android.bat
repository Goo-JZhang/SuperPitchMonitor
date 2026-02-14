@echo off
chcp 65001 >nul
setlocal EnableDelayedExpansion

REM SuperPitchMonitor Android Build Script
REM Auto-detect SDK and NDK configuration

set PROJECT_ROOT=%~dp0..
cd /d %PROJECT_ROOT%

REM Log configuration
set LOG_DIR=%PROJECT_ROOT%\build_logs
set LOG_FILE=%LOG_DIR%\build_android_%date:~0,4%%date:~5,2%%date:~8,2%_%time:~0,2%%time:~3,2%%time:~6,2%.log
set LOG_FILE=%LOG_FILE: =0%

if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"

echo ========================================
echo SuperPitchMonitor Android Builder
echo ========================================
echo Log file: %LOG_FILE%
echo.

echo [%date% %time%] Build started > "%LOG_FILE%"

call :log "INFO" "========================================"
call :log "INFO" "Starting Android build process"
call :log "INFO" "========================================"

REM 1. Detect Android SDK
call :log "INFO" ""
call :log "INFO" "[1/8] Detecting Android SDK..."

set SDK_FOUND=0
set SDK_PATH=

REM Check environment variables
if defined ANDROID_SDK_ROOT (
    call :log "INFO" "Found ANDROID_SDK_ROOT: %ANDROID_SDK_ROOT%"
    if exist "%ANDROID_SDK_ROOT%\platform-tools" (
        set SDK_PATH=%ANDROID_SDK_ROOT%
        set SDK_FOUND=1
    )
)

if %SDK_FOUND%==0 if defined ANDROID_HOME (
    call :log "INFO" "Found ANDROID_HOME: %ANDROID_HOME%"
    if exist "%ANDROID_HOME%\platform-tools" (
        set SDK_PATH=%ANDROID_HOME%
        set SDK_FOUND=1
    )
)

REM Check common paths
if %SDK_FOUND%==0 (
    set COMMON_PATHS=^
        "%LOCALAPPDATA%\Android\Sdk" ^
        "%USERPROFILE%\AppData\Local\Android\Sdk" ^
        "C:\Android\Sdk" ^
        "C:\Users\%USERNAME%\AppData\Local\Android\Sdk"
    
    for %%p in (%COMMON_PATHS%) do (
        if exist "%%~p\platform-tools" (
            set SDK_PATH=%%~p
            set SDK_FOUND=1
            call :log "INFO" "Found SDK at common path: %%~p"
            goto :sdk_found
        )
    )
)

:sdk_found
if %SDK_FOUND%==0 (
    call :log "ERROR" "Android SDK not found!"
    call :log "INFO" "Please set ANDROID_SDK_ROOT environment variable"
    call :log "INFO" "Or install SDK via Android Studio"
    goto :error
)

call :log "SUCCESS" "Android SDK found: %SDK_PATH%"

REM 2. Check NDK
call :log "INFO" ""
call :log "INFO" "[2/8] Checking Android NDK..."

set NDK_PATH=
set NDK_VERSION=

REM Check NDK directory
if exist "%SDK_PATH%\ndk" (
    for /d %%D in ("%SDK_PATH%\ndk\*") do (
        set NDK_PATH=%%~D
        for /f "tokens=1 delims=\" %%a in ("%%~nxD") do set NDK_VERSION=%%a
        call :log "INFO" "Found NDK: !NDK_VERSION!"
        goto :ndk_found
    )
)

REM Check ndk-bundle (old version)
if exist "%SDK_PATH%\ndk-bundle" (
    set NDK_PATH=%SDK_PATH%\ndk-bundle
    call :log "INFO" "Found NDK (ndk-bundle)"
    goto :ndk_found
)

REM Check side-by-side NDK
if exist "%SDK_PATH%\ndk\26.0.10792818" (
    set NDK_PATH=%SDK_PATH%\ndk\26.0.10792818
    call :log "INFO" "Found NDK 26.0.10792818"
    goto :ndk_found
)

if exist "%SDK_PATH%\ndk\25.2.9519653" (
    set NDK_PATH=%SDK_PATH%\ndk\25.2.9519653
    call :log "INFO" "Found NDK 25.2.9519653"
    goto :ndk_found
)

:ndk_found
if not defined NDK_PATH (
    call :log "ERROR" "NDK not found!"
    call :log "INFO" ""
    call :log "INFO" "Solution 1 - Install via Android Studio:"
    call :log "INFO" "  1. Open Android Studio"
    call :log "INFO" "  2. Tools -> SDK Manager"
    call :log "INFO" "  3. Switch to SDK Tools tab"
    call :log "INFO" "  4. Check NDK (Side by side)"
    call :log "INFO" "  5. Click Apply to install"
    call :log "INFO" ""
    call :log "INFO" "Solution 2 - Command line install:"
    call :log "INFO" "  sdkmanager --install ndk;25.2.9519653"
    call :log "INFO" ""
    goto :error
)

call :log "SUCCESS" "NDK found: %NDK_PATH%"

REM 3. Check CMake
call :log "INFO" ""
call :log "INFO" "[3/8] Checking CMake..."

set CMAKE_PATH=

REM Check system CMake
where cmake >nul 2>&1
if %errorlevel%==0 (
    for /f "tokens=*" %%a in ('where cmake') do (
        set CMAKE_PATH=%%a
        call :log "INFO" "Found system CMake: %%a"
        goto :cmake_found
    )
)

REM Check Android SDK CMake
if exist "%SDK_PATH%\cmake" (
    for /d %%D in ("%SDK_PATH%\cmake\*") do (
        if exist "%%~D\bin\cmake.exe" (
            set CMAKE_PATH=%%~D\bin\cmake.exe
            call :log "INFO" "Found SDK CMake: %%~D"
            goto :cmake_found
        )
    )
)

:cmake_found
if not defined CMAKE_PATH (
    call :log "ERROR" "CMake not found!"
    call :log "INFO" "Please install CMake 3.22+"
    goto :error
)

call :log "SUCCESS" "CMake found"
"%CMAKE_PATH%" --version >> "%LOG_FILE%" 2>&1

REM 4. Check Ninja
call :log "INFO" ""
call :log "INFO" "[4/8] Checking Ninja..."

if exist "%SDK_PATH%\cmake\3.22.1\bin\ninja.exe" (
    set NINJA_PATH=%SDK_PATH%\cmake\3.22.1\bin\ninja.exe
    call :log "INFO" "Found Ninja: %NINJA_PATH%"
) else (
    call :log "WARNING" "Ninja not found, CMake will try to auto-detect"
)

REM 5. Check JUCE
call :log "INFO" ""
call :log "INFO" "[5/8] Checking JUCE..."

if not exist "JUCE\CMakeLists.txt" (
    call :log "ERROR" "JUCE not found!"
    call :log "INFO" "Please ensure JUCE directory exists"
    goto :error
)
call :log "SUCCESS" "JUCE found"

REM 6. Configure Gradle / Use CMake directly
call :log "INFO" ""
call :log "INFO" "[6/8] Configuring build..."

REM Detect ABI
set ABI=arm64-v8a
call :log "INFO" "Target ABI: %ABI%"

REM Detect platform version - compatible with SDK 36 (Android 16)
set PLATFORM=android-26
if exist "%SDK_PATH%\platforms\android-36" (
    set PLATFORM=android-36
    call :log "INFO" "Using platform: android-36 (Android 16)"
) else if exist "%SDK_PATH%\platforms\android-34" (
    set PLATFORM=android-34
    call :log "INFO" "Using platform: android-34"
) else (
    call :log "INFO" "Using platform: android-26 (default)"
)

REM 7. Run CMake configuration
call :log "INFO" ""
call :log "INFO" "[7/8] Configuring CMake..."

if not exist "build-android" mkdir "build-android"
cd "build-android"

call :log "INFO" "CMake configuration parameters:"
call :log "INFO" "  - System: Android"
call :log "INFO" "  - NDK: %NDK_PATH%"
call :log "INFO" "  - Platform: %PLATFORM%"
call :log "INFO" "  - ABI: %ABI%"

echo. >> "%LOG_FILE%" 2>&1
echo CMake command: >> "%LOG_FILE%" 2>&1
echo cmake .. ^
    -DCMAKE_SYSTEM_NAME=Android ^
    -DCMAKE_ANDROID_NDK="%NDK_PATH:\=/%" ^
    -DCMAKE_ANDROID_ARCH_ABI=%ABI% ^
    -DCMAKE_ANDROID_PLATFORM=%PLATFORM% ^
    -DCMAKE_BUILD_TYPE=Debug ^
    -DJUCE_DIR="%PROJECT_ROOT:\=/%" ^
    -DCMAKE_MAKE_PROGRAM="%NINJA_PATH:\=/%" >> "%LOG_FILE%" 2>&1
echo. >> "%LOG_FILE%" 2>&1

cmake .. ^
    -DCMAKE_SYSTEM_NAME=Android ^
    -DCMAKE_ANDROID_NDK="%NDK_PATH%" ^
    -DCMAKE_ANDROID_ARCH_ABI=%ABI% ^
    -DCMAKE_ANDROID_PLATFORM=%PLATFORM% ^
    -DCMAKE_BUILD_TYPE=Debug ^
    -DJUCE_DIR="%PROJECT_ROOT%\JUCE" ^
    -DCMAKE_MAKE_PROGRAM="%NINJA_PATH%" >> "%LOG_FILE%" 2>&1

if %errorlevel% neq 0 (
    call :log "ERROR" "CMake configuration failed!"
    call :log "INFO" "Common causes:"
    call :log "INFO" "  1. NDK version incompatible (need r25+)"
    call :log "INFO" "  2. CMake version too old (need 3.22+)"
    call :log "INFO" "  3. Platform version not available"
    goto :error
)
call :log "SUCCESS" "CMake configured successfully"

REM 8. Build
call :log "INFO" ""
call :log "INFO" "[8/8] Building project..."
call :log "INFO" "This may take a few minutes..."

cmake --build . --parallel >> "%LOG_FILE%" 2>&1

if %errorlevel% neq 0 (
    call :log "ERROR" "Build failed!"
    call :log "INFO" "Please check log file for detailed error information"
    goto :error
)

call :log "SUCCESS" "Build successful"

cd "%PROJECT_ROOT%"

call :log "INFO" ""
call :log "INFO" "========================================"
call :log "SUCCESS" "Android build complete!"
call :log "INFO" "========================================"
call :log "INFO" "Log file: %LOG_FILE%"
call :log "INFO" ""
call :log "INFO" "Output directory: build-android/"
call :log "INFO" ""
call :log "INFO" "Install to device:"
call :log "INFO" "  adb install build-android\SuperPitchMonitor.apk"
call :log "INFO" ""

echo.
echo ========================================
echo Build successful!
echo ========================================
echo.
echo Log file: %LOG_FILE%
echo.

pause
exit /b 0

:error
cd "%PROJECT_ROOT%"
call :log "ERROR" ""
call :log "ERROR" "========================================"
call :log "ERROR" "Android build failed"
call :log "ERROR" "========================================"
call :log "ERROR" "Log file: %LOG_FILE%"
call :log "ERROR" ""
call :log "ERROR" "Please check error messages above, or see full log"
call :log "ERROR" ""

echo.
echo ========================================
echo Build failed - See log for details
echo ========================================
echo.
echo Log file: %LOG_FILE%
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


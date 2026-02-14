#!/usr/bin/env pwsh
# SuperPitchMonitor Auto Build Script
# Automatically builds and fixes common issues

$ErrorActionPreference = "Stop"

$PROJECT_ROOT = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
Set-Location $PROJECT_ROOT

$LOG_DIR = "$PROJECT_ROOT\build_logs"
$MAX_RETRIES = 3

if (!(Test-Path $LOG_DIR)) {
    New-Item -ItemType Directory -Force -Path $LOG_DIR | Out-Null
}

function Write-Log {
    param([string]$Message, [string]$Level = "INFO")
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $logLine = "[$timestamp] [$Level] $Message"
    Write-Host $logLine
}

function Build-Windows {
    param([int]$RetryCount = 0)
    
    Write-Log "========================================"
    Write-Log "Building Windows Desktop (Debug)"
    Write-Log "Attempt: $($RetryCount + 1) / $MAX_RETRIES"
    Write-Log "========================================"
    
    $logFile = "$LOG_DIR\build_windows_$(Get-Date -Format 'yyyyMMdd_HHmmss').log"
    
    try {
        # Clean previous build if it exists and this is a retry
        if ($RetryCount -gt 0 -and (Test-Path "build-windows")) {
            Write-Log "Cleaning previous build..." "WARNING"
            Remove-Item -Recurse -Force "build-windows" -ErrorAction SilentlyContinue
        }
        
        # Create build directory
        if (!(Test-Path "build-windows")) {
            New-Item -ItemType Directory -Force -Path "build-windows" | Out-Null
        }
        Set-Location "build-windows"
        
        # Configure
        Write-Log "Configuring CMake..."
        $cmakeOutput = cmake .. -G "Visual Studio 17 2022" -A x64 `
            -DJUCE_DIR="$PROJECT_ROOT\JUCE" `
            -DCMAKE_BUILD_TYPE=Debug 2>&1
        
        $cmakeOutput | Tee-Object -FilePath $logFile
        
        if ($LASTEXITCODE -ne 0) {
            throw "CMake configuration failed"
        }
        
        # Build
        Write-Log "Building..."
        $buildOutput = cmake --build . --config Debug --parallel 2>&1
        $buildOutput | Tee-Object -FilePath $logFile -Append
        
        if ($LASTEXITCODE -ne 0) {
            throw "Build failed"
        }
        
        # Verify output
        if (Test-Path "Debug\SuperPitchMonitor.exe") {
            Write-Log "SUCCESS: Build completed!" "SUCCESS"
            Write-Log "Output: $PWD\Debug\SuperPitchMonitor.exe"
            Set-Location $PROJECT_ROOT
            return $true
        } else {
            throw "Executable not found"
        }
        
    } catch {
        Write-Log "Build failed: $_" "ERROR"
        Write-Log "Log file: $logFile" "ERROR"
        
        # Read log and try to identify common issues
        $logContent = Get-Content $logFile -Raw -ErrorAction SilentlyContinue
        
        if ($logContent -match "JUCE directory not found" -or $logContent -match "JUCE CMakeLists.txt not found") {
            Write-Log "Issue: JUCE not found!" "ERROR"
            Write-Log "Fix: Run: git clone https://github.com/juce-framework/JUCE.git" "INFO"
            return $false
        }
        
        if ($logContent -match "No CMAKE_C_COMPILER could be found" -or $logContent -match "Could not find any instance of Visual Studio") {
            Write-Log "Issue: Visual Studio not found!" "ERROR"
            Write-Log "Fix: Install Visual Studio 2022 with 'Desktop development with C++' workload" "INFO"
            return $false
        }
        
        if ($RetryCount -lt ($MAX_RETRIES - 1)) {
            Write-Log "Retrying in 3 seconds..." "WARNING"
            Start-Sleep -Seconds 3
            Set-Location $PROJECT_ROOT
            return Build-Windows -RetryCount ($RetryCount + 1)
        }
        
        Set-Location $PROJECT_ROOT
        return $false
    }
}

function Build-Android {
    param([int]$RetryCount = 0)
    
    Write-Log "========================================"
    Write-Log "Building Android (Debug)"
    Write-Log "Attempt: $($RetryCount + 1) / $MAX_RETRIES"
    Write-Log "========================================"
    
    $logFile = "$LOG_DIR\build_android_$(Get-Date -Format 'yyyyMMdd_HHmmss').log"
    
    # Find SDK and NDK
    $sdkPaths = @(
        $env:ANDROID_SDK_ROOT,
        $env:ANDROID_HOME,
        "$env:LOCALAPPDATA\Android\Sdk"
    )
    
    $sdkPath = $null
    foreach ($p in $sdkPaths) {
        if ($p -and (Test-Path "$p\platform-tools")) {
            $sdkPath = $p
            break
        }
    }
    
    if (-not $sdkPath) {
        Write-Log "Android SDK not found!" "ERROR"
        Write-Log "Set ANDROID_SDK_ROOT environment variable" "INFO"
        return $false
    }
    
    # Find NDK
    $ndkPath = $null
    if (Test-Path "$sdkPath\ndk") {
        $ndks = Get-ChildItem "$sdkPath\ndk" -Directory | Select-Object -First 1
        if ($ndks) {
            $ndkPath = $ndks.FullName
        }
    }
    
    if (-not $ndkPath) {
        Write-Log "NDK not found!" "ERROR"
        Write-Log "Install via: Android Studio -> SDK Manager -> NDK" "INFO"
        return $false
    }
    
    Write-Log "SDK: $sdkPath"
    Write-Log "NDK: $ndkPath"
    
    try {
        # Clean previous build if retry
        if ($RetryCount -gt 0 -and (Test-Path "build-android")) {
            Write-Log "Cleaning previous build..." "WARNING"
            Remove-Item -Recurse -Force "build-android" -ErrorAction SilentlyContinue
        }
        
        if (!(Test-Path "build-android")) {
            New-Item -ItemType Directory -Force -Path "build-android" | Out-Null
        }
        Set-Location "build-android"
        
        # Configure
        Write-Log "Configuring CMake for Android..."
        $cmakeOutput = cmake .. `
            -DCMAKE_SYSTEM_NAME=Android `
            -DCMAKE_ANDROID_NDK="$ndkPath" `
            -DCMAKE_ANDROID_ARCH_ABI=arm64-v8a `
            -DCMAKE_ANDROID_PLATFORM=android-26 `
            -DCMAKE_BUILD_TYPE=Debug `
            -DJUCE_DIR="$PROJECT_ROOT\JUCE" 2>&1
        
        $cmakeOutput | Tee-Object -FilePath $logFile
        
        if ($LASTEXITCODE -ne 0) {
            throw "CMake configuration failed"
        }
        
        # Build
        Write-Log "Building..."
        $buildOutput = cmake --build . --parallel 2>&1
        $buildOutput | Tee-Object -FilePath $logFile -Append
        
        if ($LASTEXITCODE -ne 0) {
            throw "Build failed"
        }
        
        Write-Log "SUCCESS: Android build completed!" "SUCCESS"
        Set-Location $PROJECT_ROOT
        return $true
        
    } catch {
        Write-Log "Build failed: $_" "ERROR"
        Write-Log "Log file: $logFile" "ERROR"
        
        if ($RetryCount -lt ($MAX_RETRIES - 1)) {
            Write-Log "Retrying in 3 seconds..." "WARNING"
            Start-Sleep -Seconds 3
            Set-Location $PROJECT_ROOT
            return Build-Android -RetryCount ($RetryCount + 1)
        }
        
        Set-Location $PROJECT_ROOT
        return $false
    }
}

# Main
Write-Log ""
Write-Log "SuperPitchMonitor Auto Build System"
Write-Log "===================================="
Write-Log ""

# Build Windows first
$windowsSuccess = Build-Windows

if ($windowsSuccess) {
    Write-Log ""
    Write-Log "Windows build successful! Proceeding to Android..."
    Write-Log ""
    
    # Build Android
    $androidSuccess = Build-Android
    
    if ($androidSuccess) {
        Write-Log ""
        Write-Log "===================================="
        Write-Log "ALL BUILDS SUCCESSFUL!" "SUCCESS"
        Write-Log "===================================="
        exit 0
    } else {
        Write-Log ""
        Write-Log "===================================="
        Write-Log "Android build failed" "ERROR"
        Write-Log "===================================="
        exit 1
    }
} else {
    Write-Log ""
    Write-Log "===================================="
    Write-Log "Windows build failed" "ERROR"
    Write-Log "===================================="
    exit 1
}

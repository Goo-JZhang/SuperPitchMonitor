#!/usr/bin/env pwsh
# Android Build Script with Path Fix

$ErrorActionPreference = "Stop"

$PROJECT_ROOT = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
Set-Location $PROJECT_ROOT

$LOG_DIR = "$PROJECT_ROOT\build_logs"
if (!(Test-Path $LOG_DIR)) { New-Item -ItemType Directory -Force -Path $LOG_DIR | Out-Null }

$LOG_FILE = "$LOG_DIR\build_android_$(Get-Date -Format 'yyyyMMdd_HHmmss').log"

function Write-Log {
    param([string]$Message)
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $logLine = "[$timestamp] $Message"
    Write-Host $logLine
    Add-Content -Path $LOG_FILE -Value $logLine
}

# Find SDK
$sdk = $env:ANDROID_SDK_ROOT
if (-not $sdk) { $sdk = $env:ANDROID_HOME }
if (-not $sdk) { $sdk = "$env:LOCALAPPDATA\Android\Sdk" }

if (-not (Test-Path "$sdk\platform-tools")) {
    Write-Log "ERROR: Android SDK not found"
    exit 1
}

Write-Log "SDK: $sdk"

# Find NDK
$ndk = "$sdk\ndk\25.2.9519653"
if (-not (Test-Path $ndk)) {
    Write-Log "ERROR: NDK not found at $ndk"
    exit 1
}

Write-Log "NDK: $ndk"

# Clean build directory
if (Test-Path "build-android") {
    Remove-Item -Recurse -Force "build-android"
    Write-Log "Cleaned build-android"
}
New-Item -ItemType Directory -Force -Path "build-android" | Out-Null
Set-Location "build-android"

# Configure with forward slashes
$ndkPath = $ndk.Replace('\', '/')
$jucePath = "$PROJECT_ROOT\JUCE".Replace('\', '/')
$cmakePath = "$sdk\cmake\3.22.1\bin\ninja.exe".Replace('\', '/')

Write-Log "Configuring CMake..."
Write-Log "NDK Path: $ndkPath"

$cmakeArgs = @(
    "..",
    "-DCMAKE_SYSTEM_NAME=Android",
    "-DCMAKE_ANDROID_NDK=$ndkPath",
    "-DCMAKE_ANDROID_ARCH_ABI=arm64-v8a",
    "-DCMAKE_ANDROID_PLATFORM=android-26",
    "-DCMAKE_BUILD_TYPE=Debug",
    "-DJUCE_DIR=$jucePath",
    "-G=Ninja",
    "-DCMAKE_MAKE_PROGRAM=$cmakePath"
)

& "C:\Program Files\CMake\bin\cmake.exe" $cmakeArgs 2>&1 | Tee-Object -FilePath $LOG_FILE

if ($LASTEXITCODE -ne 0) {
    Write-Log "ERROR: CMake configuration failed"
    exit 1
}

Write-Log "Building..."
& "C:\Program Files\CMake\bin\cmake.exe" --build . --parallel 2>&1 | Tee-Object -FilePath $LOG_FILE -Append

if ($LASTEXITCODE -ne 0) {
    Write-Log "ERROR: Build failed"
    exit 1
}

Write-Log "SUCCESS: Android build completed!"

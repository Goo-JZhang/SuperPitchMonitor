#!/usr/bin/env pwsh
# SuperPitchMonitor Environment Check Script

$ErrorActionPreference = "Continue"
$LogFile = "build_logs/environment_check.log"

# Create log directory
New-Item -ItemType Directory -Force -Path (Split-Path $LogFile) | Out-Null

function Write-Log {
    param([string]$Message, [string]$Level = "INFO")
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $logLine = "[$timestamp] [$Level] $Message"
    Write-Host $logLine
    Add-Content -Path $LogFile -Value $logLine -Encoding UTF8
}

function Test-Command {
    param([string]$Command)
    try {
        $null = & $Command --version 2>&1
        return $true
    } catch {
        return $false
    }
}

# Clear log
"" | Set-Content -Path $LogFile -Encoding UTF8

Write-Log "========================================"
Write-Log "SuperPitchMonitor Environment Check"
Write-Log "========================================"
Write-Log ""

# 1. OS
Write-Log "[1/8] Checking OS..."
$os = [System.Environment]::OSVersion
Write-Log "  OS: $($os.Platform)"
Write-Log "  Version: $($os.Version)"

# 2. PowerShell
Write-Log ""
Write-Log "[2/8] Checking PowerShell..."
Write-Log "  Version: $($PSVersionTable.PSVersion)"

# 3. CMake
Write-Log ""
Write-Log "[3/8] Checking CMake..."
if (Test-Command "cmake") {
    Write-Log "  [OK] CMake found" "SUCCESS"
    $ver = cmake --version 2>&1 | Select-Object -First 1
    Write-Log "    $ver"
} else {
    Write-Log "  [FAIL] CMake not found!" "ERROR"
    Write-Log "    Install from: https://cmake.org/download/"
}

# 4. Visual Studio
Write-Log ""
Write-Log "[4/8] Checking Visual Studio..."
$vsPaths = @(
    "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat",
    "C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvars64.bat",
    "C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvars64.bat"
)
$foundVS = $false
foreach ($p in $vsPaths) {
    if (Test-Path $p) {
        Write-Log "  [OK] VS found: $p" "SUCCESS"
        $foundVS = $true
        break
    }
}
if (-not $foundVS) {
    Write-Log "  [WARN] VS not found in standard paths" "WARNING"
}

# 5. Android SDK
Write-Log ""
Write-Log "[5/8] Checking Android SDK..."
$sdkPaths = @(
    $env:ANDROID_SDK_ROOT,
    $env:ANDROID_HOME,
    "$env:LOCALAPPDATA\Android\Sdk"
)
$foundSdk = $null
foreach ($p in $sdkPaths) {
    if ($p -and (Test-Path "$p\platform-tools")) {
        $foundSdk = $p
        break
    }
}
if ($foundSdk) {
    Write-Log "  [OK] SDK found: $foundSdk" "SUCCESS"
    
    # Check NDK
    if (Test-Path "$foundSdk\ndk") {
        $ndks = Get-ChildItem "$foundSdk\ndk" -Directory | Select-Object -ExpandProperty Name
        Write-Log "  [OK] NDK: $($ndks -join ', ')" "SUCCESS"
    } else {
        Write-Log "  [FAIL] NDK not installed!" "ERROR"
        Write-Log "    Install via: Android Studio -> SDK Manager -> NDK"
    }
    
    # Check CMake
    if (Test-Path "$foundSdk\cmake") {
        $cms = Get-ChildItem "$foundSdk\cmake" -Directory | Select-Object -ExpandProperty Name
        Write-Log "  [OK] CMake: $($cms -join ', ')" "SUCCESS"
    } else {
        Write-Log "  [FAIL] CMake not installed!" "ERROR"
        Write-Log "    Install via: Android Studio -> SDK Manager -> CMake"
    }
} else {
    Write-Log "  [FAIL] SDK not found!" "ERROR"
    Write-Log "    Set ANDROID_SDK_ROOT environment variable"
}

# 6. Git
Write-Log ""
Write-Log "[6/8] Checking Git..."
if (Test-Command "git") {
    Write-Log "  [OK] Git found" "SUCCESS"
} else {
    Write-Log "  [WARN] Git not found (optional)" "WARNING"
}

# 7. JUCE
Write-Log ""
Write-Log "[7/8] Checking JUCE..."
$juceDir = Join-Path $PSScriptRoot "..\JUCE"
if (Test-Path "$juceDir\CMakeLists.txt") {
    Write-Log "  [OK] JUCE found" "SUCCESS"
} else {
    Write-Log "  [FAIL] JUCE not found!" "ERROR"
    Write-Log "    Run: git clone https://github.com/juce-framework/JUCE.git"
}

# 8. Project files
Write-Log ""
Write-Log "[8/8] Checking project files..."
$files = @("..\CMakeLists.txt", "..\Source\Main.cpp")
foreach ($f in $files) {
    $fp = Join-Path $PSScriptRoot $f
    if (Test-Path $fp) {
        Write-Log "  [OK] $f"
    } else {
        Write-Log "  [FAIL] $f missing" "ERROR"
    }
}

# Summary
Write-Log ""
Write-Log "========================================"
Write-Log "Check Complete"
Write-Log "========================================"
Write-Log "Log: $((Resolve-Path $LogFile))"
Write-Log ""
if ($foundSdk -and $foundVS) {
    Write-Log "Status: READY for Windows and Android builds" "SUCCESS"
} elseif ($foundVS) {
    Write-Log "Status: READY for Windows only" "WARNING"
} else {
    Write-Log "Status: MISSING required tools" "ERROR"
}
Write-Log ""
Write-Log "Next steps:"
Write-Log "  1. Run: scripts\build_windows.bat"
Write-Log "  2. Or: scripts\build_android.bat"

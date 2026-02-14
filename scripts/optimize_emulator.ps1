# Android Emulator Optimization Script
# Reduces resource usage to prevent system crashes

$logFile = "C:\SuperPitchMonitor\build_logs\emulator_optimization.log"
$androidHome = $env:ANDROID_SDK_ROOT
if (-not $androidHome) {
    $androidHome = $env:ANDROID_HOME
}
if (-not $androidHome) {
    $androidHome = "$env:LOCALAPPDATA\Android\Sdk"
}

function Write-Log {
    param($Message, $Level = "INFO")
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $logEntry = "[$timestamp] [$Level] $Message"
    Write-Host $logEntry
    Add-Content -Path $logFile -Value $logEntry -ErrorAction SilentlyContinue
}

# Ensure log directory exists
$logDir = Split-Path $logFile -Parent
if (!(Test-Path $logDir)) {
    New-Item -ItemType Directory -Path $logDir -Force | Out-Null
}

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "   Android Emulator Optimizer" -ForegroundColor Cyan
Write-Host "========================================`n" -ForegroundColor Cyan

Write-Log "Emulator optimization started"

# Find AVD directory
$avdDir = "$env:USERPROFILE\.android\avd"
if (!(Test-Path $avdDir)) {
    Write-Host "❌ No AVD directory found at: $avdDir" -ForegroundColor Red
    Write-Host "   You may not have created any virtual devices yet." -ForegroundColor Yellow
    exit 1
}

# List available AVDs
$avdConfigs = Get-ChildItem -Path $avdDir -Filter "*.ini" | Where-Object { $_.Name -notmatch "^hardware" }

if ($avdConfigs.Count -eq 0) {
    Write-Host "❌ No AVD configurations found." -ForegroundColor Red
    Write-Host "   Please create a virtual device in Android Studio first." -ForegroundColor Yellow
    exit 1
}

Write-Host "Found AVDs:" -ForegroundColor Green
for ($i = 0; $i -lt $avdConfigs.Count; $i++) {
    $avdName = $avdConfigs[$i].BaseName
    Write-Host "  [$i] $avdName"
}

Write-Host ""
$selection = Read-Host "Select AVD to optimize (0-$($avdConfigs.Count-1), or 'a' for all)"

$avdsToOptimize = @()
if ($selection -eq 'a') {
    $avdsToOptimize = $avdConfigs
} else {
    $idx = [int]$selection
    if ($idx -ge 0 -and $idx -lt $avdConfigs.Count) {
        $avdsToOptimize = @($avdConfigs[$idx])
    } else {
        Write-Host "❌ Invalid selection" -ForegroundColor Red
        exit 1
    }
}

foreach ($avdConfig in $avdsToOptimize) {
    $avdName = $avdConfig.BaseName
    $configIni = $avdConfig.FullName
    $hardwareIni = "$avdDir\$avdName.avd\hardware-qemu.ini"
    
    Write-Host "`nOptimizing: $avdName" -ForegroundColor Cyan
    Write-Log "Optimizing AVD: $avdName"
    
    # Backup original config
    $backupPath = "$configIni.backup.$(Get-Date -Format 'yyyyMMddHHmmss')"
    Copy-Item $configIni $backupPath -Force
    Write-Host "  📦 Backup created: $backupPath" -ForegroundColor DarkGray
    
    # Read current config
    $config = Get-Content $configIni
    $newConfig = @()
    
    # Optimization settings
    $optimizations = @{
        "hw.ramSize" = "2048"           # Reduced from typical 4-6GB
        "hw.heapSize" = "576"           # Reduced heap
        "hw.cpu.ncore" = "2"            # Limit to 2 cores (was probably 4-8)
        "hw.gpu.enabled" = "yes"        # Keep GPU accel but...
        "hw.gpu.mode" = "swiftshader_indirect"  # Use safer GPU mode
        "hw.audioInput" = "no"          # Disable audio input if not needed
        "hw.mainKeys" = "no"            # Software keys use less resources
        "hw.keyboard" = "yes"           # Use hardware keyboard
        "hw.dPad" = "no"
        "hw.trackBall" = "no"
        "hw.sensors.orientation" = "no" # Disable unnecessary sensors
        "hw.sensors.proximity" = "no"
        "disk.dataPartition.size" = "2048M"  # Limit disk size
    }
    
    $modifiedSettings = @()
    $existingKeys = @{}
    
    # Parse existing config
    foreach ($line in $config) {
        if ($line -match "^\s*([^=]+)\s*=\s*(.+)\s*$") {
            $key = $matches[1].Trim()
            $existingKeys[$key] = $true
            
            if ($optimizations.ContainsKey($key)) {
                $oldValue = $matches[2].Trim()
                $newValue = $optimizations[$key]
                $newConfig += "$key=$newValue"
                $modifiedSettings += "$key`: $oldValue -> $newValue"
            } else {
                $newConfig += $line
            }
        } else {
            $newConfig += $line
        }
    }
    
    # Add missing settings
    foreach ($key in $optimizations.Keys) {
        if (-not $existingKeys.ContainsKey($key)) {
            $newConfig += "$key=$($optimizations[$key])"
            $modifiedSettings += "$key`: (added) -> $($optimizations[$key])"
        }
    }
    
    # Write new config
    $newConfig | Out-File $configIni -Encoding UTF8
    
    # Display changes
    Write-Host "  Changes applied:" -ForegroundColor Green
    foreach ($change in $modifiedSettings) {
        Write-Host "    ✓ $change" -ForegroundColor DarkGray
    }
    
    Write-Log "Optimization applied to $avdName"
}

# Additional system-wide emulator settings
Write-Host "`n📋 Additional Recommendations:" -ForegroundColor Cyan
Write-Host @"

  1. In Android Studio:
     - File → Settings → Emulator
     - ☑️ 'Launch in a tool window' (reduces window management overhead)
     - ☐ 'Enable Emulator in Tool Window' (uncheck if crashing)

  2. Emulator startup flags (add to launcher):
     -gpu swiftshader_indirect  (Software rendering, more stable)
     -no-snapshot-save         (Don't save state, faster shutdown)
     -no-boot-anim             (Skip boot animation)

  3. Windows-specific:
     - Add Android Studio and emulator to Windows Defender exclusions
     - Close browser tabs before starting emulator
     - Use 'Balanced' power plan instead of 'High Performance'

  4. Alternative: Use Windows desktop build for development:
     scripts\build_windows.bat
     
     Only use emulator for final Android-specific testing.

"@

Write-Host "✅ Optimization complete!" -ForegroundColor Green
Write-Log "Emulator optimization completed"

Write-Host "`nWould you like to:" -ForegroundColor Cyan
Write-Host "  [1] Start system health monitoring"
Write-Host "  [2] Open Android Studio AVD Manager"
Write-Host "  [3] Exit"
$choice = Read-Host "Select (1-3)"

switch ($choice) {
    "1" { 
        Write-Host "`nStarting system monitoring... (Ctrl+C to stop)`n" -ForegroundColor Cyan
        & "$PSScriptRoot\check_system_health.ps1" -Continuous 
    }
    "2" { 
        if (Test-Path "$androidHome\..\Android Studio\bin\studio64.exe") {
            Start-Process "$androidHome\..\Android Studio\bin\studio64.exe" -ArgumentList "--navigate", "avd-manager"
        } else {
            Write-Host "Please open Android Studio manually and go to Tools → Device Manager" -ForegroundColor Yellow
        }
    }
}

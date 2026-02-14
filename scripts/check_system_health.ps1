# System Health Check Script for Android Emulator Issues
# Run this before using Android Emulator to check for potential problems

param(
    [switch]$Continuous
)

$logFile = "C:\SuperPitchMonitor\build_logs\system_health.log"

function Write-Log {
    param($Message, $Level = "INFO")
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $logEntry = "[$timestamp] [$Level] $Message"
    Write-Host $logEntry
    Add-Content -Path $logFile -Value $logEntry -ErrorAction SilentlyContinue
}

function Get-CpuTemperature {
    try {
        $temp = Get-WmiObject MSAcpi_ThermalZoneTemperature -Namespace "root/wmi" -ErrorAction SilentlyContinue | 
                Select-Object -First 1 -ExpandProperty CurrentTemperature
        if ($temp) {
            return [math]::Round($temp / 10 - 273.15, 1)
        }
    } catch {}
    return $null
}

function Get-AvailableMemory {
    $os = Get-WmiObject -Class Win32_OperatingSystem
    $total = [math]::Round($os.TotalVisibleMemorySize / 1MB, 2)
    $free = [math]::Round($os.FreePhysicalMemory / 1MB, 2)
    $usedPercent = [math]::Round((($total - $free) / $total) * 100, 1)
    return @{ Total = $total; Free = $free; UsedPercent = $usedPercent }
}

function Get-CpuInfo {
    $cpu = Get-WmiObject -Class Win32_Processor | Select-Object -First 1
    $load = Get-WmiObject -Class Win32_Processor | Measure-Object -Property LoadPercentage -Average | Select-Object -ExpandProperty Average
    return @{ Name = $cpu.Name; Load = [math]::Round($load, 1) }
}

function Get-PowerPlan {
    try {
        $plan = powercfg /getactivescheme 2>$null
        return $plan
    } catch {
        return "Unknown"
    }
}

# Ensure log directory exists
$logDir = Split-Path $logFile -Parent
if (!(Test-Path $logDir)) {
    New-Item -ItemType Directory -Path $logDir -Force | Out-Null
}

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "   System Health Check for Android Dev" -ForegroundColor Cyan
Write-Host "========================================`n" -ForegroundColor Cyan

Write-Log "System Health Check Started" "INFO"

# CPU Info
$cpu = Get-CpuInfo
Write-Host "CPU: " -NoNewline
Write-Host $cpu.Name -ForegroundColor Green
Write-Host "Current Load: " -NoNewline
if ($cpu.Load -gt 80) {
    Write-Host "$($cpu.Load)%" -ForegroundColor Red
} elseif ($cpu.Load -gt 50) {
    Write-Host "$($cpu.Load)%" -ForegroundColor Yellow
} else {
    Write-Host "$($cpu.Load)%" -ForegroundColor Green
}
Write-Log "CPU Load: $($cpu.Load)%"

# Memory
$mem = Get-AvailableMemory
Write-Host "`nMemory:" -ForegroundColor Cyan
Write-Host "  Total: $($mem.Total) GB"
Write-Host "  Free: " -NoNewline
if ($mem.Free -lt 4) {
    Write-Host "$($mem.Free) GB" -ForegroundColor Red
    Write-Log "Low free memory: $($mem.Free) GB" "WARNING"
} else {
    Write-Host "$($mem.Free) GB" -ForegroundColor Green
}
Write-Host "  Used: $($mem.UsedPercent)%"

# Temperature
$temp = Get-CpuTemperature
Write-Host "`nTemperature:" -ForegroundColor Cyan
if ($temp) {
    Write-Host "  CPU: " -NoNewline
    if ($temp -gt 85) {
        Write-Host "$temp°C - TOO HOT!" -ForegroundColor Red
        Write-Log "CPU temperature critical: $temp°C" "ERROR"
    } elseif ($temp -gt 70) {
        Write-Host "$temp°C - Warm" -ForegroundColor Yellow
        Write-Log "CPU temperature high: $temp°C" "WARNING"
    } else {
        Write-Host "$temp°C - Normal" -ForegroundColor Green
    }
} else {
    Write-Host "  Unable to read temperature (may need admin rights)" -ForegroundColor Yellow
}

# Power Plan
Write-Host "`nPower Plan:" -ForegroundColor Cyan
$powerPlan = Get-PowerPlan
if ($powerPlan -match "High performance|卓越性能") {
    Write-Host "  $powerPlan" -ForegroundColor Yellow
    Write-Host "  Note: High performance mode generates more heat" -ForegroundColor DarkYellow
} else {
    Write-Host "  $powerPlan" -ForegroundColor Green
}

# Check for Android Studio processes
Write-Host "`nAndroid Studio Processes:" -ForegroundColor Cyan
$androidProcesses = Get-Process | Where-Object { 
    $_.ProcessName -match "android|studio|emulator|qemu|gradle" -and 
    $_.ProcessName -notmatch "androidssh|androidssl"
} | Select-Object ProcessName, Id, @{N="Memory(MB)";E={[math]::Round($_.WorkingSet64/1MB,1)}}

if ($androidProcesses) {
    $androidProcesses | Format-Table -AutoSize
    $totalAndroidMem = ($androidProcesses | Measure-Object -Property "Memory(MB)" -Sum).Sum
    Write-Host "Total Android-related Memory: $totalAndroidMem MB" -ForegroundColor Yellow
    Write-Log "Android processes using $totalAndroidMem MB"
    
    if ($totalAndroidMem -gt 4096) {
        Write-Host "WARNING: Android processes using >4GB RAM" -ForegroundColor Red
        Write-Log "High memory usage from Android processes" "WARNING"
    }
} else {
    Write-Host "  No Android processes running" -ForegroundColor Green
}

# Recommendations
Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "   Recommendations" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

$issues = @()

if ($cpu.Load -gt 80) {
    $issues += "⚠ CPU load is high. Close unnecessary programs before starting emulator."
}

if ($mem.Free -lt 4) {
    $issues += "⚠ Low memory. Consider closing browser tabs or other apps."
}

if ($temp -and $temp -gt 80) {
    $issues += "⚠ CPU temperature is high. Check cooling before intensive tasks."
}

if ($totalAndroidMem -gt 4096) {
    $issues += "⚠ Android Studio is already using significant memory."
}

if ($issues.Count -eq 0) {
    Write-Host "✅ System looks healthy for Android development!" -ForegroundColor Green
} else {
    foreach ($issue in $issues) {
        Write-Host $issue -ForegroundColor Yellow
    }
}

Write-Host "`n💡 Tips to prevent crashes:" -ForegroundColor Cyan
Write-Host "  1. Use Windows desktop build for daily development:"
Write-Host "     scripts\build_windows.bat" -ForegroundColor DarkGray
Write-Host "  2. Reduce emulator RAM to 2GB in AVD Manager"
Write-Host "  3. Limit emulator to 2 CPU cores"
Write-Host "  4. Use physical device for testing when possible"
Write-Host ""

Write-Log "System Health Check Completed" "INFO"

# Continuous monitoring mode
if ($Continuous) {
    Write-Host "Starting continuous monitoring (Ctrl+C to stop)...`n" -ForegroundColor Cyan
    while ($true) {
        $temp = Get-CpuTemperature
        $mem = Get-AvailableMemory
        $cpu = Get-CpuInfo
        
        $status = "CPU: $($cpu.Load)% | Temp: $temp°C | Free RAM: $($mem.Free)GB"
        
        if ($temp -gt 90 -or $mem.Free -lt 2 -or $cpu.Load -gt 95) {
            Write-Host "⚠ CRITICAL: $status" -ForegroundColor Red
            Write-Log "CRITICAL: $status" "ERROR"
        } elseif ($temp -gt 80 -or $mem.Free -lt 4) {
            Write-Host "⚠ WARNING: $status" -ForegroundColor Yellow
        } else {
            Write-Host "✅ OK: $status" -ForegroundColor Green
        }
        
        Start-Sleep -Seconds 5
    }
}

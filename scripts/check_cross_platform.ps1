# Cross-Platform Compatibility Checker
# This script scans source code for potential platform compatibility issues

param(
    [switch]$Strict,  # Fail on warnings too
    [switch]$Fix      # Try to auto-fix issues
)

$projectRoot = "C:\SuperPitchMonitor"
$sourceDir = "$projectRoot\Source"
$logFile = "$projectRoot\build_logs\cross_platform_check.log"

$ErrorCount = 0
$WarningCount = 0

function Write-Log {
    param($Message, $Level = "INFO")
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $logEntry = "[$timestamp] [$Level] $Message"
    Write-Host $logEntry
    Add-Content -Path $logFile -Value $logEntry -ErrorAction SilentlyContinue
}

function Add-Issue {
    param($File, $Line, $Message, $IsError = $true)
    
    if ($IsError) {
        $script:ErrorCount++
        Write-Host "  ❌ ERROR: $File`:$Line - $Message" -ForegroundColor Red
        Write-Log "$File`:$Line - $Message" "ERROR"
    } else {
        $script:WarningCount++
        Write-Host "  ⚠️ WARNING: $File`:$Line - $Message" -ForegroundColor Yellow
        Write-Log "$File`:$Line - $Message" "WARNING"
    }
}

# Ensure log directory exists
$logDir = Split-Path $logFile -Parent
if (!(Test-Path $logDir)) {
    New-Item -ItemType Directory -Path $logDir -Force | Out-Null
}

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Cross-Platform Compatibility Check" -ForegroundColor Cyan
Write-Host "========================================`n" -ForegroundColor Cyan

Write-Log "Starting cross-platform compatibility check"

# Get all source files
$sourceFiles = Get-ChildItem -Path $sourceDir -Include "*.cpp","*.h","*.hpp" -Recurse

Write-Host "Scanning $($sourceFiles.Count) source files...`n" -ForegroundColor White

foreach ($file in $sourceFiles) {
    $content = Get-Content $file.FullName
    $relativePath = $file.FullName.Replace($projectRoot, "").TrimStart("\")
    
    for ($i = 0; $i -lt $content.Count; $i++) {
        $line = $content[$i]
        $lineNum = $i + 1
        
        # Skip comments
        if ($line -match "^\s*//" -or $line -match "^\s*/\*") {
            continue
        }
        
        # Check 1: Windows API calls
        if ($line -match "\b(CreateFile|CreateDirectory|RegOpen|WinExec|ShellExecute)\s*\\(") {
            Add-Issue $relativePath $lineNum "Windows API call detected" $true
        }
        
        # Check 2: POSIX API calls (not covered by JUCE)
        if ($line -match "\b(fork|exec|pthread_|open\\(|close\\()\s*\\(") {
            Add-Issue $relativePath $lineNum "POSIX API call detected" $true
        }
        
        # Check 3: Hardcoded paths with backslash
        if ($line -match '"[^"]*\\[^"]*"' -and $line -notmatch "\\\\[rn]") {
            # Exclude escape sequences like \n, \r, \t
            if ($line -notmatch "\\\\[nrt0]") {
                Add-Issue $relativePath $lineNum "Hardcoded backslash path" $false
            }
        }
        
        # Check 4: std::filesystem (C++17, may not be fully supported)
        if ($line -match "std::filesystem|std::path") {
            Add-Issue $relativePath $lineNum "std::filesystem usage (use juce::File instead)" $false
        }
        
        # Check 5: std::thread (use juce::Thread)
        if ($line -match "std::thread|std::async|std::mutex|std::lock_guard") {
            Add-Issue $relativePath $lineNum "std::thread usage (use juce::Thread for consistency)" $false
        }
        
        # Check 6: Raw platform defines outside PlatformUtils
        if ($line -match "#if.*(WIN32|WINDOWS|__ANDROID__|__linux__|__APPLE__)") {
            # Allow in PlatformUtils files
            if ($relativePath -notmatch "PlatformUtils") {
                Add-Issue $relativePath $lineNum "Platform define outside PlatformUtils" $false
            }
        }
        
        # Check 7: JUCE platform defines are OK
        # JUCE_WINDOWS, JUCE_ANDROID, etc. are preferred
        
        # Check 8: fopen/fclose/fread (use juce::FileInputStream)
        if ($line -match "\b(fopen|fclose|fread|fwrite)\s*\\(") {
            Add-Issue $relativePath $lineNum "C file API (use juce::File classes)" $true
        }
        
        # Check 9: printf/sprintf (use juce::Logger or DBG)
        if ($line -match "\b(sprintf|printf|fprintf|scanf)\s*\\(") {
            Add-Issue $relativePath $lineNum "C printf functions (use juce facilities)" $false
        }
        
        # Check 10: getenv (use JUCE facilities)
        if ($line -match "\bgetenv\s*\\(") {
            Add-Issue $relativePath $lineNum "getenv usage (not portable)" $false
        }
    }
}

# Check for PlatformUtils existence
Write-Host "`nChecking PlatformUtils integration..." -ForegroundColor Yellow

if (!(Test-Path "$sourceDir\Utils\PlatformUtils.h")) {
    Add-Issue "Missing" 0 "PlatformUtils.h not found" $true
} else {
    Write-Host "  ✅ PlatformUtils.h found" -ForegroundColor Green
}

# Summary
Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "  Summary" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

if ($ErrorCount -eq 0 -and $WarningCount -eq 0) {
    Write-Host "  ✅ No compatibility issues found!" -ForegroundColor Green
    Write-Log "Check completed - no issues found"
    exit 0
} else {
    Write-Host "  Errors: $ErrorCount" -ForegroundColor $(if ($ErrorCount -gt 0) { "Red" } else { "Green" })
    Write-Host "  Warnings: $WarningCount" -ForegroundColor $(if ($WarningCount -gt 0) { "Yellow" } else { "Green" })
    
    if ($ErrorCount -gt 0) {
        Write-Host "`n  ❌ FAILED: Critical compatibility issues detected!" -ForegroundColor Red
        Write-Log "Check completed with $ErrorCount errors and $WarningCount warnings" "ERROR"
        exit 1
    } elseif ($Strict) {
        Write-Host "`n  ⚠️ FAILED: Warnings treated as errors (Strict mode)" -ForegroundColor Yellow
        Write-Log "Check completed with $WarningCount warnings (Strict mode)" "WARNING"
        exit 1
    } else {
        Write-Host "`n  ⚠️ PASSED with warnings" -ForegroundColor Yellow
        Write-Log "Check completed with $WarningCount warnings"
        exit 0
    }
}

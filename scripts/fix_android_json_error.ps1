# Fix Android Studio JSON Parsing Error in C/C++ Configuration
# This script cleans corrupted .cxx cache files

$projectRoot = "C:\SuperPitchMonitor"
$logFile = "$projectRoot\build_logs\fix_json_error.log"

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

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Android JSON Error Fix Tool" -ForegroundColor Cyan
Write-Host "========================================`n" -ForegroundColor Cyan

Write-Log "Starting JSON error fix"

# Step 1: Close Android Studio processes
Write-Host "Step 1: Checking for Android Studio processes..." -ForegroundColor Yellow
$androidProcesses = Get-Process | Where-Object { 
    $_.ProcessName -match "studio|emulator|qemu|gradle" -and 
    $_.ProcessName -notmatch "androidssh|androidssl"
}

if ($androidProcesses) {
    Write-Host "Found Android processes. Please close Android Studio manually and press Enter..." -ForegroundColor Red
    Read-Host
}

# Step 2: Clean .cxx directories
Write-Host "`nStep 2: Cleaning .cxx cache directories..." -ForegroundColor Yellow

$cxxPaths = @(
    "$projectRoot\build-android\.cxx",
    "$projectRoot\build-android\app\.cxx",
    "$projectRoot\Builds\Android\.cxx",
    "$projectRoot\app\.cxx"
)

foreach ($path in $cxxPaths) {
    if (Test-Path $path) {
        Write-Host "  Removing: $path" -ForegroundColor DarkGray
        try {
            Remove-Item -Path $path -Recurse -Force -ErrorAction Stop
            Write-Log "Removed: $path" "SUCCESS"
        } catch {
            Write-Log "Failed to remove $path`: $_" "ERROR"
            Write-Host "  ⚠️ Failed to remove (may be in use)" -ForegroundColor Red
        }
    }
}

# Step 3: Clean build directories
Write-Host "`nStep 3: Cleaning build directories..." -ForegroundColor Yellow

$buildPaths = @(
    "$projectRoot\build-android\app\intermediates\cmake",
    "$projectRoot\build-android\app\intermediates\cxx",
    "$projectRoot\build-android\app\.gradle"
)

foreach ($path in $buildPaths) {
    if (Test-Path $path) {
        Write-Host "  Removing: $path" -ForegroundColor DarkGray
        try {
            Remove-Item -Path $path -Recurse -Force -ErrorAction SilentlyContinue
            Write-Log "Removed: $path" "SUCCESS"
        } catch {
            Write-Log "Failed to remove $path`: $_" "WARNING"
        }
    }
}

# Step 4: Clean Gradle daemon cache
Write-Host "`nStep 4: Stopping Gradle daemon..." -ForegroundColor Yellow
try {
    & gradlew --stop 2>$null
    Start-Sleep -Seconds 2
    Write-Log "Gradle daemon stopped"
} catch {
    Write-Log "Gradle daemon may not be running"
}

# Step 5: Clean externalNativeBuild
Write-Host "`nStep 5: Cleaning external native build files..." -ForegroundColor Yellow

$nativeBuildPaths = @(
    "$projectRoot\build-android\app\externalNativeBuild",
    "$projectRoot\Builds\Android\app\externalNativeBuild"
)

foreach ($path in $nativeBuildPaths) {
    if (Test-Path $path) {
        Write-Host "  Removing: $path" -ForegroundColor DarkGray
        Remove-Item -Path $path -Recurse -Force -ErrorAction SilentlyContinue
    }
}

# Step 6: Verify CMakeLists.txt exists and is valid
Write-Host "`nStep 6: Verifying CMakeLists.txt..." -ForegroundColor Yellow
$cmakeFile = "$projectRoot\CMakeLists.txt"
if (Test-Path $cmakeFile) {
    $content = Get-Content $cmakeFile -Raw -ErrorAction SilentlyContinue
    if ($content -and $content.Length -gt 0) {
        Write-Host "  ✅ CMakeLists.txt is valid" -ForegroundColor Green
        Write-Log "CMakeLists.txt validation passed"
    } else {
        Write-Host "  ❌ CMakeLists.txt is empty!" -ForegroundColor Red
        Write-Log "CMakeLists.txt is empty" "ERROR"
    }
} else {
    Write-Host "  ❌ CMakeLists.txt not found!" -ForegroundColor Red
    Write-Log "CMakeLists.txt not found" "ERROR"
}

# Step 7: Create a proper local.properties if needed
Write-Host "`nStep 7: Checking Android SDK configuration..." -ForegroundColor Yellow

$localProps = "$projectRoot\local.properties"
$androidSdk = $env:ANDROID_SDK_ROOT
if (-not $androidSdk) {
    $androidSdk = $env:ANDROID_HOME
}
if (-not $androidSdk) {
    $androidSdk = "$env:LOCALAPPDATA\Android\Sdk"
}

if (Test-Path $androidSdk) {
    $propContent = "sdk.dir=$($androidSdk -replace '\\', '\\\\')"
    Set-Content -Path $localProps -Value $propContent -Force
    Write-Host "  ✅ Created/updated local.properties" -ForegroundColor Green
    Write-Host "     SDK Path: $androidSdk" -ForegroundColor DarkGray
    Write-Log "local.properties configured with SDK: $androidSdk"
} else {
    Write-Host "  ⚠️ Android SDK not found at: $androidSdk" -ForegroundColor Red
    Write-Log "Android SDK not found" "ERROR"
}

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "  Cleanup Complete!" -ForegroundColor Green
Write-Host "========================================`n" -ForegroundColor Cyan

Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "  1. Open Android Studio" -ForegroundColor White
Write-Host "  2. File → Invalidate Caches..." -ForegroundColor White
Write-Host "  3. Check 'Invalidate and Restart'" -ForegroundColor White
Write-Host "  4. Wait for Gradle sync to complete" -ForegroundColor White
Write-Host "  5. Build → Make Project (Ctrl+F9)" -ForegroundColor White
Write-Host ""

Write-Log "JSON error fix completed"

# Optional: Run gradlew clean if available
Write-Host "Would you like to run 'gradlew clean' now? (y/n)" -ForegroundColor Yellow
$choice = Read-Host
if ($choice -eq 'y') {
    Write-Host "Running gradlew clean..." -ForegroundColor Cyan
    Set-Location "$projectRoot\build-android"
    & .\gradlew clean 2>&1 | Tee-Object -FilePath $logFile -Append
    Write-Host "`ngradlew clean completed" -ForegroundColor Green
}

Write-Host "`nPress any key to exit..." -ForegroundColor DarkGray
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")

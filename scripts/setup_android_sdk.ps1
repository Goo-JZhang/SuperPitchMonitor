#!/usr/bin/env pwsh
# Android SDK 环境配置助手
# 帮助安装缺少的 NDK 和 Command-line Tools

$ErrorActionPreference = "Stop"

function Write-ColorOutput {
    param(
        [string]$Message,
        [string]$Color = "White"
    )
    Write-Host $Message -ForegroundColor $Color
}

Write-ColorOutput "========================================" "Cyan"
Write-ColorOutput "SuperPitchMonitor Android SDK 配置助手" "Cyan"
Write-ColorOutput "========================================" "Cyan"
Write-ColorOutput ""

# 1. 找到 Android SDK
Write-ColorOutput "[1/5] 查找 Android SDK..." "Yellow"

$sdkPaths = @(
    $env:ANDROID_SDK_ROOT,
    $env:ANDROID_HOME,
    "$env:LOCALAPPDATA\Android\Sdk",
    "$env:USERPROFILE\AppData\Local\Android\Sdk",
    "C:\Android\Sdk"
)

$foundSdk = $null
foreach ($path in $sdkPaths) {
    if ($path -and (Test-Path "$path\platform-tools\adb.exe" -ErrorAction SilentlyContinue)) {
        $foundSdk = $path
        break
    }
}

if (-not $foundSdk) {
    Write-ColorOutput "错误: 无法找到 Android SDK!" "Red"
    Write-ColorOutput "请确保 Android Studio 已安装，或手动设置 ANDROID_SDK_ROOT 环境变量" "Yellow"
    exit 1
}

Write-ColorOutput "找到 SDK: $foundSdk" "Green"

# 2. 检查 sdkmanager
Write-ColorOutput ""
Write-ColorOutput "[2/5] 检查 Command-line Tools..." "Yellow"

$sdkManagerPaths = @(
    "$foundSdk\cmdline-tools\latest\bin\sdkmanager.bat",
    "$foundSdk\cmdline-tools\bin\sdkmanager.bat",
    "$foundSdk\tools\bin\sdkmanager.bat"
)

$sdkManager = $null
foreach ($path in $sdkManagerPaths) {
    if (Test-Path $path) {
        $sdkManager = $path
        break
    }
}

if (-not $sdkManager) {
    Write-ColorOutput "Command-line Tools 未安装!" "Red"
    Write-ColorOutput ""
    Write-ColorOutput "解决方法:" "Yellow"
    Write-ColorOutput "1. 打开 Android Studio" "White"
    Write-ColorOutput "2. Tools -> SDK Manager" "White"
    Write-ColorOutput "3. 切换到 'SDK Tools' 标签" "White"
    Write-ColorOutput "4. 勾选 'Android SDK Command-line Tools (latest)'" "White"
    Write-ColorOutput "5. 点击 'Apply' 安装" "White"
    Write-ColorOutput ""
    Write-ColorOutput "或者手动下载:" "Yellow"
    Write-ColorOutput "https://developer.android.com/studio#command-tools" "White"
    exit 1
}

Write-ColorOutput "找到 sdkmanager: $sdkManager" "Green"

# 3. 检查并安装 NDK
Write-ColorOutput ""
Write-ColorOutput "[3/5] 检查 NDK..." "Yellow"

$ndkPath = "$foundSdk\ndk"
$hasNdk = $false
$ndkVersion = $null

if (Test-Path $ndkPath) {
    $ndkVersions = Get-ChildItem $ndkPath -Directory | Select-Object -ExpandProperty Name
    if ($ndkVersions) {
        $hasNdk = $true
        $ndkVersion = $ndkVersions[0]
        Write-ColorOutput "找到 NDK 版本: $ndkVersion" "Green"
    }
}

if (-not $hasNdk) {
    Write-ColorOutput "NDK 未安装!" "Yellow"
    Write-ColorOutput ""
    Write-ColorOutput "是否自动安装 NDK r25c? (Y/n)" "Cyan"
    $response = Read-Host
    
    if ($response -eq '' -or $response -eq 'Y' -or $response -eq 'y') {
        Write-ColorOutput "正在安装 NDK r25c..." "Yellow"
        & $sdkManager "ndk;25.2.9519653" --sdk_root=$foundSdk
        
        if ($LASTEXITCODE -eq 0) {
            Write-ColorOutput "NDK 安装成功!" "Green"
            $ndkVersion = "25.2.9519653"
        } else {
            Write-ColorOutput "NDK 安装失败，请手动安装" "Red"
            Write-ColorOutput "在 Android Studio 中: SDK Manager -> SDK Tools -> NDK (Side by side)" "Yellow"
        }
    }
} else {
    # 检查 NDK 版本
    $ndkMajorVersion = [int]($ndkVersion.Split('.')[0])
    if ($ndkMajorVersion -lt 25) {
        Write-ColorOutput "警告: NDK 版本较旧 ($ndkVersion)，建议升级到 r25+" "Yellow"
        Write-ColorOutput ""
        Write-ColorOutput "是否升级 NDK? (y/N)" "Cyan"
        $response = Read-Host
        
        if ($response -eq 'Y' -or $response -eq 'y') {
            & $sdkManager "ndk;25.2.9519653" --sdk_root=$foundSdk
        }
    }
}

# 4. 检查 CMake
Write-ColorOutput ""
Write-ColorOutput "[4/5] 检查 CMake..." "Yellow"

$cmakePath = "$foundSdk\cmake"
$hasCmake = $false

if (Test-Path $cmakePath) {
    $cmakeVersions = Get-ChildItem $cmakePath -Directory | Select-Object -ExpandProperty Name
    if ($cmakeVersions) {
        $hasCmake = $true
        Write-ColorOutput "找到 CMake 版本: $($cmakeVersions -join ', ')" "Green"
    }
}

if (-not $hasCmake) {
    Write-ColorOutput "CMake 工具未安装!" "Yellow"
    Write-ColorOutput ""
    Write-ColorOutput "是否安装 CMake 3.22? (Y/n)" "Cyan"
    $response = Read-Host
    
    if ($response -eq '' -or $response -eq 'Y' -or $response -eq 'y') {
        Write-ColorOutput "正在安装 CMake 3.22.1..." "Yellow"
        & $sdkManager "cmake;3.22.1" --sdk_root=$foundSdk
        
        if ($LASTEXITCODE -eq 0) {
            Write-ColorOutput "CMake 安装成功!" "Green"
        } else {
            Write-ColorOutput "CMake 安装失败" "Red"
        }
    }
}

# 5. 检查平台
Write-ColorOutput ""
Write-ColorOutput "[5/5] 检查 Android 平台..." "Yellow"

$platformsPath = "$foundSdk\platforms"
$hasPlatform = $false

if (Test-Path $platformsPath) {
    $platforms = Get-ChildItem $platformsPath -Directory | Select-Object -ExpandProperty Name
    Write-ColorOutput "已安装的平台: $($platforms -join ', ')" "Green"
    
    # 检查是否有 android-26 或更高
    $targetPlatform = $platforms | Where-Object { $_ -match 'android-2[6-9]' -or $_ -match 'android-3[0-6]' } | Sort-Object | Select-Object -Last 1
    
    if ($targetPlatform) {
        $hasPlatform = $true
        Write-ColorOutput "使用平台: $targetPlatform" "Green"
    }
}

if (-not $hasPlatform) {
    Write-ColorOutput "缺少 Android 平台!" "Yellow"
    Write-ColorOutput ""
    Write-ColorOutput "是否安装 Android 8.0 (API 26)? (Y/n)" "Cyan"
    $response = Read-Host
    
    if ($response -eq '' -or $response -eq 'Y' -or $response -eq 'y') {
        Write-ColorOutput "正在安装平台 android-26..." "Yellow"
        & $sdkManager "platforms;android-26" --sdk_root=$foundSdk
        
        if ($LASTEXITCODE -eq 0) {
            Write-ColorOutput "平台安装成功!" "Green"
        }
    }
}

# 总结
Write-ColorOutput ""
Write-ColorOutput "========================================" "Cyan"
Write-ColorOutput "配置完成!" "Green"
Write-ColorOutput "========================================" "Cyan"
Write-ColorOutput ""
Write-ColorOutput "环境变量设置:" "Yellow"
Write-ColorOutput "  [System Environment Variables]" "White"
Write-ColorOutput "  ANDROID_SDK_ROOT = $foundSdk" "Green"
Write-ColorOutput ""
Write-ColorOutput "建议添加到系统 PATH:" "Yellow"
Write-ColorOutput "  %ANDROID_SDK_ROOT%\platform-tools" "Green"
Write-ColorOutput "  %ANDROID_SDK_ROOT%\cmdline-tools\latest\bin" "Green"
Write-ColorOutput ""
Write-ColorOutput "下一步:" "Yellow"
Write-ColorOutput "  运行 scripts\build_android.bat 开始构建" "White"
Write-ColorOutput ""

# 询问是否设置环境变量
Write-ColorOutput "是否自动设置环境变量? (需要管理员权限) (y/N)" "Cyan"
$response = Read-Host

if ($response -eq 'Y' -or $response -eq 'y') {
    try {
        [Environment]::SetEnvironmentVariable("ANDROID_SDK_ROOT", $foundSdk, "User")
        Write-ColorOutput "环境变量已设置!" "Green"
        
        # 刷新当前会话
        $env:ANDROID_SDK_ROOT = $foundSdk
    } catch {
        Write-ColorOutput "设置环境变量失败 (可能需要管理员权限)" "Red"
        Write-ColorOutput "请手动设置 ANDROID_SDK_ROOT = $foundSdk" "Yellow"
    }
}

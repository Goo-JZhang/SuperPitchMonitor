# Android 平台适配与性能优化

## 1. Android 音频架构

### 1.1 音频路径概览

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         Android 音频架构                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    Application Layer                                 │   │
│  │                         │                                           │   │
│  │                         ▼                                           │   │
│  │              ┌─────────────────────┐                                │   │
│  │              │   SuperPitchMonitor │                                │   │
│  │              │   (JUCE + 原生代码)  │                                │   │
│  │              └──────────┬──────────┘                                │   │
│  └─────────────────────────┼───────────────────────────────────────────┘   │
│                            │                                               │
│  ┌─────────────────────────┼───────────────────────────────────────────┐   │
│  │              Native Layer (JNI)                                      │   │
│  │                         │                                           │   │
│  │            ┌────────────┴────────────┐                              │   │
│  │            ▼                         ▼                              │   │
│  │   ┌─────────────┐           ┌─────────────┐                         │   │
│  │   │   AAudio    │           │  OpenSL ES  │                         │   │
│  │   │  (API 26+)  │           │  (Fallback) │                         │   │
│  │   │             │           │             │                         │   │
│  │   │ • 低延迟    │           │ • 兼容性好  │                         │   │
│  │   │ • MMAP模式  │           │ • 全版本    │                         │   │
│  │   │ • 独占模式  │           │   支持      │                         │   │
│  │   └──────┬──────┘           └──────┬──────┘                         │   │
│  └──────────┼─────────────────────────┼─────────────────────────────────┘   │
│             │                         │                                     │
│  ┌──────────┴─────────────────────────┴─────────────────────────────────┐   │
│  │                    HAL (Hardware Abstraction Layer)                  │   │
│  │                         │                                           │   │
│  │                         ▼                                           │   │
│  │              ┌─────────────────────┐                                │   │
│  │              │    Audio HAL        │                                │   │
│  │              │  (厂商实现)          │                                │   │
│  │              └──────────┬──────────┘                                │   │
│  └─────────────────────────┼───────────────────────────────────────────┘   │
│                            │                                               │
│  ┌─────────────────────────┼───────────────────────────────────────────┐   │
│  │                    Kernel Layer                                      │   │
│  │                         │                                           │   │
│  │                         ▼                                           │   │
│  │              ┌─────────────────────┐                                │   │
│  │              │   ALSA / TinyALSA   │                                │   │
│  │              └─────────────────────┘                                │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 1.2 JUCE Android 音频配置

```cpp
// Android 音频引擎配置
class AndroidAudioConfiguration
{
public:
    struct AudioConfig {
        // 音频API选择
        enum class Api {
            AAudio,      // 推荐: Android 8.0+ (API 26+)
            OpenSLES,    // 兼容: 所有 Android 版本
            Oboe         // 可选: Google 的低延迟库 (需单独集成)
        };
        
        Api preferredApi = Api::AAudio;
        bool fallbackToOpenSLES = true;
        
        // AAudio 特定配置
        struct AAudioSettings {
            aaudio_performance_mode_t performanceMode = 
                AAUDIO_PERFORMANCE_MODE_LOW_LATENCY;
            aaudio_sharing_mode_t sharingMode = 
                AAUDIO_SHARING_MODE_EXCLUSIVE;
            int32_t bufferCapacity = 0;  // 0 = 自动
            int32_t burstSize = 0;       // 0 = 自动
        } aaudio;
        
        // 通用音频设置
        double sampleRate = 44100.0;
        int bufferSize = 512;
        int inputChannels = 1;
        int outputChannels = 0;  // 仅输入模式
    };
    
    static void applyToDeviceManager(AudioDeviceManager& deviceManager,
                                     const AudioConfig& config)
    {
        AudioDeviceManager::AudioDeviceSetup setup;
        setup.sampleRate = config.sampleRate;
        setup.bufferSize = config.bufferSize;
        setup.inputChannels = config.inputChannels;
        setup.outputChannels = config.outputChannels;
        setup.useDefaultInputChannels = false;
        setup.useDefaultOutputChannels = false;
        
        // Android 特定设置
       #if JUCE_ANDROID
        StringArray androidProps;
        androidProps.add("android/audioTrackMode=lowLatency");
        
        if (config.preferredApi == AudioConfig::Api::AAudio)
        {
            androidProps.add("android/useAAudio=1");
        }
        else
        {
            androidProps.add("android/useAAudio=0");
        }
        
        setup.inputDeviceName = "Android Input";
        setup.outputDeviceName = "";
       #endif
        
        String error = deviceManager.setAudioDeviceSetup(setup, true);
        if (error.isNotEmpty())
        {
            DBG("Audio setup error: " << error);
        }
    }
};
```

---

## 2. Android 项目配置

### 2.1 CMake 配置

```cmake
# CMakeLists.txt

cmake_minimum_required(VERSION 3.22.1)
project(SuperPitchMonitor VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# JUCE 设置
set(JUCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../JUCE" CACHE PATH "Path to JUCE")
add_subdirectory(${JUCE_DIR} JUCE)

# Android 特定配置
if(ANDROID)
    # NDK 配置
    set(CMAKE_ANDROID_STL_TYPE c++_shared)
    
    # 架构优化
    if(${ANDROID_ABI} STREQUAL "armeabi-v7a")
        add_compile_options(-mfloat-abi=softfp -mfpu=neon)
        add_definitions(-DUSE_NEON=1)
    elseif(${ANDROID_ABI} STREQUAL "arm64-v8a")
        add_compile_options(-march=armv8-a)
        add_definitions(-DUSE_NEON=1)
    endif()
    
    # AAudio 支持 (Android 8.0+)
    if(ANDROID_PLATFORM_LEVEL GREATER_EQUAL 26)
        add_definitions(-DUSE_AAUDIO=1)
    else()
        add_definitions(-DUSE_AAUDIO=0)
    endif()
endif()

# 源文件
set(SOURCE_FILES
    Source/Main.cpp
    Source/MainComponent.cpp
    Source/Audio/AudioEngine.cpp
    Source/Audio/SpectrumAnalyzer.cpp
    Source/Audio/PolyphonicDetector.cpp
    Source/UI/SpectrumDisplay.cpp
    Source/UI/PitchDisplay.cpp
    Source/Utils/AndroidHelper.cpp
)

# 创建应用
juce_add_gui_app(SuperPitchMonitor
    PRODUCT_NAME "SuperPitchMonitor"
    COMPANY_NAME "YourCompany"
    BUNDLE_ID com.yourcompany.superpitchmonitor
)

target_sources(SuperPitchMonitor PRIVATE ${SOURCE_FILES})

# 链接 JUCE 模块
target_link_libraries(SuperPitchMonitor
    PRIVATE
        juce::juce_core
        juce::juce_data_structures
        juce::juce_events
        juce::juce_graphics
        juce::juce_gui_basics
        juce::juce_gui_extra
        juce::juce_audio_basics
        juce::juce_audio_devices
        juce::juce_audio_formats
        juce::juce_audio_processors
        juce::juce_audio_utils
        juce::juce_dsp
    PUBLIC
        juce::juce_recommended_config_flags
        juce::juce_recommended_lto_flags
        juce::juce_recommended_warning_flags
)

# Android 特定链接
target_link_libraries(SuperPitchMonitor PRIVATE log android)
```

### 2.2 Gradle 配置

```gradle
// build.gradle (Module: app)

android {
    compileSdk 34
    
    defaultConfig {
        applicationId "com.yourcompany.superpitchmonitor"
        minSdk 24  // Android 7.0 - 支持 AAudio
        targetSdk 34
        
        versionCode 1
        versionName "1.0"
        
        // 多架构支持
        ndk {
            abiFilters 'arm64-v8a', 'armeabi-v7a', 'x86_64'
        }
        
        externalNativeBuild {
            cmake {
                cppFlags "-O3 -ffast-math -funroll-loops"
                arguments "-DANDROID_STL=c++_shared"
            }
        }
    }
    
    buildTypes {
        release {
            minifyEnabled true
            proguardFiles getDefaultProguardFile('proguard-android-optimize.txt')
            
            externalNativeBuild {
                cmake {
                    cppFlags "-O3 -DNDEBUG -ffast-math"
                }
            }
        }
        
        debug {
            externalNativeBuild {
                cmake {
                    cppFlags "-O0 -g"
                }
            }
        }
    }
    
    externalNativeBuild {
        cmake {
            path "src/main/cpp/CMakeLists.txt"
            version "3.22.1"
        }
    }
    
    // 避免打包不必要的文件
    packagingOptions {
        pickFirst 'lib/arm64-v8a/libc++_shared.so'
        pickFirst 'lib/armeabi-v7a/libc++_shared.so'
    }
}

dependencies {
    implementation 'androidx.appcompat:appcompat:1.6.1'
    implementation 'com.google.android.material:material:1.9.0'
}
```

---

## 3. 运行时权限管理

```cpp
// AndroidPermissions.h
#pragma once

#include <juce_core/juce_core.h>

namespace spm {

class AndroidPermissions
{
public:
    using PermissionCallback = std::function<void(bool granted)>;
    
    // 请求音频录制权限
    static void requestAudioPermission(PermissionCallback callback)
    {
       #if JUCE_ANDROID
        auto& runtimePermissions = RuntimePermissions::getInstance();
        
        runtimePermissions.request(RuntimePermissions::recordAudio,
            [callback](bool granted)
            {
                if (callback)
                    callback(granted);
            });
       #else
        callback(true);  // 非 Android 平台直接允许
       #endif
    }
    
    // 检查权限状态
    static bool hasAudioPermission()
    {
       #if JUCE_ANDROID
        return RuntimePermissions::isGranted(RuntimePermissions::recordAudio);
       #else
        return true;
       #endif
    }
    
    // 请求前台服务权限 (Android 9.0+ 后台音频)
    static void requestForegroundServicePermission()
    {
       #if JUCE_ANDROID
        if (getAndroidSDKVersion() >= 28)
        {
            // 需要在 AndroidManifest.xml 中声明 FOREGROUND_SERVICE 权限
            // 并启动前台服务
        }
       #endif
    }
    
private:
    static int getAndroidSDKVersion()
    {
       #if JUCE_ANDROID
        // 通过 JNI 获取 SDK 版本
        return android_get_device_api_level();
       #else
        return 0;
       #endif
    }
};

} // namespace spm
```

### AndroidManifest.xml 配置

```xml
<?xml version="1.0" encoding="utf-8"?>
<manifest xmlns:android="http://schemas.android.com/apk/res/android"
    package="com.yourcompany.superpitchmonitor">

    <!-- 音频权限 -->
    <uses-permission android:name="android.permission.RECORD_AUDIO" />
    <uses-permission android:name="android.permission.MODIFY_AUDIO_SETTINGS" />
    
    <!-- 前台服务权限 (Android 9.0+) -->
    <uses-permission android:name="android.permission.FOREGROUND_SERVICE" />
    <uses-permission android:name="android.permission.FOREGROUND_SERVICE_MICROPHONE" />
    
    <!-- 唤醒锁 (防止屏幕关闭时音频中断) -->
    <uses-permission android:name="android.permission.WAKE_LOCK" />
    
    <!-- 低延迟音频特性声明 -->
    <uses-feature android:name="android.hardware.audio.low_latency" android:required="false" />
    <uses-feature android:name="android.hardware.audio.pro" android:required="false" />
    <uses-feature android:name="android.hardware.microphone" android:required="true" />

    <application
        android:allowBackup="true"
        android:icon="@mipmap/ic_launcher"
        android:label="@string/app_name"
        android:roundIcon="@mipmap/ic_launcher_round"
        android:supportsRtl="true"
        android:theme="@style/Theme.SuperPitchMonitor">
        
        <activity
            android:name="com.rmsl.juce.JuceApp"
            android:exported="true"
            android:screenOrientation="portrait"
            android:configChanges="orientation|screenSize|keyboardHidden">
            <intent-filter>
                <action android:name="android.intent.action.MAIN" />
                <category android:name="android.intent.category.LAUNCHER" />
            </intent-filter>
        </activity>
        
        <!-- 前台服务 (可选，用于后台检测) -->
        <service
            android:name=".AudioDetectionService"
            android:enabled="true"
            android:exported="false"
            android:foregroundServiceType="microphone" />
            
    </application>

</manifest>
```

---

## 4. 性能优化策略

### 4.1 实时性能保障

```cpp
// RealTimePerformanceOptimizer.h
#pragma once

namespace spm {

class RealTimePerformanceOptimizer
{
public:
    struct PerformanceProfile {
        int fftOrder;           // FFT 阶数
        int hopSize;            // 步长
        int maxPolyphony;       // 最大检测音数
        bool useMultiResolution;// 是否使用多分辨率
        bool useNMF;            // 是否使用 NMF
        float targetCpuUsage;   // 目标 CPU 使用率
    };
    
    void initialize()
    {
        // 检测设备性能等级
        deviceTier_ = detectDeviceTier();
        
        // 根据设备等级设置初始配置
        currentProfile_ = getProfileForTier(deviceTier_);
        
        // 启动性能监控
        startMonitoring();
    }
    
    void adaptToCpuLoad(float currentCpuUsage)
    {
        // 自适应调整
        if (currentCpuUsage > currentProfile_.targetCpuUsage + 0.1f)
        {
            // CPU 过载，降低质量
            degradeQuality();
        }
        else if (currentCpuUsage < currentProfile_.targetCpuUsage - 0.2f)
        {
            // CPU 有余量，提升质量
            improveQuality();
        }
    }
    
    const PerformanceProfile& getCurrentProfile() const
    {
        return currentProfile_;
    }
    
private:
    enum class DeviceTier {
        Low,      // 低端设备
        Medium,   // 中端设备
        High      // 高端设备
    };
    
    DeviceTier deviceTier_;
    PerformanceProfile currentProfile_;
    
    DeviceTier detectDeviceTier()
    {
       #if JUCE_ANDROID
        // 通过 JNI 获取设备信息
        auto cpuCores = std::thread::hardware_concurrency();
        auto ramMB = getTotalRAM();
        
        if (cpuCores >= 8 && ramMB >= 6144)
            return DeviceTier::High;
        else if (cpuCores >= 4 && ramMB >= 3072)
            return DeviceTier::Medium;
        else
            return DeviceTier::Low;
       #else
        return DeviceTier::High;
       #endif
    }
    
    PerformanceProfile getProfileForTier(DeviceTier tier)
    {
        switch (tier)
        {
            case DeviceTier::High:
                return {13, 256, 8, true, true, 0.3f};   // 8192 FFT
                
            case DeviceTier::Medium:
                return {12, 512, 4, true, false, 0.25f}; // 4096 FFT
                
            case DeviceTier::Low:
                return {11, 1024, 2, false, false, 0.2f}; // 2048 FFT
        }
        return {};
    }
    
    void degradeQuality()
    {
        if (currentProfile_.useNMF)
        {
            currentProfile_.useNMF = false;
        }
        else if (currentProfile_.useMultiResolution)
        {
            currentProfile_.useMultiResolution = false;
        }
        else if (currentProfile_.fftOrder > 11)
        {
            currentProfile_.fftOrder--;
        }
        else if (currentProfile_.hopSize < 1024)
        {
            currentProfile_.hopSize *= 2;
        }
    }
    
    void improveQuality()
    {
        // 逆向 degradeQuality 的操作
    }
    
    void startMonitoring()
    {
        // 启动 CPU 使用监控线程
    }
    
    long getTotalRAM()
    {
       #if JUCE_ANDROID
        // 通过 JNI 获取
        struct sysinfo info;
        sysinfo(&info);
        return info.totalram / (1024 * 1024);
       #else
        return 8192; // 默认值
       #endif
    }
};

} // namespace spm
```

### 4.2 内存管理

```cpp
// MemoryPool.h
#pragma once

namespace spm {

// 固定大小的内存池，避免实时线程中的堆分配
template<size_t BlockSize, size_t NumBlocks>
class MemoryPool
{
public:
    MemoryPool()
    {
        for (size_t i = 0; i < NumBlocks - 1; ++i)
        {
            blocks_[i].next = &blocks_[i + 1];
        }
        blocks_[NumBlocks - 1].next = nullptr;
        freeList_ = &blocks_[0];
    }
    
    void* allocate()
    {
        if (!freeList_)
            return nullptr;  // 池耗尽
            
        Block* block = freeList_;
        freeList_ = block->next;
        return block->data;
    }
    
    void deallocate(void* ptr)
    {
        if (!ptr) return;
        
        Block* block = reinterpret_cast<Block*>(
            static_cast<char*>(ptr) - offsetof(Block, data));
        block->next = freeList_;
        freeList_ = block;
    }
    
private:
    struct Block {
        alignas(alignof(std::max_align_t)) char data[BlockSize];
        Block* next;
    };
    
    std::array<Block, NumBlocks> blocks_;
    Block* freeList_;
};

// FFT 缓冲区池
template<int MaxFFTOrder>
class FFTBufferPool
{
public:
    static constexpr size_t MaxSize = 2 * (1 << MaxFFTOrder) * sizeof(float);
    
    using PoolType = MemoryPool<MaxSize, 4>;
    
    static FFTBufferPool& getInstance()
    {
        static FFTBufferPool instance;
        return instance;
    }
    
    void* acquire() { return pool_.allocate(); }
    void release(void* ptr) { pool_.deallocate(ptr); }
    
private:
    FFTBufferPool() = default;
    PoolType pool_;
};

} // namespace spm
```

---

## 5. 电池优化

```cpp
// BatteryOptimizer.h
#pragma once

namespace spm {

class BatteryOptimizer
{
public:
    void initialize()
    {
        // 注册电池状态监听
        startBatteryMonitoring();
    }
    
    void onBatteryLevelChanged(float level, bool isCharging)
    {
        if (level < 0.2f && !isCharging)
        {
            // 低电量且未充电，启用省电模式
            enablePowerSaveMode();
        }
        else if (isCharging || level > 0.3f)
        {
            // 充电中或电量充足，禁用省电模式
            disablePowerSaveMode();
        }
    }
    
    bool isPowerSaveMode() const { return powerSaveMode_; }
    
private:
    bool powerSaveMode_ = false;
    
    void enablePowerSaveMode()
    {
        powerSaveMode_ = true;
        
        // 降低采样率
        // 增加 FFT hop size
        // 降低 UI 刷新率
        // 减少多音检测数量
    }
    
    void disablePowerSaveMode()
    {
        powerSaveMode_ = false;
        
        // 恢复正常设置
    }
    
    void startBatteryMonitoring()
    {
       #if JUCE_ANDROID
        // 通过 JNI 注册电池状态广播接收器
       #endif
    }
};

} // namespace spm
```

---

## 6. 延迟校准

```cpp
// LatencyCalibrator.h
#pragma once

namespace spm {

class LatencyCalibrator
{
public:
    struct LatencyInfo {
        int inputLatencyMs;
        int outputLatencyMs;
        int totalLatencyMs;
        bool isLowLatency;
    };
    
    LatencyInfo calibrate(AudioDeviceManager& deviceManager)
    {
        LatencyInfo info{};
        
       #if JUCE_ANDROID
        if (auto* device = deviceManager.getCurrentAudioDevice())
        {
            // 获取底层延迟信息
            info.inputLatencyMs = device->getInputLatencyInSamples() / 
                                  device->getCurrentSampleRate() * 1000;
            info.outputLatencyMs = device->getOutputLatencyInSamples() / 
                                   device->getCurrentSampleRate() * 1000;
            info.totalLatencyMs = info.inputLatencyMs + info.outputLatencyMs;
            
            // 判断是否低延迟
            info.isLowLatency = info.totalLatencyMs < 20;
        }
       #endif
        
        return info;
    }
    
    // 设备延迟数据库 (用于无法直接获取延迟的设备)
    static int estimateLatencyForDevice(const String& deviceName)
    {
        // 知名设备的延迟数据
        static const std::map<String, int> knownDevices = {
            {"Pixel 6", 12},
            {"Pixel 7", 10},
            {"Galaxy S23", 15},
            {"Xiaomi 13", 18},
            {"OnePlus 11", 14}
        };
        
        for (const auto& [name, latency] : knownDevices)
        {
            if (deviceName.containsIgnoreCase(name))
                return latency;
        }
        
        return 25; // 默认估计值
    }
};

} // namespace spm
```

---

## 7. 测试与调试

### 7.1 性能测试指标

| 指标 | 目标值 | 说明 |
|-----|-------|------|
| 端到端延迟 | < 30ms | 音频输入到显示输出 |
| CPU 使用率 | < 25% | 持续负载 |
| 内存使用 | < 100MB | 峰值内存 |
| 电池消耗 | < 5%/小时 | 后台运行 |
| 检测准确率 | > 85% | 单音场景 |
| 多音分离率 | > 70% | 和弦场景 |

### 7.2 Android Profiler 配置

```cpp
// 在调试构建中启用详细日志
#if defined(DEBUG) && JUCE_ANDROID
    #define SPM_LOG_AUDIO_TIMING 1
    #define SPM_LOG_DETECTION_RESULTS 1
    #define SPM_LOG_PERFORMANCE 1
#endif
```

#pragma once

#include <juce_core/juce_core.h>
#include <juce_gui_basics/juce_gui_basics.h>
#include <functional>

namespace spm {
namespace Platform {

// =============================================================================
// Platform Abstraction Layer
// 
// All platform-specific functionality is exposed through this module
// to avoid using platform APIs directly in business logic
// =============================================================================

// =============================================================================
// Permission Management
// =============================================================================

enum class Permission {
    AudioInput,     // Microphone permission (Android requires runtime request)
    Storage,        // Storage permission
    Camera          // Camera permission (future use)
};

/**
 * Request a runtime permission
 * On Windows: immediately calls callback with true
 * On Android: shows permission dialog
 */
void requestPermission(Permission permission, 
                       std::function<void(bool granted)> callback);

/**
 * Check if permission is already granted
 */
bool hasPermission(Permission permission);

// =============================================================================
// File System Paths
// =============================================================================

/**
 * Get application data directory
 * Windows: %APPDATA%/SuperPitchMonitor
 * Android: /data/data/com.superpitchmonitor/files
 * macOS: ~/Library/Application Support/SuperPitchMonitor
 * iOS: Documents/SuperPitchMonitor
 */
juce::File getAppDataDirectory();

/**
 * Get cache directory (can be cleared by system)
 */
juce::File getCacheDirectory();

/**
 * Get documents directory (user accessible)
 * Windows: %USERPROFILE%/Documents
 * Android: /sdcard/Documents (or scoped storage)
 * macOS: ~/Documents
 * iOS: Documents (visible in Files app)
 */
juce::File getDocumentsDirectory();

// =============================================================================
// Display & Window Management
// =============================================================================

/**
 * Set fullscreen mode
 */
void setFullscreen(bool fullscreen);

/**
 * Check if currently fullscreen
 */
bool isFullscreen();

/**
 * Get display DPI scale factor
 * Windows: per-monitor DPI awareness
 * Android: densityDpi / 160
 * macOS/iOS: 1.0 (Retina handled by JUCE)
 */
float getDisplayScale();

// =============================================================================
// Audio
// =============================================================================

/**
 * Get recommended audio buffer size for platform
 */
int getRecommendedBufferSize();

/**
 * Check if low latency mode is available
 */
bool supportsLowLatencyMode();

// =============================================================================
// Performance & Power
// =============================================================================

/**
 * Set low power mode (reduces CPU/GPU usage)
 */
void setLowPowerMode(bool enabled);

/**
 * Check if low power mode is enabled
 */
bool isLowPowerModeEnabled();

/**
 * Get number of CPU cores to use for processing
 * Respects platform limits and user settings
 */
int getOptimalThreadCount();

// =============================================================================
// Debug & Development
// =============================================================================

/**
 * Check if this is a debug build
 */
inline constexpr bool isDebugBuild() {
#if defined(DEBUG) || defined(_DEBUG)
    return true;
#else
    return false;
#endif
}

/**
 * Check if audio simulator is allowed
 * Debug builds: always allowed
 * Release builds: only on desktop platforms
 */
inline bool isSimulatorAllowed() {
#if defined(DEBUG) || defined(_DEBUG)
    return true;  // Debug build always allows simulator
#else
    // Release build: only desktop platforms
    #if JUCE_ANDROID
        return false;  // Android release: no simulator
    #else
        return true;   // Desktop release: allow simulator
    #endif
#endif
}

/**
 * Get platform name for logging
 */
juce::String getPlatformName();

/**
 * Check if running on emulator/simulator
 */
bool isRunningOnEmulator();

// =============================================================================
// UI & Window Management
// =============================================================================

/**
 * Configure main application window based on platform
 * Android/iOS: Fullscreen
 * Desktop: Centered, resizable
 */
void configureMainWindow(juce::DocumentWindow* window);

// =============================================================================
// Platform Info
// =============================================================================

struct PlatformInfo {
    juce::String osName;
    juce::String osVersion;
    juce::String deviceModel;
    int cpuCores;
    juce::String cpuArchitecture;
    int totalMemoryMB;
};

PlatformInfo getPlatformInfo();

} // namespace Platform
} // namespace spm

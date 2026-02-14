#include "PlatformUtils.h"

namespace spm {
namespace Platform {

// =============================================================================
// Universal Implementations (work on all platforms)
// =============================================================================

juce::String getPlatformName() {
#if JUCE_ANDROID
    return "Android";
#elif JUCE_WINDOWS
    return "Windows";
#elif JUCE_MAC
    return "macOS";
#elif JUCE_LINUX
    return "Linux";
#elif JUCE_IOS
    return "iOS";
#else
    return "Unknown";
#endif
}

PlatformInfo getPlatformInfo() {
    PlatformInfo info;
    info.osName = getPlatformName();
    
#if JUCE_ANDROID
    // Android-specific info gathering would go here
    info.osVersion = juce::String(juce::SystemStats::getOperatingSystemType());
    info.deviceModel = "Unknown";  // Would need JNI call
#elif JUCE_WINDOWS
    info.osVersion = juce::SystemStats::getOperatingSystemName();
    info.deviceModel = juce::SystemStats::getComputerName();
#endif
    
    info.cpuCores = juce::SystemStats::getNumCpus();
    info.cpuArchitecture = juce::SystemStats::getCpuVendor();
    // Memory size not directly available in JUCE, set to 0 for now
    info.totalMemoryMB = 0;
    
    return info;
}

// Default implementations (can be overridden in platform-specific files)
int getOptimalThreadCount() {
    int cores = juce::SystemStats::getNumCpus();
    // Leave one core free for UI
    return juce::jmax(1, cores - 1);
}

bool isRunningOnEmulator() {
#if JUCE_ANDROID
    // Check for emulator indicators
    auto vendor = juce::SystemStats::getCpuVendor();
    return vendor.containsIgnoreCase("hypervisor") || 
           vendor.containsIgnoreCase("qemu");
#else
    return false;  // Desktop is always "real"
#endif
}

// macOS Desktop implementations
#if JUCE_MAC

void requestPermission(Permission permission, std::function<void(bool)> callback) {
    // macOS permissions are typically handled by the system or not required for desktop
    // Audio permissions on macOS are handled by the system automatically
    if (callback)
        callback(true);
}

bool hasPermission(Permission permission) {
    return true;  // Desktop typically has permissions by default
}

juce::File getAppDataDirectory() {
    return juce::File::getSpecialLocation(juce::File::userApplicationDataDirectory)
           .getChildFile("SuperPitchMonitor");
}

juce::File getCacheDirectory() {
    return juce::File::getSpecialLocation(juce::File::tempDirectory)
           .getChildFile("SuperPitchMonitor");
}

juce::File getDocumentsDirectory() {
    return juce::File::getSpecialLocation(juce::File::userDocumentsDirectory);
}

void setFullscreen(bool fullscreen) {
    // Handled by JUCE component methods
}

bool isFullscreen() {
    return false;
}

float getDisplayScale() {
    return 1.0f;  // macOS Retina handling is done by JUCE automatically
}

int getRecommendedBufferSize() {
    return 512;  // Default buffer size for macOS
}

bool supportsLowLatencyMode() {
    return true;  // macOS supports low latency audio
}

void setLowPowerMode(bool enabled) {
    // No-op on desktop
    juce::ignoreUnused(enabled);
}

bool isLowPowerModeEnabled() {
    return false;
}

void configureMainWindow(juce::DocumentWindow* window) {
    if (window != nullptr) {
        window->setResizable(true, true);
        window->setResizeLimits(800, 600, 4096, 2160);
        window->centreWithSize(window->getWidth(), window->getHeight());
    }
}

#endif  // JUCE_MAC

} // namespace Platform
} // namespace spm

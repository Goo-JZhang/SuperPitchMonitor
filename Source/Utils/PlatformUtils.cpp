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

} // namespace Platform
} // namespace spm

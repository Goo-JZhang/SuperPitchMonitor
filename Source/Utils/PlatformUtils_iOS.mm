#include "PlatformUtils.h"

#if JUCE_IOS || JUCE_MAC

#import <Foundation/Foundation.h>

namespace spm {
namespace Platform {

// =============================================================================
// Permission Management (iOS/macOS)
// =============================================================================

void requestPermission(Permission permission, 
                       std::function<void(bool granted)> callback) {
    switch (permission) {
        case Permission::AudioInput:
            // iOS requires microphone permission
            // macOS may require it depending on sandbox settings
            #if JUCE_IOS
            {
                // Request microphone access via AVFoundation
                // This is handled by JUCE's AudioDeviceManager
                // Just verify audio is available
                callback(true);
            }
            #else
                callback(true);  // macOS desktop usually doesn't require explicit permission
            #endif
            break;
            
        case Permission::Storage:
            // iOS: Sandboxed, no permission needed for app directories
            // macOS: May prompt for Documents/Desktop access
            callback(true);
            break;
            
        case Permission::Camera:
            #if JUCE_IOS
            // iOS camera permission would be requested here
            callback(true);
            #else
            callback(true);  // macOS
            #endif
            break;
            
        default:
            callback(false);
    }
}

bool hasPermission(Permission permission) {
    // iOS/macOS permissions are typically checked via system APIs
    // For audio, we assume granted if the device exists
    switch (permission) {
        case Permission::AudioInput:
            return true;
        case Permission::Camera:
            return true;
        default:
            return true;
    }
}

// =============================================================================
// File System Paths (iOS/macOS)
// =============================================================================

juce::File getAppDataDirectory() {
    #if JUCE_IOS
    // iOS: Documents directory for user data
    return juce::File::getSpecialLocation(
        juce::File::userDocumentsDirectory
    ).getChildFile("SuperPitchMonitor");
    #else
    // macOS: Application Support directory
    return juce::File::getSpecialLocation(
        juce::File::userApplicationDataDirectory
    ).getChildFile("SuperPitchMonitor");
    #endif
}

juce::File getCacheDirectory() {
    return juce::File::getSpecialLocation(
        juce::File::tempDirectory
    ).getChildFile("SuperPitchMonitor");
}

juce::File getDocumentsDirectory() {
    #if JUCE_IOS
    // iOS: Documents directory is user-accessible via Files app
    return juce::File::getSpecialLocation(
        juce::File::userDocumentsDirectory
    ).getChildFile("SuperPitchMonitor");
    #else
    // macOS: Documents folder
    return juce::File::getSpecialLocation(
        juce::File::userDocumentsDirectory
    ).getChildFile("SuperPitchMonitor");
    #endif
}

// =============================================================================
// Display & Window Management (iOS/macOS)
// =============================================================================

void setFullscreen(bool fullscreen) {
    auto& desktop = juce::Desktop::getInstance();
    if (auto* window = desktop.getComponent(0)) {
        if (auto* documentWindow = dynamic_cast<juce::DocumentWindow*>(window)) {
            #if JUCE_MAC
            documentWindow->setFullScreen(fullscreen);
            #endif
            // iOS is always fullscreen
            juce::ignoreUnused(fullscreen);
        }
    }
}

bool isFullscreen() {
    #if JUCE_IOS
    return true;  // iOS is always fullscreen
    #else
    auto& desktop = juce::Desktop::getInstance();
    if (auto* window = desktop.getComponent(0)) {
        if (auto* documentWindow = dynamic_cast<juce::DocumentWindow*>(window)) {
            return documentWindow->isFullScreen();
        }
    }
    return false;
    #endif
}

float getDisplayScale() {
    // Handle Retina displays
    return juce::Desktop::getInstance().getGlobalScaleFactor();
}

// =============================================================================
// Audio (iOS/macOS)
// =============================================================================

int getRecommendedBufferSize() {
    #if JUCE_IOS
    // iOS can handle low latency with 256-512 samples
    return 256;  // Good for iOS with low latency
    #else
    // macOS can handle even smaller buffers
    return 256;
    #endif
}

bool supportsLowLatencyMode() {
    #if JUCE_IOS
    // iOS supports low latency audio via Core Audio
    return true;
    #else
    // macOS definitely supports low latency
    return true;
    #endif
}

// =============================================================================
// UI & Window Management (iOS/macOS)
// =============================================================================

void configureMainWindow(juce::DocumentWindow* window)
{
    if (window == nullptr)
        return;
        
    #if JUCE_IOS
    // iOS: Fullscreen on device
    window->setFullScreen(true);
    #else
    // macOS: Centered, resizable window
    window->centreWithSize(1000, 800);
    window->setResizable(true, true);
    #endif
}

// =============================================================================
// Performance & Power (iOS/macOS)
// =============================================================================

namespace {
    bool lowPowerMode = false;
}

void setLowPowerMode(bool enabled) {
    lowPowerMode = enabled;
    // On iOS, could check NSProcessInfo.processInfo.lowPowerModeEnabled
}

bool isLowPowerModeEnabled() {
    #if JUCE_IOS
    // Could check actual system state via:
    // [[NSProcessInfo processInfo] isLowPowerModeEnabled]
    #endif
    return lowPowerMode;
}

int getOptimalThreadCount() {
    #if JUCE_IOS
    // iOS: Be conservative to save battery
    int cores = juce::SystemStats::getNumCpus();
    return juce::jmin(2, juce::jmax(1, cores - 1));
    #else
    // macOS: Can use more threads
    int cores = juce::SystemStats::getNumCpus();
    return juce::jmax(1, cores - 1);
    #endif
}

} // namespace Platform
} // namespace spm

#endif // JUCE_IOS || JUCE_MAC

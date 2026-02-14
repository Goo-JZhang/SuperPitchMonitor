#include "PlatformUtils.h"

#if JUCE_WINDOWS

#include <windows.h>

namespace spm {
namespace Platform {

// =============================================================================
// Permission Management (Windows)
// =============================================================================

void requestPermission(Permission permission, 
                       std::function<void(bool granted)> callback) {
    // Windows doesn't require runtime permissions like Android
    // Just verify the capability exists
    
    switch (permission) {
        case Permission::AudioInput:
            // Check if audio input is available
            // On Windows, this is always true if there's a device
            callback(true);
            break;
            
        case Permission::Storage:
            callback(true);
            break;
            
        case Permission::Camera:
            // Could enumerate cameras to verify
            callback(true);
            break;
            
        default:
            callback(false);
    }
}

bool hasPermission(Permission permission) {
    // Windows doesn't have runtime permission system
    return true;
}

// =============================================================================
// File System Paths (Windows)
// =============================================================================

juce::File getAppDataDirectory() {
    return juce::File::getSpecialLocation(
        juce::File::userApplicationDataDirectory
    ).getChildFile("SuperPitchMonitor");
}

juce::File getCacheDirectory() {
    return juce::File::getSpecialLocation(
        juce::File::tempDirectory
    ).getChildFile("SuperPitchMonitor");
}

juce::File getDocumentsDirectory() {
    return juce::File::getSpecialLocation(
        juce::File::userDocumentsDirectory
    ).getChildFile("SuperPitchMonitor");
}

// =============================================================================
// Display & Window Management (Windows)
// =============================================================================

void setFullscreen(bool fullscreen) {
    // Get the main window
    auto& desktop = juce::Desktop::getInstance();
    if (auto* window = desktop.getComponent(0)) {
        if (auto* documentWindow = dynamic_cast<juce::DocumentWindow*>(window)) {
            documentWindow->setFullScreen(fullscreen);
        }
    }
}

bool isFullscreen() {
    auto& desktop = juce::Desktop::getInstance();
    if (auto* window = desktop.getComponent(0)) {
        if (auto* documentWindow = dynamic_cast<juce::DocumentWindow*>(window)) {
            return documentWindow->isFullScreen();
        }
    }
    return false;
}

float getDisplayScale() {
    return juce::Desktop::getInstance().getGlobalScaleFactor();
}

// =============================================================================
// Audio (Windows)
// =============================================================================

int getRecommendedBufferSize() {
    // Windows can handle smaller buffers with ASIO or WASAPI
    // Default to 512 for good balance
    return 512;
}

bool supportsLowLatencyMode() {
    // Windows with ASIO or WASAPI exclusive mode supports low latency
    return true;
}

// =============================================================================
// UI & Window Management (Windows)
// =============================================================================

void configureMainWindow(juce::DocumentWindow* window)
{
    if (window == nullptr)
        return;
        
    // Desktop: Centered, resizable window
    window->centreWithSize(800, 1200);
    window->setResizable(true, true);
}

// =============================================================================
// Performance & Power (Windows)
// =============================================================================

namespace {
    bool lowPowerMode = false;
}

void setLowPowerMode(bool enabled) {
    lowPowerMode = enabled;
    
    // Could adjust process priority here
    if (enabled) {
        SetPriorityClass(GetCurrentProcess(), BELOW_NORMAL_PRIORITY_CLASS);
    } else {
        SetPriorityClass(GetCurrentProcess(), NORMAL_PRIORITY_CLASS);
    }
}

bool isLowPowerModeEnabled() {
    return lowPowerMode;
}

} // namespace Platform
} // namespace spm

#endif // JUCE_WINDOWS

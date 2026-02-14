#include "PlatformUtils.h"

#if JUCE_ANDROID

namespace spm {
namespace Platform {

// =============================================================================
// Permission Management (Android)
// =============================================================================

namespace {
    std::function<void(bool)> pendingAudioCallback;
}

void requestPermission(Permission permission, 
                       std::function<void(bool granted)> callback) {
    switch (permission) {
        case Permission::AudioInput:
            pendingAudioCallback = callback;
            juce::RuntimePermissions::request(
                juce::RuntimePermissions::recordAudio,
                [](bool granted) {
                    if (pendingAudioCallback) {
                        pendingAudioCallback(granted);
                        pendingAudioCallback = nullptr;
                    }
                }
            );
            break;
            
        case Permission::Storage:
            // Scoped storage changes in Android 10+ 
            // For now, just grant
            callback(true);
            break;
            
        case Permission::Camera:
            juce::RuntimePermissions::request(
                juce::RuntimePermissions::camera,
                [callback](bool granted) { callback(granted); }
            );
            break;
    }
}

bool hasPermission(Permission permission) {
    switch (permission) {
        case Permission::AudioInput:
            return juce::RuntimePermissions::isGranted(
                juce::RuntimePermissions::recordAudio);
                
        case Permission::Camera:
            return juce::RuntimePermissions::isGranted(
                juce::RuntimePermissions::camera);
                
        default:
            return true;
    }
}

// =============================================================================
// File System Paths (Android)
// =============================================================================

juce::File getAppDataDirectory() {
    // Internal app files directory
    return juce::File::getSpecialLocation(
        juce::File::userApplicationDataDirectory
    );
}

juce::File getCacheDirectory() {
    return juce::File::getSpecialLocation(
        juce::File::tempDirectory
    );
}

juce::File getDocumentsDirectory() {
    // On Android 10+, use app-specific external directory
    // Users should use scoped storage APIs for shared files
    return juce::File::getSpecialLocation(
        juce::File::userDocumentsDirectory
    );
}

// =============================================================================
// Display & Window Management (Android)
// =============================================================================

void setFullscreen(bool fullscreen) {
    // JUCE handles Android fullscreen via AndroidActivity
    // This is typically set in Main.cpp for the main window
    auto& desktop = juce::Desktop::getInstance();
    if (auto* window = desktop.getComponent(0)) {
        // Android fullscreen is typically set at window creation
        // Runtime changes would need JNI calls
        juce::ignoreUnused(fullscreen);
    }
}

bool isFullscreen() {
    // Android app typically always runs fullscreen
    return true;
}

float getDisplayScale() {
    return juce::Desktop::getInstance().getGlobalScaleFactor();
}

// =============================================================================
// Audio (Android)
// =============================================================================

int getRecommendedBufferSize() {
    // Android typically needs larger buffers for stability
    // Low latency mode (AAudio) can use smaller buffers
    return 512;  // AAudio can handle this
}

bool supportsLowLatencyMode() {
    // AAudio is available on Android 8.0+
    // We can check for FEATURE_AUDIO_PRO
    return true;  // Assume support, handle fallback in audio engine
}

// =============================================================================
// UI & Window Management (Android)
// =============================================================================

void configureMainWindow(juce::DocumentWindow* window)
{
    if (window == nullptr)
        return;
        
    // Android: Fullscreen
    window->setFullScreen(true);
}

// =============================================================================
// Performance & Power (Android)
// =============================================================================

namespace {
    bool lowPowerMode = false;
}

void setLowPowerMode(bool enabled) {
    lowPowerMode = enabled;
    // Could trigger hints to the system
    // Or reduce processing load
}

bool isLowPowerModeEnabled() {
    // Could also check PowerManager.isPowerSaveMode()
    return lowPowerMode;
}

int getOptimalThreadCount() {
    // Android devices vary widely
    // Be conservative on mobile
    int cores = juce::SystemStats::getNumCpus();
    
    // Use at most 2 threads on mobile to save battery
    // and avoid thermal throttling
    return juce::jmin(2, juce::jmax(1, cores - 1));
}

} // namespace Platform
} // namespace spm

#endif // JUCE_ANDROID

#pragma once

#include <juce_core/juce_core.h>
#include <juce_data_structures/juce_data_structures.h>
#include <juce_events/juce_events.h>
#include <juce_graphics/juce_graphics.h>
#include <juce_gui_basics/juce_gui_basics.h>
#include <juce_gui_extra/juce_gui_extra.h>
#include <juce_audio_basics/juce_audio_basics.h>
#include <juce_audio_devices/juce_audio_devices.h>
#include <juce_audio_formats/juce_audio_formats.h>
#include <juce_audio_processors/juce_audio_processors.h>
#include <juce_audio_utils/juce_audio_utils.h>
#include <juce_dsp/juce_dsp.h>

namespace spm {

/**
 * Application Configuration Constants
 */
namespace Config {

// =============================================================================
// Version Information
// =============================================================================
inline constexpr const char* AppName = "SuperPitchMonitor";
inline constexpr const char* AppVersion = "1.0.0";
inline constexpr int VersionCode = 1;

// =============================================================================
// Audio Settings
// =============================================================================
namespace Audio {
    inline constexpr double DefaultSampleRate = 44100.0;
    inline constexpr int DefaultBufferSize = 512;
    inline constexpr int InputChannels = 1;
    inline constexpr int OutputChannels = 0;  // Input only
}

// =============================================================================
// Spectrum Analysis Settings
// =============================================================================
namespace Spectrum {
    // FFT configuration
    inline constexpr int DefaultFFTOrder = 12;        // 4096 samples
    inline constexpr int HighQualityFFTOrder = 13;    // 8192 samples
    inline constexpr int LowLatencyFFTOrder = 11;     // 2048 samples
    
    // Hop size configuration (overlap)
    inline constexpr int DefaultHopSize = 512;        // 87.5% overlap @ 4096
    inline constexpr float OverlapRatio = 0.875f;
    
    // Fixed frequency range for all analysis (20-5000 Hz)
    // This range covers most instrument fundamentals and voice
    inline constexpr float MinFrequency = 20.0f;
    inline constexpr float MaxFrequency = 5000.0f;
    
    // Fixed display range for spectrum visualization (dB)
    inline constexpr float MinDecibels = -90.0f;
    inline constexpr float MaxDecibels = -10.0f;
    
    // Display bands
    inline constexpr int NumDisplayBands = 128;
}



// =============================================================================
// Pitch Detection Settings
// =============================================================================
namespace Pitch {
    // Fixed detection range (Hz) - Fundamental frequency range (not including harmonics)
    // Range: 20-5000 Hz (E0 to B7, covers most instrument fundamentals and voice)
    inline constexpr float MinFrequency = 20.0f;   // ~20 Hz (E0, below piano A0)
    inline constexpr float MaxFrequency = 5000.0f; // ~5 kHz (B7, covers most instrument fundamentals)
    
    // MIDI conversion range (corresponding to 20-5000 Hz)
    inline constexpr float MinMidiNote = 16.0f;   // ~20 Hz (E0)
    inline constexpr float MaxMidiNote = 107.0f;  // ~5 kHz (B7, just below C8)
    
    // Polyphony settings
    inline constexpr int MaxPolyphony = 6;
    inline constexpr float ConfidenceThreshold = 0.35f;  // Lower for better sensitivity with strong peaks
    
    // Cents tolerance
    inline constexpr int CentsTolerance = 50;  // Half semitone
    
    // Harmonic detection
    inline constexpr int MaxHarmonics = 8;
    inline constexpr float HarmonicTolerance = 0.03f;  // 3% frequency tolerance
}

// =============================================================================
// Performance Settings
// =============================================================================
namespace Performance {
    // Target latency (milliseconds)
    inline constexpr int TargetLatencyMs = 20;
    
    // UI refresh rate
    inline constexpr int DisplayRefreshRateHz = 60;
    
    // Performance levels
    enum class QualityLevel {
        Fast = 0,       // Low latency, single pitch detection
        Balanced = 1,   // Balanced mode
        Accurate = 2    // High precision, polyphonic detection
    };
    
    // Adaptive quality adjustment
    inline constexpr bool EnableAdaptiveQuality = true;
    inline constexpr float TargetCpuUsage = 0.25f;  // 25% CPU target
}

// =============================================================================
// UI Settings
// =============================================================================
namespace UI {
    // Window dimensions
    inline constexpr int DefaultWidth = 800;
    inline constexpr int DefaultHeight = 1200;
    inline constexpr int MinWidth = 360;
    inline constexpr int MinHeight = 640;
    
    // Colors
    namespace Colors {
        inline const juce::Colour BackgroundDark = juce::Colour(0xFF1A1A20);
        inline const juce::Colour BackgroundCard = juce::Colour(0xFF2A2A35);
        inline const juce::Colour Primary = juce::Colour(0xFF00C8FF);
        inline const juce::Colour Secondary = juce::Colour(0xFF9664FF);
        inline const juce::Colour Success = juce::Colour(0xFF32C864);
        inline const juce::Colour Warning = juce::Colour(0xFFFFB432);
        inline const juce::Colour Error = juce::Colour(0xFFFF5050);
    }
}

// =============================================================================
// File Paths
// =============================================================================
namespace Paths {
    inline const juce::String SettingsFile = "settings.json";
    inline const juce::String LogFile = "app.log";
}

} // namespace Config

// =============================================================================
// Performance Profiler (Debug Mode)
// =============================================================================
#if defined(DEBUG) || defined(_DEBUG)
    #define SPM_PROFILE_SCOPE(name) ProfileScope _profileScope(name)
    
    class ProfileScope {
    public:
        explicit ProfileScope(const char* name) : name_(name), start_(juce::Time::getMillisecondCounterHiRes()) {}
        ~ProfileScope() {
            double elapsed = juce::Time::getMillisecondCounterHiRes() - start_;
            DBG("[PROFILE] " << name_ << ": " << elapsed << " ms");
        }
    private:
        const char* name_;
        double start_;
    };
#else
    #define SPM_PROFILE_SCOPE(name)
#endif

} // namespace spm


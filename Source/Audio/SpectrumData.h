#pragma once

#include <vector>
#include <juce_core/juce_core.h>

namespace spm {

/**
 * Spectrum Data Structure
 * 独立定义以避免头文件循环依赖
 */
struct SpectrumData {
    std::vector<float> frequencies;     // FFT bin center frequencies (Hz)
    std::vector<float> magnitudes;      // Magnitude spectrum
    std::vector<float> refinedFreqs;    // Phase vocoder refined frequencies (Hz)
    double timestamp;
    float sampleRate = 44100.0f;        // Sample rate used for FFT
    int fftSize = 4096;                 // FFT size
    int hopSize = 512;                  // Hop size for phase calculation
    bool hasRefinedFreqs = false;       // Whether refined frequencies are available
    
    // Optional: raw audio data for time-domain algorithms (YIN)
    std::vector<float> rawAudio;
    bool hasRawAudio = false;
};

/**
 * Pitch Candidate Structure
 */
struct PitchCandidate {
    float frequency = 0.0f;       // Frequency (Hz)
    float midiNote = 0.0f;        // MIDI note number (with decimal for cents)
    float centsDeviation = 0.0f;  // Deviation from standard pitch in cents
    float confidence = 0.0f;      // Confidence level (0-1)
    float amplitude = 0.0f;       // Relative amplitude
    int harmonicCount = 0;        // Number of detected harmonics
};

using PitchVector = std::vector<PitchCandidate>;

} // namespace spm

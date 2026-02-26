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
    std::vector<float> magnitudes;      // Magnitude spectrum (for backward compatibility)
    std::vector<float> refinedFreqs;    // Phase vocoder refined frequencies (Hz)
    double timestamp;
    float sampleRate = 44100.0f;        // Sample rate used for FFT
    int fftSize = 4096;                 // FFT size
    int hopSize = 512;                  // Hop size for phase calculation
    bool hasRefinedFreqs = false;       // Whether refined frequencies are available
    
    // Optional: raw audio data for time-domain algorithms (YIN)
    std::vector<float> rawAudio;
    bool hasRawAudio = false;
    
    // ML Mode: Separate confidence and energy data
    std::vector<float> mlConfidence;    // ML confidence for each bin (0-1)
    std::vector<float> mlEnergy;        // ML energy for each bin
    bool isMLMode = false;              // Whether this data is from ML model
    bool isFFTMode = false;             // Whether this data is from FFT analysis
};

/**
 * Pitch Candidate Structure
 */
struct PitchCandidate {
    float frequency = 0.0f;       // Frequency (Hz)
    float midiNote = 0.0f;        // MIDI note number (with decimal for cents)
    float centsDeviation = 0.0f;  // Deviation from standard pitch in cents
    float confidence = 0.0f;      // Confidence level (0-1)
    float amplitude = 0.0f;       // Relative amplitude (FFT) or energy (ML)
    int harmonicCount = 0;        // Number of detected harmonics
    bool isMLEnergy = false;      // If true, amplitude is ML energy (not dB)
};

using PitchVector = std::vector<PitchCandidate>;

} // namespace spm

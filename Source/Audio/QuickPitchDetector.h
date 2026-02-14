#pragma once

#include <juce_core/juce_core.h>
#include <juce_audio_basics/juce_audio_basics.h>
#include <vector>

namespace spm {

/**
 * Quick Pitch Detector for fast note detection
 * 
 * Uses small window (512-1024 samples) for low latency.
 * Optimized for detecting fast note changes rather than high precision.
 */
class QuickPitchDetector
{
public:
    QuickPitchDetector();
    ~QuickPitchDetector();

    /**
     * Initialize the detector
     * @param sampleRate Audio sample rate
     * @param minFreq Minimum detectable frequency (Hz)
     * @param maxFreq Maximum detectable frequency (Hz)
     */
    void prepare(double sampleRate, float minFreq, float maxFreq);

    /**
     * Detect pitch from audio buffer using small window
     * @param audio Input audio samples (mono)
     * @param numSamples Number of samples (should be >= 512)
     * @return Detected frequency in Hz, or 0 if no pitch found
     */
    float detectPitch(const float* audio, int numSamples);

    /**
     * Get the confidence of the last detection (0-1)
     */
    float getLastConfidence() const { return lastConfidence_; }

    /**
     * Set the detection threshold (default 0.3, lower = more sensitive)
     */
    void setThreshold(float threshold) { threshold_ = threshold; }

private:
    // Simplified normalized autocorrelation
    float calculateNACF(const float* audio, int numSamples, int lag);
    
    // Find peak in NACF
    int findPeakLag();
    
    // Parabolic interpolation
    float interpolatePeak(int peakLag);

    double sampleRate_ = 44100.0;
    float minFreq_ = 20.0f;
    float maxFreq_ = 5000.0f;
    float threshold_ = 0.3f;
    
    int minLag_ = 0;
    int maxLag_ = 0;
    
    std::vector<float> nacfBuffer_;
    float lastConfidence_ = 0.0f;
    bool prepared_ = false;
};

} // namespace spm

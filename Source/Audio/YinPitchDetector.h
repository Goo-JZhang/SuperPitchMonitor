#pragma once

#include <juce_core/juce_core.h>
#include <juce_audio_basics/juce_audio_basics.h>
#include <vector>
#include <algorithm>
#include <cmath>

namespace spm {

/**
 * YIN Pitch Detection Algorithm
 * 
 * Optimized for monophonic pitch detection with high accuracy.
 * Uses cumulative mean normalized difference function (CMND).
 * 
 * Reference: "YIN, a fundamental frequency estimator for speech and music"
 *            by de Cheveigne and Kawahara (2002)
 */
class YinPitchDetector
{
public:
    YinPitchDetector();
    ~YinPitchDetector();

    /**
     * Initialize the detector
     * @param sampleRate Audio sample rate
     * @param minFreq Minimum detectable frequency (Hz)
     * @param maxFreq Maximum detectable frequency (Hz)
     * @param bufferSize Input buffer size (default 2048)
     */
    void prepare(double sampleRate, float minFreq, float maxFreq, int bufferSize = 2048);

    /**
     * Detect pitch from audio buffer
     * @param audio Input audio samples (mono)
     * @param numSamples Number of samples
     * @return Detected frequency in Hz, or 0 if no pitch found
     */
    float detectPitch(const float* audio, int numSamples);

    /**
     * Get the confidence/probability of the last detection (0-1)
     */
    float getLastConfidence() const { return lastConfidence_; }

    /**
     * Get the periodicity of the last detection (0-1, higher = more periodic)
     */
    float getLastPeriodicity() const { return lastPeriodicity_; }

    /**
     * Set the threshold for voiced/unvoiced decision (default 0.1)
     * Lower = more sensitive to weak pitches, but more noise
     * Higher = stricter, rejects more noise
     */
    void setThreshold(float threshold) { threshold_ = threshold; }

private:
    // Core YIN algorithm steps
    void calculateDifference(const float* audio, int numSamples);
    void cumulativeMeanNormalizedDifference();
    int findBestTau();
    float parabolicInterpolation(int tauEstimate);

    // Parameters
    double sampleRate_ = 44100.0;
    float minFreq_ = 20.0f;
    float maxFreq_ = 5000.0f;
    float threshold_ = 0.15f;  // YIN threshold for voiced detection
    
    // Calculated search range
    int minTau_ = 0;
    int maxTau_ = 0;
    int bufferSize_ = 2048;

    // Buffers
    std::vector<float> difference_;  // Difference function D(tau)
    std::vector<float> cmnd_;        // Cumulative mean normalized difference
    std::vector<float> audioBuffer_; // Padded audio buffer

    // Results
    float lastConfidence_ = 0.0f;
    float lastPeriodicity_ = 0.0f;
    float lastTau_ = 0.0f;
    
    bool prepared_ = false;
};

} // namespace spm

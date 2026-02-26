#pragma once

#include <juce_core/juce_core.h>
#include <juce_dsp/juce_dsp.h>
#include <vector>
#include <complex>
#include "../Utils/Config.h"

namespace spm {

/**
 * Nonlinear Fourier Analyzer
 * 
 * Computes a "Nonlinear Fourier Transform" with logarithmically-spaced frequency bins.
 * Unlike standard FFT which uses linear frequency spacing, this analyzer allows
 * arbitrary frequency bin placement - specifically log-spaced bins matching the ML model.
 * 
 * Key features:
 * - 2048 frequency bins logarithmically spaced from 20Hz to 5000Hz
 * - 4096-sample window (matches ML model input)
 * - Pre-computed basis matrices (cosine/sine) for efficient computation
 * - Provides magnitude, phase, and refined frequency estimates (phase vocoder)
 * - Compatible with YIN algorithm (raw audio buffer access)
 * 
 * Mathematical basis:
 * For each target frequency f_k, the Fourier coefficient is:
 *   X(f_k) = sum_{n=0}^{N-1} x[n] * exp(-j*2*pi*f_k*n/fs)
 *          = sum_{n=0}^{N-1} x[n] * cos(2*pi*f_k*n/fs) - j*sum_{n=0}^{N-1} x[n] * sin(2*pi*f_k*n/fs)
 * 
 * The cosine and sine terms are pre-computed into matrices:
 *   cosMatrix[k][n] = cos(2*pi*f_k*n/fs)
 *   sinMatrix[k][n] = sin(2*pi*f_k*n/fs)
 */
class NonlinearFourierAnalyzer
{
public:
    NonlinearFourierAnalyzer();
    ~NonlinearFourierAnalyzer();

    /**
     * Initialize the analyzer
     * @param sampleRate Audio sample rate (typically 44100 Hz)
     */
    void prepare(double sampleRate);
    
    /**
     * Check if the analyzer has been prepared
     */
    bool isPrepared() const { return prepared_; }
    
    /**
     * Process audio buffer and compute nonlinear Fourier transform
     * @param input Input audio buffer (mono)
     * @param magnitudes Output magnitude spectrum (size = NumBins)
     * @param phases Output phase spectrum in radians (size = NumBins)
     * @param frequencies Output bin center frequencies (size = NumBins)
     */
    void process(const juce::AudioBuffer<float>& input,
                 std::vector<float>& magnitudes,
                 std::vector<float>& phases,
                 std::vector<float>& frequencies);
    
    /**
     * Get the current hop size
     */
    int getHopSize() const { return hopSize_; }
    
    /**
     * Get the window size
     */
    int getWindowSize() const { return windowSize_; }
    
    /**
     * Get the number of frequency bins
     */
    static constexpr int getNumBins() { return Config::Spectrum::NonlinearFourierBins; }
    
    /**
     * Get frequency for a specific bin
     */
    float getBinFrequency(int binIndex) const;
    
    /**
     * Compute refined frequencies using phase vocoder technique
     * @param currentPhases Current frame phases
     * @param prevPhases Previous frame phases  
     * @param hopSize Hop size between frames
     * @param refinedFreqs Output refined frequencies
     */
    void computeRefinedFrequencies(const std::vector<float>& currentPhases,
                                   const std::vector<float>& prevPhases,
                                   int hopSize,
                                   std::vector<float>& refinedFreqs) const;
    
    /**
     * Get the raw audio window (for YIN algorithm compatibility)
     * @return Vector containing the most recent window of audio samples
     */
    const std::vector<float>& getRawAudioWindow() const { return rawAudioWindow_; }

private:
    // Parameters
    double sampleRate_ = Config::Audio::DefaultSampleRate;
    int windowSize_ = Config::Spectrum::NonlinearFourierWindowSize;
    int hopSize_ = Config::Spectrum::DefaultHopSize;
    int numBins_ = Config::Spectrum::NonlinearFourierBins;
    float minFreq_ = Config::Spectrum::NonlinearFourierMinFreq;
    float maxFreq_ = Config::Spectrum::NonlinearFourierMaxFreq;
    
    // Pre-computed basis matrices (numBins x windowSize)
    // Using float for memory efficiency and cache performance
    std::vector<float> cosMatrix_;  // Flattened: [bin][sample] = cosMatrix_[bin * windowSize_ + sample]
    std::vector<float> sinMatrix_;
    
    // Frequency bin center frequencies
    std::vector<float> binFrequencies_;
    
    // Circular buffer for sliding window processing
    std::vector<float> circularBuffer_;
    int writePos_ = 0;
    int samplesSinceTransform_ = 0;
    
    // Raw audio window (for YIN compatibility)
    std::vector<float> rawAudioWindow_;
    
    // Previous phases for phase vocoder
    std::vector<float> prevPhases_;
    bool firstFrame_ = true;
    
    // Window function (Hann window)
    std::vector<float> windowFunction_;
    
    // State
    bool prepared_ = false;
    
    // Internal methods
    void precomputeBasisMatrices();
    void computeLogSpacedFrequencies();
    void createWindowFunction();
    void performTransform(const float* audio, std::vector<float>& magnitudes, std::vector<float>& phases);
    void copyToWindowBuffer(const float* input, int numSamples);
    void applyWindow(float* buffer, int numSamples);
};

/**
 * Nonlinear Fourier Spectrum Data
 * Structured output compatible with existing pitch detection algorithms
 */
struct NonlinearFourierData {
    std::vector<float> frequencies;     // Bin center frequencies (Hz) - log spaced
    std::vector<float> magnitudes;      // Magnitude spectrum
    std::vector<float> phases;          // Phase spectrum (radians)
    std::vector<float> refinedFreqs;    // Phase vocoder refined frequencies
    std::vector<float> rawAudio;        // Raw audio window (for YIN)
    
    double timestamp = 0.0;
    float sampleRate = 44100.0f;
    bool hasRefinedFreqs = false;
    
    void resize(size_t numBins) {
        frequencies.resize(numBins);
        magnitudes.resize(numBins);
        phases.resize(numBins);
        refinedFreqs.resize(numBins);
        rawAudio.resize(Config::Spectrum::NonlinearFourierWindowSize);
    }
    
    void clear() {
        frequencies.clear();
        magnitudes.clear();
        phases.clear();
        refinedFreqs.clear();
        rawAudio.clear();
        hasRefinedFreqs = false;
    }
};

} // namespace spm

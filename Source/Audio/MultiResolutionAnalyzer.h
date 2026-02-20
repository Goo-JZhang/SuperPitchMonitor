#pragma once

#include <juce_core/juce_core.h>
#include <juce_dsp/juce_dsp.h>
#include <array>
#include <future>
#include <functional>
#include "SpectrumData.h"

namespace spm {

// Forward declaration
class YinPitchDetector;

/**
 * Resolution Band Configuration
 * Defines processing parameters for different frequency bands
 */
struct ResolutionBand {
    // Frequency range
    float minFreq = 0.0f;
    float maxFreq = 20000.0f;
    
    // FFT parameters
    int fftOrder = 12;        // Default 4096 points
    int fftSize = 4096;
    int hopSize = 512;
    
    // Window function type
    enum WindowType { Hann, Hamming, Blackman, FlatTop };
    WindowType windowType = Hann;
    
    // Processing strategy
    enum Strategy { 
        HighPrecision,    // Pursue frequency precision (long window)
        FastResponse,     // Pursue time precision (short window)
        Balanced          // Balanced mode
    };
    Strategy strategy = Balanced;
    
    // Pre-computed parameters
    float binWidth = 0.0f;    // Hz per bin
    float timeResolution = 0.0f;  // ms
    
    void calculateParams(double sampleRate) {
        fftSize = 1 << fftOrder;
        binWidth = static_cast<float>(sampleRate) / fftSize;
        timeResolution = 1000.0f * hopSize / static_cast<float>(sampleRate);
    }
};

/**
 * Single band analysis result
 * Independent data structure designed for parallel processing
 */
struct BandSpectrumData {
    int bandIndex = 0;
    
    // Spectrum data
    std::vector<float> frequencies;
    std::vector<float> magnitudes;
    std::vector<float> phases;
    
    // Phase vocoder refined frequencies
    std::vector<float> refinedFreqs;
    bool hasRefinedFreqs = false;
    
    // Time domain data (for YIN algorithm)
    std::vector<float> timeDomain;
    bool hasTimeDomain = false;
    
    // YIN refined pitch (if available)
    float yinFrequency = 0.0f;
    float yinConfidence = 0.0f;
    bool hasYinResult = false;
    
    // Metadata
    double timestamp = 0.0;
    float sampleRate = 44100.0f;
    
    // Clear data
    void clear() {
        frequencies.clear();
        magnitudes.clear();
        phases.clear();
        refinedFreqs.clear();
        timeDomain.clear();
        hasRefinedFreqs = false;
        hasTimeDomain = false;
        hasYinResult = false;
        yinFrequency = 0.0f;
        yinConfidence = 0.0f;
    }
};

/**
 * Multi-resolution spectrum data
 */
struct MultiResolutionData {
    // Independent results for each band
    std::array<BandSpectrumData, 3> bands;
    
    // Fused unified spectrum (for compatibility with existing code)
    SpectrumData fusedSpectrum;
    
    // Processing status
    bool isComplete = false;
    double processingTimeMs = 0.0;
    
    // Quick access to each band
    BandSpectrumData& lowBand() { return bands[0]; }      // < 500Hz
    BandSpectrumData& midBand() { return bands[1]; }      // 500-4000Hz
    BandSpectrumData& highBand() { return bands[2]; }     // > 4000Hz
    
    const BandSpectrumData& lowBand() const { return bands[0]; }
    const BandSpectrumData& midBand() const { return bands[1]; }
    const BandSpectrumData& highBand() const { return bands[2]; }
};

/**
 * Band analysis task (reserved for parallelization)
 */
class BandAnalysisTask {
public:
    BandAnalysisTask(const ResolutionBand& config, int bandIdx);
    ~BandAnalysisTask();
    
    // Prepare for processing
    void prepare(double sampleRate);
    
    // Process audio block (current synchronous implementation)
    void process(const juce::AudioBuffer<float>& input, BandSpectrumData& output);
    
    // Future parallel interface (placeholder for now)
    // std::future<BandSpectrumData> processAsync(const juce::AudioBuffer<float>& input);
    
private:
    ResolutionBand config_;
    int bandIndex_ = 0;
    double sampleRate_ = 44100.0;
    
    // FFT related
    std::unique_ptr<juce::dsp::FFT> fft_;
    juce::AudioBuffer<float> fftBuffer_;
    juce::AudioBuffer<float> windowBuffer_;
    std::vector<float> windowCoeffs_;
    
    // Phase vocoder history
    std::vector<float> prevPhases_;
    bool firstFrame_ = true;
    
    // Circular buffer (supports sliding window)
    juce::AudioBuffer<float> circularBuffer_;
    int writePos_ = 0;
    int samplesSinceFFT_ = 0;
    
    // YIN detector for this band
    std::unique_ptr<YinPitchDetector> yinDetector_;
    int yinBufferSize_ = 2048;
    
    void createWindow();
    void performFFT(BandSpectrumData& output);
    void extractMagnitudesAndPhases(BandSpectrumData& output);
    void calculateRefinedFrequencies(BandSpectrumData& output);
    void copyToFFTBuffer();
    void performYinAnalysis(const juce::AudioBuffer<float>& input, BandSpectrumData& output);
};

/**
 * Multi-resolution spectrum analyzer
 * 
 * Design goals:
 * 1. Support different resolution processing for low/mid/high frequency bands
 * 2. Interface reserved for parallel optimization
 * 3. Flexible configuration for each band
 */
class MultiResolutionAnalyzer {
public:
    MultiResolutionAnalyzer();
    ~MultiResolutionAnalyzer();

    // Initialize
    void prepare(double sampleRate);
    
    // Process audio block (main entry)
    void process(const juce::AudioBuffer<float>& input, MultiResolutionData& output);
    
    // Get fused standard spectrum (compatible with existing code)
    void getFusedSpectrum(const MultiResolutionData& multiData, SpectrumData& output);
    
    // Configuration interface
    void setLowBandConfig(const ResolutionBand& config);
    void setMidBandConfig(const ResolutionBand& config);
    void setHighBandConfig(const ResolutionBand& config);
    
    // Parallel control (reserved for future)
    void setParallelEnabled(bool enabled) { parallelEnabled_ = enabled; }
    bool isParallelEnabled() const { return parallelEnabled_; }
    
    // Get band configuration
    const ResolutionBand& getBandConfig(int bandIndex) const;

private:
    // Three band processing tasks
    std::array<std::unique_ptr<BandAnalysisTask>, 3> bandTasks_;
    
    // Sample rate
    double sampleRate_ = 44100.0;
    
    // Parallel flag (placeholder for future versions)
    bool parallelEnabled_ = false;
    
    // Default configuration
    void setupDefaultConfigs();
    
    // Spectrum fusion
    void fuseSpectrums(MultiResolutionData& data);
    
    // Frequency to band index mapping
    int freqToBandIndex(float freq) const;
};

} // namespace spm

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
#include "../Utils/Config.h"

namespace spm {

/**
 * Spectrum Data Structure
 */
struct SpectrumData;

/**
 * Spectrum Analyzer with Sliding Window (Hop-based) Processing
 * 
 * Key improvements for fast note detection:
 * - Overlapping windows: Each FFT uses 4096 samples with 512-sample hop
 * - Update rate: ~86Hz @ 44.1kHz (11.6ms interval)
 * - Always provides fresh data for pitch detection
 */
class SpectrumAnalyzer
{
public:
    SpectrumAnalyzer();
    ~SpectrumAnalyzer();

    // Initialize - call when sample rate or FFT size changes
    void prepare(double sampleRate, int fftOrder);
    
    // Process audio buffer - performs FFT when hop size is reached
    void process(const juce::AudioBuffer<float>& input, SpectrumData& output);

private:
    // FFT related
    std::unique_ptr<juce::dsp::FFT> fft_;
    int fftOrder_ = Config::Spectrum::DefaultFFTOrder;
    int fftSize_ = 1 << Config::Spectrum::DefaultFFTOrder;
    
    // Sliding window buffers
    juce::AudioBuffer<float> windowBuffer_;     // Window function
    juce::AudioBuffer<float> fftBuffer_;        // FFT working buffer
    juce::AudioBuffer<float> inputBuffer_;      // Circular buffer for sliding window
    
    // Sliding window state
    int bufferWritePos_ = 0;                    // Write position in circular buffer
    int samplesSinceLastFFT_ = 0;               // Counter for hop size
    
    // Parameters
    double sampleRate_ = Config::Audio::DefaultSampleRate;
    int hopSize_ = 512;                         // 512 samples = ~11.6ms @ 44.1kHz
    
    // Frequency cache
    std::vector<float> frequencyCache_;
    
    // Cached spectrum data
    std::vector<float> lastMagnitudes_;
    std::vector<float> prevPhases_;
    std::vector<float> refinedFreqs_;
    double lastTimestamp_ = 0;
    
    // Initialization flag
    bool prepared_ = false;
    
    void createWindow();
    void performFFT();
    void extractMagnitudes(std::vector<float>& magnitudes);
    void calculateRefinedFrequencies();
    
    // Copy samples from circular buffer to FFT buffer with windowing
    void copyToFFTBuffer();
};

} // namespace spm

#include "SpectrumAnalyzer.h"
#include "Audio/AudioEngine.h"
#include "../Utils/Logger.h"

namespace spm {

SpectrumAnalyzer::SpectrumAnalyzer() = default;
SpectrumAnalyzer::~SpectrumAnalyzer() = default;

void SpectrumAnalyzer::prepare(double sampleRate, int fftOrder)
{
    sampleRate_ = sampleRate;
    fftOrder_ = fftOrder;
    fftSize_ = 1 << fftOrder;
    hopSize_ = fftSize_ / 8;  // 87.5% overlap (512 hop @ 4096 FFT)
    
    // Create FFT
    fft_ = std::make_unique<juce::dsp::FFT>(fftOrder_);
    
    // Allocate buffers
    windowBuffer_.setSize(1, fftSize_);
    fftBuffer_.setSize(1, fftSize_ * 2);  // FFT needs 2x space for complex
    inputBuffer_.setSize(1, fftSize_);    // Circular buffer for sliding window
    
    // Create window (Hann window)
    createWindow();
    
    // Pre-calculate frequencies
    frequencyCache_.resize(fftSize_ / 2 + 1);
    float binWidth = static_cast<float>(sampleRate_ / fftSize_);
    for (int i = 0; i <= fftSize_ / 2; ++i)
    {
        frequencyCache_[i] = i * binWidth;
    }
    
    // Initialize phase vocoder state
    prevPhases_.resize(fftSize_ / 2 + 1, 0.0f);
    refinedFreqs_.resize(fftSize_ / 2 + 1, 0.0f);
    
    // Reset state
    bufferWritePos_ = 0;
    samplesSinceLastFFT_ = 0;
    inputBuffer_.clear();
    
    prepared_ = true;
    
    SPM_LOG_INFO("[SpectrumAnalyzer] Prepared: FFT=" + juce::String(fftSize_) 
                 + " hop=" + juce::String(hopSize_)
                 + " overlap=" + juce::String(87.5f, 1) + "%"
                 + " rate=" + juce::String(sampleRate_ / hopSize_, 1) + "Hz");
}

void SpectrumAnalyzer::process(const juce::AudioBuffer<float>& input, SpectrumData& output)
{
    if (!prepared_)
    {
        SPM_LOG_DEBUG("[SpectrumAnalyzer] Not prepared");
        return;
    }
    
    const float* inputData = input.getReadPointer(0);
    int numSamples = input.getNumSamples();
    bool fftPerformed = false;
    
    // Write samples to circular buffer
    for (int i = 0; i < numSamples; ++i)
    {
        // Write to circular buffer
        inputBuffer_.setSample(0, bufferWritePos_, inputData[i]);
        bufferWritePos_ = (bufferWritePos_ + 1) % fftSize_;
        samplesSinceLastFFT_++;
        
        // Check if we've accumulated enough samples for next FFT
        if (samplesSinceLastFFT_ >= hopSize_)
        {
            samplesSinceLastFFT_ = 0;
            
            // Copy from circular buffer to FFT buffer with windowing
            copyToFFTBuffer();
            
            // Perform FFT
            performFFT();
            
            // Extract results
            extractMagnitudes(lastMagnitudes_);
            calculateRefinedFrequencies();
            
            lastTimestamp_ = juce::Time::getMillisecondCounterHiRes();
            fftPerformed = true;
            
            // Debug logging
            static int fftCount = 0;
            if (++fftCount % 50 == 0)
            {
                float maxMag = 0.0f;
                int maxBin = 0;
                for (int j = 0; j < (int)lastMagnitudes_.size(); ++j)
                {
                    if (lastMagnitudes_[j] > maxMag)
                    {
                        maxMag = lastMagnitudes_[j];
                        maxBin = j;
                    }
                }
                SPM_LOG_INFO("[SpectrumAnalyzer] FFT #" + juce::String(fftCount) 
                             + " peak=" + juce::String(frequencyCache_[maxBin], 1) + "Hz"
                             + " mag=" + juce::String(maxMag, 4));
            }
        }
    }
    
    // Output latest available data
    if (!lastMagnitudes_.empty())
    {
        output.frequencies = frequencyCache_;
        output.magnitudes = lastMagnitudes_;
        output.refinedFreqs = refinedFreqs_;
        output.timestamp = lastTimestamp_;
        output.sampleRate = (float)sampleRate_;
        output.fftSize = fftSize_;
        output.hopSize = hopSize_;
        output.hasRefinedFreqs = true;
        
        // Copy raw audio for time-domain algorithms
        // Provide most recent fftSize_ samples in chronological order
        output.rawAudio.resize(fftSize_);
        for (int i = 0; i < fftSize_; ++i)
        {
            // Read from circular buffer: most recent samples
            int readPos = (bufferWritePos_ - fftSize_ + i + fftSize_) % fftSize_;
            output.rawAudio[i] = inputBuffer_.getSample(0, readPos);
        }
        output.hasRawAudio = true;
    }
}

void SpectrumAnalyzer::copyToFFTBuffer()
{
    auto* dest = fftBuffer_.getWritePointer(0);
    auto* window = windowBuffer_.getReadPointer(0);
    
    // Copy fftSize_ samples from circular buffer, starting from (writePos - fftSize_)
    // These are the most recent samples
    int startPos = (bufferWritePos_ - fftSize_ + fftSize_) % fftSize_;
    
    for (int i = 0; i < fftSize_; ++i)
    {
        int readPos = (startPos + i) % fftSize_;
        dest[i] = inputBuffer_.getSample(0, readPos) * window[i];
    }
}

void SpectrumAnalyzer::createWindow()
{
    auto* w = windowBuffer_.getWritePointer(0);
    
    // Hann window
    for (int i = 0; i < fftSize_; ++i)
    {
        w[i] = 0.5f - 0.5f * std::cos(2.0f * juce::MathConstants<float>::pi * i / (fftSize_ - 1));
    }
}

void SpectrumAnalyzer::performFFT()
{
    auto* dest = fftBuffer_.getWritePointer(0);
    fft_->performRealOnlyForwardTransform(dest, true);
}

void SpectrumAnalyzer::extractMagnitudes(std::vector<float>& magnitudes)
{
    int numBins = fftSize_ / 2 + 1;
    magnitudes.resize(numBins);
    
    auto* fftData = fftBuffer_.getReadPointer(0);
    
    for (int i = 0; i < numBins; ++i)
    {
        float real = fftData[i * 2];
        float imag = fftData[i * 2 + 1];
        magnitudes[i] = std::sqrt(real * real + imag * imag);
    }
}

void SpectrumAnalyzer::calculateRefinedFrequencies()
{
    int numBins = fftSize_ / 2 + 1;
    float binWidth = static_cast<float>(sampleRate_ / fftSize_);
    
    auto* fftData = fftBuffer_.getReadPointer(0);
    const float twoPi = 2.0f * juce::MathConstants<float>::pi;
    
    for (int i = 0; i < numBins; ++i)
    {
        float real = fftData[i * 2];
        float imag = fftData[i * 2 + 1];
        float currentPhase = std::atan2(imag, real);
        
        // Expected phase advance for this bin
        float expectedPhaseDiff = twoPi * hopSize_ * i / fftSize_;
        
        // Phase difference
        float phaseDiff = currentPhase - prevPhases_[i] - expectedPhaseDiff;
        
        // Wrap to -PI ~ PI
        while (phaseDiff > juce::MathConstants<float>::pi)
            phaseDiff -= twoPi;
        while (phaseDiff < -juce::MathConstants<float>::pi)
            phaseDiff += twoPi;
        
        // Frequency deviation
        float freqDeviation = phaseDiff * sampleRate_ / (twoPi * hopSize_);
        refinedFreqs_[i] = i * binWidth + freqDeviation;
        
        prevPhases_[i] = currentPhase;
    }
}

} // namespace spm

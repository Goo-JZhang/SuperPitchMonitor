#include "NonlinearFourierAnalyzer.h"
#include "../Utils/Logger.h"
#include <cmath>

namespace {
    constexpr double PI = 3.14159265358979323846;
}

namespace spm {

NonlinearFourierAnalyzer::NonlinearFourierAnalyzer()
{
}

NonlinearFourierAnalyzer::~NonlinearFourierAnalyzer()
{
}

void NonlinearFourierAnalyzer::prepare(double sampleRate)
{
    sampleRate_ = sampleRate;
    
    // Resize buffers
    circularBuffer_.resize(windowSize_, 0.0f);
    rawAudioWindow_.resize(windowSize_, 0.0f);
    prevPhases_.resize(numBins_, 0.0f);
    binFrequencies_.resize(numBins_);
    
    // Pre-compute basis matrices
    precomputeBasisMatrices();
    
    // Compute frequency bins
    computeLogSpacedFrequencies();
    
    // Create window function
    createWindowFunction();
    
    // Reset state
    writePos_ = 0;
    samplesSinceTransform_ = 0;
    firstFrame_ = true;
    std::fill(circularBuffer_.begin(), circularBuffer_.end(), 0.0f);
    std::fill(prevPhases_.begin(), prevPhases_.end(), 0.0f);
    
    prepared_ = true;
    
    SPM_LOG_INFO("[NonlinearFourierAnalyzer] Prepared: " + juce::String(numBins_) + 
                 " bins, " + juce::String(windowSize_) + " samples, " +
                 juce::String(sampleRate_) + " Hz");
}

void NonlinearFourierAnalyzer::precomputeBasisMatrices()
{
    // Allocate flattened matrices
    const size_t matrixSize = static_cast<size_t>(numBins_) * windowSize_;
    cosMatrix_.resize(matrixSize);
    sinMatrix_.resize(matrixSize);
    
    // Compute log-spaced frequencies first
    computeLogSpacedFrequencies();
    
    // Pre-compute basis functions for each frequency bin
    const double twoPiOverFs = 2.0 * PI / sampleRate_;
    
    for (int bin = 0; bin < numBins_; ++bin)
    {
        const double freq = binFrequencies_[bin];
        const double phaseIncrement = twoPiOverFs * freq;
        
        const size_t baseIndex = static_cast<size_t>(bin) * windowSize_;
        
        for (int n = 0; n < windowSize_; ++n)
        {
            const double phase = phaseIncrement * n;
            cosMatrix_[baseIndex + n] = static_cast<float>(std::cos(phase));
            sinMatrix_[baseIndex + n] = static_cast<float>(std::sin(phase));
        }
    }
    
    SPM_LOG_INFO("[NonlinearFourierAnalyzer] Precomputed basis matrices: " +
                 juce::String(matrixSize * 2 * sizeof(float) / 1024 / 1024) + " MB");
}

void NonlinearFourierAnalyzer::computeLogSpacedFrequencies()
{
    // Logarithmically spaced frequencies from minFreq to maxFreq
    const double logMin = std::log(minFreq_);
    const double logMax = std::log(maxFreq_);
    const double logStep = (logMax - logMin) / (numBins_ - 1);
    
    for (int i = 0; i < numBins_; ++i)
    {
        const double logFreq = logMin + i * logStep;
        binFrequencies_[i] = static_cast<float>(std::exp(logFreq));
    }
}

void NonlinearFourierAnalyzer::createWindowFunction()
{
    windowFunction_.resize(windowSize_);
    
    // Hann window
    for (int n = 0; n < windowSize_; ++n)
    {
        const double hann = 0.5 * (1.0 - std::cos(2.0 * PI * n / (windowSize_ - 1)));
        windowFunction_[n] = static_cast<float>(hann);
    }
}

float NonlinearFourierAnalyzer::getBinFrequency(int binIndex) const
{
    if (binIndex < 0 || binIndex >= numBins_)
        return 0.0f;
    return binFrequencies_[binIndex];
}

void NonlinearFourierAnalyzer::process(const juce::AudioBuffer<float>& input,
                                       std::vector<float>& magnitudes,
                                       std::vector<float>& phases,
                                       std::vector<float>& frequencies)
{
    if (!prepared_)
    {
        SPM_LOG_ERROR("[NonlinearFourierAnalyzer] Not prepared!");
        return;
    }
    
    const int numSamples = input.getNumSamples();
    const float* inputData = input.getReadPointer(0);
    
    // Resize output vectors
    magnitudes.resize(numBins_);
    phases.resize(numBins_);
    frequencies.resize(numBins_);
    
    // Copy frequencies (they don't change)
    std::copy(binFrequencies_.begin(), binFrequencies_.end(), frequencies.begin());
    
    // Write to circular buffer
    for (int i = 0; i < numSamples; ++i)
    {
        circularBuffer_[writePos_] = inputData[i];
        writePos_ = (writePos_ + 1) % windowSize_;
        samplesSinceTransform_++;
        
        // Perform transform when hop size is reached
        if (samplesSinceTransform_ >= hopSize_)
        {
            performTransform(nullptr, magnitudes, phases);
            samplesSinceTransform_ = 0;
        }
    }
}

void NonlinearFourierAnalyzer::performTransform(const float* /*audio*/,
                                                std::vector<float>& magnitudes,
                                                std::vector<float>& phases)
{
    // Copy circular buffer to linear buffer (oldest to newest)
    std::vector<float> windowBuffer(windowSize_);
    for (int i = 0; i < windowSize_; ++i)
    {
        int idx = (writePos_ + i) % windowSize_;
        windowBuffer[i] = circularBuffer_[idx];
        rawAudioWindow_[i] = circularBuffer_[idx];  // Save for YIN
    }
    
    // Apply window function
    applyWindow(windowBuffer.data(), windowSize_);
    
    // Compute DFT for each frequency bin using pre-computed basis
    for (int bin = 0; bin < numBins_; ++bin)
    {
        float real = 0.0f;
        float imag = 0.0f;
        
        const size_t baseIndex = static_cast<size_t>(bin) * windowSize_;
        
        // Matrix multiplication: X[k] = sum(x[n] * basis[k][n])
        // Using manual unrolling for better performance
        int n = 0;
        
        // Process 4 samples at a time
        for (; n <= windowSize_ - 4; n += 4)
        {
            const float x0 = windowBuffer[n];
            const float x1 = windowBuffer[n + 1];
            const float x2 = windowBuffer[n + 2];
            const float x3 = windowBuffer[n + 3];
            
            real += x0 * cosMatrix_[baseIndex + n]
                  + x1 * cosMatrix_[baseIndex + n + 1]
                  + x2 * cosMatrix_[baseIndex + n + 2]
                  + x3 * cosMatrix_[baseIndex + n + 3];
                  
            imag -= x0 * sinMatrix_[baseIndex + n]
                  + x1 * sinMatrix_[baseIndex + n + 1]
                  + x2 * sinMatrix_[baseIndex + n + 2]
                  + x3 * sinMatrix_[baseIndex + n + 3];
        }
        
        // Process remaining samples
        for (; n < windowSize_; ++n)
        {
            real += windowBuffer[n] * cosMatrix_[baseIndex + n];
            imag -= windowBuffer[n] * sinMatrix_[baseIndex + n];
        }
        
        // Compute magnitude and phase
        magnitudes[bin] = std::sqrt(real * real + imag * imag);
        phases[bin] = std::atan2(imag, real);
    }
}

void NonlinearFourierAnalyzer::applyWindow(float* buffer, int numSamples)
{
    for (int i = 0; i < numSamples; ++i)
    {
        buffer[i] *= windowFunction_[i];
    }
}

void NonlinearFourierAnalyzer::computeRefinedFrequencies(const std::vector<float>& currentPhases,
                                                          const std::vector<float>& prevPhases,
                                                          int hopSize,
                                                          std::vector<float>& refinedFreqs) const
{
    if (currentPhases.size() != static_cast<size_t>(numBins_) || 
        prevPhases.size() != static_cast<size_t>(numBins_))
    {
        refinedFreqs.resize(numBins_);
        std::copy(binFrequencies_.begin(), binFrequencies_.end(), refinedFreqs.begin());
        return;
    }
    
    refinedFreqs.resize(numBins_);
    
    const float hopSizeF = static_cast<float>(hopSize);
    const float sampleRateF = static_cast<float>(sampleRate_);
    const float twoPi = 2.0f * static_cast<float>(PI);
    
    for (int bin = 0; bin < numBins_; ++bin)
    {
        // Phase difference
        float phaseDiff = currentPhases[bin] - prevPhases[bin];
        
        // Unwrap phase (handle 2*pi jumps)
        while (phaseDiff > PI) phaseDiff -= twoPi;
        while (phaseDiff < -PI) phaseDiff += twoPi;
        
        // Expected phase increment for this bin frequency
        const float expectedPhaseInc = twoPi * binFrequencies_[bin] * hopSizeF / sampleRateF;
        
        // True frequency = expected + deviation
        // deviation = phaseDiff / (2*pi*hopSize/fs)
        const float deviation = phaseDiff * sampleRateF / (twoPi * hopSizeF);
        refinedFreqs[bin] = binFrequencies_[bin] + deviation;
    }
}

} // namespace spm

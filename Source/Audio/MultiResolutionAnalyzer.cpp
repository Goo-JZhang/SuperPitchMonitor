#include "MultiResolutionAnalyzer.h"
#include "YinPitchDetector.h"
#include "../Utils/Logger.h"

namespace spm {

//==============================================================================
// BandAnalysisTask Implementation
//==============================================================================

BandAnalysisTask::BandAnalysisTask(const ResolutionBand& config, int bandIdx)
    : config_(config)
    , bandIndex_(bandIdx)
{
}

BandAnalysisTask::~BandAnalysisTask() = default;

void BandAnalysisTask::prepare(double sampleRate)
{
    sampleRate_ = sampleRate;
    config_.calculateParams(sampleRate);
    
    // Create FFT instance
    fft_ = std::make_unique<juce::dsp::FFT>(config_.fftOrder);
    
    // Allocate buffers
    fftBuffer_.setSize(1, config_.fftSize * 2);  // FFT needs 2x space for complex numbers
    windowBuffer_.setSize(1, config_.fftSize);
    windowCoeffs_.resize(config_.fftSize);
    prevPhases_.resize(config_.fftSize / 2 + 1, 0.0f);
    
    // Create circular buffer (stores 2 FFT window lengths, supports sliding window)
    circularBuffer_.setSize(1, config_.fftSize * 2);
    circularBuffer_.clear();
    writePos_ = 0;
    samplesSinceFFT_ = 0;
    
    // Initialize YIN detector - strictly limited to frequency band
    // Avoid cross-band detection (e.g., low-band YIN shouldn't detect C2 when input is C4 chord)
    yinBufferSize_ = 2048;  // Use 2048 uniformly to balance accuracy and latency
    
    yinDetector_ = std::make_unique<YinPitchDetector>();
    
    // Strictly limit YIN detection range within band to avoid false detection
    float yinMinFreq = config_.minFreq;
    float yinMaxFreq = config_.maxFreq;
    
    // Special handling: low band doesn't use YIN (FFT is accurate enough)
    if (bandIndex_ == 0) {
        yinMinFreq = 80.0f;  // Low-band YIN only detects above 80Hz (lowest E2)
        yinMaxFreq = 400.0f; // Don't detect too high to avoid band conflicts
    }
    
    yinDetector_->prepare(sampleRate, yinMinFreq, yinMaxFreq, yinBufferSize_);
    yinDetector_->setThreshold(0.2f);  // Increase threshold to reduce false positives
    
    createWindow();
    
    SPM_LOG_INFO("[BandAnalysisTask] Band " + juce::String(bandIndex_) + 
                 " prepared: FFT=" + juce::String(config_.fftSize) + 
                 ", YIN=" + juce::String(yinMinFreq, 0) + "-" + juce::String(yinMaxFreq, 0) + "Hz" +
                 ", binWidth=" + juce::String(config_.binWidth, 2) + "Hz");
}

void BandAnalysisTask::createWindow()
{
    const int N = config_.fftSize;
    
    switch (config_.windowType) {
        case ResolutionBand::Hann:
            for (int i = 0; i < N; ++i)
                windowCoeffs_[i] = 0.5f * (1.0f - std::cos(2.0f * juce::MathConstants<float>::pi * i / (N - 1)));
            break;
            
        case ResolutionBand::Hamming:
            for (int i = 0; i < N; ++i)
                windowCoeffs_[i] = 0.54f - 0.46f * std::cos(2.0f * juce::MathConstants<float>::pi * i / (N - 1));
            break;
            
        case ResolutionBand::Blackman:
            for (int i = 0; i < N; ++i) {
                float a0 = 0.42f;
                float a1 = 0.5f;
                float a2 = 0.08f;
                windowCoeffs_[i] = a0 - a1 * std::cos(2.0f * juce::MathConstants<float>::pi * i / (N - 1))
                                     + a2 * std::cos(4.0f * juce::MathConstants<float>::pi * i / (N - 1));
            }
            break;
            
        case ResolutionBand::FlatTop:
            for (int i = 0; i < N; ++i) {
                float a0 = 0.21557895f;
                float a1 = 0.41663158f;
                float a2 = 0.277263158f;
                float a3 = 0.083578947f;
                float a4 = 0.006947368f;
                windowCoeffs_[i] = a0 - a1 * std::cos(2.0f * juce::MathConstants<float>::pi * i / (N - 1))
                                     + a2 * std::cos(4.0f * juce::MathConstants<float>::pi * i / (N - 1))
                                     - a3 * std::cos(6.0f * juce::MathConstants<float>::pi * i / (N - 1))
                                     + a4 * std::cos(8.0f * juce::MathConstants<float>::pi * i / (N - 1));
            }
            break;
    }
}

void BandAnalysisTask::process(const juce::AudioBuffer<float>& input, BandSpectrumData& output)
{
    const int numSamples = input.getNumSamples();
    const float* inputData = input.getReadPointer(0);
    
    // Write to circular buffer
    for (int i = 0; i < numSamples; ++i) {
        circularBuffer_.setSample(0, writePos_, inputData[i]);
        writePos_ = (writePos_ + 1) % circularBuffer_.getNumSamples();
    }
    
    samplesSinceFFT_ += numSamples;
    
    // Check if hop size reached
    if (samplesSinceFFT_ < config_.hopSize) {
        // Try running YIN even without new FFT (using history buffer)
        // But only when previous FFT data exists
        if (bandIndex_ > 0 && !output.magnitudes.empty()) {
            performYinAnalysis(input, output);
        }
        output.hasRefinedFreqs = false;  // Mark FFT data as not updated
        return;
    }
    
    samplesSinceFFT_ = 0;
    
    // Copy data to FFT buffer and apply window
    copyToFFTBuffer();
    
    // Execute FFT (must be before YIN, YIN needs spectrum verification)
    performFFT(output);
    
    // Extract magnitude and phase
    extractMagnitudesAndPhases(output);
    
    // Phase vocoder refinement
    calculateRefinedFrequencies(output);
    
    // Set metadata
    output.bandIndex = bandIndex_;
    output.sampleRate = static_cast<float>(sampleRate_);
    output.hasRefinedFreqs = true;
    
    // YIN analysis: all bands use time-domain for precise frequency
    // Run after FFT for spectrum verification
    // Key fix: low band also needs YIN to distinguish dense fundamentals (e.g., C4-E4-G4 chord)
    performYinAnalysis(input, output);
}

void BandAnalysisTask::performYinAnalysis(const juce::AudioBuffer<float>& input, 
                                           BandSpectrumData& output)
{
    // Key: first check if target band has energy in spectrum
    // If spectrum shows no peaks in band, YIN shouldn't detect pitch
    if (output.magnitudes.empty()) {
        output.hasYinResult = false;
        return;
    }
    
    // Calculate spectral energy in target band
    float bandEnergy = 0.0f;
    float totalEnergy = 0.0f;
    int bandBinStart = static_cast<int>(config_.minFreq / config_.binWidth);
    int bandBinEnd = static_cast<int>(config_.maxFreq / config_.binWidth);
    bandBinStart = std::max(1, bandBinStart);
    bandBinEnd = std::min(static_cast<int>(output.magnitudes.size()) - 1, bandBinEnd);
    
    for (size_t i = 0; i < output.magnitudes.size(); ++i) {
        float mag = output.magnitudes[i];
        totalEnergy += mag * mag;
        if (i >= bandBinStart && i <= bandBinEnd) {
            bandEnergy += mag * mag;
        }
    }
    
    // If target band energy ratio too low, don't run YIN
    float energyRatio = (totalEnergy > 0) ? bandEnergy / totalEnergy : 0.0f;
    if (energyRatio < 0.1f || bandEnergy < 1.0f) {  // Need at least 10% energy in band
        output.hasYinResult = false;
        return;
    }
    
    // Check for clear peaks in target band
    bool hasPeakInBand = false;
    float maxBandMag = 0.0f;
    for (int i = bandBinStart; i <= bandBinEnd && i < (int)output.magnitudes.size(); ++i) {
        if (output.magnitudes[i] > maxBandMag) {
            maxBandMag = output.magnitudes[i];
        }
    }
    
    // Need at least one local max significantly above surroundings
    for (int i = bandBinStart + 2; i < bandBinEnd - 2 && i < (int)output.magnitudes.size() - 2; ++i) {
        float mag = output.magnitudes[i];
        if (mag > maxBandMag * 0.5f &&  // Is one of main peaks
            mag > output.magnitudes[i-1] && mag > output.magnitudes[i-2] &&
            mag > output.magnitudes[i+1] && mag > output.magnitudes[i+2]) {
            hasPeakInBand = true;
            break;
        }
    }
    
    if (!hasPeakInBand) {
        output.hasYinResult = false;
        return;
    }
    
    // Read last yinBufferSize_ samples from circular buffer
    std::vector<float> yinBuffer(yinBufferSize_);
    
    for (int i = 0; i < yinBufferSize_; ++i) {
        int readPos = (writePos_ - yinBufferSize_ + i + circularBuffer_.getNumSamples()) 
                      % circularBuffer_.getNumSamples();
        yinBuffer[i] = circularBuffer_.getSample(0, readPos);
    }
    
    // Check signal energy (avoid false detection in silence)
    float rms = 0.0f;
    for (float s : yinBuffer) rms += s * s;
    rms = std::sqrt(rms / yinBufferSize_);
    
    if (rms < 0.005f) {  // Raise noise gate
        output.hasYinResult = false;
        return;
    }
    
    // Execute YIN detection
    float yinFreq = yinDetector_->detectPitch(yinBuffer.data(), yinBufferSize_);
    float confidence = yinDetector_->getLastConfidence();
    
    // Verify YIN result matches spectrum: energy must exist near YIN frequency
    bool spectrumConfirms = false;
    if (yinFreq >= config_.minFreq && yinFreq <= config_.maxFreq) {
        int expectedBin = static_cast<int>(yinFreq / config_.binWidth);
        for (int offset = -2; offset <= 2; ++offset) {
            int checkBin = expectedBin + offset;
            if (checkBin >= 0 && checkBin < (int)output.magnitudes.size()) {
                if (output.magnitudes[checkBin] > maxBandMag * 0.3f) {
                    spectrumConfirms = true;
                    break;
                }
            }
        }
    }
    
    // Strict verification: result must be in band range and match spectrum
    if (yinFreq >= config_.minFreq && yinFreq <= config_.maxFreq && 
        confidence > 0.5f && spectrumConfirms)  // Raise confidence threshold
    {
        output.yinFrequency = yinFreq;
        output.yinConfidence = confidence;
        output.hasYinResult = true;
        
        // Save time-domain data for later use
        output.timeDomain = std::move(yinBuffer);
        output.hasTimeDomain = true;
    }
    else
    {
        output.hasYinResult = false;
    }
}

void BandAnalysisTask::copyToFFTBuffer()
{
    const int N = config_.fftSize;
    
    // Read last N samples from circular buffer
    for (int i = 0; i < N; ++i) {
        int readPos = (writePos_ - N + i + circularBuffer_.getNumSamples()) 
                      % circularBuffer_.getNumSamples();
        float sample = circularBuffer_.getSample(0, readPos);
        
        // Apply window and store to FFT buffer (interleaved real/imag)
        fftBuffer_.setSample(0, i * 2, sample * windowCoeffs_[i]);
        fftBuffer_.setSample(0, i * 2 + 1, 0.0f);
    }
}

void BandAnalysisTask::performFFT(BandSpectrumData& output)
{
    // Execute FFT
    fft_->performRealOnlyForwardTransform(fftBuffer_.getWritePointer(0), true);
    
    // Prepare output arrays
    const int numBins = config_.fftSize / 2 + 1;
    output.frequencies.resize(numBins);
    output.magnitudes.resize(numBins);
    output.phases.resize(numBins);
    output.refinedFreqs.resize(numBins);
    
    // Calculate frequency scale
    for (int i = 0; i < numBins; ++i) {
        output.frequencies[i] = i * config_.binWidth;
    }
}

void BandAnalysisTask::extractMagnitudesAndPhases(BandSpectrumData& output)
{
    const int numBins = static_cast<int>(output.magnitudes.size());
    
    for (int i = 0; i < numBins; ++i) {
        float real = fftBuffer_.getSample(0, i * 2);
        float imag = fftBuffer_.getSample(0, i * 2 + 1);
        
        output.magnitudes[i] = std::sqrt(real * real + imag * imag);
        output.phases[i] = std::atan2(imag, real);
    }
}

void BandAnalysisTask::calculateRefinedFrequencies(BandSpectrumData& output)
{
    const int numBins = static_cast<int>(output.magnitudes.size());
    const float sampleRate = config_.binWidth * config_.fftSize;
    const float hopSize = static_cast<float>(config_.hopSize);
    const float twoPi = 2.0f * juce::MathConstants<float>::pi;
    
    for (int i = 0; i < numBins; ++i) {
        float phaseCurrent = output.phases[i];
        float phasePrev = prevPhases_[i];
        
        // Phase difference
        float phaseDiff = phaseCurrent - phasePrev;
        
        // Unwrap
        float expectedPhaseDiff = twoPi * hopSize * i / config_.fftSize;
        phaseDiff -= expectedPhaseDiff;
        
        while (phaseDiff > juce::MathConstants<float>::pi)
            phaseDiff -= twoPi;
        while (phaseDiff < -juce::MathConstants<float>::pi)
            phaseDiff += twoPi;
        
        // Calculate refined frequency
        float binFreq = output.frequencies[i];
        float refinedFreq = binFreq + phaseDiff * sampleRate / (twoPi * hopSize);
        
        // Limit refined frequency to reasonable range (±50% of bin freq)
        if (std::abs(refinedFreq - binFreq) > binFreq * 0.5f) {
            refinedFreq = binFreq;
        }
        
        output.refinedFreqs[i] = refinedFreq;
        prevPhases_[i] = phaseCurrent;
    }
}

//==============================================================================
// MultiResolutionAnalyzer Implementation
//==============================================================================

MultiResolutionAnalyzer::MultiResolutionAnalyzer()
{
    setupDefaultConfigs();
}

MultiResolutionAnalyzer::~MultiResolutionAnalyzer() = default;

void MultiResolutionAnalyzer::setupDefaultConfigs()
{
    // Low band: < 400Hz, long window for high frequency precision
    // No YIN, rely on FFT only (8192 points give 5.4Hz resolution, accurate enough)
    ResolutionBand low;
    low.minFreq = 50.0f;      // Minimum 50Hz (lowest range)
    low.maxFreq = 400.0f;     // Upper limit 400Hz, no overlap with mid band
    low.fftOrder = 13;        // 8192 points
    low.hopSize = 512;
    low.windowType = ResolutionBand::Blackman;
    low.strategy = ResolutionBand::HighPrecision;
    
    // Mid band: 400-2000Hz, balanced settings
    // This is main fundamental region for piano chords
    ResolutionBand mid;
    mid.minFreq = 400.0f;     // No overlap with low band
    mid.maxFreq = 2000.0f;    // Upper limit 2000Hz, covers to C7
    mid.fftOrder = 12;        // 4096 points, 10.8Hz resolution
    mid.hopSize = 512;
    mid.windowType = ResolutionBand::Hann;
    mid.strategy = ResolutionBand::Balanced;
    
    // High band: 2000-6000Hz, for overtone verification
    // Short window for fast response, not for fundamental detection
    ResolutionBand high;
    high.minFreq = 2000.0f;   // No overlap with mid band
    high.maxFreq = 6000.0f;   // Upper limit 6000Hz
    high.fftOrder = 11;       // 2048 points, 21.5Hz resolution
    high.hopSize = 512;
    high.windowType = ResolutionBand::Hann;
    high.strategy = ResolutionBand::FastResponse;
    
    bandTasks_[0] = std::make_unique<BandAnalysisTask>(low, 0);
    bandTasks_[1] = std::make_unique<BandAnalysisTask>(mid, 1);
    bandTasks_[2] = std::make_unique<BandAnalysisTask>(high, 2);
}

void MultiResolutionAnalyzer::prepare(double sampleRate)
{
    sampleRate_ = sampleRate;
    
    for (int i = 0; i < 3; ++i) {
        if (bandTasks_[i]) {
            bandTasks_[i]->prepare(sampleRate);
        }
    }
    
    SPM_LOG_INFO("[MultiResolutionAnalyzer] Prepared at " + 
                 juce::String(sampleRate) + "Hz");
}

void MultiResolutionAnalyzer::process(const juce::AudioBuffer<float>& input, 
                                       MultiResolutionData& output)
{
    auto startTime = juce::Time::getHighResolutionTicks();
    
    // Current version: sequential processing (structure reserved for future parallelization)
    for (int i = 0; i < 3; ++i) {
        bandTasks_[i]->process(input, output.bands[i]);
    }
    
    // Merge results
    fuseSpectrums(output);
    
    // Calculate processing time
    auto endTime = juce::Time::getHighResolutionTicks();
    output.processingTimeMs = juce::Time::highResolutionTicksToSeconds(
        endTime - startTime) * 1000.0;
    output.isComplete = true;
}

void MultiResolutionAnalyzer::fuseSpectrums(MultiResolutionData& data)
{
    // Build unified merged spectrum
    const int totalBins = 2048;
    
    auto& fused = data.fusedSpectrum;
    fused.frequencies.resize(totalBins);
    fused.magnitudes.resize(totalBins);
    fused.refinedFreqs.resize(totalBins);
    fused.sampleRate = static_cast<float>(sampleRate_);
    fused.fftSize = totalBins * 2;
    
    // Calculate merged frequency scale
    for (int i = 0; i < totalBins; ++i) {
        fused.frequencies[i] = i * sampleRate_ / (2.0 * totalBins);
    }
    
    // Fill data from each band
    // Low frequency (<400Hz)
    auto& low = data.lowBand();
    if (low.hasRefinedFreqs && !low.frequencies.empty()) {
        float lowBinWidth = low.frequencies[1] - low.frequencies[0];
        int lowBins = std::min((int)low.magnitudes.size(), 
                                static_cast<int>(400.0f * 2.0f * totalBins / sampleRate_));
        for (int i = 0; i < lowBins; ++i) {
            if (i < (int)low.magnitudes.size()) {
                float freq = i * lowBinWidth;
                int targetBin = static_cast<int>(freq * 2.0 * totalBins / sampleRate_);
                if (targetBin < totalBins) {
                    fused.magnitudes[targetBin] = low.magnitudes[i];
                    fused.refinedFreqs[targetBin] = low.refinedFreqs[i];
                }
            }
        }
    }
    
    // Mid frequency (400-2000Hz) - use YIN results for refinement
    auto& mid = data.midBand();
    if (mid.hasRefinedFreqs) {
        int midStart = static_cast<int>(400.0f * 2.0 * totalBins / sampleRate_);
        int midEnd = static_cast<int>(2000.0f * 2.0 * totalBins / sampleRate_);
        
        for (size_t i = 0; i < mid.magnitudes.size(); ++i) {
            float freq = mid.frequencies[i];
            if (freq < 400.0f || freq > 2000.0f) continue;
            
            int targetBin = static_cast<int>(freq * 2.0 * totalBins / sampleRate_);
            if (targetBin >= midStart && targetBin < midEnd && targetBin < totalBins) {
                fused.magnitudes[targetBin] = mid.magnitudes[i];
                
                // Mid band prefers YIN results (if available)
                if (mid.hasYinResult && mid.yinConfidence > 0.5f &&
                    std::abs(freq - mid.yinFrequency) / mid.yinFrequency < 0.03f) {
                    fused.refinedFreqs[targetBin] = mid.yinFrequency;
                } else {
                    fused.refinedFreqs[targetBin] = mid.refinedFreqs[i];
                }
            }
        }
    }
    
    // High frequency (2000-6000Hz) - for overtone verification only
    auto& high = data.highBand();
    if (high.hasRefinedFreqs) {
        int highStart = static_cast<int>(2000.0f * 2.0 * totalBins / sampleRate_);
        
        for (size_t i = 0; i < high.magnitudes.size(); ++i) {
            float freq = high.frequencies[i];
            if (freq < 2000.0f || freq > 6000.0f) continue;
            
            int targetBin = static_cast<int>(freq * 2.0 * totalBins / sampleRate_);
            if (targetBin >= highStart && targetBin < totalBins) {
                fused.magnitudes[targetBin] = high.magnitudes[i];
                fused.refinedFreqs[targetBin] = high.refinedFreqs[i];
            }
        }
    }
    
    fused.hasRefinedFreqs = true;
}

void MultiResolutionAnalyzer::getFusedSpectrum(const MultiResolutionData& multiData, 
                                                SpectrumData& output)
{
    output = multiData.fusedSpectrum;
}

void MultiResolutionAnalyzer::setLowBandConfig(const ResolutionBand& config)
{
    bandTasks_[0] = std::make_unique<BandAnalysisTask>(config, 0);
    if (sampleRate_ > 0) {
        bandTasks_[0]->prepare(sampleRate_);
    }
}

void MultiResolutionAnalyzer::setMidBandConfig(const ResolutionBand& config)
{
    bandTasks_[1] = std::make_unique<BandAnalysisTask>(config, 1);
    if (sampleRate_ > 0) {
        bandTasks_[1]->prepare(sampleRate_);
    }
}

void MultiResolutionAnalyzer::setHighBandConfig(const ResolutionBand& config)
{
    bandTasks_[2] = std::make_unique<BandAnalysisTask>(config, 2);
    if (sampleRate_ > 0) {
        bandTasks_[2]->prepare(sampleRate_);
    }
}

const ResolutionBand& MultiResolutionAnalyzer::getBandConfig(int bandIndex) const
{
    static ResolutionBand dummy;
    return dummy;
}

int MultiResolutionAnalyzer::freqToBandIndex(float freq) const
{
    if (freq < 400.0f) return 0;
    if (freq < 2000.0f) return 1;
    return 2;
}

} // namespace spm

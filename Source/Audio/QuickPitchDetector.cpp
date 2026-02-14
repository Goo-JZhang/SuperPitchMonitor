#include "QuickPitchDetector.h"
#include "../Utils/Logger.h"
#include <cmath>

namespace spm {

QuickPitchDetector::QuickPitchDetector() = default;
QuickPitchDetector::~QuickPitchDetector() = default;

void QuickPitchDetector::prepare(double sampleRate, float minFreq, float maxFreq)
{
    sampleRate_ = sampleRate;
    minFreq_ = minFreq;
    maxFreq_ = maxFreq;
    
    // Calculate lag range
    // lag = sampleRate / frequency
    minLag_ = static_cast<int>(sampleRate_ / maxFreq_);
    maxLag_ = static_cast<int>(sampleRate_ / minFreq_);
    
    minLag_ = std::max(2, minLag_);
    maxLag_ = std::min(512, maxLag_);  // Limit for quick detection
    
    nacfBuffer_.resize(maxLag_ + 2);
    
    prepared_ = true;
    
    SPM_LOG_INFO("[QuickPitchDetector] Prepared: sampleRate=" + juce::String(sampleRate, 0)
                 + " minFreq=" + juce::String(minFreq_, 1)
                 + " maxFreq=" + juce::String(maxFreq_, 1)
                 + " minLag=" + juce::String(minLag_)
                 + " maxLag=" + juce::String(maxLag_));
}

float QuickPitchDetector::detectPitch(const float* audio, int numSamples)
{
    if (!prepared_ || numSamples < 512)
    {
        return 0.0f;
    }
    
    // Use only first 512 samples for quick detection (~11.6ms @ 44.1kHz)
    int samplesToUse = std::min(numSamples, 512);
    
    // Calculate NACF for each lag
    int searchEnd = std::min(maxLag_, samplesToUse / 2);
    
    for (int lag = minLag_; lag <= searchEnd; ++lag)
    {
        nacfBuffer_[lag] = calculateNACF(audio, samplesToUse, lag);
    }
    
    // Find peak
    int peakLag = findPeakLag();
    
    if (peakLag <= 0)
    {
        lastConfidence_ = 0.0f;
        return 0.0f;
    }
    
    // Interpolate for better precision
    float refinedLag = interpolatePeak(peakLag);
    
    // Calculate frequency
    float frequency = static_cast<float>(sampleRate_ / refinedLag);
    
    // Confidence based on peak height
    lastConfidence_ = nacfBuffer_[peakLag];
    
    // Clamp confidence
    if (lastConfidence_ < threshold_)
    {
        lastConfidence_ *= 0.5f;
    }
    
    return frequency;
}

float QuickPitchDetector::calculateNACF(const float* audio, int numSamples, int lag)
{
    // Normalized Autocorrelation Function
    // NACF(lag) = sum(x[n] * x[n+lag]) / sqrt(sum(x[n]^2) * sum(x[n+lag]^2))
    
    float sumProduct = 0.0f;
    float sumX1 = 0.0f;
    float sumX2 = 0.0f;
    
    int count = numSamples - lag;
    
    for (int i = 0; i < count; ++i)
    {
        float x1 = audio[i];
        float x2 = audio[i + lag];
        
        sumProduct += x1 * x2;
        sumX1 += x1 * x1;
        sumX2 += x2 * x2;
    }
    
    float denominator = std::sqrt(sumX1 * sumX2);
    
    if (denominator < 1e-10f)
    {
        return 0.0f;
    }
    
    return sumProduct / denominator;
}

int QuickPitchDetector::findPeakLag()
{
    int searchEnd = std::min(maxLag_, static_cast<int>(nacfBuffer_.size()) - 2);
    
    float maxValue = threshold_;
    int maxLag = -1;
    
    for (int lag = minLag_; lag <= searchEnd; ++lag)
    {
        float val = nacfBuffer_[lag];
        
        // Check if local maximum
        if (val > maxValue && val > nacfBuffer_[lag - 1] && val > nacfBuffer_[lag + 1])
        {
            maxValue = val;
            maxLag = lag;
        }
    }
    
    return maxLag;
}

float QuickPitchDetector::interpolatePeak(int peakLag)
{
    if (peakLag <= 0 || peakLag >= static_cast<int>(nacfBuffer_.size()) - 1)
    {
        return static_cast<float>(peakLag);
    }
    
    float alpha = nacfBuffer_[peakLag - 1];
    float beta = nacfBuffer_[peakLag];
    float gamma = nacfBuffer_[peakLag + 1];
    
    float denominator = alpha - 2.0f * beta + gamma;
    
    if (std::abs(denominator) < 1e-10f)
    {
        return static_cast<float>(peakLag);
    }
    
    float delta = 0.5f * (alpha - gamma) / denominator;
    delta = juce::jlimit(-0.5f, 0.5f, delta);
    
    return static_cast<float>(peakLag) + delta;
}

} // namespace spm

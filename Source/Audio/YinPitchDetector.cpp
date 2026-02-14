#include "YinPitchDetector.h"
#include "../Utils/Logger.h"

namespace spm {

YinPitchDetector::YinPitchDetector() = default;
YinPitchDetector::~YinPitchDetector() = default;

void YinPitchDetector::prepare(double sampleRate, float minFreq, float maxFreq, int bufferSize)
{
    sampleRate_ = sampleRate;
    minFreq_ = minFreq;
    maxFreq_ = maxFreq;
    bufferSize_ = bufferSize;
    
    // Calculate tau range based on frequency range
    // tau = sampleRate / frequency
    maxTau_ = static_cast<int>(sampleRate_ / minFreq_);
    minTau_ = static_cast<int>(sampleRate_ / maxFreq_);
    
    // Ensure valid range
    minTau_ = std::max(2, minTau_);
    maxTau_ = std::min(bufferSize_ / 2, maxTau_);
    
    // Allocate buffers
    int maxTauSize = maxTau_ + 2;  // Extra for interpolation
    difference_.resize(maxTauSize);
    cmnd_.resize(maxTauSize);
    audioBuffer_.resize(bufferSize_ + maxTauSize);
    
    prepared_ = true;
    
    SPM_LOG_INFO("[YinPitchDetector] Prepared: sampleRate=" + juce::String(sampleRate, 0)
                 + " minFreq=" + juce::String(minFreq_, 1)
                 + " maxFreq=" + juce::String(maxFreq_, 1)
                 + " minTau=" + juce::String(minTau_)
                 + " maxTau=" + juce::String(maxTau_));
}

float YinPitchDetector::detectPitch(const float* audio, int numSamples)
{
    if (!prepared_ || numSamples < maxTau_ * 2)
    {
        SPM_LOG_DEBUG("[YinPitchDetector] Not prepared or insufficient samples");
        return 0.0f;
    }
    
    // Limit to our buffer size
    int samplesToUse = std::min(numSamples, bufferSize_);
    
    // Step 1: Calculate difference function
    calculateDifference(audio, samplesToUse);
    
    // Step 2: Cumulative mean normalized difference
    cumulativeMeanNormalizedDifference();
    
    // Step 3: Find best tau (period estimate)
    int tau = findBestTau();
    
    if (tau <= 0)
    {
        lastConfidence_ = 0.0f;
        lastPeriodicity_ = 0.0f;
        return 0.0f;  // No pitch detected
    }
    
    // Step 4: Parabolic interpolation for sub-sample precision
    float refinedTau = parabolicInterpolation(tau);
    
    // Calculate frequency
    float frequency = static_cast<float>(sampleRate_ / refinedTau);
    
    // Calculate confidence based on CMND value at tau
    float cmndAtTau = cmnd_[tau];
    lastPeriodicity_ = 1.0f - cmndAtTau;
    
    // Confidence calculation
    // For pure sine wave, CMND should be very close to 0 at the correct tau
    float confidence = 1.0f - cmndAtTau;
    
    // For strong periodic signals (like sine wave), boost confidence
    if (cmndAtTau < 0.1f)
    {
        confidence = 0.9f + 0.1f * (1.0f - cmndAtTau / 0.1f);
    }
    else if (cmndAtTau < threshold_)
    {
        confidence = 0.7f + 0.2f * (1.0f - (cmndAtTau - 0.1f) / (threshold_ - 0.1f));
    }
    else
    {
        confidence *= 0.5f;
    }
    
    lastConfidence_ = juce::jlimit(0.0f, 1.0f, confidence);
    lastTau_ = refinedTau;
    
    // Debug logging for 440Hz test
    static int debugCount = 0;
    if (++debugCount % 30 == 0)
    {
        SPM_LOG_INFO("[YinPitchDetector] f=" + juce::String(frequency, 2) 
                     + "Hz tau=" + juce::String(tau)
                     + " cmnd=" + juce::String(cmndAtTau, 4)
                     + " conf=" + juce::String(lastConfidence_, 2));
    }
    
    return frequency;
}

void YinPitchDetector::calculateDifference(const float* audio, int numSamples)
{
    // YIN difference function: D(tau) = sum((x[t] - x[t+tau])^2)
    
    // Clear buffers
    std::fill(difference_.begin(), difference_.end(), 0.0f);
    
    // For small tau values, direct calculation is accurate enough
    int maxTau = std::min(maxTau_, numSamples / 2);
    
    // Pre-calculate energy for normalization (optional optimization)
    float totalEnergy = 0.0f;
    for (int i = 0; i < numSamples; ++i)
    {
        totalEnergy += audio[i] * audio[i];
    }
    
    for (int tau = minTau_; tau <= maxTau; ++tau)
    {
        float sum = 0.0f;
        int count = numSamples - tau;
        
        for (int i = 0; i < count; ++i)
        {
            float diff = audio[i] - audio[i + tau];
            sum += diff * diff;
        }
        
        // Normalize by count to make values comparable across different tau
        difference_[tau] = sum / count;
    }
}

void YinPitchDetector::cumulativeMeanNormalizedDifference()
{
    // CMND: d'(tau) = d(tau) / ((1/tau) * sum(d(j)))
    // Special case: at tau=0, d'(0) = 1
    
    // Fill initial values with 1.0 (no periodicity)
    std::fill(cmnd_.begin(), cmnd_.end(), 1.0f);
    
    float runningSum = 0.0f;
    int maxTau = std::min(maxTau_, static_cast<int>(difference_.size()) - 1);
    
    for (int tau = minTau_; tau <= maxTau; ++tau)
    {
        runningSum += difference_[tau];
        
        if (runningSum > 0.0f && tau > 0)
        {
            float mean = runningSum / static_cast<float>(tau - minTau_ + 1);
            if (mean > 0.0f)
            {
                cmnd_[tau] = difference_[tau] / mean;
            }
        }
    }
}

int YinPitchDetector::findBestTau()
{
    // Find first local minimum below threshold
    int maxTau = std::min(maxTau_, static_cast<int>(cmnd_.size()) - 2);
    
    float globalMinValue = 1.0f;
    int globalMinTau = -1;
    
    for (int tau = minTau_; tau <= maxTau; ++tau)
    {
        float current = cmnd_[tau];
        
        // Track global minimum
        if (current < globalMinValue)
        {
            globalMinValue = current;
            globalMinTau = tau;
        }
        
        // Check if below threshold and is a local minimum
        if (current < threshold_)
        {
            // Local minimum: lower than neighbors
            bool isLocalMin = (current <= cmnd_[tau - 1]) && (current <= cmnd_[tau + 1]);
            
            if (isLocalMin)
            {
                // Additional check: should be significantly lower than neighbors
                float neighborAvg = (cmnd_[tau - 1] + cmnd_[tau + 1]) * 0.5f;
                if (neighborAvg > current * 1.1f)  // At least 10% lower
                {
                    return tau;
                }
            }
        }
    }
    
    // If no clear local minimum found, use global minimum if it's reasonably low
    if (globalMinTau > 0 && globalMinValue < threshold_ * 1.5f)
    {
        return globalMinTau;
    }
    
    return -1;  // No valid pitch
}

float YinPitchDetector::parabolicInterpolation(int tauEstimate)
{
    // Parabolic interpolation around the minimum for sub-sample precision
    if (tauEstimate <= 0 || tauEstimate >= static_cast<int>(cmnd_.size()) - 1)
    {
        return static_cast<float>(tauEstimate);
    }
    
    float alpha = cmnd_[tauEstimate - 1];
    float beta = cmnd_[tauEstimate];
    float gamma = cmnd_[tauEstimate + 1];
    
    float denominator = alpha - 2.0f * beta + gamma;
    
    if (std::abs(denominator) < 1e-10f)
    {
        return static_cast<float>(tauEstimate);
    }
    
    float delta = 0.5f * (alpha - gamma) / denominator;
    
    // Limit interpolation to reasonable range
    delta = juce::jlimit(-0.5f, 0.5f, delta);
    
    return static_cast<float>(tauEstimate) + delta;
}

} // namespace spm

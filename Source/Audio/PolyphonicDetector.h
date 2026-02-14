#pragma once

#include "SpectrumData.h"
#include <vector>
#include <memory>

namespace spm {

// Forward declarations
class YinPitchDetector;
class QuickPitchDetector;
struct MultiResolutionData;
struct BandSpectrumData;

struct Peak {
    int bin;
    float frequency;
    float magnitude;
    int bandIndex;  // 0=low, 1=mid, 2=high
    
    Peak() : bin(0), frequency(0.0f), magnitude(0.0f), bandIndex(1) {}
};

/**
 * Advanced polyphonic pitch detection with multi-resolution support.
 * 
 * Key improvements for orchestral music:
 * 1. Low band (<500Hz): High precision for bass/fundamental separation
 * 2. Mid band (500-4000Hz): Balanced for main melodic content
 * 3. High band (>4000Hz): Fast response for overtone analysis
 * 
 * Uses "fundamentalness" scoring to distinguish true fundamentals from harmonics.
 */
class PolyphonicDetector
{
public:
    PolyphonicDetector();
    ~PolyphonicDetector();

    void prepare(double sampleRate, float minFreq, float maxFreq);
    
    // Traditional interface (backward compatible)
    void detect(const SpectrumData& spectrum, PitchVector& results);
    
    // Multi-resolution interface (new)
    void detectMultiResolution(const MultiResolutionData& multiData, PitchVector& results);

    // Enable/disable multi-resolution mode
    void setMultiResolutionEnabled(bool enabled) { useMultiRes_ = enabled; }
    bool isMultiResolutionEnabled() const { return useMultiRes_; }

private:
    // Detection strategies
    void detectTimeDomain(const SpectrumData& spectrum, PitchVector& results);
    void detectPolyphonicFFT(const SpectrumData& spectrum, PitchVector& results);
    
    // Multi-resolution detection
    void detectMultiResolutionImpl(const MultiResolutionData& multiData, PitchVector& results);
    void detectLowBand(const BandSpectrumData& lowBand, std::vector<Peak>& allPeaks);
    void detectMidBand(const BandSpectrumData& midBand, std::vector<Peak>& allPeaks);
    void detectHighBand(const BandSpectrumData& highBand, std::vector<Peak>& allPeaks);
    
    // Peak processing (unified processing for peaks from different bands)
    void findPeaksInBand(const BandSpectrumData& bandData, 
                          std::vector<Peak>& peaks,
                          int bandIndex,
                          float threshold);
    PitchCandidate evaluateAsFundamental(
        const std::vector<Peak>& allPeaks,
        size_t peakIndex,
        const MultiResolutionData* multiData = nullptr);
    
    // Frequency refinement
    float refineFrequency(const BandSpectrumData& bandData, int peakBin);
    float interpolateFrequency(int bin, float alpha, float beta, float gamma) const;
    
    // Harmonic verification (using multi-resolution info)
    bool verifyHarmonicsInMultiRes(const PitchCandidate& candidate,
                                    const MultiResolutionData& multiData);
    float calculateHarmonicScore(const PitchCandidate& candidate,
                                  const MultiResolutionData& multiData);
    
    // Utilities
    float binToFreq(int bin, int fftSize) const;
    float freqToMidi(float freq) const;

    std::unique_ptr<YinPitchDetector> yinDetector_;
    std::unique_ptr<QuickPitchDetector> quickDetector_;
    
    double sampleRate_ = 44100.0;
    float minFreq_ = 20.0f;
    float maxFreq_ = 5000.0f;
    
    float thresholdDb_ = -30.0f;
    float noiseGate_ = 0.001f;
    
    // Multi-resolution support
    bool useMultiRes_ = false;
};

} // namespace spm

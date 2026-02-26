#include "PolyphonicDetector.h"
#include "YinPitchDetector.h"
#include "QuickPitchDetector.h"
#include "MultiResolutionAnalyzer.h"
#include "../Utils/Config.h"
#include "../Utils/Logger.h"

namespace spm {

PolyphonicDetector::PolyphonicDetector() 
    : yinDetector_(std::make_unique<YinPitchDetector>())
    , quickDetector_(std::make_unique<QuickPitchDetector>())
{
}

PolyphonicDetector::~PolyphonicDetector() = default;

void PolyphonicDetector::prepare(double sampleRate, float minFreq, float maxFreq)
{
    sampleRate_ = sampleRate;
    minFreq_ = minFreq;
    maxFreq_ = maxFreq;
    
    yinDetector_->prepare(sampleRate, minFreq, maxFreq, 2048);
    yinDetector_->setThreshold(0.15f);
    
    quickDetector_->prepare(sampleRate, minFreq, maxFreq);
    quickDetector_->setThreshold(0.3f);
    
    useMultiRes_ = false;
}

void PolyphonicDetector::detect(const SpectrumData& spectrum, PitchVector& results)
{
    results.clear();
    
    static int debugCount = 0;
    int frameNum = ++debugCount;
    
    if (spectrum.magnitudes.empty()) return;
    
    // Check noise gate
    float maxMag = *std::max_element(spectrum.magnitudes.begin(), spectrum.magnitudes.end());
    if (maxMag < noiseGate_) return;
    
    // Prepare single band data for detection
    BandSpectrumData singleBand;
    singleBand.frequencies = spectrum.frequencies;
    singleBand.magnitudes = spectrum.magnitudes;
    singleBand.refinedFreqs = spectrum.refinedFreqs;
    singleBand.hasRefinedFreqs = spectrum.hasRefinedFreqs;
    singleBand.sampleRate = spectrum.sampleRate;
    
    // Always use FFT-based polyphonic detection
    detectPolyphonicFFT(spectrum, results);
    
    // For monophonic signals, supplement with time-domain precision
    if (spectrum.hasRawAudio && results.size() == 1)
    {
        PitchVector timeDomainResults;
        detectTimeDomain(spectrum, timeDomainResults);
        
        if (!timeDomainResults.empty())
        {
            float fftFreq = results[0].frequency;
            float tdFreq = timeDomainResults[0].frequency;
            float diff = std::abs(fftFreq - tdFreq) / std::max(fftFreq, tdFreq);
            
            if (diff < 0.05f)
            {
                results[0].frequency = tdFreq;
                results[0].midiNote = freqToMidi(tdFreq);
                float roundedMidi = std::round(results[0].midiNote);
                results[0].centsDeviation = (results[0].midiNote - roundedMidi) * 100.0f;
            }
        }
    }
    
    // Log final results (frequency/amplitude/confidence)
    if (!results.empty())
    {
        juce::String logMsg = "[PITCH][Frame " + juce::String(frameNum) + "] " 
                            + juce::String(results.size()) + " pitches:";
        for (const auto& p : results)
        {
            logMsg += " " + juce::String(p.frequency, 1) + "Hz/" 
                    + juce::String(p.amplitude, 3) + "/" 
                    + juce::String(p.confidence, 2);
        }
        SPM_LOG_INFO(logMsg);
    }
}

void PolyphonicDetector::detectTimeDomain(const SpectrumData& spectrum, PitchVector& results)
{
    float quickFreq = quickDetector_->detectPitch(
        spectrum.rawAudio.data(), 
        static_cast<int>(spectrum.rawAudio.size())
    );
    float quickConf = quickDetector_->getLastConfidence();
    
    if (quickFreq > minFreq_ && quickFreq < maxFreq_ && quickConf > 0.5f)
    {
        PitchCandidate candidate;
        candidate.frequency = quickFreq;
        candidate.midiNote = freqToMidi(quickFreq);
        candidate.confidence = quickConf;
        candidate.amplitude = quickConf;
        candidate.harmonicCount = 1;
        
        float roundedMidi = std::round(candidate.midiNote);
        candidate.centsDeviation = (candidate.midiNote - roundedMidi) * 100.0f;
        
        results.push_back(candidate);
    }
}

void PolyphonicDetector::detectPolyphonicFFT(const SpectrumData& spectrum, PitchVector& results)
{
    BandSpectrumData singleBand;
    singleBand.frequencies = spectrum.frequencies;
    singleBand.magnitudes = spectrum.magnitudes;
    singleBand.refinedFreqs = spectrum.refinedFreqs;
    singleBand.hasRefinedFreqs = spectrum.hasRefinedFreqs;
    singleBand.sampleRate = spectrum.sampleRate;
    
    std::vector<Peak> peaks;
    float maxMag = *std::max_element(spectrum.magnitudes.begin(), spectrum.magnitudes.end());
    
    float threshold = std::max(maxMag * 0.005f, 0.0005f);
    findPeaksInBand(singleBand, peaks, 1, threshold);
    
    if (peaks.empty()) return;
    
    std::vector<PitchCandidate> candidates;
    
    for (size_t i = 0; i < peaks.size() && i < 20; ++i)
    {
        // Skip only very low frequencies (avoid DC and noise)
        if (peaks[i].frequency < 30.0f) continue;
        
        auto candidate = evaluateAsFundamental(peaks, i, nullptr);
        
        if (candidate.confidence > 0.2f)
        {
            candidates.push_back(candidate);
        }
    }
    
    std::sort(candidates.begin(), candidates.end(),
              [](const PitchCandidate& a, const PitchCandidate& b) {
                  return a.confidence > b.confidence;
              });
    
    std::vector<bool> peakUsed(peaks.size(), false);
    
    for (const auto& candidate : candidates)
    {
        if (results.size() >= static_cast<size_t>(Config::Pitch::MaxPolyphony))
            break;
        
        bool overlap = false;
        for (const auto& selected : results)
        {
            float ratio = candidate.frequency / selected.frequency;
            float deviation = std::abs(ratio - std::round(ratio));
            
            if (deviation < 0.03f) 
            {
                overlap = true;
                break;
            }
            
            float inverseRatio = selected.frequency / candidate.frequency;
            float inverseDev = std::abs(inverseRatio - std::round(inverseRatio));
            if (inverseDev < 0.03f && inverseRatio > 1.5f)
            {
                overlap = true;
                break;
            }
            
            float midiDiff = std::abs(candidate.midiNote - selected.midiNote);
            if (midiDiff < 0.5f)
            {
                overlap = true;
                break;
            }
        }
        
        if (!overlap)
        {
            results.push_back(candidate);
        }
    }
}

void PolyphonicDetector::detectMultiResolution(const MultiResolutionData& multiData, 
                                                PitchVector& results)
{
    results.clear();
    detectMultiResolutionImpl(multiData, results);
}

void PolyphonicDetector::detectMultiResolutionImpl(const MultiResolutionData& multiData,
                                                    PitchVector& results)
{
    static int debugCount = 0;
    int frameNum = ++debugCount;
    
    std::vector<Peak> allPeaks;
    
    // Low frequency band
    std::vector<Peak> lowPeaks;
    if (multiData.lowBand().hasRefinedFreqs) {
        detectLowBand(multiData.lowBand(), lowPeaks);
    }
    
    // Mid frequency band
    std::vector<Peak> midPeaks;
    if (multiData.midBand().hasRefinedFreqs) {
        detectMidBand(multiData.midBand(), midPeaks);
    }
    
    allPeaks.insert(allPeaks.end(), lowPeaks.begin(), lowPeaks.end());
    allPeaks.insert(allPeaks.end(), midPeaks.begin(), midPeaks.end());
    
    if (multiData.highBand().hasRefinedFreqs) {
        detectHighBand(multiData.highBand(), allPeaks);
    }
    
    if (allPeaks.empty()) return;
    
    std::sort(allPeaks.begin(), allPeaks.end(),
              [](const Peak& a, const Peak& b) { return a.magnitude > b.magnitude; });
    
    std::vector<PitchCandidate> candidates;
    
    for (size_t i = 0; i < allPeaks.size() && i < 15; ++i)
    {
        // Skip only very low frequencies (avoid DC and noise)
        if (allPeaks[i].frequency < 30.0f) continue;
        
        auto candidate = evaluateAsFundamental(allPeaks, i, &multiData);
        
        if (candidate.confidence > 0.25f)
        {
            candidates.push_back(candidate);
        }
    }
    
    // Sort by frequency from low to high (critical fix: start detection from low frequencies)
    std::sort(candidates.begin(), candidates.end(),
              [](const PitchCandidate& a, const PitchCandidate& b) {
                  return a.frequency < b.frequency;
              });
    
    // Candidates will be logged with final results
    
    // New strategy: progressive harmonic penalty
    // Process candidates from low to high, check remaining confidence after harmonic penalty
    // Lower threshold: better false positive than false negative
    const float minConfidence = 0.15f;  // Reduced from 0.25 to 0.15
    const float minConfidenceAfterPenalty = 0.10f;  // Reduced from 0.20 to 0.10
    
    std::sort(candidates.begin(), candidates.end(),
              [](const PitchCandidate& a, const PitchCandidate& b) {
                  return a.frequency < b.frequency;
              });
    
    std::vector<PitchCandidate> selectedCandidates;
    
    for (auto& cand : candidates)
    {
        if (selectedCandidates.size() >= static_cast<size_t>(Config::Pitch::MaxPolyphony))
            break;
        
        if (cand.confidence < minConfidence)
            continue;
        
        // Pre-screening: verify harmonic structure independence
        // Strategy: reduce confidence if low-freq candidate's harmonics overlap with high-freq candidates'
        if (selectedCandidates.empty()) {
            // Calculate current candidate's harmonic set
            std::vector<float> candidateHarmonics;
            for (int h = 2; h <= 6; ++h) {
                candidateHarmonics.push_back(cand.frequency * h);
            }
            
            // Calculate combined harmonic set of all other candidates
            std::vector<float> otherHarmonics;
            for (const auto& other : candidates) {
                if (other.frequency <= cand.frequency) continue;
                // Add fundamental and harmonics for each other candidate
                otherHarmonics.push_back(other.frequency);  // Fundamental
                for (int h = 2; h <= 4; ++h) {
                    otherHarmonics.push_back(other.frequency * h);
                }
            }
            
            // Check how many candidate harmonics match other candidates' harmonics (loose matching)
            int overlappingCount = 0;
            float totalOverlapScore = 0.0f;
            for (float candHarm : candidateHarmonics) {
                for (float otherHarm : otherHarmonics) {
                    float freqDiff = std::abs(candHarm - otherHarm);
                    float relativeDiff = freqDiff / std::max(candHarm, otherHarm);
                    
                    // Relax tolerance to 10%
                    if (relativeDiff < 0.10f) {
                        overlappingCount++;
                        float overlapScore = 1.0f - (relativeDiff / 0.10f);
                        totalOverlapScore += overlapScore;
                        break;  // Found one match
                    }
                }
            }
            
            // Relax pre-screening: even with harmonic overlap, only slightly reduce confidence
            // Better false positive than false negative - low-energy false positives acceptable
            float overlapRatio = candidateHarmonics.empty() ? 0.0f : 
                                (float)overlappingCount / candidateHarmonics.size();
            
            // Only slight reduction with high overlap (keep more candidates)
            if (overlapRatio >= 0.6f && totalOverlapScore >= 2.5f) {
                // Slight reduction: confidence drops to 70% at 60% overlap
                float reductionFactor = 0.70f + 0.30f * (1.0f - overlapRatio);  // 0.7~1.0
                cand.confidence *= reductionFactor;
            }
        }
        
        // Calculate harmonic penalty: based on frequency distance to selected fundamentals
        float penalty = 0.0f;
        
        for (const auto& selected : selectedCandidates)
        {
            float fundFreq = selected.frequency;
            
            // Check distance to fundamental (0th harmonic)
            float fundDiff = std::abs(cand.frequency - fundFreq);
            float fundRelativeDiff = fundDiff / fundFreq;
            
            if (fundRelativeDiff < 0.10f) {  // 10% tolerance
                // Close to fundamental itself, heavy penalty (possibly same note detected differently)
                float fundPenalty = cand.confidence * (1.0f - fundRelativeDiff / 0.10f) * 0.9f;
                penalty += fundPenalty;
            }
            
            // Check distance to each harmonic
            for (int h = 2; h <= 6; ++h)
            {
                float harmonicFreq = fundFreq * h;
                float harmonicDiff = std::abs(cand.frequency - harmonicFreq);
                float harmonicRelativeDiff = harmonicDiff / harmonicFreq;
                
                // Harmonic penalty window: based on harmonic order (higher orders wider)
                float harmonicTolerance = 0.06f + h * 0.01f;  // H2: 8%, H6: 12%
                
                if (harmonicRelativeDiff < harmonicTolerance)
                {
                    // Closer to harmonic, larger penalty
                    // Base penalty factor: 1/h (higher harmonics have smaller penalty)
                    float baseHarmonicPenalty = 1.0f / h;
                    float proximityFactor = 1.0f - (harmonicRelativeDiff / harmonicTolerance);
                    float harmonicPenalty = cand.confidence * baseHarmonicPenalty * proximityFactor;
                    
                    penalty += harmonicPenalty;
                }
            }
        }
        
        // Limit penalty to not exceed original confidence
        penalty = std::min(penalty, cand.confidence * 0.95f);
        float remainingConfidence = cand.confidence - penalty;
        
        // Decision: is remaining confidence sufficient
        bool acceptAsFundamental = false;
        if (remainingConfidence >= minConfidenceAfterPenalty) {
            acceptAsFundamental = true;
        }
        
        if (acceptAsFundamental) {
            // Remove post-verification rejection - better false positive than false negative
            // Low-energy "borrowed" harmonics acceptable (faint traces on main panel)
            
            // Update candidate confidence to remaining value
            cand.confidence = remainingConfidence;
            selectedCandidates.push_back(cand);
        }
    }
    
    // Copy results
    for (const auto& cand : selectedCandidates) {
        results.push_back(cand);
    }
    
    // Log final results (frequency/amplitude/confidence)
    if (!results.empty())
    {
        juce::String logMsg = "[PITCH][MR][Frame " + juce::String(frameNum) + "] " 
                            + juce::String(results.size()) + " pitches:";
        for (const auto& p : results)
        {
            logMsg += " " + juce::String(p.frequency, 1) + "Hz/" 
                    + juce::String(p.amplitude, 3) + "/" 
                    + juce::String(p.confidence, 2);
        }
        SPM_LOG_INFO(logMsg);
    }
}

void PolyphonicDetector::detectLowBand(const BandSpectrumData& lowBand, 
                                        std::vector<Peak>& peaks)
{
    float maxMag = *std::max_element(lowBand.magnitudes.begin(), lowBand.magnitudes.end());
    float threshold = std::max(maxMag * 0.003f, 0.0003f);
    
    findPeaksInBand(lowBand, peaks, 0, threshold);
}

void PolyphonicDetector::detectMidBand(const BandSpectrumData& midBand,
                                        std::vector<Peak>& peaks)
{
    float maxMag = *std::max_element(midBand.magnitudes.begin(), midBand.magnitudes.end());
    float threshold = std::max(maxMag * 0.003f, 0.0003f);
    
    findPeaksInBand(midBand, peaks, 1, threshold);
}

void PolyphonicDetector::detectHighBand(const BandSpectrumData& highBand,
                                         std::vector<Peak>& peaks)
{
    float maxMag = *std::max_element(highBand.magnitudes.begin(), highBand.magnitudes.end());
    float threshold = maxMag * 0.02f;
    
    findPeaksInBand(highBand, peaks, 2, threshold);
}

void PolyphonicDetector::findPeaksInBand(const BandSpectrumData& bandData,
                                          std::vector<Peak>& peaks,
                                          int bandIndex,
                                          float threshold)
{
    const auto& mags = bandData.magnitudes;
    if (mags.empty()) return;
    
    float maxMag = *std::max_element(mags.begin(), mags.end());
    float actualThreshold = std::max(threshold, maxMag * 0.0005f);
    
    float minFreq = 50.0f, maxFreq = 6000.0f;
    switch (bandIndex) {
        case 0: minFreq = 50.0f; maxFreq = 400.0f; break;
        case 1: minFreq = 400.0f; maxFreq = 2000.0f; break;
        case 2: minFreq = 2000.0f; maxFreq = 6000.0f; break;
    }
    
    // Find start and end bins that cover the frequency range
    int startBin = 2;  // Skip DC and first bin
    int endBin = static_cast<int>(mags.size()) - 2;
    
    // Find actual frequency boundaries by scanning
    if (!bandData.frequencies.empty()) {
        for (int i = 0; i < static_cast<int>(bandData.frequencies.size()); ++i) {
            if (bandData.frequencies[i] >= minFreq) {
                startBin = std::max(2, i);
                break;
            }
        }
        for (int i = static_cast<int>(bandData.frequencies.size()) - 1; i >= 0; --i) {
            if (bandData.frequencies[i] <= maxFreq) {
                endBin = std::min(static_cast<int>(mags.size()) - 2, i);
                break;
            }
        }
    }
    

    for (int i = startBin; i < endBin; ++i)
    {
        float mag = mags[i];
        if (mag < actualThreshold) continue;
        
        // Local max detection: strictly higher than both neighbors
        if (!(mag > mags[i-1] && mag > mags[i+1])) continue;
        
        // Check peak prominence: should be at least 5% above neighbors
        float neighborAvg = (mags[i-1] + mags[i+1]) * 0.5f;
        if (mag < neighborAvg * 1.05f) {
            continue;
        }
        
        Peak peak;
        peak.bin = i;
        peak.bandIndex = bandIndex;
        
        // Use raw bin frequency for peak detection
        peak.frequency = bandData.frequencies[i];
            
            // Calculate magnitude and frequency with parabolic interpolation
        float alpha = mags[i-1];
        float beta = mag;
        float gamma = mags[i+1];
        float denom = alpha - 2.0f * beta + gamma;
        if (std::abs(denom) > 1e-10f) {
            peak.magnitude = beta - 0.25f * (alpha - gamma) * (alpha - gamma) / denom;
            
            // Apply frequency correction using parabolic interpolation
            float p = 0.5f * (alpha - gamma) / denom;
            float binWidth = bandData.frequencies[i+1] - bandData.frequencies[i];
            peak.frequency = bandData.frequencies[i] + p * binWidth;
        } else {
            peak.magnitude = mag;
        }
            
            peaks.push_back(peak);
    }
}

PitchCandidate PolyphonicDetector::evaluateAsFundamental(
    const std::vector<Peak>& allPeaks,
    size_t peakIndex,
    const MultiResolutionData* multiData)
{
    PitchCandidate candidate;
    const Peak& peak = allPeaks[peakIndex];
    
    candidate.frequency = peak.frequency;
    candidate.midiNote = freqToMidi(candidate.frequency);
    candidate.amplitude = peak.magnitude;
    candidate.harmonicCount = 1;
    
    float fundFreq = candidate.frequency;
    float maxMag = allPeaks.empty() ? 1.0f : allPeaks[0].magnitude;
    
    // NEW APPROACH: Only look FORWARD for harmonics (not backward for sub-harmonics)
    // Check if current candidate's harmonics exist in the spectrum
    std::vector<std::pair<int, float>> foundHarmonics;
    juce::String harmonicsLog;
    
    for (int h = 2; h <= 8; ++h)
    {
        float expectedFreq = fundFreq * h;
        if (expectedFreq > 8000.0f) break;
        
        for (const auto& p : allPeaks)
        {
            float deviation = std::abs(p.frequency - expectedFreq) / expectedFreq;
            if (deviation < 0.03f)
            {
                foundHarmonics.push_back({h, p.magnitude});
                harmonicsLog += "H" + juce::String(h) + "=" + juce::String(p.frequency, 1) + "Hz ";
                break;
            }
        }
    }
    
    // Check for consecutive harmonics (H2, H3, H4...)
    bool hasConsecutiveHarmonics = false;
    int maxConsecutive = 0;
    if (foundHarmonics.size() >= 2)
    {
        int consecutiveCount = 1;
        for (size_t i = 1; i < foundHarmonics.size(); ++i)
        {
            if (foundHarmonics[i].first == foundHarmonics[i-1].first + 1)
            {
                consecutiveCount++;
                maxConsecutive = std::max(maxConsecutive, consecutiveCount);
            }
            else
            {
                consecutiveCount = 1;
            }
        }
        hasConsecutiveHarmonics = (maxConsecutive >= 2);
    }
    
    // Calculate harmonic quality score
    float harmonicQuality = 0.0f;
    for (const auto& [h, mag] : foundHarmonics)
    {
        float expectedMag = candidate.amplitude / h;
        float magRatio = mag / expectedMag;
        
        if (magRatio > 0.3f && magRatio < 3.0f)
        {
            harmonicQuality += 1.0f / h;
        }
    }
    
    candidate.harmonicCount = 1 + (int)foundHarmonics.size();
    
    // Calculate base confidence
    float relativeAmp = candidate.amplitude / maxMag;
    float baseScore = relativeAmp * 0.3f;
    float harmonicScore = std::min(0.6f, harmonicQuality * 0.3f + maxConsecutive * 0.1f);
    
    // Penalty: if current candidate is a harmonic of a LOWER frequency peak
    // (This means we're looking at a higher harmonic, not the true fundamental)
    float subHarmonicPenalty = 0.0f;
    for (const auto& p : allPeaks)
    {
        if (p.frequency >= fundFreq) continue;  // Only check lower frequencies
        
        float ratio = fundFreq / p.frequency;
        int nearestInt = juce::roundToInt(ratio);
        
        if (nearestInt >= 2 && nearestInt <= 8)
        {
            float deviation = std::abs(ratio - nearestInt) / nearestInt;
            if (deviation < 0.03f)
            {
                // Current frequency is a harmonic of a lower peak
                // Strong penalty: H2 (octave) gets highest penalty
                // H2: 0.35, H3: 0.25, H4: 0.18, H5+: 0.12
                if (nearestInt == 2) subHarmonicPenalty = 0.35f;
                else if (nearestInt == 3) subHarmonicPenalty = 0.25f;
                else if (nearestInt == 4) subHarmonicPenalty = 0.18f;
                else subHarmonicPenalty = 0.12f;
                break;
            }
        }
    }
    
    float rawConfidence = baseScore + harmonicScore - subHarmonicPenalty;
    
    // Adjust minimum confidence based on harmonic count
    if (foundHarmonics.size() >= 4)
    {
        rawConfidence = std::max(rawConfidence, 0.75f);
    }
    else if (foundHarmonics.size() >= 2)
    {
        rawConfidence = std::max(rawConfidence, 0.55f);
    }
    
    candidate.confidence = juce::jlimit(0.0f, 1.0f, rawConfidence);
    
    // Debug logging
    juce::String dbgLog = "[EVAL] " + juce::String(fundFreq, 1) + "Hz (" + 
                          juce::String(candidate.midiNote, 1) + ") mag=" + juce::String(peak.magnitude, 3) +
                          " | harmonics=" + juce::String((int)foundHarmonics.size()) + " [" + harmonicsLog + "]" +
                          " | base=" + juce::String(baseScore, 2) + 
                          " harm=" + juce::String(harmonicScore, 2) + 
                          " subPen=" + juce::String(subHarmonicPenalty, 2) + 
                          " conf=" + juce::String(candidate.confidence, 2);
    
    float roundedMidi = std::round(candidate.midiNote);
    candidate.centsDeviation = (candidate.midiNote - roundedMidi) * 100.0f;
    
    return candidate;
}

float PolyphonicDetector::interpolateFrequency(int bin, float alpha, float beta, float gamma) const
{
    float p = 0.5f * (alpha - gamma) / (alpha - 2*beta + gamma);
    return binToFreq(bin, 4096) + p * (sampleRate_ / 4096.0);
}

float PolyphonicDetector::binToFreq(int bin, int fftSize) const
{
    return bin * static_cast<float>(sampleRate_) / fftSize;
}

float PolyphonicDetector::freqToMidi(float freq) const
{
    return 69.0f + 12.0f * std::log2(freq / 440.0f);
}

} // namespace spm

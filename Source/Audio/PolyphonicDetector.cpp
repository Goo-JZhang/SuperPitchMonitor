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
    
    if (spectrum.magnitudes.empty())
    {
        SPM_LOG_INFO("[Pitch][Frame " + juce::String(frameNum) + "] No spectrum data");
        return;
    }
    
    // Check noise gate
    float maxMag = *std::max_element(spectrum.magnitudes.begin(), spectrum.magnitudes.end());
    if (maxMag < noiseGate_)
    {
        SPM_LOG_INFO("[Pitch][Frame " + juce::String(frameNum) + "] Below noise gate (maxMag=" + 
                     juce::String(maxMag, 6) + ")");
        return;
    }
    
    // Output FFT peaks (before YIN)
    std::vector<Peak> fftPeaks;
    BandSpectrumData singleBand;
    singleBand.frequencies = spectrum.frequencies;
    singleBand.magnitudes = spectrum.magnitudes;
    singleBand.refinedFreqs = spectrum.refinedFreqs;
    singleBand.hasRefinedFreqs = spectrum.hasRefinedFreqs;
    singleBand.sampleRate = spectrum.sampleRate;
    
    float threshold = maxMag * 0.005f;
    findPeaksInBand(singleBand, fftPeaks, 1, threshold);
    
    // Output FFT peaks
    juce::String fftLog = "[Pitch][Frame " + juce::String(frameNum) + "] FFT Peaks: ";
    if (fftPeaks.empty()) {
        fftLog += "None";
    } else {
        for (size_t i = 0; i < std::min(size_t(8), fftPeaks.size()); ++i) {
            if (i > 0) fftLog += " | ";
            fftLog += "P" + juce::String((int)i) + ":";
            fftLog += juce::String(fftPeaks[i].frequency, 1) + "Hz";
            fftLog += "(" + juce::String((int)(fftPeaks[i].magnitude / maxMag * 100)) + "%)";
        }
    }
    SPM_LOG_INFO(fftLog);
    
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
    
    // Output final results
    juce::String logMsg = "[Pitch][Frame " + juce::String(frameNum) + "] Final Results: ";
    
    if (results.empty())
    {
        logMsg += "None";
    }
    else
    {
        for (size_t i = 0; i < results.size(); ++i)
        {
            if (i > 0) logMsg += " | ";
            logMsg += "P" + juce::String((int)i) + ":";
            logMsg += "F=" + juce::String(results[i].frequency, 2) + "Hz";
            logMsg += "/M=" + juce::String(results[i].midiNote, 2);
            logMsg += "/C=" + juce::String((int)results[i].centsDeviation) + "ct";
            logMsg += "/Conf=" + juce::String(results[i].confidence, 2);
            logMsg += "/H=" + juce::String(results[i].harmonicCount);
        }
    }
    
    SPM_LOG_INFO(logMsg);
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
    
    juce::String lowDebug = "[MR][Frame " + juce::String(frameNum) + "] LowBand peaks: ";
    if (lowPeaks.empty()) {
        lowDebug += "None";
    } else {
        float maxLowMag = 0.0001f;
        for (const auto& p : lowPeaks) maxLowMag = std::max(maxLowMag, p.magnitude);
        for (size_t i = 0; i < std::min(size_t(5), lowPeaks.size()); ++i) {
            if (i > 0) lowDebug += ", ";
            lowDebug += juce::String(lowPeaks[i].frequency, 1) + "Hz";
            lowDebug += "(" + juce::String((int)(lowPeaks[i].magnitude / maxLowMag * 100)) + "%)";
        }
    }
    SPM_LOG_INFO(lowDebug);
    
    // Mid frequency band
    std::vector<Peak> midPeaks;
    if (multiData.midBand().hasRefinedFreqs) {
        detectMidBand(multiData.midBand(), midPeaks);
    }
    
    juce::String midDebug = "[MR][Frame " + juce::String(frameNum) + "] MidBand peaks: ";
    if (midPeaks.empty()) {
        midDebug += "None";
    } else {
        float maxMidMag = 0.0001f;
        for (const auto& p : midPeaks) maxMidMag = std::max(maxMidMag, p.magnitude);
        for (size_t i = 0; i < std::min(size_t(5), midPeaks.size()); ++i) {
            if (i > 0) midDebug += ", ";
            midDebug += juce::String(midPeaks[i].frequency, 1) + "Hz";
            midDebug += "(" + juce::String((int)(midPeaks[i].magnitude / maxMidMag * 100)) + "%)";
        }
    }
    SPM_LOG_INFO(midDebug);
    
    allPeaks.insert(allPeaks.end(), lowPeaks.begin(), lowPeaks.end());
    allPeaks.insert(allPeaks.end(), midPeaks.begin(), midPeaks.end());
    
    if (multiData.highBand().hasRefinedFreqs) {
        detectHighBand(multiData.highBand(), allPeaks);
    }
    
    if (allPeaks.empty()) {
        SPM_LOG_INFO("[Pitch][MR][Frame " + juce::String(frameNum) + "] No peaks found");
        return;
    }
    
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
    
    juce::String candLog = "[MR][Frame " + juce::String(frameNum) + "] Candidates: ";
    if (candidates.empty()) {
        candLog += "None";
    } else {
        for (size_t i = 0; i < std::min(size_t(5), candidates.size()); ++i) {
            if (i > 0) candLog += " | ";
            candLog += "C" + juce::String((int)i) + ":";
            candLog += juce::String(candidates[i].frequency, 1) + "Hz";
            candLog += "/Conf=" + juce::String(candidates[i].confidence, 2);
            candLog += "/H=" + juce::String(candidates[i].harmonicCount);
        }
    }
    SPM_LOG_INFO(candLog);
    
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
                float originalConf = cand.confidence;
                // Slight reduction: confidence drops to 70% at 60% overlap
                float reductionFactor = 0.70f + 0.30f * (1.0f - overlapRatio);  // 0.7~1.0
                cand.confidence *= reductionFactor;
                SPM_LOG_INFO("[PRE] " + juce::String(cand.frequency, 1) + "Hz overlap=" + 
                            juce::String(overlappingCount) + "/" + juce::String(candidateHarmonics.size()) +
                            " score=" + juce::String(totalOverlapScore, 2) +
                            " conf=" + juce::String(originalConf, 2) + 
                            "->" + juce::String(cand.confidence, 2) + " [SLIGHT_REDUCED]");
            }
        }
        
        // Calculate harmonic penalty: based on frequency distance to selected fundamentals
        float penalty = 0.0f;
        juce::String penaltyDetails;
        
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
                penaltyDetails += "F" + juce::String((int)fundFreq) + "(" + juce::String(fundPenalty, 2) + ") ";
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
                    penaltyDetails += "H" + juce::String(h) + "(" + juce::String(harmonicPenalty, 2) + ") ";
                }
            }
        }
        
        // Limit penalty to not exceed original confidence
        penalty = std::min(penalty, cand.confidence * 0.95f);
        float remainingConfidence = cand.confidence - penalty;
        
        juce::String penalLog = "[PENALTY] " + juce::String(cand.frequency, 1) + "Hz " +
                                "orig=" + juce::String(cand.confidence, 2) + 
                                " penalty=" + juce::String(penalty, 2) + 
                                " remain=" + juce::String(remainingConfidence, 2);
        if (!penaltyDetails.isEmpty()) {
            penalLog += " from:" + penaltyDetails;
        }
        
        // Decision: is remaining confidence sufficient
        bool acceptAsFundamental = false;
        if (remainingConfidence >= minConfidenceAfterPenalty) {
            // Optional: YIN verification for remaining confidence
            // For simplicity, accept for now but mark as needing verification
            acceptAsFundamental = true;
            penalLog += " [ACCEPT]";
        } else {
            penalLog += " [REJECT]";
        }
        SPM_LOG_INFO(penalLog);
        
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
    
    juce::String logMsg = "[Pitch][MR][Frame " + juce::String(frameNum) + "] Final: ";
    
    if (results.empty())
    {
        logMsg += "None (rejected " + juce::String((int)candidates.size()) + " candidates)";
    }
    else
    {
        for (size_t i = 0; i < results.size(); ++i)
        {
            if (i > 0) logMsg += " | ";
            logMsg += "P" + juce::String((int)i) + ":";
            logMsg += "F=" + juce::String(results[i].frequency, 2) + "Hz";
            logMsg += "/M=" + juce::String(results[i].midiNote, 2);
            logMsg += "/C=" + juce::String((int)results[i].centsDeviation) + "ct";
            logMsg += "/Conf=" + juce::String(results[i].confidence, 2);
            logMsg += "/H=" + juce::String(results[i].harmonicCount);
        }
    }
    
    SPM_LOG_INFO(logMsg);
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
    
    int startBin = static_cast<int>(minFreq / bandData.sampleRate * bandData.frequencies.size() * 2);
    int endBin = static_cast<int>(maxFreq / bandData.sampleRate * bandData.frequencies.size() * 2);
    startBin = std::max(2, startBin);
    endBin = std::min(static_cast<int>(mags.size()) - 2, endBin);
    

    for (int i = startBin; i < endBin; ++i)
    {
        float mag = mags[i];
        if (mag < actualThreshold) continue;
        
        // Relax local max detection: higher than 1 point on each side
        if (mag > mags[i-1] && mag > mags[i+1])
        {
            // Relax check: peak should be above surroundings (at least 5%)
            float neighborAvg = (mags[i-1] + mags[i+1]) * 0.5f;
            if (mag < neighborAvg * 1.05f) {
                continue;
            }
            
            Peak peak;
            peak.bin = i;
            peak.bandIndex = bandIndex;
            
            // Use raw FFT bin frequency for peak detection
            // Phase-vocoder refined frequencies can be unreliable for polyphonic signals
            // because each bin may contain energy from multiple sources
            peak.frequency = bandData.frequencies[i];
            
            float alpha = mags[i-1];
            float beta = mag;
            float gamma = mags[i+1];
            float denom = alpha - 2.0f * beta + gamma;
            if (std::abs(denom) > 1e-10f) {
                peak.magnitude = beta - 0.25f * (alpha - gamma) * (alpha - gamma) / denom;
            } else {
                peak.magnitude = mag;
            }
            
            peaks.push_back(peak);
        }
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
    
    // ========== Detailed logging start ==========
    juce::String dbgLog = "[EVAL] Evaluating " + juce::String(fundFreq, 1) + "Hz (" + 
                          juce::String(candidate.midiNote, 1) + ") mag=" + juce::String(peak.magnitude, 3);
    
    // 1. Check if current candidate is sub-harmonic of a stronger peak
    // Key fix: if a stronger peak exists where current freq ≈ stronger peak / n, it's sub-harmonic
    std::vector<size_t> possibleFundamentals;  // Store indices of peaks that could be fundamentals
    juce::String possibleFundLog;
    bool isStrongSubHarmonic = false;
    float strongestFundamentalMag = 0.0f;
    
    for (size_t i = 0; i < allPeaks.size(); ++i)
    {
        if (i == peakIndex) continue;
        
        float otherFreq = allPeaks[i].frequency;
        float otherMag = allPeaks[i].magnitude;
        
        // Check if current frequency is integer fraction of otherFreq (i.e., otherFreq = n * fundFreq)
        if (otherFreq > fundFreq * 1.5f)  // otherFreq significantly higher
        {
            float ratio = otherFreq / fundFreq;
            int nearestInt = juce::roundToInt(ratio);
            
            if (nearestInt >= 2 && nearestInt <= 8)
            {
                float deviation = std::abs(ratio - nearestInt) / nearestInt;
                if (deviation < 0.03f)
                {
                    possibleFundamentals.push_back(i);
                    possibleFundLog += juce::String(otherFreq, 1) + "Hz(H" + 
                                      juce::String(nearestInt) + ") ";
                    
                    // Key fix: if "fundamental" is much stronger than current, current is likely sub-harmonic
                    if (otherMag > candidate.amplitude * 1.5f)
                    {
                        isStrongSubHarmonic = true;
                        strongestFundamentalMag = std::max(strongestFundamentalMag, otherMag);
                    }
                }
            }
        }
    }
    
    if (!possibleFundLog.isEmpty())
    {
        dbgLog += " | Possible fundamentals: " + possibleFundLog;
        dbgLog += " | isStrongSubHarmonic=" + juce::String(isStrongSubHarmonic ? "YES" : "NO");
    }
    
    // 2. Search harmonics, check if "borrowed" or "independent fundamental"
    std::vector<std::pair<int, float>> ownHarmonics;      // Harmonics truly belonging to current frequency
    std::vector<std::pair<int, float>> borrowedHarmonics; // Harmonics belonging to other "fundamentals"
    juce::String ownLog, borrowedLog;
    
    // Pre-calculate which peaks are "independent fundamental" candidates (have multiple own harmonics)
    std::vector<bool> isPeakIndependentFundamental(allPeaks.size(), false);
    for (size_t i = 0; i < allPeaks.size(); ++i)
    {
        int harmonicsCount = 0;
        for (int h2 = 2; h2 <= 6; ++h2)
        {
            float expectedFreq2 = allPeaks[i].frequency * h2;
            for (const auto& p2 : allPeaks)
            {
                float dev2 = std::abs(p2.frequency - expectedFreq2) / expectedFreq2;
                if (dev2 < 0.03f && p2.magnitude > allPeaks[i].magnitude * 0.3f)
                {
                    harmonicsCount++;
                    break;
                }
            }
        }
        // If 3 or more harmonics, mark as independent fundamental
        if (harmonicsCount >= 3)
        {
            isPeakIndependentFundamental[i] = true;
        }
    }
    
    for (int h = 2; h <= 10; ++h)
    {
        float expectedFreq = fundFreq * h;
        if (expectedFreq > 8000.0f) break;
        
        for (size_t i = 0; i < allPeaks.size(); ++i)
        {
            const auto& p = allPeaks[i];
            float deviation = std::abs(p.frequency - expectedFreq) / expectedFreq;
            
            if (deviation < 0.03f)
            {
                // Check if this peak is harmonic of a stronger peak
                bool isBorrowed = false;
                size_t borrowedFromIdx = 0;
                
                for (size_t fundIdx : possibleFundamentals)
                {
                    float fundFreq2 = allPeaks[fundIdx].frequency;
                    float ratio2 = p.frequency / fundFreq2;
                    int nearestInt2 = juce::roundToInt(ratio2);
                    
                    if (nearestInt2 >= 1 && nearestInt2 <= 8)
                    {
                        float dev2 = std::abs(ratio2 - nearestInt2) / nearestInt2;
                        if (dev2 < 0.03f)
                        {
                            // If "fundamental" is stronger or comparable, this harmonic is borrowed
                            if (allPeaks[fundIdx].magnitude >= candidate.amplitude * 0.8f)
                            {
                                isBorrowed = true;
                                borrowedFromIdx = fundIdx;
                                borrowedHarmonics.push_back({h, p.magnitude});
                                borrowedLog += "H" + juce::String(h) + "=" + 
                                              juce::String(p.frequency, 1) + "Hz(from " +
                                              juce::String(fundFreq2, 1) + ") ";
                                break;
                            }
                        }
                    }
                }
                
                // Key fix: if this "harmonic" itself is independent fundamental, mark as borrowed
                if (!isBorrowed && isPeakIndependentFundamental[i])
                {
                    borrowedHarmonics.push_back({h, p.magnitude});
                    borrowedLog += "H" + juce::String(h) + "=" + 
                                  juce::String(p.frequency, 1) + "Hz(INDEP) ";
                    isBorrowed = true;
                }
                
                if (!isBorrowed)
                {
                    ownHarmonics.push_back({h, p.magnitude});
                    ownLog += "H" + juce::String(h) + "=" + juce::String(p.frequency, 1) + "Hz ";
                }
                break;  // Only take first matching peak
            }
        }
    }
    
    // 3. Determine if truly sub-harmonic - using multiple conditions
    int totalFoundHarmonics = (int)ownHarmonics.size() + (int)borrowedHarmonics.size();
    float borrowedRatio = totalFoundHarmonics > 0 ? (float)borrowedHarmonics.size() / totalFoundHarmonics : 0.0f;
    
    // Check if harmonics are consecutive (key fix: true fundamental has consecutive H2, H3, H4...)
    bool hasConsecutiveHarmonics = false;
    if (ownHarmonics.size() >= 3)
    {
        // Check if at least 3 consecutive harmonics
        int consecutiveCount = 1;
        for (size_t i = 1; i < ownHarmonics.size(); ++i)
        {
            if (ownHarmonics[i].first == ownHarmonics[i-1].first + 1)
            {
                consecutiveCount++;
                if (consecutiveCount >= 3)
                {
                    hasConsecutiveHarmonics = true;
                    break;
                }
            }
            else
            {
                consecutiveCount = 1;
            }
        }
    }
    
    // Condition 1: if >50% "harmonics" are borrowed, current is sub-harmonic
    bool isSubHarmonic = !possibleFundamentals.empty() && borrowedRatio >= 0.5f;
    
    // Condition 2: if all found harmonics are borrowed, definitely sub-harmonic
    if (totalFoundHarmonics > 0 && ownHarmonics.size() == 0)
    {
        isSubHarmonic = true;
    }
    
    // Condition 3: if a much stronger fundamental exists, current is sub-harmonic
    if (isStrongSubHarmonic)
    {
        isSubHarmonic = true;
    }
    
    // Condition 4 (key fix): if no consecutive harmonic series, may be spurious fundamental
    if (ownHarmonics.size() >= 3 && !hasConsecutiveHarmonics && !possibleFundamentals.empty())
    {
        isSubHarmonic = true;
    }
    
    dbgLog += " | Own: " + juce::String((int)ownHarmonics.size()) + " [" + ownLog + "]";
    dbgLog += " | Borrowed: " + juce::String((int)borrowedHarmonics.size()) + " [" + borrowedLog + "]";
    dbgLog += " | borrowedRatio=" + juce::String(borrowedRatio, 2);
    dbgLog += " | isSubHarmonic=" + juce::String(isSubHarmonic ? "YES" : "NO");
    
    // 4. Calculate quality using truly own harmonics
    float harmonicQuality = 0.0f;
    int consecutiveCount = 0;
    int maxConsecutive = 0;
    int lastHarmonic = 0;
    
    for (const auto& [h, mag] : ownHarmonics)
    {
        float expectedMag = candidate.amplitude / h;
        float magRatio = mag / expectedMag;
        
        if (lastHarmonic == 0 || h == lastHarmonic + 1)
        {
            consecutiveCount++;
            maxConsecutive = std::max(maxConsecutive, consecutiveCount);
        }
        else
        {
            consecutiveCount = 1;
        }
        lastHarmonic = h;
        
        if (magRatio > 0.5f && magRatio < 2.0f)
        {
            harmonicQuality += 1.0f / h;
        }
    }
    
    candidate.harmonicCount = 1 + (int)ownHarmonics.size();
    
    // 5. Calculate confidence - based on truly own harmonics
    float relativeAmp = candidate.amplitude / maxMag;
    float baseScore = relativeAmp * 0.3f;
    float harmonicScore = std::min(0.5f, harmonicQuality * 0.25f + maxConsecutive * 0.08f);
    
    // Sub-harmonic penalty - based on borrowed ratio (relaxed: better false positive)
    float penalty = 0.0f;
    if (isSubHarmonic)
    {
        int totalHarmonics = (int)ownHarmonics.size() + (int)borrowedHarmonics.size();
        float borrowedRatio = (float)borrowedHarmonics.size() / totalHarmonics;
        // Reduce penalty: sub-harmonics still have chance to be detected (low confidence)
        penalty = 0.15f + 0.25f * borrowedRatio;  // Reduced from 0.3+0.5 to 0.15+0.25
        
        // If has few true own harmonics, slightly increase penalty
        if (ownHarmonics.size() <= 2)
        {
            penalty += 0.1f;  // Reduced from 0.2 to 0.1
        }
    }
    
    float rawConfidence = baseScore + harmonicScore - penalty;
    
    // Adjust minimum confidence
    if (ownHarmonics.size() >= 4 && !isSubHarmonic)
    {
        rawConfidence = std::max(rawConfidence, 0.75f);
    }
    else if (ownHarmonics.size() >= 2 && !isSubHarmonic)
    {
        rawConfidence = std::max(rawConfidence, 0.55f);
    }
    
    candidate.confidence = juce::jlimit(0.0f, 1.0f, rawConfidence);
    
    // Output detailed calculation log
    dbgLog += " | base=" + juce::String(baseScore, 2) + 
              " harm=" + juce::String(harmonicScore, 2) + 
              " penalty=" + juce::String(penalty, 2) + 
              " conf=" + juce::String(candidate.confidence, 2);
    
    // Only output in multi-resolution mode or debugging
    static int logCounter = 0;
    if (++logCounter % 5 == 0 || multiData != nullptr)  // Output every 5 frames to avoid log spam
    {
        SPM_LOG_INFO(dbgLog);
    }
    
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

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
    
    // 输出FFT峰值（YIN之前）
    std::vector<Peak> fftPeaks;
    BandSpectrumData singleBand;
    singleBand.frequencies = spectrum.frequencies;
    singleBand.magnitudes = spectrum.magnitudes;
    singleBand.refinedFreqs = spectrum.refinedFreqs;
    singleBand.hasRefinedFreqs = spectrum.hasRefinedFreqs;
    singleBand.sampleRate = spectrum.sampleRate;
    
    float threshold = maxMag * 0.005f;
    findPeaksInBand(singleBand, fftPeaks, 1, threshold);
    
    // 输出FFT峰值
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
    
    // 输出最终结果
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
        // 只跳过极低频率（避免DC分量和噪声）
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
    
    // 低频带
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
    
    // 中频带
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
        // 只跳过极低频率（避免DC分量和噪声）
        if (allPeaks[i].frequency < 30.0f) continue;
        
        auto candidate = evaluateAsFundamental(allPeaks, i, &multiData);
        
        if (candidate.confidence > 0.25f)
        {
            candidates.push_back(candidate);
        }
    }
    
    // 按频率从低到高排序（关键修复：从低频开始检测）
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
    
    // 新策略：渐进式谐波惩罚
    // 从低到高处理候选，应用谐波惩罚后检查剩余置信度
    // 降低阈值：宁可错判，不可漏判
    const float minConfidence = 0.15f;  // 从 0.25 降低到 0.15
    const float minConfidenceAfterPenalty = 0.10f;  // 从 0.20 降低到 0.10
    
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
        
        // 预筛选：验证候选的谐波结构独立性
        // 策略：如果低频候选的谐波与高频候选的谐波集合重叠，则降低其置信度
        if (selectedCandidates.empty()) {
            // 计算当前候选的谐波集合
            std::vector<float> candidateHarmonics;
            for (int h = 2; h <= 6; ++h) {
                candidateHarmonics.push_back(cand.frequency * h);
            }
            
            // 计算所有其他候选的合并谐波集合
            std::vector<float> otherHarmonics;
            for (const auto& other : candidates) {
                if (other.frequency <= cand.frequency) continue;
                // 为每个其他候选添加其基频和谐波
                otherHarmonics.push_back(other.frequency);  // 基频
                for (int h = 2; h <= 4; ++h) {
                    otherHarmonics.push_back(other.frequency * h);
                }
            }
            
            // 检查候选的谐波有多少与其他候选的谐波匹配（宽松匹配）
            int overlappingCount = 0;
            float totalOverlapScore = 0.0f;
            for (float candHarm : candidateHarmonics) {
                for (float otherHarm : otherHarmonics) {
                    float freqDiff = std::abs(candHarm - otherHarm);
                    float relativeDiff = freqDiff / std::max(candHarm, otherHarm);
                    
                    // 放宽容差到 10%
                    if (relativeDiff < 0.10f) {
                        overlappingCount++;
                        float overlapScore = 1.0f - (relativeDiff / 0.10f);
                        totalOverlapScore += overlapScore;
                        break;  // 找到一个匹配即可
                    }
                }
            }
            
            // 放宽预筛选：即使有谐波重叠，也只轻微降低置信度
            // 宁可错判，不可漏判 - 低能量错判可以接受
            float overlapRatio = candidateHarmonics.empty() ? 0.0f : 
                                (float)overlappingCount / candidateHarmonics.size();
            
            // 只有高度重叠时才轻微削减（保留更多候选）
            if (overlapRatio >= 0.6f && totalOverlapScore >= 2.5f) {
                float originalConf = cand.confidence;
                // 轻微削减：60%重叠时置信度降至原来的 70%
                float reductionFactor = 0.70f + 0.30f * (1.0f - overlapRatio);  // 0.7~1.0
                cand.confidence *= reductionFactor;
                SPM_LOG_INFO("[PRE] " + juce::String(cand.frequency, 1) + "Hz overlap=" + 
                            juce::String(overlappingCount) + "/" + juce::String(candidateHarmonics.size()) +
                            " score=" + juce::String(totalOverlapScore, 2) +
                            " conf=" + juce::String(originalConf, 2) + 
                            "->" + juce::String(cand.confidence, 2) + " [SLIGHT_REDUCED]");
            }
        }
        
        // 计算谐波惩罚：基于与已选基频及其谐波的频率距离
        float penalty = 0.0f;
        juce::String penaltyDetails;
        
        for (const auto& selected : selectedCandidates)
        {
            float fundFreq = selected.frequency;
            
            // 检查与基频的距离（0次谐波）
            float fundDiff = std::abs(cand.frequency - fundFreq);
            float fundRelativeDiff = fundDiff / fundFreq;
            
            if (fundRelativeDiff < 0.10f) {  // 10% 容差
                // 接近基频本身，大幅惩罚（可能是同一个音的不同检测）
                float fundPenalty = cand.confidence * (1.0f - fundRelativeDiff / 0.10f) * 0.9f;
                penalty += fundPenalty;
                penaltyDetails += "F" + juce::String((int)fundFreq) + "(" + juce::String(fundPenalty, 2) + ") ";
            }
            
            // 检查与各次谐波的距离
            for (int h = 2; h <= 6; ++h)
            {
                float harmonicFreq = fundFreq * h;
                float harmonicDiff = std::abs(cand.frequency - harmonicFreq);
                float harmonicRelativeDiff = harmonicDiff / harmonicFreq;
                
                // 谐波惩罚窗口：基于谐波次数（高次谐波更宽）
                float harmonicTolerance = 0.06f + h * 0.01f;  // H2: 8%, H6: 12%
                
                if (harmonicRelativeDiff < harmonicTolerance)
                {
                    // 距离谐波越近，惩罚越大
                    // 基础惩罚系数：1/h（高次谐波惩罚较小）
                    float baseHarmonicPenalty = 1.0f / h;
                    float proximityFactor = 1.0f - (harmonicRelativeDiff / harmonicTolerance);
                    float harmonicPenalty = cand.confidence * baseHarmonicPenalty * proximityFactor;
                    
                    penalty += harmonicPenalty;
                    penaltyDetails += "H" + juce::String(h) + "(" + juce::String(harmonicPenalty, 2) + ") ";
                }
            }
        }
        
        // 限制惩罚不超过原始置信度
        penalty = std::min(penalty, cand.confidence * 0.95f);
        float remainingConfidence = cand.confidence - penalty;
        
        juce::String penalLog = "[PENALTY] " + juce::String(cand.frequency, 1) + "Hz " +
                                "orig=" + juce::String(cand.confidence, 2) + 
                                " penalty=" + juce::String(penalty, 2) + 
                                " remain=" + juce::String(remainingConfidence, 2);
        if (!penaltyDetails.isEmpty()) {
            penalLog += " from:" + penaltyDetails;
        }
        
        // 决策：剩余置信度是否足够
        bool acceptAsFundamental = false;
        if (remainingConfidence >= minConfidenceAfterPenalty) {
            // 可选：对剩余置信度进行 YIN 验证
            // 为简化，暂时接受，但标记为需要验证
            acceptAsFundamental = true;
            penalLog += " [ACCEPT]";
        } else {
            penalLog += " [REJECT]";
        }
        SPM_LOG_INFO(penalLog);
        
        if (acceptAsFundamental) {
            // 移除后验证拒绝逻辑 - 宁可错判，不可漏判
            // 低能量的"借来"谐波可以接受（主面板上痕迹淡）
            
            // 更新候选的置信度为剩余值
            cand.confidence = remainingConfidence;
            selectedCandidates.push_back(cand);
        }
    }
    
    // 复制结果
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
        
        // 放宽局部最大值检测：比左右各1个点都高即可
        if (mag > mags[i-1] && mag > mags[i+1])
        {
            // 放宽检查：峰值应该高于周围（至少5%即可）
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
    
    // ========== 详细日志开始 ==========
    juce::String dbgLog = "[EVAL] Evaluating " + juce::String(fundFreq, 1) + "Hz (" + 
                          juce::String(candidate.midiNote, 1) + ") mag=" + juce::String(peak.magnitude, 3);
    
    // 1. 检查当前候选是否是某个更强峰值的次谐波
    // 关键修复：如果存在一个更强的峰值，使得当前频率 ≈ 更强峰值 / n，则当前是次谐波
    std::vector<size_t> possibleFundamentals;  // 存储可能作为基频的峰值索引
    juce::String possibleFundLog;
    bool isStrongSubHarmonic = false;
    float strongestFundamentalMag = 0.0f;
    
    for (size_t i = 0; i < allPeaks.size(); ++i)
    {
        if (i == peakIndex) continue;
        
        float otherFreq = allPeaks[i].frequency;
        float otherMag = allPeaks[i].magnitude;
        
        // 检查当前频率是否是 otherFreq 的整数分之一（即 otherFreq = n * fundFreq）
        if (otherFreq > fundFreq * 1.5f)  // otherFreq 显著更高
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
                    
                    // 关键修复：如果"基频"比当前候选强很多，当前候选很可能是次谐波
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
    
    // 2. 搜索谐波，同时检查是否是"借来的"或"独立基频"
    std::vector<std::pair<int, float>> ownHarmonics;      // 真正属于当前频率的谐波
    std::vector<std::pair<int, float>> borrowedHarmonics; // 属于其他"基频"的谐波
    juce::String ownLog, borrowedLog;
    
    // 预先计算哪些峰值是"独立基频"候选（有多个自己的谐波）
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
        // 如果有3个或更多谐波，标记为独立基频
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
                // 检查这个峰值是否是某个更强峰值的谐波
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
                            // 如果"基频"更强或相当，这个谐波是借来的
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
                
                // 关键修复：如果这个"谐波"本身是独立基频，标记为借来的
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
                break;  // 只取第一个匹配的峰值
            }
        }
    }
    
    // 3. 判断是否真的是次谐波 - 使用多重条件
    int totalFoundHarmonics = (int)ownHarmonics.size() + (int)borrowedHarmonics.size();
    float borrowedRatio = totalFoundHarmonics > 0 ? (float)borrowedHarmonics.size() / totalFoundHarmonics : 0.0f;
    
    // 检查谐波是否连续（关键修复：真正的基频应该有连续的谐波 H2, H3, H4...）
    bool hasConsecutiveHarmonics = false;
    if (ownHarmonics.size() >= 3)
    {
        // 检查是否有至少3个连续的谐波
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
    
    // 条件1：如果超过50%的"谐波"是借来的，则当前候选是次谐波
    bool isSubHarmonic = !possibleFundamentals.empty() && borrowedRatio >= 0.5f;
    
    // 条件2：如果所有找到的谐波都是借来的，确定是次谐波
    if (totalFoundHarmonics > 0 && ownHarmonics.size() == 0)
    {
        isSubHarmonic = true;
    }
    
    // 条件3：如果存在一个强得多的基频，当前是次谐波
    if (isStrongSubHarmonic)
    {
        isSubHarmonic = true;
    }
    
    // 条件4（关键修复）：如果没有连续的谐波序列，可能是虚假基频
    if (ownHarmonics.size() >= 3 && !hasConsecutiveHarmonics && !possibleFundamentals.empty())
    {
        isSubHarmonic = true;
    }
    
    dbgLog += " | Own: " + juce::String((int)ownHarmonics.size()) + " [" + ownLog + "]";
    dbgLog += " | Borrowed: " + juce::String((int)borrowedHarmonics.size()) + " [" + borrowedLog + "]";
    dbgLog += " | borrowedRatio=" + juce::String(borrowedRatio, 2);
    dbgLog += " | isSubHarmonic=" + juce::String(isSubHarmonic ? "YES" : "NO");
    
    // 4. 使用真正属于自己的谐波计算质量
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
    
    // 5. 计算置信度 - 基于真正属于自己的谐波
    float relativeAmp = candidate.amplitude / maxMag;
    float baseScore = relativeAmp * 0.3f;
    float harmonicScore = std::min(0.5f, harmonicQuality * 0.25f + maxConsecutive * 0.08f);
    
    // 次谐波惩罚 - 基于借来的谐波比例（放宽：宁可错判，不可漏判）
    float penalty = 0.0f;
    if (isSubHarmonic)
    {
        int totalHarmonics = (int)ownHarmonics.size() + (int)borrowedHarmonics.size();
        float borrowedRatio = (float)borrowedHarmonics.size() / totalHarmonics;
        // 降低惩罚：次谐波仍有机会被检测（低置信度）
        penalty = 0.15f + 0.25f * borrowedRatio;  // 从 0.3+0.5 降低到 0.15+0.25
        
        // 如果自己有很少的真正谐波，轻微增加惩罚
        if (ownHarmonics.size() <= 2)
        {
            penalty += 0.1f;  // 从 0.2 降低到 0.1
        }
    }
    
    float rawConfidence = baseScore + harmonicScore - penalty;
    
    // 调整最小置信度
    if (ownHarmonics.size() >= 4 && !isSubHarmonic)
    {
        rawConfidence = std::max(rawConfidence, 0.75f);
    }
    else if (ownHarmonics.size() >= 2 && !isSubHarmonic)
    {
        rawConfidence = std::max(rawConfidence, 0.55f);
    }
    
    candidate.confidence = juce::jlimit(0.0f, 1.0f, rawConfidence);
    
    // 输出详细计算日志
    dbgLog += " | base=" + juce::String(baseScore, 2) + 
              " harm=" + juce::String(harmonicScore, 2) + 
              " penalty=" + juce::String(penalty, 2) + 
              " conf=" + juce::String(candidate.confidence, 2);
    
    // 只在多分辨率模式或调试时输出
    static int logCounter = 0;
    if (++logCounter % 5 == 0 || multiData != nullptr)  // 每5帧输出一次避免日志过多
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

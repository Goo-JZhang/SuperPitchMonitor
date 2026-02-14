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
    
    // 创建FFT实例
    fft_ = std::make_unique<juce::dsp::FFT>(config_.fftOrder);
    
    // 分配缓冲区
    fftBuffer_.setSize(1, config_.fftSize * 2);  // FFT需要2倍空间（复数）
    windowBuffer_.setSize(1, config_.fftSize);
    windowCoeffs_.resize(config_.fftSize);
    prevPhases_.resize(config_.fftSize / 2 + 1, 0.0f);
    
    // 创建环形缓冲区（存储2个FFT窗长度的数据，支持滑动窗口）
    circularBuffer_.setSize(1, config_.fftSize * 2);
    circularBuffer_.clear();
    writePos_ = 0;
    samplesSinceFFT_ = 0;
    
    // 初始化YIN检测器 - 严格限制在频带范围内
    // 避免跨频带检测（如低频YIN不应检测到C2当输入是C4和弦时）
    yinBufferSize_ = 2048;  // 统一使用2048，平衡精度和延迟
    
    yinDetector_ = std::make_unique<YinPitchDetector>();
    
    // 严格限制YIN检测范围在频带内，避免误检其他频带信号
    float yinMinFreq = config_.minFreq;
    float yinMaxFreq = config_.maxFreq;
    
    // 特殊处理：低频带不使用YIN（FFT已足够精确）
    if (bandIndex_ == 0) {
        yinMinFreq = 80.0f;  // 低频YIN只检测80Hz以上（最低E2）
        yinMaxFreq = 400.0f; // 不检测太高，避免与其他频带冲突
    }
    
    yinDetector_->prepare(sampleRate, yinMinFreq, yinMaxFreq, yinBufferSize_);
    yinDetector_->setThreshold(0.2f);  // 提高阈值，减少误判
    
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
    
    // 写入环形缓冲区
    for (int i = 0; i < numSamples; ++i) {
        circularBuffer_.setSample(0, writePos_, inputData[i]);
        writePos_ = (writePos_ + 1) % circularBuffer_.getNumSamples();
    }
    
    samplesSinceFFT_ += numSamples;
    
    // 检查是否达到hop size
    if (samplesSinceFFT_ < config_.hopSize) {
        // 即使没有新FFT，也尝试运行YIN（使用历史缓冲区）
        // 但只在有之前FFT数据的情况下
        if (bandIndex_ > 0 && !output.magnitudes.empty()) {
            performYinAnalysis(input, output);
        }
        output.hasRefinedFreqs = false;  // 标记FFT数据未更新
        return;
    }
    
    samplesSinceFFT_ = 0;
    
    // 复制数据到FFT缓冲区并加窗
    copyToFFTBuffer();
    
    // 执行FFT（必须先于YIN，YIN需要频谱验证）
    performFFT(output);
    
    // 提取幅度和相位
    extractMagnitudesAndPhases(output);
    
    // 相位声码器精化
    calculateRefinedFrequencies(output);
    
    // 设置元数据
    output.bandIndex = bandIndex_;
    output.sampleRate = static_cast<float>(sampleRate_);
    output.hasRefinedFreqs = true;
    
    // YIN分析：所有频带使用时域分析提供精确频率
    // 在FFT之后运行，以便验证频谱
    // 关键修复：低频带也需要YIN来区分密集的基频（如C4-E4-G4和弦）
    performYinAnalysis(input, output);
}

void BandAnalysisTask::performYinAnalysis(const juce::AudioBuffer<float>& input, 
                                           BandSpectrumData& output)
{
    // 关键：先检查频谱中对应频段是否有能量
    // 如果频谱显示该频段没有峰值，YIN不应该检测出pitch
    if (output.magnitudes.empty()) {
        output.hasYinResult = false;
        return;
    }
    
    // 计算目标频段内的频谱能量
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
    
    // 如果目标频段能量占比太低，不运行YIN
    float energyRatio = (totalEnergy > 0) ? bandEnergy / totalEnergy : 0.0f;
    if (energyRatio < 0.1f || bandEnergy < 1.0f) {  // 需要至少10%能量在该频段
        output.hasYinResult = false;
        return;
    }
    
    // 检查目标频段内是否有明显的峰值
    bool hasPeakInBand = false;
    float maxBandMag = 0.0f;
    for (int i = bandBinStart; i <= bandBinEnd && i < (int)output.magnitudes.size(); ++i) {
        if (output.magnitudes[i] > maxBandMag) {
            maxBandMag = output.magnitudes[i];
        }
    }
    
    // 需要至少一个局部最大值显著高于周围
    for (int i = bandBinStart + 2; i < bandBinEnd - 2 && i < (int)output.magnitudes.size() - 2; ++i) {
        float mag = output.magnitudes[i];
        if (mag > maxBandMag * 0.5f &&  // 是主要峰值之一
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
    
    // 从环形缓冲区读取最近的yinBufferSize_个样本
    std::vector<float> yinBuffer(yinBufferSize_);
    
    for (int i = 0; i < yinBufferSize_; ++i) {
        int readPos = (writePos_ - yinBufferSize_ + i + circularBuffer_.getNumSamples()) 
                      % circularBuffer_.getNumSamples();
        yinBuffer[i] = circularBuffer_.getSample(0, readPos);
    }
    
    // 检查信号能量（避免静音时误判）
    float rms = 0.0f;
    for (float s : yinBuffer) rms += s * s;
    rms = std::sqrt(rms / yinBufferSize_);
    
    if (rms < 0.005f) {  // 提高噪声门限
        output.hasYinResult = false;
        return;
    }
    
    // 执行YIN检测
    float yinFreq = yinDetector_->detectPitch(yinBuffer.data(), yinBufferSize_);
    float confidence = yinDetector_->getLastConfidence();
    
    // 验证YIN结果与频谱一致：频谱中必须在YIN频率附近有能量
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
    
    // 严格验证：检测结果必须在频带范围内，且与频谱一致
    if (yinFreq >= config_.minFreq && yinFreq <= config_.maxFreq && 
        confidence > 0.5f && spectrumConfirms)  // 提高置信度阈值
    {
        output.yinFrequency = yinFreq;
        output.yinConfidence = confidence;
        output.hasYinResult = true;
        
        // 保存时域数据供后续使用
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
    
    // 从环形缓冲区读取最近的N个样本
    for (int i = 0; i < N; ++i) {
        int readPos = (writePos_ - N + i + circularBuffer_.getNumSamples()) 
                      % circularBuffer_.getNumSamples();
        float sample = circularBuffer_.getSample(0, readPos);
        
        // 加窗并存储到FFT缓冲区（实部和虚部交错）
        fftBuffer_.setSample(0, i * 2, sample * windowCoeffs_[i]);
        fftBuffer_.setSample(0, i * 2 + 1, 0.0f);
    }
}

void BandAnalysisTask::performFFT(BandSpectrumData& output)
{
    // 执行FFT
    fft_->performRealOnlyForwardTransform(fftBuffer_.getWritePointer(0), true);
    
    // 准备输出数组
    const int numBins = config_.fftSize / 2 + 1;
    output.frequencies.resize(numBins);
    output.magnitudes.resize(numBins);
    output.phases.resize(numBins);
    output.refinedFreqs.resize(numBins);
    
    // 计算频率刻度
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
        
        // 相位差
        float phaseDiff = phaseCurrent - phasePrev;
        
        // 解卷绕
        float expectedPhaseDiff = twoPi * hopSize * i / config_.fftSize;
        phaseDiff -= expectedPhaseDiff;
        
        while (phaseDiff > juce::MathConstants<float>::pi)
            phaseDiff -= twoPi;
        while (phaseDiff < -juce::MathConstants<float>::pi)
            phaseDiff += twoPi;
        
        // 计算精化频率
        float binFreq = output.frequencies[i];
        float refinedFreq = binFreq + phaseDiff * sampleRate / (twoPi * hopSize);
        
        // 限制精化频率在合理范围内（bin频率的±50%）
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
    // 低频带：< 400Hz，长窗高频率精度
    // 不使用YIN，仅依赖FFT（8192点提供5.4Hz分辨率，足够精确）
    ResolutionBand low;
    low.minFreq = 50.0f;      // 最低检测50Hz（最低音域）
    low.maxFreq = 400.0f;     // 上限400Hz，与中频带无重叠
    low.fftOrder = 13;        // 8192点
    low.hopSize = 512;
    low.windowType = ResolutionBand::Blackman;
    low.strategy = ResolutionBand::HighPrecision;
    
    // 中频带：400-2000Hz，平衡设置
    // 这是钢琴和弦的主要基频区域
    ResolutionBand mid;
    mid.minFreq = 400.0f;     // 与低频带无重叠
    mid.maxFreq = 2000.0f;    // 上限2000Hz，覆盖到C7
    mid.fftOrder = 12;        // 4096点，10.8Hz分辨率
    mid.hopSize = 512;
    mid.windowType = ResolutionBand::Hann;
    mid.strategy = ResolutionBand::Balanced;
    
    // 高频带：2000-6000Hz，用于泛音验证
    // 短窗快速响应，不用于基频检测
    ResolutionBand high;
    high.minFreq = 2000.0f;   // 与中频带无重叠
    high.maxFreq = 6000.0f;   // 上限6000Hz
    high.fftOrder = 11;       // 2048点，21.5Hz分辨率
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
    
    // 当前版本：顺序处理（为后续并行化预留结构）
    for (int i = 0; i < 3; ++i) {
        bandTasks_[i]->process(input, output.bands[i]);
    }
    
    // 融合结果
    fuseSpectrums(output);
    
    // 计算处理时间
    auto endTime = juce::Time::getHighResolutionTicks();
    output.processingTimeMs = juce::Time::highResolutionTicksToSeconds(
        endTime - startTime) * 1000.0;
    output.isComplete = true;
}

void MultiResolutionAnalyzer::fuseSpectrums(MultiResolutionData& data)
{
    // 构建统一的融合频谱
    const int totalBins = 2048;
    
    auto& fused = data.fusedSpectrum;
    fused.frequencies.resize(totalBins);
    fused.magnitudes.resize(totalBins);
    fused.refinedFreqs.resize(totalBins);
    fused.sampleRate = static_cast<float>(sampleRate_);
    fused.fftSize = totalBins * 2;
    
    // 计算融合后的频率刻度
    for (int i = 0; i < totalBins; ++i) {
        fused.frequencies[i] = i * sampleRate_ / (2.0 * totalBins);
    }
    
    // 从各频带填充数据
    // 低频 (<400Hz)
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
    
    // 中频 (400-2000Hz) - 使用YIN结果精化
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
                
                // 中频带优先使用YIN结果（如果有）
                if (mid.hasYinResult && mid.yinConfidence > 0.5f &&
                    std::abs(freq - mid.yinFrequency) / mid.yinFrequency < 0.03f) {
                    fused.refinedFreqs[targetBin] = mid.yinFrequency;
                } else {
                    fused.refinedFreqs[targetBin] = mid.refinedFreqs[i];
                }
            }
        }
    }
    
    // 高频 (2000-6000Hz) - 仅用于泛音验证
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

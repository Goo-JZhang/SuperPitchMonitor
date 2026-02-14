#pragma once

#include <juce_core/juce_core.h>
#include <functional>
#include <queue>
#include <memory>
#include <atomic>
#include "../Audio/SpectrumData.h"

namespace spm {

// Forward declarations
class AudioEngine;
class MainComponent;

/**
 * AutoTestManager - 自动化测试管理器
 * 
 * 启动参数:
 *   -AutoTest          启用测试模式（不显示UI，通过命名管道接收命令）
 *   -TestPipe <name>   指定命名管道名称（默认: SPM_TestPipe）
 * 
 * 命令格式 (JSON):
 *   {"cmd": "setMultiRes", "enabled": true}
 *   {"cmd": "loadFile", "filename": "chord_c_major_7_piano.wav"}
 *   {"cmd": "start"}
 *   {"cmd": "stop"}
 *   {"cmd": "getPitches"}
 *   {"cmd": "getSpectrumPeaks", "freqMin": 50, "freqMax": 1000}
 *   {"cmd": "wait", "frames": 30}
 *   {"cmd": "exit"}
 */
class AutoTestManager
{
public:
    AutoTestManager();
    ~AutoTestManager();
    
    // 解析命令行参数，返回是否启用测试模式
    bool parseCommandLine(const juce::String& commandLine);
    bool isAutoTestMode() const { return autoTestMode_; }
    
    // 启动/停止测试模式
    bool startTestMode(AudioEngine* engine, MainComponent* main);
    void stopTestMode();
    
    // 处理待处理的命令（在主线程调用）
    void processPendingCommands();
    
    // 更新检测结果（从 AudioEngine 回调）
    void updatePitchResults(const std::vector<PitchCandidate>& pitches);
    void updateSpectrumData(const SpectrumData& data);
    void onFrameProcessed();
    
    // 获取当前状态
    juce::String getStatusJson() const;
    juce::String getPitchesJson() const;
    juce::String getSpectrumPeaksJson(float freqMin, float freqMax) const;
    
    // 等待帧数
    bool waitForFrames(int frameCount, int timeoutMs);
    
    // 检查是否应该退出
    bool shouldExit() const { return shouldExit_; }

private:
    struct TestCommand {
        juce::String cmd;
        juce::DynamicObject::Ptr params;
    };
    
    void runPipeServer();
    juce::String processCommand(const TestCommand& command);
    
    std::atomic<bool> autoTestMode_{false};
    std::atomic<bool> running_{false};
    juce::String pipeName_ = "SPM_TestPipe";
    
    AudioEngine* audioEngine_ = nullptr;
    MainComponent* mainComponent_ = nullptr;
    
    // 命令队列
    juce::CriticalSection commandLock_;
    std::queue<TestCommand> commandQueue_;
    
    // 结果缓存
    juce::CriticalSection resultsLock_;
    std::vector<PitchCandidate> latestPitches_;
    SpectrumData latestSpectrum_;
    std::atomic<int> frameCount_{0};
    
    // 等待帧数同步
    std::atomic<int> targetFrame_{0};
    juce::WaitableEvent frameEvent_;
    
    // 退出标志
    std::atomic<bool> shouldExit_{false};
    
    // Windows 命名管道句柄
    void* pipeHandle_ = nullptr;
    std::unique_ptr<juce::Thread> pipeThread_;
};

} // namespace spm

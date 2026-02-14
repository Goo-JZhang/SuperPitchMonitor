#pragma once

#include <juce_core/juce_core.h>
#include <juce_events/juce_events.h>
#include <memory>
#include <functional>
#include <vector>
#include <atomic>

namespace spm {

// Forward declarations
class AudioEngine;
class MainComponent;

/**
 * TCP-based test server for automated testing
 * Protocol: JSON-based command/response
 */
class TestServer : public juce::Thread
{
public:
    TestServer();
    ~TestServer() override;
    
    // Start server on specified port
    bool start(int port = 9999);
    void stop();
    
    // Set callbacks for command handling
    void setAudioEngine(AudioEngine* engine) { audioEngine_ = engine; }
    void setMainComponent(MainComponent* main) { mainComponent_ = main; }
    
    // Get latest detection results
    struct DetectionResult {
        float frequency;
        float midiNote;
        float confidence;
        float centsDeviation;
        int harmonicCount;
        juce::int64 timestamp;
    };
    
    void updateResults(const std::vector<DetectionResult>& results);
    std::vector<DetectionResult> getLatestResults() const;
    
    juce::int64 getFrameCount() const { return frameCount_; }
    void incrementFrameCount() { ++frameCount_; }
    
    void waitForFrames(int count, int timeoutMs);

private:
    void run() override;
    void handleClient(juce::StreamingSocket* clientSocket);
    juce::String processCommand(const juce::String& jsonCommand);
    
    // Command handlers
    juce::String handleGetStatus();
    juce::String handleSetMultiRes(bool enabled);
    juce::String handleLoadFile(const juce::String& filename);
    juce::String handleStartPlayback();
    juce::String handleStopPlayback();
    juce::String handleGetPitches();
    juce::String handleWaitForFrames(int frameCount);
    juce::String handleGetSpectrumPeaks();
    
    std::atomic<bool> running_{false};
    std::unique_ptr<juce::StreamingSocket> serverSocket_;
    AudioEngine* audioEngine_ = nullptr;
    MainComponent* mainComponent_ = nullptr;
    
    mutable juce::CriticalSection resultsLock_;
    std::vector<DetectionResult> latestResults_;
    std::atomic<juce::int64> frameCount_{0};
    
    int port_ = 9999;
};

} // namespace spm

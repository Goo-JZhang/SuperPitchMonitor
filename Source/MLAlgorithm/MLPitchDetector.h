/*
  =============================================================================-
    MLPitchDetector.h
    Pitch detection using ONNX Runtime neural network inference
  =============================================================================-
*/

#pragma once

#include <JuceHeader.h>
#include <onnxruntime_cxx_api.h>

// Platform-specific execution provider headers
#if defined(JUCE_MAC) || defined(JUCE_IOS)
    // CoreML EP header (if available in the build)
    #if __has_include(<coreml_provider_factory.h>)
        #include <coreml_provider_factory.h>
        #define HAS_COREML_EP 1
    #else
        #define HAS_COREML_EP 0
    #endif
#elif defined(JUCE_WINDOWS)
    // CUDA EP header (preferred for NVIDIA GPUs like RTX 4080S)
    #if __has_include(<cuda_provider_factory.h>)
        #include <cuda_provider_factory.h>
        #define HAS_CUDA_EP 1
    #else
        #define HAS_CUDA_EP 0
    #endif
    // DirectML EP header (fallback for AMD/Intel GPUs)
    #if __has_include(<dml_provider_factory.h>)
        #include <dml_provider_factory.h>
        #define HAS_DML_EP 1
    #else
        #define HAS_DML_EP 0
    #endif
#elif defined(JUCE_ANDROID)
    // NNAPI EP is usually available
    #define HAS_NNAPI_EP 1
#else
    #define HAS_COREML_EP 0
    #define HAS_DML_EP 0
    #define HAS_NNAPI_EP 0
#endif

#include <vector>
#include <memory>
#include <atomic>
#include <thread>
#include <queue>

namespace spm
{

/**
 * Neural network-based pitch detector
 * 
 * Input: 4096 samples @ 44.1kHz (raw audio waveform)
 * Output: 2048 frequency bins (20-5000Hz, log-spaced)
 *         Each bin: (confidence, energy)
 */
class MLPitchDetector
{
public:
    //==============================================================================
    struct Config
    {
        int sampleRate = 44100;
        int inputSamples = 4096;
        int numFreqBins = 2048;
        float minFreq = 20.0f;
        float maxFreq = 5000.0f;
        juce::String modelPath;
        
        // 运行时配置
        int threadPoolSize = 2;
        bool useGPU = true;
        
        // 检测阈值 (根据模型调整，训练不足的模型需要较低阈值如0.1)
        float confidenceThreshold = 0.1f;
    };
    
    struct Detection
    {
        float frequency = 0.0f;      // Hz
        float confidence = 0.0f;     // 0.0 - 1.0
        float energy = 0.0f;         // relative energy
        uint64_t frameIndex = 0;     // frame counter
    };
    
    //==============================================================================
    MLPitchDetector();
    ~MLPitchDetector();
    
    // 禁止拷贝
    MLPitchDetector(const MLPitchDetector&) = delete;
    MLPitchDetector& operator=(const MLPitchDetector&) = delete;
    
    //==============================================================================
    /** Initialize with configuration */
    bool initialize(const Config& config);
    
    /** Release resources */
    void release();
    
    /** Check if initialized */
    bool isInitialized() const { return initialized_.load(); }
    
    //==============================================================================
    /** 
     * Submit audio for inference (non-blocking)
     * @param audioData: pointer to 4096 float samples
     * @param numSamples: must be 4096
     */
    void submitAudio(const float* audioData, int numSamples);
    
    /** 
     * Synchronous inference (blocking, for testing)
     * @param audioData: pointer to 4096 float samples
     * @param outputBuffer: pre-allocated [2048 * 2] float buffer
     * @return true if successful
     */
    bool inferenceSync(const float* audioData, float* outputBuffer);
    
    //==============================================================================
    /** Get latest detection results (thread-safe, confidence > threshold only) */
    std::vector<Detection> getLatestResults() const;
    
    /** Get full spectrum data (all bins, for spectrum display) */
    std::vector<Detection> getFullSpectrum() const;
    
    /** Set callback for new results */
    using ResultsCallback = std::function<void(const std::vector<Detection>&)>;
    void setResultsCallback(ResultsCallback callback);
    
    //==============================================================================
    /** Get frequency for bin index (log-spaced 20-5000Hz) */
    static float binIndexToFrequency(int binIndex, int totalBins = 2048);
    
    /** Get frequency for fractional bin index (with sub-bin interpolation) */
    static float binIndexToFrequencyInterpolated(float binIndex, int totalBins = 2048);
    
    /** Get bin index for frequency */
    static int frequencyToBinIndex(float frequency, int totalBins = 2048);
    
    //==============================================================================
    /** Get last inference time in milliseconds */
    double getLastInferenceTimeMs() const { return lastInferenceTimeMs_; }
    
    /** Get model info string */
    juce::String getModelInfo() const;

private:
    //==============================================================================
    void inferenceThreadFunc();
    std::vector<Detection> postProcess(const float* rawOutput);
    
    //==============================================================================
    // ONNX Runtime
    std::unique_ptr<Ort::Env> ortEnv_;
    std::unique_ptr<Ort::Session> ortSession_;
    std::unique_ptr<Ort::MemoryInfo> memoryInfo_;
    
    // Model I/O info
    std::vector<int64_t> inputShape_;
    std::vector<int64_t> outputShape_;
    std::string inputName_;
    std::string outputName_;
    
    //==============================================================================
    // Threading
    std::unique_ptr<std::thread> inferenceThread_;
    std::atomic<bool> shouldExit_{false};
    std::atomic<bool> initialized_{false};
    
    // Lock-free queue for audio chunks
    struct AudioChunk
    {
        std::vector<float> data;
        uint64_t frameIndex;
    };
    static constexpr size_t maxQueueSize = 4;
    std::queue<AudioChunk> audioQueue_;
    mutable std::mutex queueMutex_;
    std::condition_variable queueCV_;
    
    //==============================================================================
    // Results
    std::vector<Detection> latestResults_;        // High confidence detections only
    std::vector<Detection> fullSpectrumResults_;  // All bins for spectrum display
    mutable std::mutex resultsMutex_;
    ResultsCallback resultsCallback_;
    std::atomic<uint64_t> frameCounter_{0};
    
    //==============================================================================
    // Performance metrics
    std::atomic<double> lastInferenceTimeMs_{0.0};
    
    //==============================================================================
    Config config_;
};

} // namespace spm

/*
  =============================================================================-
    MLPitchDetector.cpp
    GPU Async Inference Implementation
  =============================================================================-
*/

#include "MLPitchDetector.h"
#include "../Utils/Logger.h"
#include <chrono>
#include <algorithm>  // for std::max, std::abs
#include <cmath>      // for std::abs, std::log2, std::pow

namespace spm
{

//==============================================================================
MLPitchDetector::MLPitchDetector()
{
}

MLPitchDetector::~MLPitchDetector()
{
    release();
}

//==============================================================================
bool MLPitchDetector::initialize(const Config& config)
{
    if (initialized_.load())
    {
        release();
    }
    
    config_ = config;
    
    try
    {
        // Initialize ONNX Runtime environment
        OrtLoggingLevel logLevel = ORT_LOGGING_LEVEL_WARNING;
        ortEnv_ = std::make_unique<Ort::Env>(logLevel, "PitchDetector");
        
        // Session options
        Ort::SessionOptions sessionOptions;
        sessionOptions.SetIntraOpNumThreads(config.threadPoolSize);
        sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        
        // Get available execution providers
        auto availableProviders = Ort::GetAvailableProviders();
        SPM_LOG_INFO("[ML] Available execution providers:");
        for (const auto& provider : availableProviders)
        {
            SPM_LOG_INFO("[ML]   - " + juce::String(provider));
        }
        
        // Log compile-time flags
        SPM_LOG_INFO("[ML] Compile-time flags:");
        SPM_LOG_INFO("[ML]   - HAS_CUDA_EP = " + juce::String(HAS_CUDA_EP));
        SPM_LOG_INFO("[ML]   - HAS_DML_EP = " + juce::String(HAS_DML_EP));
        
        // Platform-specific GPU execution providers
        bool gpuEnabled = false;
        
        if (config.useGPU)
        {
#if defined(JUCE_MAC) || defined(JUCE_IOS)
            // Try to enable CoreML
            bool hasCoreML = false;
            for (const auto& provider : availableProviders)
            {
                if (std::string(provider) == "CoreMLExecutionProvider")
                {
                    hasCoreML = true;
                    break;
                }
            }
            
            if (hasCoreML && HAS_COREML_EP)
            {
                DBG("ML: Enabling CoreML execution provider");
                Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_CoreML(sessionOptions, 0));
                gpuEnabled = true;
                DBG("ML: CoreML enabled successfully");
            }
            else if (hasCoreML)
            {
                DBG("ML: CoreML available in library but headers not found");
                DBG("ML: Rebuild with CoreML headers for GPU support");
            }
            else
            {
                DBG("ML: CoreML not available (using CPU)");
            }
            
#elif defined(JUCE_WINDOWS)
            // Windows: Try CUDA first (best for NVIDIA RTX 4080S, etc.)
            bool hasCUDA = false;
            bool hasDML = false;
            
            for (const auto& provider : availableProviders)
            {
                std::string p(provider);
                if (p == "CUDAExecutionProvider")
                    hasCUDA = true;
                else if (p == "DmlExecutionProvider")
                    hasDML = true;
            }
            
            SPM_LOG_INFO("[ML] Runtime detection:");
            SPM_LOG_INFO("[ML]   - hasCUDA = " + juce::String(hasCUDA ? "true" : "false"));
            SPM_LOG_INFO("[ML]   - hasDML = " + juce::String(hasDML ? "true" : "false"));
            
            // 1. Try CUDA (NVIDIA GPUs)
            if (hasCUDA && HAS_CUDA_EP)
            {
                SPM_LOG_INFO("[ML] Attempting to enable CUDA execution provider...");
                try
                {
                    OrtCUDAProviderOptionsV2* cudaOptions = nullptr;
                    Ort::ThrowOnError(Ort::GetApi().CreateCUDAProviderOptions(&cudaOptions));
                    SPM_LOG_INFO("[ML] CUDA provider options created successfully");
                    
                    // Use AppendExecutionProvider_CUDA_V2
                    sessionOptions.AppendExecutionProvider_CUDA_V2(*cudaOptions);
                    gpuEnabled = true;
                    SPM_LOG_INFO("[ML] CUDA enabled successfully - RTX 4080S ready!");
                }
                catch (const Ort::Exception& e)
                {
                    SPM_LOG_ERROR("[ML] ERROR - CUDA initialization failed: " + juce::String(e.what()));
                    SPM_LOG_INFO("[ML] This may indicate:");
                    SPM_LOG_INFO("[ML]   1. CUDA runtime libraries (cudart64_*.dll) are not found");
                    SPM_LOG_INFO("[ML]   2. cuDNN libraries are missing");
                    SPM_LOG_INFO("[ML]   3. NVIDIA driver is outdated");
                    SPM_LOG_INFO("[ML]   4. GPU memory allocation failed");
                }
                catch (const std::exception& e)
                {
                    SPM_LOG_ERROR("[ML] ERROR - CUDA initialization failed (std): " + juce::String(e.what()));
                }
            }
            else if (hasCUDA && !HAS_CUDA_EP)
            {
                SPM_LOG_WARNING("[ML] WARNING - CUDA available in library but headers not found at compile time");
                SPM_LOG_INFO("[ML] HAS_CUDA_EP = 0, cannot enable CUDA");
                SPM_LOG_INFO("[ML] Rebuild with cuda_provider_factory.h in include path");
            }
            else if (!hasCUDA && HAS_CUDA_EP)
            {
                SPM_LOG_WARNING("[ML] WARNING - CUDA headers found at compile time but not in runtime library");
                SPM_LOG_INFO("[ML] The ONNX Runtime library may be CPU-only version");
            }
            else
            {
                SPM_LOG_INFO("[ML] CUDA not available (hasCUDA=false, HAS_CUDA_EP=0)");
            }
            
            // 2. Fallback to DirectML (AMD/Intel GPUs)
            if (!gpuEnabled && hasDML && HAS_DML_EP)
            {
                DBG("ML: Attempting to enable DirectML execution provider (fallback)...");
                try
                {
                    Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_DML(sessionOptions, 0));
                    gpuEnabled = true;
                    DBG("ML: DirectML enabled successfully");
                }
                catch (const Ort::Exception& e)
                {
                    DBG("ML: ERROR - DirectML initialization failed: " + juce::String(e.what()));
                }
            }
            else if (!gpuEnabled && hasDML && !HAS_DML_EP)
            {
                DBG("ML: WARNING - DirectML available in library but headers not found");
            }
            
            if (!gpuEnabled)
            {
                DBG("ML: No GPU execution provider available (using CPU)");
                DBG("ML: For RTX 4080S, ensure:");
                DBG("     1. ONNX Runtime GPU version is used (onnxruntime-win-x64-gpu)");
                DBG("     2. CUDA/cuDNN DLLs are in PATH or same directory as exe");
                DBG("     3. NVIDIA driver >= 520.00");
            }
            
#elif defined(JUCE_ANDROID)
            // Try to enable NNAPI
            bool hasNNAPI = false;
            for (const auto& provider : availableProviders)
            {
                if (std::string(provider) == "NnapiExecutionProvider")
                {
                    hasNNAPI = true;
                    break;
                }
            }
            
            if (hasNNAPI)
            {
                DBG("ML: Enabling NNAPI execution provider");
                Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_Nnapi(sessionOptions, 0));
                gpuEnabled = true;
                DBG("ML: NNAPI enabled successfully");
            }
            else
            {
                DBG("ML: NNAPI not available (using CPU)");
            }
#else
            DBG("ML: GPU not supported on this platform");
#endif
        }
        
        if (!gpuEnabled)
        {
            DBG("ML: Using CPU execution provider");
        }
        
        // Create session
        std::string modelPathStr = config.modelPath.toStdString();
        
        if (modelPathStr.empty())
        {
            SPM_LOG_ERROR("[ML] Model path is empty");
            return false;
        }
        
        if (!juce::File(modelPathStr).existsAsFile())
        {
            SPM_LOG_ERROR("[ML] Model file not found: " + juce::String(modelPathStr));
            return false;
        }
        
        SPM_LOG_INFO("[ML] Loading model from: " + juce::String(modelPathStr));
        
        // Check if onnxruntime library is available
        try
        {
#ifdef _WIN32
            // Windows uses wchar_t for paths
            std::wstring wModelPath(modelPathStr.begin(), modelPathStr.end());
            ortSession_ = std::make_unique<Ort::Session>(*ortEnv_, wModelPath.c_str(), sessionOptions);
#else
            ortSession_ = std::make_unique<Ort::Session>(*ortEnv_, modelPathStr.c_str(), sessionOptions);
#endif
        }
        catch (const Ort::Exception& e)
        {
            SPM_LOG_ERROR("[ML] Failed to create ONNX Session: " + juce::String(e.what()));
            SPM_LOG_ERROR("[ML] This may indicate the onnxruntime library is not properly loaded");
            return false;
        }
        
        // Get model I/O info
        Ort::AllocatorWithDefaultOptions allocator;
        
        // Input info
        size_t numInputs = ortSession_->GetInputCount();
        if (numInputs != 1)
        {
            DBG("ML: Warning - Model has " + juce::String(numInputs) + " inputs, expected 1");
        }
        
        inputName_ = ortSession_->GetInputNameAllocated(0, allocator).get();
        Ort::TypeInfo inputTypeInfo = ortSession_->GetInputTypeInfo(0);
        auto inputTensorInfo = inputTypeInfo.GetTensorTypeAndShapeInfo();
        inputShape_ = inputTensorInfo.GetShape();
        
        // Output info
        size_t numOutputs = ortSession_->GetOutputCount();
        if (numOutputs != 1)
        {
            DBG("ML: Warning - Model has " + juce::String(numOutputs) + " outputs, expected 1");
        }
        
        outputName_ = ortSession_->GetOutputNameAllocated(0, allocator).get();
        Ort::TypeInfo outputTypeInfo = ortSession_->GetOutputTypeInfo(0);
        auto outputTensorInfo = outputTypeInfo.GetTensorTypeAndShapeInfo();
        outputShape_ = outputTensorInfo.GetShape();
        
        // Validate shapes
        juce::String inputShapeStr = "[" + juce::String(inputShape_[0]) + ", " + 
                                     juce::String(inputShape_[1]) + ", " + 
                                     juce::String(inputShape_[2]) + "]";
        juce::String outputShapeStr = "[" + juce::String(outputShape_[0]) + ", " +
                                      juce::String(outputShape_[1]) + ", " +
                                      juce::String(outputShape_[2]) + "]";
        DBG("ML: Model input: " + juce::String(inputName_.c_str()) + " shape " + inputShapeStr);
        DBG("ML: Model output: " + juce::String(outputName_.c_str()) + " shape " + outputShapeStr);
        
        // Expected: input [1, 1, 4096], output [1, 2048, 2]
        if (inputShape_.size() != 3 || outputShape_.size() != 3)
        {
            DBG("ML: Error - Unexpected tensor dimensions, expected 3D tensors");
            return false;
        }
        
        // Create memory info
        memoryInfo_ = std::make_unique<Ort::MemoryInfo>(Ort::MemoryInfo::CreateCpu(
            OrtArenaAllocator, OrtMemTypeDefault));
        
        // Start inference thread
        shouldExit_ = false;
        inferenceThread_ = std::make_unique<std::thread>(&MLPitchDetector::inferenceThreadFunc, this);
        
        initialized_.store(true);
        DBG("ML: MLPitchDetector initialized (" + juce::String(gpuEnabled ? "GPU" : "CPU") + ")");
        return true;
        
    }
    catch (const Ort::Exception& e)
    {
        SPM_LOG_ERROR("[ML] ONNX Runtime error: " + juce::String(e.what()));
        return false;
    }
    catch (const std::exception& e)
    {
        SPM_LOG_ERROR("[ML] Initialization error: " + juce::String(e.what()));
        return false;
    }
}

void MLPitchDetector::release()
{
    if (!initialized_.load())
        return;
    
    // Signal thread to exit
    shouldExit_ = true;
    queueCV_.notify_all();
    
    // Wait for thread
    if (inferenceThread_ && inferenceThread_->joinable())
    {
        inferenceThread_->join();
        inferenceThread_.reset();
    }
    
    // Release ONNX resources
    memoryInfo_.reset();
    ortSession_.reset();
    ortEnv_.reset();
    
    initialized_.store(false);
    DBG("ML: MLPitchDetector released");
}

//==============================================================================
void MLPitchDetector::submitAudio(const float* audioData, int numSamples)
{
    if (!initialized_.load() || numSamples != config_.inputSamples)
    {
        jassertfalse;
        return;
    }
    
    // Copy audio to chunk
    AudioChunk chunk;
    chunk.data.assign(audioData, audioData + numSamples);
    chunk.frameIndex = frameCounter_.fetch_add(1);
    
    // Push to queue (drop old if full)
    {
        std::lock_guard<std::mutex> lock(queueMutex_);
        
        if (audioQueue_.size() >= maxQueueSize)
        {
            audioQueue_.pop();  // Drop oldest
            DBG("ML: Audio queue full, dropped frame");
        }
        
        audioQueue_.push(std::move(chunk));
    }
    
    queueCV_.notify_one();
}

bool MLPitchDetector::inferenceSync(const float* audioData, float* outputBuffer)
{
    if (!initialized_.load())
        return false;
    
    try
    {
        // Normalize input audio: Z-score normalization (mean=0, std=1)
        // This matches Python training preprocessing
        thread_local std::vector<float> normalizedAudio;
        normalizedAudio.resize(config_.inputSamples);
        
        // Step 1: Remove DC offset (make mean = 0)
        float mean = 0.0f;
        for (int i = 0; i < config_.inputSamples; ++i)
        {
            mean += audioData[i];
        }
        mean /= config_.inputSamples;
        
        // Step 2: Calculate standard deviation
        float variance = 0.0f;
        for (int i = 0; i < config_.inputSamples; ++i)
        {
            float diff = audioData[i] - mean;
            variance += diff * diff;
        }
        variance /= config_.inputSamples;
        float std = std::sqrt(variance);
        
        // Step 3: Check for silence (very low std)
        // If silence, skip inference and return zero confidence + uniform energy
        if (std < 1e-6f)
        {
            const int numBins = config_.numFreqBins;
            float uniformEnergy = 1.0f / numBins;
            for (int i = 0; i < numBins; ++i)
            {
                outputBuffer[i * 2] = 0.0f;        // confidence = 0
                outputBuffer[i * 2 + 1] = uniformEnergy;  // uniform energy
            }
            lastInferenceTimeMs_ = 0.0;  // No inference performed
            return true;
        }
        
        // Step 4: Z-score normalization
        float scale = 1.0f / std;
        for (int i = 0; i < config_.inputSamples; ++i)
        {
            normalizedAudio[i] = (audioData[i] - mean) * scale;
        }
        
        // Prepare input tensor [1, 1, 4096]
        std::vector<int64_t> inputDims = {1, 1, config_.inputSamples};
        Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
            *memoryInfo_, 
            normalizedAudio.data(), 
            config_.inputSamples, 
            inputDims.data(), 
            inputDims.size()
        );
        
        // Run inference
        const char* inputNames[] = {inputName_.c_str()};
        const char* outputNames[] = {outputName_.c_str()};
        
        auto startTime = std::chrono::high_resolution_clock::now();
        
        Ort::RunOptions runOptions;
        std::vector<Ort::Value> outputTensors = ortSession_->Run(
            runOptions,
            inputNames, &inputTensor, 1,
            outputNames, 1
        );
        
        auto endTime = std::chrono::high_resolution_clock::now();
        lastInferenceTimeMs_ = std::chrono::duration<double, std::milli>(endTime - startTime).count();
        
        // Copy output
        float* outputData = outputTensors[0].GetTensorMutableData<float>();
        size_t outputCount = outputTensors[0].GetTensorTypeAndShapeInfo().GetElementCount();
        std::memcpy(outputBuffer, outputData, outputCount * sizeof(float));
        
        return true;
        
    }
    catch (const Ort::Exception& e)
    {
        DBG("ML: Inference error: " + juce::String(e.what()));
        return false;
    }
}

//==============================================================================
void MLPitchDetector::inferenceThreadFunc()
{
    DBG("ML: Inference thread started");
    
    std::vector<float> outputBuffer(config_.numFreqBins * 2);
    
    while (!shouldExit_)
    {
        AudioChunk chunk;
        
        // Wait for audio data
        {
            std::unique_lock<std::mutex> lock(queueMutex_);
            queueCV_.wait(lock, [this] { return !audioQueue_.empty() || shouldExit_; });
            
            if (shouldExit_)
                break;
            
            chunk = std::move(audioQueue_.front());
            audioQueue_.pop();
        }
        
        // Run inference
        if (inferenceSync(chunk.data.data(), outputBuffer.data()))
        {
            // Note: Model output already includes softmax on energy channel
            // Post-process for high-confidence detections
            auto detections = postProcess(outputBuffer.data());
            
            // Generate full spectrum data (all bins)
            std::vector<Detection> fullSpectrum;
            fullSpectrum.reserve(config_.numFreqBins);
            for (int i = 0; i < config_.numFreqBins; ++i)
            {
                float confidence = outputBuffer[i * 2];
                float energy = outputBuffer[i * 2 + 1];
                Detection det;
                det.frequency = binIndexToFrequency(i, config_.numFreqBins);
                det.confidence = confidence;
                det.energy = energy;
                det.frameIndex = frameCounter_.load();
                fullSpectrum.push_back(det);
            }
            
            // Update results
            {
                std::lock_guard<std::mutex> lock(resultsMutex_);
                latestResults_ = std::move(detections);
                fullSpectrumResults_ = std::move(fullSpectrum);
            }
            
            // Callback
            if (resultsCallback_)
            {
                std::vector<Detection> resultsCopy;
                {
                    std::lock_guard<std::mutex> lock(resultsMutex_);
                    resultsCopy = latestResults_;
                }
                resultsCallback_(resultsCopy);
            }
        }
    }
    
    DBG("ML: Inference thread exited");
}

std::vector<MLPitchDetector::Detection> MLPitchDetector::postProcess(const float* rawOutput)
{
    std::vector<Detection> detections;
    detections.reserve(16);
    
    const int numBins = config_.numFreqBins;
    const float threshold = config_.confidenceThreshold;
    
    // Step 1: Find local peaks (low-high-low pattern)
    // A bin is a peak if its confidence is higher than both neighbors
    for (int i = 1; i < numBins - 1; ++i)
    {
        float confCenter = rawOutput[i * 2];
        float confLeft = rawOutput[(i - 1) * 2];
        float confRight = rawOutput[(i + 1) * 2];
        
        // Skip if below threshold
        if (confCenter < threshold)
            continue;
        
        // Check for local peak: left < center > right
        if (confCenter > confLeft && confCenter > confRight)
        {
            Detection det;
            det.frameIndex = frameCounter_.load();
            
            // Step 2: Quadratic interpolation for sub-bin accuracy
            // Fit parabola: y = ax^2 + bx + c through three points
            // At x = -1: y = confLeft
            // At x = 0:  y = confCenter  
            // At x = 1:  y = confRight
            
            // Peak offset from center bin (in bin units)
            float peakOffset = 0.0f;
            float peakConfidence = confCenter;
            
            // Only interpolate if we have valid neighbors
            if (confLeft > 0 && confRight > 0)
            {
                // Parabola vertex: x = -b / (2a)
                // Using finite differences: a = (confLeft + confRight - 2*confCenter) / 2
                // b = (confRight - confLeft) / 2
                float a = (confLeft + confRight - 2.0f * confCenter) * 0.5f;
                float b = (confRight - confLeft) * 0.5f;
                
                if (std::abs(a) > 1e-6f)  // Avoid division by near-zero
                {
                    peakOffset = -b / (2.0f * a);
                    
                    // Clamp offset to [-1, 1] (between neighbors)
                    peakOffset = juce::jlimit(-1.0f, 1.0f, peakOffset);
                    
                    // Calculate interpolated confidence at peak
                    peakConfidence = a * peakOffset * peakOffset + b * peakOffset + confCenter;
                    
                    // Clamp confidence to [0, 1] as it may exceed 1.0 after fitting
                    peakConfidence = juce::jlimit(0.0f, 1.0f, peakConfidence);
                }
            }
            
            // Step 3: Calculate interpolated frequency
            // Use log-scale interpolation for frequency
            float binIndexWithOffset = static_cast<float>(i) + peakOffset;
            det.frequency = binIndexToFrequencyInterpolated(binIndexWithOffset, numBins);
            det.confidence = peakConfidence;
            
            // Step 4: Accumulate energy from neighborhood (卤2 bins)
            // This gives total energy of the detected pitch
            float totalEnergy = 0.0f;
            int energyWindow = 2;
            for (int j = std::max(0, i - energyWindow); 
                 j <= std::min(numBins - 1, i + energyWindow); ++j)
            {
                totalEnergy += rawOutput[j * 2 + 1];  // energy channel
            }
            det.energy = totalEnergy;
            
            detections.push_back(det);
        }
    }
    
    // Step 5: Sort by confidence descending for output
    std::sort(detections.begin(), detections.end(),
        [](const Detection& a, const Detection& b) { return a.confidence > b.confidence; });
    
    return detections;
}

//==============================================================================
std::vector<MLPitchDetector::Detection> MLPitchDetector::getLatestResults() const
{
    std::lock_guard<std::mutex> lock(resultsMutex_);
    // Return high-confidence results sorted by confidence descending
    return latestResults_;
}

std::vector<MLPitchDetector::Detection> MLPitchDetector::getFullSpectrum() const
{
    std::lock_guard<std::mutex> lock(resultsMutex_);
    // Return all bins sorted by frequency (ascending) for spectrum display
    std::vector<Detection> sortedResults = fullSpectrumResults_;
    std::sort(sortedResults.begin(), sortedResults.end(),
        [](const Detection& a, const Detection& b) { return a.frequency < b.frequency; });
    return sortedResults;
}

void MLPitchDetector::setResultsCallback(ResultsCallback callback)
{
    resultsCallback_ = callback;
}

//==============================================================================
float MLPitchDetector::binIndexToFrequency(int binIndex, int totalBins)
{
    const float minFreq = 20.0f;
    const float maxFreq = 5000.0f;
    const float ratio = static_cast<float>(binIndex) / (totalBins - 1);
    return minFreq * std::pow(maxFreq / minFreq, ratio);
}

float MLPitchDetector::binIndexToFrequencyInterpolated(float binIndex, int totalBins)
{
    const float minFreq = 20.0f;
    const float maxFreq = 5000.0f;
    const float ratio = binIndex / (totalBins - 1);
    return minFreq * std::pow(maxFreq / minFreq, ratio);
}

int MLPitchDetector::frequencyToBinIndex(float frequency, int totalBins)
{
    const float minFreq = 20.0f;
    const float maxFreq = 5000.0f;
    
    float logFreq = std::log(frequency / minFreq) / std::log(maxFreq / minFreq);
    int bin = static_cast<int>(logFreq * (totalBins - 1) + 0.5f);
    
    return juce::jlimit(0, totalBins - 1, bin);
}

//==============================================================================
juce::String MLPitchDetector::getModelInfo() const
{
    if (!initialized_.load())
        return "Not initialized";
    
    juce::String info;
    info += "Input: [" + juce::String(inputShape_[0]) + ", " + 
            juce::String(inputShape_[1]) + ", " + juce::String(inputShape_[2]) + "]\n";
    info += "Output: [" + juce::String(outputShape_[0]) + ", " +
            juce::String(outputShape_[1]) + ", " + juce::String(outputShape_[2]) + "]\n";
    info += "Last inference: " + juce::String(lastInferenceTimeMs_, 2) + " ms";
    return info;
}

} // namespace spm

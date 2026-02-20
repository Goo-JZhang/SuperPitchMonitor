#include "AudioEngine.h"
#include "SpectrumAnalyzer.h"
#include "PolyphonicDetector.h"
#include "../Debug/AudioSimulator.h"
#include "../Utils/Logger.h"

namespace spm {

AudioEngine::AudioEngine()
    : Thread("AudioProcessingThread")
{
    // Pre-allocate FIFO buffers
    audioBuffers_.allocate(FIFOSize, 1);
    for (int i = 0; i < FIFOSize; ++i)
    {
        audioBuffers_[i].setSize(1, 4096);  // Max FFT size
    }
    
    // Initialize processors
    spectrumAnalyzer_ = std::make_unique<SpectrumAnalyzer>();
    polyphonicDetector_ = std::make_unique<PolyphonicDetector>();
    mlDetector_ = std::make_unique<MLPitchDetector>();
    
    // Initialize spectrum analyzer with default values
    spectrumAnalyzer_->prepare(Config::Audio::DefaultSampleRate, 
                               Config::Spectrum::DefaultFFTOrder);
    
    // Initialize polyphonic detector with default values
    polyphonicDetector_->prepare(Config::Audio::DefaultSampleRate,
                                 Config::Pitch::MinFrequency,
                                 Config::Pitch::MaxFrequency);
    
    // Initialize ML detector (async GPU inference by default)
    MLPitchDetector::Config mlConfig;
    mlConfig.sampleRate = Config::Audio::DefaultSampleRate;
    mlConfig.useGPU = true;
    mlConfig.threadPoolSize = 2;
    mlConfig.confidenceThreshold = 0.1f;  // Lower threshold for current model
    // Model path will be set when enabling ML analysis
    mlDetector_->initialize(mlConfig);
    
    // Initialize level smoother
    inputLevel_.setCurrentAndTargetValue(0.0f);
    inputLevel_.reset(sampleRate_, 0.05);  // 50ms smoothing time
}

AudioEngine::~AudioEngine()
{
    stop();
    shutdownAudio();
}

juce::String AudioEngine::initialize()
{
    // Only initialize audio device in real device mode
    if (currentMode_ == Mode::RealDevice)
    {
        // Setup audio device manager
        juce::AudioDeviceManager::AudioDeviceSetup setup;
        setup.sampleRate = Config::Audio::DefaultSampleRate;
        setup.bufferSize = Config::Audio::DefaultBufferSize;
        setup.inputChannels = Config::Audio::InputChannels;
        setup.outputChannels = Config::Audio::OutputChannels;
        setup.useDefaultInputChannels = true;
        setup.useDefaultOutputChannels = false;
        
        // Android specific settings
       #if JUCE_ANDROID
        setup.inputDeviceName = "Android Input";
       #endif
        
        juce::String error = deviceManager.initialise(
            Config::Audio::InputChannels,
            Config::Audio::OutputChannels,
            nullptr,
            true,
            {},
            &setup
        );
        
        if (error.isEmpty())
        {
            auto* device = deviceManager.getCurrentAudioDevice();
            if (device != nullptr)
            {
                sampleRate_ = device->getCurrentSampleRate();
                bufferSize_ = device->getCurrentBufferSizeSamples();
                
                DBG("Audio initialized: " << sampleRate_ << " Hz, " 
                    << bufferSize_ << " samples");
                    
                // Initialize detector with actual sample rate
                polyphonicDetector_->prepare(sampleRate_, Config::Pitch::MinFrequency,
                                              Config::Pitch::MaxFrequency);
                
                // Reinitialize ML detector with actual sample rate
                mlDetector_->release();
                MLPitchDetector::Config mlConfig;
                mlConfig.sampleRate = static_cast<int>(sampleRate_);
                mlConfig.useGPU = true;
                mlConfig.threadPoolSize = 2;
                mlConfig.confidenceThreshold = 0.1f;  // Lower threshold for current model
                mlDetector_->initialize(mlConfig);
            }
            else
            {
                error = "No audio device available";
            }
        }
        else
        {
            // Real device init failed, suggest switching to simulation mode
            DBG("Real device init failed: " << error);
        }
        
        return error;
    }
    else
    {
        // Simulation mode uses default sample rate
        sampleRate_ = 44100.0;
        bufferSize_ = 512;
        return {};
    }
}

void AudioEngine::setMode(Mode mode)
{
    if (currentMode_ == mode)
        return;
    
    bool wasRunning = isRunning_;
    
    if (wasRunning)
        stop();
    
    currentMode_ = mode;
    
    if (currentMode_ == Mode::Simulated && simulator_ != nullptr)
    {
        sampleRate_ = simulator_->getSampleRate();
        bufferSize_ = 512;
    }
    
    // Reinitialize processors
    spectrumAnalyzer_->prepare(sampleRate_, Config::Spectrum::DefaultFFTOrder);
    polyphonicDetector_->prepare(sampleRate_, Config::Pitch::MinFrequency, 
                                  Config::Pitch::MaxFrequency);
    
    if (wasRunning)
        start();
}

void AudioEngine::setInputSource(std::shared_ptr<AudioInputSource> source)
{
    if (isRunning_)
    {
        SPM_LOG_WARNING("[AudioEngine] Cannot change input source while running");
        return;
    }
    
    // Disconnect old source
    if (inputSource_)
    {
        SPM_LOG_INFO("[AudioEngine] Disconnecting old source: " + inputSource_->getName());
        inputSource_->setAudioCallback(nullptr);
        inputSource_->setLevelCallback(nullptr);
    }
    
    inputSource_ = source;
    
    if (inputSource_)
    {
        SPM_LOG_INFO("[AudioEngine] ====== INPUT SOURCE CHANGED ======");
        SPM_LOG_INFO("[AudioEngine] Source type: " + juce::String((int)inputSource_->getType()));
        SPM_LOG_INFO("[AudioEngine] Source name: " + inputSource_->getName());
        
        // Configure new source
        inputSource_->prepare(sampleRate_, bufferSize_);
        
        // Connect callbacks
        inputSource_->setAudioCallback([this](const juce::AudioBuffer<float>& buffer) {
            this->processAudioBlock(buffer);
        });
        
        // Only set level callback if we have one (avoid GUI updates from non-GUI threads)
        if (inputLevelCallback_)
        {
            inputSource_->setLevelCallback([this](float level) {
                if (inputLevelCallback_)
                    inputLevelCallback_(level);
            });
        }
        
       #if defined(DEBUG) || defined(_DEBUG)
        DBG("[AudioEngine] Input source set: " << inputSource_->getName());
       #endif
        
        // Update mode based on source type
        switch (inputSource_->getType())
        {
            case AudioInputSource::Type::Device:
                currentMode_ = Mode::RealDevice;
                break;
            case AudioInputSource::Type::SystemAudio:
            case AudioInputSource::Type::FilePlayback:
                currentMode_ = Mode::Simulated;
                break;
        }
    }
}

void AudioEngine::setSimulator(AudioSimulator* simulator)
{
    simulator_ = simulator;
    
    if (currentMode_ == Mode::Simulated && simulator_ != nullptr)
    {
        sampleRate_ = simulator->getSampleRate();
        
        // Setup simulator callbacks
        simulator_->setAudioCallback([this](const juce::AudioBuffer<float>& buffer) {
            processAudioBlock(buffer);
        });
        
        simulator_->setLevelCallback([this](float level) {
            if (inputLevelCallback_)
                inputLevelCallback_(level);
        });
        
        // Reinitialize processors
        spectrumAnalyzer_->prepare(sampleRate_, Config::Spectrum::DefaultFFTOrder);
        polyphonicDetector_->prepare(sampleRate_, Config::Pitch::MinFrequency, 
                                      Config::Pitch::MaxFrequency);
    }
}

void AudioEngine::start()
{
    if (isRunning_)
    {
       #if defined(DEBUG) || defined(_DEBUG)
        DBG("[AudioEngine] Start called but already running");
       #endif
        return;
    }
    
    isRunning_ = true;
    shouldExit_ = false;
    
   #if defined(DEBUG) || defined(_DEBUG)
    DBG("[AudioEngine] Starting in mode=" << (currentMode_ == Mode::Simulated ? "Simulated" : "RealDevice"));
   #endif
    
    // Prefer new input source if available
    if (inputSource_)
    {
        inputSource_->start();
        SPM_LOG_INFO("[AudioEngine] Input source started: " + inputSource_->getName());
       #if defined(DEBUG) || defined(_DEBUG)
        DBG("[AudioEngine] Input source started: " << inputSource_->getName());
       #endif
    }
    else if (currentMode_ == Mode::RealDevice)
    {
        // Legacy mode
        startThread(juce::Thread::Priority::high);
        setAudioChannels(Config::Audio::InputChannels, Config::Audio::OutputChannels);
       #if defined(DEBUG) || defined(_DEBUG)
        DBG("[AudioEngine] Real device started (legacy mode)");
       #endif
    }
    else if (currentMode_ == Mode::Simulated && simulator_ != nullptr)
    {
        // Legacy simulator
        simulator_->start();
       #if defined(DEBUG) || defined(_DEBUG)
        DBG("[AudioEngine] Simulator started (legacy mode)");
       #endif
    }
    else if (currentMode_ == Mode::Simulated && simulator_ == nullptr && !inputSource_)
    {
       #if defined(DEBUG) || defined(_DEBUG)
        DBG("[AudioEngine] ERROR: No input source available!");
       #endif
    }
}

void AudioEngine::stop()
{
    if (!isRunning_)
        return;
    
    isRunning_ = false;
    shouldExit_ = true;
    
    // Stop new input source if available
    if (inputSource_)
    {
        inputSource_->stop();
       #if defined(DEBUG) || defined(_DEBUG)
        DBG("[AudioEngine] Input source stopped");
       #endif
    }
    else if (currentMode_ == Mode::RealDevice)
    {
        // Legacy mode
        shutdownAudio();
        stopThread(2000);
    }
    else if (currentMode_ == Mode::Simulated && simulator_ != nullptr)
    {
        // Legacy simulator
        simulator_->stop();
    }
}

void AudioEngine::prepareToPlay(int samplesPerBlockExpected, double newSampleRate)
{
    sampleRate_ = newSampleRate;
    bufferSize_ = samplesPerBlockExpected;
    
    inputLevel_.reset(sampleRate_, 0.05);
}

void AudioEngine::releaseResources()
{
    // Cleanup resources
}

void AudioEngine::getNextAudioBlock(const juce::AudioSourceChannelInfo& bufferToFill)
{
    if (!isRunning_ || bufferToFill.buffer == nullptr || 
        currentMode_ != Mode::RealDevice)
    {
        bufferToFill.clearActiveBufferRegion();
        return;
    }
    
    // Calculate input level
    float rms = 0.0f;
    if (bufferToFill.buffer->getNumChannels() > 0)
    {
        rms = bufferToFill.buffer->getRMSLevel(0, bufferToFill.startSample, 
                                                bufferToFill.numSamples);
        inputLevel_.setTargetValue(rms);
    }
    
    // Notify input level
    if (inputLevelCallback_ && inputLevel_.isSmoothing())
    {
        inputLevelCallback_(inputLevel_.getNextValue());
    }
    
    // Write to FIFO
    int startIndex1, blockSize1, startIndex2, blockSize2;
    fifo_.prepareToWrite(1, startIndex1, blockSize1, startIndex2, blockSize2);
    
    if (blockSize1 > 0)
    {
        auto& targetBuffer = audioBuffers_[startIndex1];
        targetBuffer.setSize(1, bufferToFill.numSamples);
        targetBuffer.copyFrom(0, 0, *bufferToFill.buffer, 0, 
                              bufferToFill.startSample, bufferToFill.numSamples);
        
        fifo_.finishedWrite(1);
    }
    
    // Clear output (we only use input)
    bufferToFill.clearActiveBufferRegion();
}

void AudioEngine::run()
{
    while (!shouldExit_ && !threadShouldExit())
    {
        // Read from FIFO
        int startIndex1, blockSize1, startIndex2, blockSize2;
        fifo_.prepareToRead(1, startIndex1, blockSize1, startIndex2, blockSize2);
        
        if (blockSize1 > 0)
        {
            const auto& inputBuffer = audioBuffers_[startIndex1];
            processAudioBlock(inputBuffer);
            fifo_.finishedRead(1);
        }
        else
        {
            // No data, wait a bit
            juce::Thread::sleep(1);
        }
    }
}

void AudioEngine::processAudioBlock(const juce::AudioBuffer<float>& buffer)
{
    SPM_PROFILE_SCOPE("AudioProcessing");
    
    static int blockCount = 0;
    ++blockCount;
    
    // ML Analysis mode (default) - GPU async inference
    if (useMLAnalysis_ && mlDetector_ && mlDetector_->isInitialized())
    {
        const int mlInputSize = 4096;
        const int numSamples = buffer.getNumSamples();
        const float* channelData = buffer.getReadPointer(0);
        
        // Initialize ring buffer on first use
        if (mlRingBuffer_.empty())
        {
            mlRingBuffer_.resize(MLWindowSize);
            mlWritePos_.store(0);
        }
        
        // Write new samples to ring buffer
        int writePos = mlWritePos_.load();
        for (int i = 0; i < numSamples; ++i)
        {
            mlRingBuffer_[writePos] = channelData[i];
            writePos = (writePos + 1) % MLWindowSize;
        }
        mlWritePos_.store(writePos);
        
        // Extract latest 4096 samples from ring buffer (if available)
        static int totalSamplesReceived = 0;
        totalSamplesReceived += numSamples;
        
        if (totalSamplesReceived >= mlInputSize)
        {
            // Prepare input buffer
            if (mlInputBuffer_.size() != mlInputSize)
                mlInputBuffer_.resize(mlInputSize);
            
            // Read backwards from write position to get latest 4096 samples
            int readPos = (writePos - mlInputSize + MLWindowSize) % MLWindowSize;
            for (int i = 0; i < mlInputSize; ++i)
            {
                mlInputBuffer_[i] = mlRingBuffer_[readPos];
                readPos = (readPos + 1) % MLWindowSize;
            }
            
            // Submit for async GPU inference
            mlDetector_->submitAudio(mlInputBuffer_.data(), mlInputSize);
        }
        
        // Get detection results (high confidence only) and full spectrum (all bins)
        auto mlDetections = mlDetector_->getLatestResults();
        auto mlSpectrum = mlDetector_->getFullSpectrum();
        
        // Convert detections to PitchVector for compatibility
        PitchVector pitches;
        for (const auto& det : mlDetections)
        {
            PitchCandidate pitch;
            pitch.frequency = det.frequency;
            pitch.amplitude = det.energy;  // ML mode: use energy directly
            pitch.confidence = det.confidence;
            pitch.isMLEnergy = true;       // Mark as ML energy
            
            // Calculate MIDI note and cents deviation
            pitch.midiNote = FFTUtils::freqToMidi(det.frequency);
            int roundedMidi = static_cast<int>(std::round(pitch.midiNote));
            pitch.centsDeviation = (pitch.midiNote - roundedMidi) * 100.0f;
            
            pitches.push_back(pitch);
        }
        
        // Build spectrum data from ML full spectrum for visualization
        SpectrumData spectrumData;
        spectrumData.sampleRate = sampleRate_;
        spectrumData.timestamp = juce::Time::getCurrentTime().toMilliseconds() / 1000.0;
        spectrumData.isMLMode = true;  // Mark as ML mode data
        
        // Convert ML full spectrum (all 2048 bins) to spectrum format
        spectrumData.frequencies.reserve(mlSpectrum.size());
        spectrumData.mlConfidence.reserve(mlSpectrum.size());
        spectrumData.mlEnergy.reserve(mlSpectrum.size());
        
        for (const auto& det : mlSpectrum)
        {
            spectrumData.frequencies.push_back(det.frequency);
            spectrumData.mlConfidence.push_back(det.confidence);
            spectrumData.mlEnergy.push_back(det.energy);
            // Keep magnitudes for backward compatibility (confidence * energy)
            spectrumData.magnitudes.push_back(det.confidence * det.energy);
        }
        
        if (spectrumCallback_)
        {
            spectrumCallback_(spectrumData);
        }
        
        if (pitchCallback_ && !pitches.empty())
        {
            pitchCallback_(pitches);
        }
        
        // Log ML inference performance
        if (blockCount % 100 == 0)
        {
            SPM_LOG_INFO("[AudioEngine] ML inference time: " 
                        + juce::String(mlDetector_->getLastInferenceTimeMs(), 2) + " ms");
        }
    }
    else
    {
        // Fallback to traditional FFT-based analysis
        SpectrumData spectrumData;
        spectrumAnalyzer_->process(buffer, spectrumData);
        spectrumData.isFFTMode = true;  // Mark as FFT mode data
        
        if (spectrumCallback_)
        {
            spectrumCallback_(spectrumData);
        }
        
        // Pitch detection
        PitchVector pitches;
        polyphonicDetector_->detect(spectrumData, pitches);
        
        if (pitchCallback_ && !pitches.empty())
        {
            pitchCallback_(pitches);
        }
    }
}

void AudioEngine::setQualityLevel(Config::Performance::QualityLevel level)
{
    // Update FFT size and other parameters
    int fftOrder = Config::Spectrum::DefaultFFTOrder;
    
    switch (level)
    {
        case Config::Performance::QualityLevel::Fast:
            fftOrder = Config::Spectrum::LowLatencyFFTOrder;
            break;
        case Config::Performance::QualityLevel::Balanced:
            fftOrder = Config::Spectrum::DefaultFFTOrder;
            break;
        case Config::Performance::QualityLevel::Accurate:
            fftOrder = Config::Spectrum::HighQualityFFTOrder;
            break;
    }
    
    spectrumAnalyzer_->prepare(sampleRate_, fftOrder);
}

void AudioEngine::setMLAnalysisEnabled(bool enabled)
{
    useMLAnalysis_ = enabled;
    
    if (enabled && mlDetector_)
    {
        // Ensure ML detector is initialized with model
        if (!mlDetector_->isInitialized())
        {
            MLPitchDetector::Config mlConfig;
            mlConfig.sampleRate = static_cast<int>(sampleRate_);
            mlConfig.useGPU = useMLGPU_;  // Use GPU or CPU based on setting
            mlConfig.threadPoolSize = 2;
            mlConfig.confidenceThreshold = 0.1f;  // Lower threshold for current model
            
            // Use configured model path, or auto-detect from MLModel directory
            if (mlModelPath_.isNotEmpty())
            {
                mlConfig.modelPath = mlModelPath_;
            }
            else
            {
                // Auto-detect any .onnx file in MLModel directory
                juce::File modelDir = juce::File::getSpecialLocation(
                    juce::File::currentApplicationFile)
                    .getParentDirectory()
                    .getChildFile("MLModel");
                
                juce::Array<juce::File> onnxFiles;
                modelDir.findChildFiles(onnxFiles, juce::File::findFiles, false, "*.onnx");
                
                if (onnxFiles.size() > 0)
                {
                    // Use first available .onnx file and save the path
                    mlModelPath_ = onnxFiles[0].getFullPathName();
                    mlConfig.modelPath = mlModelPath_;
                    SPM_LOG_INFO("[AudioEngine] Auto-selected model: " + onnxFiles[0].getFileName());
                }
                else
                {
                    // Fallback to default name (will fail with clear error)
                    mlConfig.modelPath = modelDir.getChildFile("pitchnet_v1.onnx").getFullPathName();
                }
            }
            
            if (mlDetector_->initialize(mlConfig))
            {
                SPM_LOG_INFO("[AudioEngine] ML Analysis enabled (" 
                            + juce::String(useMLGPU_ ? "GPU" : "CPU") + " mode)");
            }
            else
            {
                SPM_LOG_ERROR("[AudioEngine] Failed to initialize ML detector");
                useMLAnalysis_ = false;
            }
        }
        else
        {
            SPM_LOG_INFO("[AudioEngine] ML Analysis enabled");
        }
    }
    else
    {
        SPM_LOG_INFO("[AudioEngine] ML Analysis disabled, using FFT fallback");
    }
}

void AudioEngine::setMLGPUEnabled(bool enabled)
{
    useMLGPU_ = enabled;
    
    // If ML detector is initialized, reinitialize with new GPU setting
    if (mlDetector_ && mlDetector_->isInitialized())
    {
        mlDetector_->release();
        
        MLPitchDetector::Config mlConfig;
        mlConfig.sampleRate = static_cast<int>(sampleRate_);
        mlConfig.useGPU = useMLGPU_;
        mlConfig.threadPoolSize = 2;
        mlConfig.confidenceThreshold = 0.1f;  // Lower threshold for current model
        
        // Use configured model path, or auto-detect from MLModel directory
        if (mlModelPath_.isNotEmpty())
        {
            mlConfig.modelPath = mlModelPath_;
        }
        else
        {
            // Auto-detect any .onnx file in MLModel directory
            juce::File modelDir = juce::File::getSpecialLocation(
                juce::File::currentApplicationFile)
                .getParentDirectory()
                .getChildFile("MLModel");
            
            juce::Array<juce::File> onnxFiles;
            modelDir.findChildFiles(onnxFiles, juce::File::findFiles, false, "*.onnx");
            
            if (onnxFiles.size() > 0)
            {
                // Use first available .onnx file and save the path
                mlModelPath_ = onnxFiles[0].getFullPathName();
                mlConfig.modelPath = mlModelPath_;
                SPM_LOG_INFO("[AudioEngine] Auto-selected model: " + onnxFiles[0].getFileName());
            }
            else
            {
                // Fallback to default name (will fail with clear error)
                mlConfig.modelPath = modelDir.getChildFile("pitchnet_v1.onnx").getFullPathName();
            }
        }
        
        if (mlDetector_->initialize(mlConfig))
        {
            SPM_LOG_INFO("[AudioEngine] ML detector reinitialized with " 
                        + juce::String(useMLGPU_ ? "GPU" : "CPU") + " mode");
        }
        else
        {
            SPM_LOG_ERROR("[AudioEngine] Failed to reinitialize ML detector");
        }
    }
    else
    {
        SPM_LOG_INFO("[AudioEngine] ML GPU mode set to: " + juce::String(enabled ? "GPU" : "CPU"));
    }
}

void AudioEngine::setBufferSize(int newBufferSize)
{
    if (bufferSize_ == newBufferSize)
        return;
    
    bufferSize_ = newBufferSize;
    
    // Update ML ring buffer size (2x the input size for safety)
    int newWindowSize = MLInputSize * 2;
    if (newWindowSize != MLWindowSize)
    {
        // This would require recompiling, so for now we just log
        // The actual buffer size change affects the callback frequency
    }
    
    // Reinitialize audio to apply new buffer size
    // Note: This requires restarting the audio stream
    if (isRunning_)
    {
        stop();
        
        // Re-prepare input source with new buffer size
        if (inputSource_)
        {
            inputSource_->prepare(sampleRate_, bufferSize_);
        }
        
        start();
        SPM_LOG_INFO("[AudioEngine] Restarted with new buffer size: " + juce::String(bufferSize_));
    }
    else
    {
        // Just update the preparation for next start
        if (inputSource_)
        {
            inputSource_->prepare(sampleRate_, bufferSize_);
        }
        SPM_LOG_INFO("[AudioEngine] Buffer size set to: " + juce::String(bufferSize_) + 
                     " (will apply on next start)");
    }
}

void AudioEngine::setMLModelPath(const juce::String& modelPath)
{
    mlModelPath_ = modelPath;
    
    // Reinitialize ML detector with new model
    if (mlDetector_ && useMLAnalysis_)
    {
        mlDetector_->release();
        
        MLPitchDetector::Config mlConfig;
        mlConfig.sampleRate = static_cast<int>(sampleRate_);
        mlConfig.useGPU = useMLGPU_;
        mlConfig.threadPoolSize = 2;
        mlConfig.modelPath = modelPath;
        mlConfig.confidenceThreshold = 0.1f;  // Lower threshold for current model
        
        SPM_LOG_INFO("[AudioEngine] ML confidence threshold: " + juce::String(mlConfig.confidenceThreshold));
        
        if (mlDetector_->initialize(mlConfig))
        {
            SPM_LOG_INFO("[AudioEngine] ML detector loaded model: " + modelPath);
        }
        else
        {
            SPM_LOG_ERROR("[AudioEngine] Failed to load ML model: " + modelPath);
        }
    }
}

void AudioEngine::setDetectionRange(float minFreq, float maxFreq)
{
    if (polyphonicDetector_)
    {
        polyphonicDetector_->prepare(sampleRate_, minFreq, maxFreq);
    }
    
    // Also update ML detector config if needed
    if (mlDetector_)
    {
        // ML detector uses fixed 20-5000Hz range in model
        // But we can filter results
    }
}

} // namespace spm

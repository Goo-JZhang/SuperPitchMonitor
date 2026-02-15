#include "AudioEngine.h"
#include "SpectrumAnalyzer.h"
#include "PolyphonicDetector.h"
#include "MultiResolutionAnalyzer.h"
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
    multiResAnalyzer_ = std::make_unique<MultiResolutionAnalyzer>();
    
    // Initialize spectrum analyzer with default values
    spectrumAnalyzer_->prepare(Config::Audio::DefaultSampleRate, 
                               Config::Spectrum::DefaultFFTOrder);
    
    // Initialize polyphonic detector with default values
    polyphonicDetector_->prepare(Config::Audio::DefaultSampleRate,
                                 Config::Pitch::MinFrequency,
                                 Config::Pitch::MaxFrequency);
    
    // Initialize multi-resolution analyzer
    multiResData_ = std::make_unique<MultiResolutionData>();
    
    // Initialize level smoother
    inputLevel_.setCurrentAndTargetValue(0.0f);
    inputLevel_.reset(0.05);  // 50ms smoothing time
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
    
    inputLevel_.reset(0.05);
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
    
    // Log every 30 frames (~1 sec at 30fps) using SPM_LOG for file logging
    static int blockCount = 0;
    if (++blockCount % 30 == 0)
    {
        SPM_LOG_INFO("[AudioEngine] Processing block #" + juce::String(blockCount) 
                     + " samples=" + juce::String(buffer.getNumSamples())
                     + " rms=" + juce::String(buffer.getRMSLevel(0, 0, buffer.getNumSamples()), 4)
                     + " multiRes=" + (useMultiResolution_ ? "ON" : "OFF"));
    }
    
   #if defined(DEBUG) || defined(_DEBUG)
    if (blockCount % 100 == 0)
    {
        DBG("[AudioEngine] Processing audio block #" << blockCount 
            << " samples=" << buffer.getNumSamples()
            << " multiRes=" << (useMultiResolution_ ? "ON" : "OFF"));
    }
   #endif
    
    // Spectrum analysis (multi-resolution or standard)
    SpectrumData spectrumData;
    
    if (useMultiResolution_ && multiResAnalyzer_)
    {
        // Multi-resolution mode
        multiResAnalyzer_->process(buffer, *multiResData_);
        multiResAnalyzer_->getFusedSpectrum(*multiResData_, spectrumData);
        
       #if defined(DEBUG) || defined(_DEBUG)
        if (blockCount % 100 == 0)
        {
            DBG("[AudioEngine] Multi-res processing time: " << multiResData_->processingTimeMs << " ms");
        }
       #endif
    }
    else
    {
        // Standard mode
        spectrumAnalyzer_->process(buffer, spectrumData);
    }
    
   #if defined(DEBUG) || defined(_DEBUG)
    if (blockCount % 100 == 0)
    {
        DBG("[AudioEngine] Spectrum data: " << spectrumData.magnitudes.size() << " bins");
    }
   #endif
    
    if (spectrumCallback_)
    {
        spectrumCallback_(spectrumData);
    }
    
    // Pitch detection
    PitchVector pitches;
    if (useMultiResolution_ && multiResData_ && multiResData_->isComplete)
    {
        // Detect using multi-resolution data
        polyphonicDetector_->detectMultiResolution(*multiResData_, pitches);
    }
    else
    {
        // Standard detection
        polyphonicDetector_->detect(spectrumData, pitches);
    }
    
    // Log every 30 frames (~1 second at 30fps) or when pitches detected
    if ((blockCount % 30 == 0) || !pitches.empty())
    {
        if (pitches.empty())
        {
            SPM_LOG_INFO("[AudioEngine] No pitches detected (frame " + juce::String(blockCount) + ")");
           #if defined(DEBUG) || defined(_DEBUG)
            DBG("[AudioEngine] No pitches detected (frame " << blockCount << ")");
           #endif
        }
        else
        {
            // Find strongest pitch
            auto strongest = *std::max_element(pitches.begin(), pitches.end(),
                [](const PitchCandidate& a, const PitchCandidate& b) {
                    return a.confidence < b.confidence;
                });
            
            SPM_LOG_INFO("[AudioEngine] Frame " + juce::String(blockCount) + ": " 
                         + juce::String(pitches.size()) + " pitches, "
                         + "strongest=" + juce::String(strongest.frequency, 1) + "Hz "
                         + "(midi=" + juce::String(strongest.midiNote, 1) + ", conf=" + juce::String(strongest.confidence, 2) + ")");
           #if defined(DEBUG) || defined(_DEBUG)
            DBG("[AudioEngine] Frame " << blockCount << ": " << pitches.size() << " pitches, "
                << "strongest=" << strongest.frequency << "Hz "
                << "(midi=" << strongest.midiNote << ", conf=" << strongest.confidence << ")");
           #endif
        }
    }
    
    if (pitchCallback_ && !pitches.empty())
    {
        pitchCallback_(pitches);
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

void AudioEngine::setMultiResolutionEnabled(bool enabled)
{
    useMultiResolution_ = enabled;
    
    if (enabled && multiResAnalyzer_ && polyphonicDetector_)
    {
        // Re-prepare when enabling multi-resolution mode
        multiResAnalyzer_->prepare(sampleRate_);
        polyphonicDetector_->setMultiResolutionEnabled(true);
        SPM_LOG_INFO("[AudioEngine] Multi-resolution analysis enabled");
    }
    else if (polyphonicDetector_)
    {
        polyphonicDetector_->setMultiResolutionEnabled(false);
        SPM_LOG_INFO("[AudioEngine] Multi-resolution analysis disabled");
    }
}

void AudioEngine::setDetectionRange(float minFreq, float maxFreq)
{
    if (polyphonicDetector_)
    {
        polyphonicDetector_->prepare(sampleRate_, minFreq, maxFreq);
    }
}

} // namespace spm


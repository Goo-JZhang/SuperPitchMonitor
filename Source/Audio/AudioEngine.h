#pragma once

#include <juce_core/juce_core.h>
#include <juce_data_structures/juce_data_structures.h>
#include <juce_events/juce_events.h>
#include <juce_graphics/juce_graphics.h>
#include <juce_gui_basics/juce_gui_basics.h>
#include <juce_gui_extra/juce_gui_extra.h>
#include <juce_audio_basics/juce_audio_basics.h>
#include <juce_audio_devices/juce_audio_devices.h>
#include <juce_audio_formats/juce_audio_formats.h>
#include <juce_audio_processors/juce_audio_processors.h>
#include <juce_audio_utils/juce_audio_utils.h>
#include <juce_dsp/juce_dsp.h>
#include "../Utils/Config.h"
#include "../Algorithms/FFTUtils.h"
#include "AudioInputSource.h"
#include "SpectrumData.h"
#include "../MLAlgorithm/MLPitchDetector.h"

// Forward declarations
namespace spm { class AudioSimulator; }

namespace spm {

// Forward declarations
class SpectrumAnalyzer;
class PolyphonicDetector;
class NonlinearFourierAnalyzer;

/**
 * Audio Engine
 * Manages audio devices, audio processing and analysis
 * Supports both real audio input and simulated input modes
 */
class AudioEngine : public juce::AudioAppComponent,
                    private juce::Thread
{
public:
    // Operating modes
    enum class Mode {
        RealDevice,     // Real audio device
        Simulated       // Simulated audio input (for debugging)
    };

    using SpectrumCallback = std::function<void(const SpectrumData&)>;
    using PitchCallback = std::function<void(const PitchVector&)>;
    using InputLevelCallback = std::function<void(float)>;

    AudioEngine();
    ~AudioEngine() override;

    // Initialize audio device
    juce::String initialize();
    
    // Start/Stop detection
    void start();
    void stop();
    bool isRunning() const { return isRunning_; }
    
    // Operating mode
    void setMode(Mode mode);
    Mode getMode() const { return currentMode_; }
    
    // Set audio source in simulation mode (legacy)
    void setSimulator(AudioSimulator* simulator);
    
    // Set audio input source (new unified interface)
    void setInputSource(std::shared_ptr<AudioInputSource> source);
    
    // Callback settings
    void setSpectrumCallback(SpectrumCallback callback) { spectrumCallback_ = callback; }
    void setPitchCallback(PitchCallback callback) { pitchCallback_ = callback; }
    void setInputLevelCallback(InputLevelCallback callback) { inputLevelCallback_ = callback; }
    
    // Configuration settings
    void setQualityLevel(Config::Performance::QualityLevel level);
    void setDetectionRange(float minFreq, float maxFreq);
    
    // ML Analysis control
    void setMLAnalysisEnabled(bool enabled);
    bool isMLAnalysisEnabled() const { return useMLAnalysis_; }
    
    // ML GPU/CPU mode control
    void setMLGPUEnabled(bool enabled);
    bool isMLGPUEnabled() const { return useMLGPU_; }
    
    // Non-ML Analysis method control
    void setAnalysisMethod(Config::TraditionalAnalysisMethod method);
    Config::TraditionalAnalysisMethod getAnalysisMethod() const { return analysisMethod_; }
    
    // Buffer size control
    void setBufferSize(int newBufferSize);
    
    // ML Model path control
    void setMLModelPath(const juce::String& modelPath);
    juce::String getMLModelPath() const { return mlModelPath_; }
    
    // Get current configuration
    double getSampleRate() const { return sampleRate_; }
    int getBufferSize() const { return bufferSize_; }
    Mode getCurrentMode() const { return currentMode_; }
    
    // Get current input source name (for display)
    juce::String getInputSourceName() const { return inputSource_ ? inputSource_->getName() : "None"; }

    // AudioAppComponent interface (only used in RealDevice mode)
    void prepareToPlay(int samplesPerBlockExpected, double newSampleRate) override;
    void releaseResources() override;
    void getNextAudioBlock(const juce::AudioSourceChannelInfo& bufferToFill) override;
    
    // Process audio block (used by simulator)
    void processAudioBlock(const juce::AudioBuffer<float>& buffer);

private:
    // Operating mode
    Mode currentMode_ = Mode::RealDevice;
    AudioSimulator* simulator_ = nullptr;
    
    // New unified input source
    std::shared_ptr<AudioInputSource> inputSource_;
    
    // Audio parameters
    double sampleRate_ = Config::Audio::DefaultSampleRate;
    int bufferSize_ = Config::Audio::DefaultBufferSize;
    
    // State
    std::atomic<bool> isRunning_{false};
    std::atomic<bool> shouldExit_{false};
    
    // Callbacks
    SpectrumCallback spectrumCallback_;
    PitchCallback pitchCallback_;
    InputLevelCallback inputLevelCallback_;
    
    // Processing components
    std::unique_ptr<SpectrumAnalyzer> spectrumAnalyzer_;
    std::unique_ptr<PolyphonicDetector> polyphonicDetector_;
    std::unique_ptr<MLPitchDetector> mlDetector_;
    std::unique_ptr<NonlinearFourierAnalyzer> nonlinearFourierAnalyzer_;
    
    // ML Analysis mode
    bool useMLAnalysis_ = true;  // Default ON
    bool useMLGPU_ = true;       // Default GPU ON
    juce::String mlModelPath_;   // Current model path
    
    // Non-ML Analysis method
    Config::TraditionalAnalysisMethod analysisMethod_ = Config::TraditionalAnalysisMethod::NonlinearFourier;
    
    // Circular buffer (used in RealDevice mode)
    static constexpr int FIFOSize = 8;
    juce::AbstractFifo fifo_{FIFOSize};
    juce::HeapBlock<juce::AudioBuffer<float>> audioBuffers_;
    std::atomic<int> writeIndex_{0};
    std::atomic<int> readIndex_{0};
    
    // ML sliding window buffer - stores recent audio for 4096-sample inference
    static constexpr int MLWindowSize = 4096 * 2;  // 8192 samples (2x for safety)
    std::vector<float> mlRingBuffer_;   // Large ring buffer for history
    std::vector<float> mlInputBuffer_;  // 4096 samples extracted for inference
    std::atomic<int> mlWritePos_{0};
    static constexpr int MLInputSize = 4096;
    
    // Input level smoother
    juce::LinearSmoothedValue<float> inputLevel_{0.0f};
    
    // Processing thread
    void run() override;  // Thread implementation
    
    // Simulation mode processing
    void startSimulation();
    void stopSimulation();
    
    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(AudioEngine)
};

} // namespace spm

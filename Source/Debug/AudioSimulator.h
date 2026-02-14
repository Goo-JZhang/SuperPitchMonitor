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

namespace spm {

/**
 * Audio Simulator
 * Reads data from audio files to simulate real-time audio input stream
 * Used for debugging UI and algorithms when no real audio device is available
 */
class AudioSimulator : public juce::Timer
{
public:
    AudioSimulator();
    ~AudioSimulator() override;

    // Load audio file
    bool loadAudioFile(const juce::File& file);
    
    // Use built-in test signals
    enum class TestSignal {
        SineWave,       // Sine wave (440Hz)
        Chord,          // Chord (C Major)
        Sweep,          // Frequency sweep
        WhiteNoise,     // White noise
        PinkNoise       // Pink noise
    };
    bool generateTestSignal(TestSignal type, double durationSeconds = 5.0);
    
    // Playback control
    void start();
    void stop();
    void pause();
    bool isPlaying() const { return isPlaying_; }
    
    // Playback position control
    void setPlayPosition(double positionSeconds);
    double getPlayPosition() const;
    double getTotalDuration() const;
    
    // Loop playback
    void setLooping(bool shouldLoop) { isLooping_ = shouldLoop; }
    bool isLooping() const { return isLooping_; }
    
    // Playback speed (1.0 = normal)
    void setPlaybackSpeed(float speed) { playbackSpeed_ = speed; }
    float getPlaybackSpeed() const { return playbackSpeed_; }
    
    // Set callback function (simulates audio device callback)
    using AudioCallback = std::function<void(const juce::AudioBuffer<float>&)>;
    void setAudioCallback(AudioCallback callback) { audioCallback_ = callback; }
    
    // Set input level callback
    using LevelCallback = std::function<void(float)>;
    void setLevelCallback(LevelCallback callback) { levelCallback_ = callback; }
    
    // Get audio info
    double getSampleRate() const { return sampleRate_; }
    int getNumChannels() const { return audioBuffer_.getNumChannels(); }
    
    // Timer callback
    void timerCallback() override;

private:
    juce::AudioBuffer<float> audioBuffer_;
    double sampleRate_ = 44100.0;
    std::atomic<bool> isPlaying_{false};
    std::atomic<bool> isLooping_{true};
    std::atomic<double> currentPosition_{0.0};  // Seconds
    float playbackSpeed_ = 1.0f;
    
    // Callbacks
    AudioCallback audioCallback_;
    LevelCallback levelCallback_;
    
    // Internal buffer
    juce::AudioBuffer<float> outputBuffer_;
    static constexpr int bufferSize_ = 512;  // Simulated audio buffer size
    
    // Generate test signals
    void generateSineWave(double duration, float frequency);
    void generateChord(double duration);
    void generateSweep(double duration);
    void generateWhiteNoise(double duration);
    void generatePinkNoise(double duration);
    
    // Process audio block
    void processAudioBlock(int numSamples);
    
    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(AudioSimulator)
};

} // namespace spm


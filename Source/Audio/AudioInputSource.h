#pragma once

#include <juce_core/juce_core.h>
#include <juce_events/juce_events.h>
#include <juce_audio_basics/juce_audio_basics.h>
#include <functional>

namespace spm {

/**
 * Audio Input Source Interface
 * Abstract base class for all audio input sources
 */
class AudioInputSource : public juce::ChangeBroadcaster
{
public:
    virtual ~AudioInputSource() = default;

    // Source type enumeration
    enum class Type
    {
        Device,         // Physical audio device (microphone/line in)
        SystemAudio,    // System audio capture (loopback)
        FilePlayback    // Audio file playback
    };

    // Get source type
    virtual Type getType() const = 0;

    // Get source name for display
    virtual juce::String getName() const = 0;

    // Initialize the source
    virtual bool prepare(double sampleRate, int bufferSize) = 0;

    // Start capturing/playing
    virtual void start() = 0;

    // Stop capturing/playing
    virtual void stop() = 0;

    // Check if active
    virtual bool isActive() const = 0;

    // Get current sample rate
    virtual double getSampleRate() const = 0;

    // Get buffer size
    virtual int getBufferSize() const = 0;

    // Set audio callback - called when new audio data is available
    using AudioCallback = std::function<void(const juce::AudioBuffer<float>&)>;
    void setAudioCallback(AudioCallback callback) { audioCallback_ = callback; }

    // Set level callback - called with RMS level
    using LevelCallback = std::function<void(float)>;
    void setLevelCallback(LevelCallback callback) { levelCallback_ = callback; }

    // Get available input devices (for Device type)
    static juce::StringArray getAvailableDevices();

    // Get available audio files (for FilePlayback type)
    static juce::StringArray getAvailableTestFiles();

protected:
    AudioCallback audioCallback_;
    LevelCallback levelCallback_;

    // Helper to calculate RMS level
    static float calculateRMS(const juce::AudioBuffer<float>& buffer);
};

} // namespace spm

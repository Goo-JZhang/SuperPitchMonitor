#pragma once

#include "AudioInputSource.h"
#include <juce_events/juce_events.h>
#include <juce_audio_formats/juce_audio_formats.h>

namespace spm {

/**
 * File Audio Input
 * Plays audio files from Resources/TestAudio directory
 * Can be used for testing and debugging
 */
class FileAudioInput : public AudioInputSource,
                       public juce::Timer
{
public:
    FileAudioInput();
    ~FileAudioInput() override;

    // AudioInputSource interface
    Type getType() const override { return Type::FilePlayback; }
    juce::String getName() const override { return "File: " + currentFileName_; }
    bool prepare(double sampleRate, int bufferSize) override;
    void start() override;
    void stop() override;
    bool isActive() const override { return isPlaying_; }
    double getSampleRate() const override { return sampleRate_; }
    int getBufferSize() const override { return bufferSize_; }

    // File management
    bool loadFile(const juce::File& file);
    bool loadTestFile(const juce::String& fileName);
    
    // Get available test files
    static juce::File getTestAudioDirectory();
    static juce::StringArray getAvailableTestFiles();

    // Playback control
    void pause();
    void setPlayPosition(double positionSeconds);
    double getPlayPosition() const;
    double getTotalDuration() const;
    void setLooping(bool shouldLoop) { isLooping_ = shouldLoop; }
    bool isLooping() const { return isLooping_; }
    void setPlaybackSpeed(float speed) { playbackSpeed_ = speed; }
    float getPlaybackSpeed() const { return playbackSpeed_; }

    juce::String getCurrentFileName() const { return currentFileName_; }

    // Timer callback for audio processing
    void timerCallback() override;

private:
    void processAudioBlock(int numSamples);

    juce::AudioBuffer<float> audioBuffer_;
    juce::AudioBuffer<float> outputBuffer_;
    double sampleRate_ = 44100.0;
    int bufferSize_ = 512;

    std::atomic<bool> isPlaying_{false};
    std::atomic<bool> isLooping_{true};
    std::atomic<double> currentPosition_{0.0};  // Seconds
    float playbackSpeed_ = 1.0f;
    juce::String currentFileName_;
    int numChannels_ = 1;

    static constexpr int timerHz = 60;  // ~16.6ms interval
};

} // namespace spm

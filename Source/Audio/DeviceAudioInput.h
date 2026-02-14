#pragma once

#include "AudioInputSource.h"
#include <juce_audio_devices/juce_audio_devices.h>

namespace spm {

/**
 * Device Audio Input
 * Captures audio from physical input devices (microphone, line in, etc.)
 */
class DeviceAudioInput : public AudioInputSource,
                         public juce::AudioIODeviceCallback
{
public:
    DeviceAudioInput();
    ~DeviceAudioInput() override;

    // AudioInputSource interface
    Type getType() const override { return Type::Device; }
    juce::String getName() const override { return "Audio Device"; }
    bool prepare(double sampleRate, int bufferSize) override;
    void start() override;
    void stop() override;
    bool isActive() const override;
    double getSampleRate() const override;
    int getBufferSize() const override;

    // Device selection
    void setDevice(const juce::String& deviceName);
    juce::String getCurrentDevice() const;

    // AudioIODeviceCallback interface
    void audioDeviceAboutToStart(juce::AudioIODevice* device) override;
    void audioDeviceIOCallbackWithContext(const float* const* inputChannelData,
                                          int numInputChannels,
                                          float* const* outputChannelData,
                                          int numOutputChannels,
                                          int numSamples,
                                          const juce::AudioIODeviceCallbackContext& context) override;
    void audioDeviceStopped() override;

private:
    juce::AudioDeviceManager deviceManager_;
    juce::String selectedDevice_;
    double sampleRate_ = 44100.0;
    int bufferSize_ = 512;
    bool isActive_ = false;

    juce::AudioBuffer<float> inputBuffer_;
    juce::CriticalSection callbackLock_;
};

} // namespace spm

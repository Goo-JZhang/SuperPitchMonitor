#pragma once

#include "AudioInputSource.h"
#include <juce_audio_devices/juce_audio_devices.h>

namespace spm {

// Forward declaration for platform-specific implementation
#if JUCE_WINDOWS
class WASAPILoopbackCapture;
#endif

/**
 * System Audio Input
 * Captures system audio output (loopback) using native APIs
 * 
 * Platform Support:
 * - Windows: WASAPI loopback (supported, no "Stereo Mix" required)
 * - Android: Not supported (requires system-level permissions)
 * - macOS/Linux: Not implemented
 */
class SystemAudioInput : public AudioInputSource
{
public:
    SystemAudioInput();
    ~SystemAudioInput() override;

    // AudioInputSource interface
    Type getType() const override { return Type::SystemAudio; }
    juce::String getName() const override { return "System Audio (Loopback)"; }
    bool prepare(double sampleRate, int bufferSize) override;
    void start() override;
    void stop() override;
    bool isActive() const override;
    double getSampleRate() const override;
    int getBufferSize() const override;

    // Check if supported on current platform
    static bool isSupported();

private:
    double sampleRate_ = 44100.0;
    int bufferSize_ = 512;
    bool isActive_ = false;
    bool isPrepared_ = false;

    juce::AudioBuffer<float> outputBuffer_;

#if JUCE_WINDOWS
    std::unique_ptr<WASAPILoopbackCapture> loopbackCapture_;
#endif
};

} // namespace spm

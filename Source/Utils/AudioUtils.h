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
 * Audio Utility Functions
 */
namespace AudioUtils {

// Calculate RMS level
inline float calculateRMS(const float* data, int numSamples)
{
    float sum = 0.0f;
    for (int i = 0; i < numSamples; ++i)
    {
        sum += data[i] * data[i];
    }
    return std::sqrt(sum / numSamples);
}

// Convert to dB
inline float amplitudeToDb(float amplitude)
{
    return 20.0f * std::log10(amplitude + 1e-10f);
}

// Convert from dB
inline float dbToAmplitude(float db)
{
    return std::pow(10.0f, db / 20.0f);
}

} // namespace AudioUtils

} // namespace spm


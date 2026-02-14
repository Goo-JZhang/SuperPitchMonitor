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
 * FFT Utility Functions
 */
namespace FFTUtils {

// Calculate FFT order for given size
inline int sizeToOrder(int size)
{
    int order = 0;
    while ((1 << order) < size)
        ++order;
    return order;
}

// Create window functions
template<typename FloatType>
void createHannWindow(FloatType* window, int size)
{
    for (int i = 0; i < size; ++i)
    {
        window[i] = static_cast<FloatType>(0.5 - 0.5 * std::cos(2.0 * juce::MathConstants<double>::pi * i / (size - 1)));
    }
}

template<typename FloatType>
void createHammingWindow(FloatType* window, int size)
{
    for (int i = 0; i < size; ++i)
    {
        window[i] = static_cast<FloatType>(0.54 - 0.46 * std::cos(2.0 * juce::MathConstants<double>::pi * i / (size - 1)));
    }
}

// Frequency to MIDI conversion
inline float freqToMidi(float freq)
{
    return 69.0f + 12.0f * std::log2(freq / 440.0f);
}

// MIDI to frequency conversion
inline float midiToFreq(float midiNote)
{
    return 440.0f * std::pow(2.0f, (midiNote - 69.0f) / 12.0f);
}

} // namespace FFTUtils

} // namespace spm


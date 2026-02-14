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
 * MIDI Utility Functions
 */
namespace MidiUtils {

// Get note name
inline juce::String getNoteName(int midiNote, bool includeOctave = true)
{
    static const char* noteNames[] = {"C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"};
    
    int note = midiNote % 12;
    int octave = midiNote / 12 - 1;
    
    if (includeOctave)
        return juce::String(noteNames[note]) + juce::String(octave);
    else
        return juce::String(noteNames[note]);
}

// Calculate cents deviation
inline float getCentsDeviation(float detectedFreq, float referenceFreq)
{
    return 1200.0f * std::log2(detectedFreq / referenceFreq);
}

} // namespace MidiUtils

} // namespace spm


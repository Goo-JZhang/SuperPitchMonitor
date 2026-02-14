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
#include "AudioEngine.h"

namespace spm {

/**
 * Pitch Tracker
 * Temporal smoothing and tracking
 */
class PitchTracker
{
public:
    struct TrackedPitch {
        float midiNote;
        float velocity;
        int age;
        int missedFrames;
    };

    void update(const PitchVector& currentCandidates);
    const std::vector<TrackedPitch>& getActivePitches() const;

private:
    std::vector<TrackedPitch> activePitches_;
};

} // namespace spm


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
#include "../Audio/AudioEngine.h"
#include "PitchCard.h"

namespace spm {

/**
 * Pitch Display Panel
 * Displays detected multiple pitches
 */
class PitchDisplay : public juce::Component
{
public:
    PitchDisplay();
    ~PitchDisplay() override;

    void paint(juce::Graphics& g) override;
    void resized() override;
    
    // Update pitch data
    void updatePitches(const PitchVector& pitches);
    void clear();  // Clear all pitch cards (e.g., when stopped)

private:
    std::vector<std::unique_ptr<PitchCard>> pitchCards_;
    juce::CriticalSection dataLock_;
    
    // Title
    juce::Label titleLabel_;
    
    void refreshCards(const PitchVector& pitches);
    
    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(PitchDisplay)
};

} // namespace spm

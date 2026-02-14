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
 * Tuner Display Component
 * Needle-style tuner interface
 */
class TunerDisplay : public juce::Component
{
public:
    TunerDisplay();
    ~TunerDisplay() override;

    void paint(juce::Graphics& g) override;
    void resized() override;
    
    // Set target pitch
    void setTargetPitch(float midiNote, float centsDeviation);

private:
    float targetMidi_ = 69.0f;      // A4
    float centsDeviation_ = 0.0f;
    
    // Drawing functions
    void drawScale(juce::Graphics& g);
    void drawNeedle(juce::Graphics& g);
    void drawNoteDisplay(juce::Graphics& g);
    
    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(TunerDisplay)
};

} // namespace spm


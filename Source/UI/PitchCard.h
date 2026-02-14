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

namespace spm {

class PitchCard : public juce::Component
{
public:
    PitchCard();
    
    void setPitchData(const PitchCandidate& data);
    void clearData();  // Clear pitch data when hidden
    void paint(juce::Graphics& g) override;
    
private:
    PitchCandidate data_;
    juce::CriticalSection lock_;
    
    // Smoothing state for temporal stability
    struct SmoothingState {
        bool initialized = false;
        float smoothedFreq = 0.0f;
        float smoothedCents = 0.0f;
        float smoothedMidi = 0.0f;
        float smoothedConfidence = 0.0f;
        juce::uint32 lastUpdateTime = 0;
        
        // EMA coefficient (higher = more responsive, lower = smoother)
        // 0.3 means 30% new value, 70% old value - good for display
        static constexpr float alpha = 0.3f;
        
        // Reset threshold - if pitch jumps more than this, reset smoothing
        static constexpr float resetThresholdSemitones = 0.5f;  // Half semitone
        
        void update(const PitchCandidate& newData);
        void reset();
    };
    
    SmoothingState smoothing_;
};

} // namespace spm

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
#include "AudioSimulator.h"

namespace spm {

class AudioEngine;

/**
 * Debug Control Panel
 * Used to control audio playback in simulator mode
 */
class DebugControlPanel : public juce::Component,
                          public juce::Button::Listener,
                          public juce::ComboBox::Listener,
                          public juce::Slider::Listener,
                          public juce::FilenameComponentListener
{
public:
    DebugControlPanel();
    ~DebugControlPanel() override;

    void paint(juce::Graphics& g) override;
    void resized() override;
    
    // Set associated AudioEngine
    void setAudioEngine(AudioEngine* engine) { audioEngine_ = engine; }
    
    // Control audio simulator
    AudioSimulator& getSimulator() { return simulator_; }
    
    // Callbacks
    void buttonClicked(juce::Button* button) override;
    void comboBoxChanged(juce::ComboBox* comboBox) override;
    void sliderValueChanged(juce::Slider* slider) override;
    void filenameComponentChanged(juce::FilenameComponent* fileComponentThatHasChanged) override;

private:
    AudioSimulator simulator_;
    AudioEngine* audioEngine_ = nullptr;
    
    // UI Controls
    juce::Label titleLabel_;
    juce::TextButton closeButton_{"Close"};
    
    // Mode selection (File only - no TestSignal)
    juce::Label modeLabel_;
    juce::ComboBox modeCombo_;
    
    // File selection
    std::unique_ptr<juce::FilenameComponent> fileChooser_;
    
    // Playback controls
    juce::TextButton playButton_{"Play"};
    juce::TextButton pauseButton_{"Pause"};
    juce::TextButton stopButton_{"Stop"};
    juce::ToggleButton loopButton_{"Loop"};
    
    // Progress bar
    juce::Slider positionSlider_;
    juce::Label timeLabel_;
    
    // Playback speed
    juce::Label speedLabel_;
    juce::Slider speedSlider_;
    
    // Status
    juce::Label statusLabel_;
    
    // Timer for UI updates
    class TimerCallback : public juce::Timer
    {
    public:
        TimerCallback(DebugControlPanel& panel) : panel_(panel) {}
        void timerCallback() override { panel_.updateUI(); }
    private:
        DebugControlPanel& panel_;
    };
    std::unique_ptr<TimerCallback> uiTimer_;
    
    void setupUI();
    void updateUI();
    void updateTimeDisplay();
    
    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(DebugControlPanel)
};

} // namespace spm

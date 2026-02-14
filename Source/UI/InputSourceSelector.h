#pragma once

#include "../Audio/AudioInputSource.h"
#include <juce_gui_basics/juce_gui_basics.h>

namespace spm {

/**
 * Input Source Selector Component
 * Allows user to select and configure audio input sources
 */
class InputSourceSelector : public juce::Component,
                           public juce::ComboBox::Listener,
                           public juce::Button::Listener
{
public:
    InputSourceSelector();
    ~InputSourceSelector() override;

    void paint(juce::Graphics& g) override;
    void resized() override;

    // Callback when input source changes
    using SourceChangedCallback = std::function<void(std::shared_ptr<AudioInputSource>)>;
    void onSourceChanged(SourceChangedCallback callback) { sourceChangedCallback_ = callback; }

    // Get current source
    std::shared_ptr<AudioInputSource> getCurrentSource() const { return currentSource_; }

    // ComboBox listener
    void comboBoxChanged(juce::ComboBox* comboBox) override;

    // Button listener
    void buttonClicked(juce::Button* button) override;

    // Refresh available sources
    void refreshSources();

private:
    void createSource(AudioInputSource::Type type);
    void updateFileSelector();

    // UI Components
    juce::Label titleLabel_;
    juce::Label sourceTypeLabel_;
    juce::ComboBox sourceTypeCombo_;
    
    juce::Label deviceLabel_;
    juce::ComboBox deviceCombo_;
    
    juce::Label fileLabel_;
    juce::ComboBox fileCombo_;
    juce::TextButton refreshButton_{"Refresh"};
    
    juce::Label statusLabel_;

    // Current source (shared ownership with AudioEngine)
    std::shared_ptr<AudioInputSource> currentSource_;
    SourceChangedCallback sourceChangedCallback_;

    // Available options
    juce::StringArray availableDevices_;
    juce::StringArray availableFiles_;
};

} // namespace spm

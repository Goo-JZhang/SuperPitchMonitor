#pragma once

#include <juce_gui_basics/juce_gui_basics.h>
#include <functional>
#include "../Audio/AudioInputSource.h"

namespace spm {

/**
 * Custom LookAndFeel for Settings Panel
 * Limits ComboBox popup menu height for better mobile experience
 */
class SettingsLookAndFeel : public juce::LookAndFeel_V4
{
public:
    SettingsLookAndFeel();
    
    // Override to limit popup menu height
    void getIdealPopupMenuItemSize(const juce::String& text, bool isSeparator,
                                   int standardMenuItemHeight,
                                   int& idealWidth, int& idealHeight) override;
    
    // Override to disable shadow (fixes VirtualDesktopWatcher crash)
    void drawPopupMenuBackground(juce::Graphics& g, int width, int height) override;
    
    // Force non-native menus to avoid Windows-specific crash
    bool isPopupMenuNativeActive();
    
    void setMaxPopupMenuHeight(int maxHeight) { maxPopupMenuHeight_ = maxHeight; }
    int getMaxPopupMenuHeight() const { return maxPopupMenuHeight_; }

private:
    int maxPopupMenuHeight_ = 400;  // Maximum popup height in pixels
};

// Inner content component for settings
class SettingsContent : public juce::Component,
                        public juce::Button::Listener,
                        public juce::Slider::Listener,
                        public juce::ComboBox::Listener
{
public:
    SettingsContent();
    ~SettingsContent() override;

    void paint(juce::Graphics& g) override;
    void resized() override;
    
    void buttonClicked(juce::Button* button) override;
    void sliderValueChanged(juce::Slider* slider) override;
    void comboBoxChanged(juce::ComboBox* comboBox) override;
    
    // Callbacks
    using A4FrequencyCallback = std::function<void(float)>;
    void onA4FrequencyChanged(A4FrequencyCallback callback) { a4Callback_ = callback; }
    
    using ScaleModeCallback = std::function<void(bool)>;
    void onScaleModeChanged(ScaleModeCallback callback) { scaleCallback_ = callback; }
    
    using TimeWindowCallback = std::function<void(float)>;
    void onTimeWindowChanged(TimeWindowCallback callback) { timeWindowCallback_ = callback; }
    
    using SourceChangedCallback = std::function<void(std::shared_ptr<AudioInputSource>)>;
    void onSourceChanged(SourceChangedCallback callback) { sourceChangedCallback_ = callback; }
    
    using CloseCallback = std::function<void()>;
    void onClose(CloseCallback callback) { closeCallback_ = callback; }
    
    // FPS callback
    using FPSCallback = std::function<void(int)>;  // -1 for unlimited, or target FPS
    void onFPSChanged(FPSCallback callback) { fpsCallback_ = callback; }
    
    // Buffer Size callback
    using BufferSizeCallback = std::function<void(int)>;  // 128, 256, 512, 1024, 2048, 4096
    void onBufferSizeChanged(BufferSizeCallback callback) { bufferSizeCallback_ = callback; }
    int getBufferSize() const;  // Returns selected buffer size
    
    // ML Analysis callback
    using MLAnalyzeCallback = std::function<void(bool)>;
    void onMLAnalyzeChanged(MLAnalyzeCallback callback) { mlAnalyzeCallback_ = callback; }
    
    // ML CPU/GPU mode callback
    using MLModeCallback = std::function<void(bool)>;  // true = GPU, false = CPU
    void onMLModeChanged(MLModeCallback callback) { mlModeCallback_ = callback; }
    
    // Get current values
    float getA4Frequency() const { return (float)a4Slider_.getValue(); }
    bool getUseLogScale() const { return logScaleButton_.getToggleState(); }
    float getDbRangeMin() const { return (float)dbMinSlider_.getValue(); }
    float getDbRangeMax() const { return (float)dbMaxSlider_.getValue(); }
    float getTimeWindow() const { return (float)timeWindowSlider_.getValue(); }
    int getTargetFPS() const;  // -1 for unlimited
    
    // Get current input source
    std::shared_ptr<AudioInputSource> getCurrentSource() const { return currentSource_; }
    
    // Set ML Analysis enabled (default ON)
    void setMLAnalysisEnabled(bool enabled);
    
    // Set ML GPU mode (default true = GPU)
    void setMLGPUEnabled(bool enabled);
    
    // ML Model selection callback
    using MLModelCallback = std::function<void(const juce::String&)>;
    void onMLModelChanged(MLModelCallback callback) { mlModelCallback_ = callback; }
    
    // Refresh available models in the dropdown
    void refreshModelList();
    
    // Set current model in UI
    void setCurrentMLModel(const juce::String& modelPath);

private:
    void setupComponents();
    void loadSettings();
    void saveSettings();
    void createSource(AudioInputSource::Type type);
    void refreshSources();
    juce::String sanitizeDeviceName(const juce::String& name);

    juce::TextButton closeButton_{"Close"};
    juce::Label titleLabel_;
    
    // === Input Source Section ===
    juce::Label inputSourceLabel_;
    juce::Label sourceTypeLabel_;
    juce::ComboBox sourceTypeCombo_;
    juce::Label deviceLabel_;
    juce::ComboBox deviceCombo_;
    juce::Label fileLabel_;
    juce::ComboBox fileCombo_;
    juce::TextButton refreshButton_{"Refresh"};
    
    // Tuning settings
    juce::Label tuningLabel_;
    juce::Slider a4Slider_;
    juce::Label a4Label_;
    juce::Label a4ValueLabel_;
    
    // Display settings
    juce::Label displayLabel_;
    juce::ToggleButton logScaleButton_;
    juce::ToggleButton showNotesButton_;
    
    // Waterfall display settings
    juce::Label waterfallLabel_;
    juce::Slider timeWindowSlider_;
    juce::Label timeWindowLabel_;
    juce::Label timeWindowValueLabel_;
    
    // Spectrum settings
    juce::Label spectrumLabel_;
    juce::Slider dbMinSlider_;
    juce::Label dbMinLabel_;
    juce::Slider dbMaxSlider_;
    juce::Label dbMaxLabel_;
    
    // Pitch detection settings
    juce::Label pitchLabel_;
    juce::Slider minFreqSlider_;
    juce::Label minFreqLabel_;
    juce::Slider maxFreqSlider_;
    juce::Label maxFreqLabel_;
    
    juce::ToggleButton mlAnalyzeButton_;  // ML Analysis toggle (default ON)
    juce::ToggleButton mlGpuButton_;      // ML GPU/CPU toggle (default ON = GPU)
    
    // ML Model selection
    juce::Label mlModelLabel_;
    juce::ComboBox mlModelCombo_;         // Dropdown for available models
    juce::String currentModelPath_;       // Currently selected model path
    
    // Performance settings
    juce::Label performanceLabel_;
    juce::Label fpsLabel_;
    juce::ComboBox fpsCombo_;
    juce::Label bufferSizeLabel_;
    juce::ComboBox bufferSizeCombo_;
    
    // Current source
    std::shared_ptr<AudioInputSource> currentSource_;
    juce::StringArray availableDevices_;
    juce::StringArray availableFiles_;
    
    // Callbacks
    A4FrequencyCallback a4Callback_;
    ScaleModeCallback scaleCallback_;
    TimeWindowCallback timeWindowCallback_;
    SourceChangedCallback sourceChangedCallback_;
    CloseCallback closeCallback_;
    FPSCallback fpsCallback_;
    BufferSizeCallback bufferSizeCallback_;
    MLAnalyzeCallback mlAnalyzeCallback_;
    MLModeCallback mlModeCallback_;
    MLModelCallback mlModelCallback_;
    
    // Custom look and feel for limited popup menu height
    SettingsLookAndFeel lookAndFeel_;
};

/**
 * Settings Panel with Scrollbar
 * Wraps SettingsContent in a Viewport for scrolling
 */
class SettingsPanel : public juce::Component
{
public:
    SettingsPanel();
    ~SettingsPanel() override;

    void paint(juce::Graphics& g) override;
    void resized() override;
    
    // Pass-through to content
    SettingsContent* getContent() const { return content_.get(); }
    
    // Forward callback to content
    void onClose(SettingsContent::CloseCallback callback) { content_->onClose(callback); }
    
    void setVisible(bool shouldBeVisible) override;

private:
    std::unique_ptr<SettingsContent> content_;
    std::unique_ptr<juce::Viewport> viewport_;
    
    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(SettingsPanel)
};

} // namespace spm

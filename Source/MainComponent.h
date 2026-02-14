#pragma once

#include <juce_core/juce_core.h>
#include <juce_events/juce_events.h>
#include <juce_graphics/juce_graphics.h>
#include <juce_gui_basics/juce_gui_basics.h>
#include <juce_audio_basics/juce_audio_basics.h>
#include <juce_audio_devices/juce_audio_devices.h>
#include "Audio/AudioEngine.h"
#include "UI/PitchWaterfallDisplay.h"
#include "UI/SpectrumDisplay.h"
#include "UI/PitchDisplay.h"
#include "UI/SettingsPanel.h"
#include "Utils/PlatformUtils.h"
#include "Test/TestServer.h"

namespace spm {

/**
 * Main Application Component
 * Layout:
 * - Top: Pitch Waterfall Display (main) + Detected Pitches (right, same height)
 * - Bottom: Real-time Spectrum (full width)
 */
class MainComponent : public juce::Component,
                      public juce::Button::Listener,
                      private juce::Timer
{
public:
    MainComponent();
    ~MainComponent() override;

    void paint(juce::Graphics& g) override;
    void resized() override;
    
    void buttonClicked(juce::Button* button) override;
    
    void timerCallback() override;

private:
    // Audio engine
    std::unique_ptr<AudioEngine> audioEngine_;
    
    // TCP Test server for cross-platform automated testing
    std::unique_ptr<TestServer> testServer_;
    int testServerPort_ = 9999;
    
    // UI Components
    std::unique_ptr<PitchWaterfallDisplay> pitchWaterfall_;  // Main display
    std::unique_ptr<PitchDisplay> pitchDisplay_;             // Right panel (same height as waterfall)
    std::unique_ptr<SpectrumDisplay> spectrumDisplay_;       // Bottom (full width)
    std::unique_ptr<SettingsPanel> settingsPanel_;
    
    // Control buttons
    juce::TextButton settingsButton_{"Settings"};
    juce::TextButton startStopButton_{"Start"};
    
    // Status labels
    juce::Label statusLabel_;
    juce::Label inputLevelLabel_;
    juce::Label fpsLabel_;  // FPS display
    
    // Layout parameters
    static constexpr int statusBarHeight = 40;
    static constexpr int spectrumHeight = 200;   // Bottom spectrum height
    static constexpr int pitchListWidth = 220;   // Right panel width
    
    // FPS monitoring
    struct FPSCounter {
        static constexpr int historySize = 30;
        float frameTimes[historySize] = {};
        int index = 0;
        int count = 0;
        double lastFrameTime = 0;
        
        void recordFrame();
        float getAverageFPS() const;
        void reset();
    };
    FPSCounter fpsCounter_;
    
    // Display refresh rate (Hz) - 0 means unlimited
    int targetRefreshRate_ = 60;
    bool unlimitedFPS_ = false;
    
    // Initialization
    void setupUI();
    void setupAudio();
    void handlePermissionDenied();
    void connectSettingsCallbacks();
    
    // Callback functions
    void onSpectrumData(const SpectrumData& data);
    void onPitchDetected(const PitchVector& pitches);
    void onInputLevel(float level);
    
    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(MainComponent)
};

} // namespace spm

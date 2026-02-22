#pragma once

#include <juce_gui_basics/juce_gui_basics.h>
#include <deque>
#include "../Audio/AudioEngine.h"
#include "../Utils/AutoTracker.h"

namespace spm {

/**
 * Pitch Waterfall Display
 * Similar to Vocal Pitch Monitor - shows pitch over time
 * Y-axis: log frequency (can display as note names)
 * X-axis: time (scrolling from right to left, right=0s, left=-timeWindow)
 */
class PitchWaterfallDisplay : public juce::Component,
                              private juce::Timer
{
public:
    PitchWaterfallDisplay();
    ~PitchWaterfallDisplay() override;

    void paint(juce::Graphics& g) override;
    void resized() override;
    void timerCallback() override;
    void mouseWheelMove(const juce::MouseEvent& event, const juce::MouseWheelDetails& wheel) override;
    void mouseMagnify(const juce::MouseEvent& event, float scaleFactor) override;
    void mouseDown(const juce::MouseEvent& event) override;
    void mouseDrag(const juce::MouseEvent& event) override;
    void mouseUp(const juce::MouseEvent& event) override;

    // Update with new pitch data
    void updatePitch(const PitchCandidate& pitch);
    void updatePitches(const PitchVector& pitches);
    void updateSpectrum(const SpectrumData& data);
    void clear();  // Clear all pitch history (e.g., when stopped)

    // Settings
    void setA4Frequency(float freq) { a4Frequency_ = freq; repaint(); }
    float getA4Frequency() const { return a4Frequency_; }
    
    void setShowNoteNames(bool show) { showNoteNames_ = show; repaint(); }
    bool getShowNoteNames() const { return showNoteNames_; }

    // Time window - X axis range from -timeWindow to 0 seconds
    void setTimeWindow(float seconds);
    float getTimeWindow() const { return timeWindow_; }

    // Frequency range
    void setFrequencyRange(float minFreq, float maxFreq);
    float getMinFreq() const { return minFreq_; }
    float getMaxFreq() const { return maxFreq_; }

    // Scroll offset (for Y axis panning)
    void setScrollOffset(float offset);
    float getScrollOffset() const { return scrollOffset_; }
    
    // Target refresh rate (Hz), -1 for unlimited
    void setTargetRefreshRate(int fps);
    int getTargetRefreshRate() const { return targetFPS_; }
    
    // Auto-tracking control
    void setAutoTrackingEnabled(bool enabled) { autoTrackingEnabled_ = enabled; }
    bool isAutoTrackingEnabled() const { return autoTrackingEnabled_; }
    
    // Reset auto-tracker cooldown (call when Start button clicked)
    void resetAutoTrackerCooldown() { autoTracker_.resetCooldown(); }
    
    // Perform jump to best pitch or reset to A4 (for double-click)
    void performJumpToBestOrReset();

private:
    // History buffer - stores pitch data over time
    struct PitchHistory {
        double timestamp;
        float midiNote;
        float confidence;
        float frequency;
        float amplitude;  // For brightness display
        bool isMLEnergy;  // true = ML energy (0-1 range), false = FFT dB
        bool hasPitch;
    };
    
    std::deque<PitchHistory> pitchHistory_;
    std::vector<float> currentSpectrum_;
    juce::CriticalSection dataLock_;
    float currentSampleRate_ = 44100.0f;
    
    // Settings
    float a4Frequency_ = 440.0f;
    float minFreq_ = 50.0f;
    float maxFreq_ = 2000.0f;
    float timeWindow_ = 5.0f;
    bool showNoteNames_ = true;
    
    // Scroll offset for Y axis (in semitones, 0 = centered on A4)
    float scrollOffset_ = 0.0f;
    
    // Visible range (zoom control) - number of semitones visible on Y axis
    float visibleSemitones_ = 24.0f;  // Default: 2 octaves
    static constexpr float minVisibleSemitones = 12.0f;   // Min: 1 octave
    static constexpr float maxVisibleSemitones = 60.0f;   // Max: 5 octaves
    
    // Target refresh rate
    int targetFPS_ = 60;
    bool unlimitedFPS_ = false;
    
    // Drag state for mouse scrolling
    bool isDragging_ = false;
    float dragStartY_ = 0.0f;
    float dragStartOffset_ = 0.0f;
    
    // Auto-tracking
    AutoTracker autoTracker_;
    bool autoTrackingEnabled_ = true;
    double lastAutoTrackUpdateTime_ = 0.0;
    
    // Current pitch data for auto-tracking (updated by updatePitches)
    std::vector<PitchCandidate> currentPitches_;
    bool hasValidDetection_ = false;
    juce::CriticalSection currentPitchesLock_;
    
    // Derived cache
    float minLogFreq_;
    float maxLogFreq_;
    float logFreqRange_;
    
    // Layout margins for axes
    static constexpr int leftMargin = 50;
    static constexpr int rightMargin = 10;
    static constexpr int topMargin = 10;
    static constexpr int bottomMargin = 40;
    
    // Helpers
    void updateLogFreqRange();
    float freqToLogY(float freq) const;
    float midiToLogY(float midiNote) const;
    float logYToFreq(float logY) const;
    juce::String midiToNoteName(float midiNote) const;
    bool isMainNote(int noteIndex) const;  // CDEFGAB = true
    
    // Drawing
    void drawBackground(juce::Graphics& g);
    void drawAxes(juce::Graphics& g);
    void drawGrid(juce::Graphics& g);
    void drawPitchHistory(juce::Graphics& g);
    void drawCurrentPosition(juce::Graphics& g);
    void drawRadialGradientDot(juce::Graphics& g, float x, float y, float radius, float hue, float brightness);
    
    // Coordinate conversion
    float timeToX(double timestamp, double now) const;
    float freqToY(float freq) const;
    float midiToY(float midiNote) const;
    
    juce::Rectangle<int> getPlotArea() const;
    
    // Get visible MIDI range based on scroll offset
    void getVisibleMidiRange(float& minMidi, float& maxMidi) const;
    
    // Constants
    static constexpr int maxHistorySize = 5000;
    static constexpr float scrollSensitivity = 2.0f;  // semitones per wheel tick
    
    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(PitchWaterfallDisplay)
};

} // namespace spm

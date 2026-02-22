#pragma once

#include <juce_gui_basics/juce_gui_basics.h>
#include "../Audio/AudioEngine.h"

namespace spm {

/**
 * Spectrum Display Component (Bottom panel)
 * Shows real-time spectrum with:
 * - Y-axis: dB magnitude (auto-ranging)
 * - X-axis: frequency (linear or log scale)
 */
class SpectrumDisplay : public juce::Component,
                        private juce::Timer
{
public:
    SpectrumDisplay();
    ~SpectrumDisplay() override;

    void paint(juce::Graphics& g) override;
    void resized() override;
    void timerCallback() override;
    
    // Update spectrum data
    void updateSpectrum(const SpectrumData& data);
    
    // Clear spectrum data (e.g., when stopped)
    void clear();

    // Settings
    void setA4Frequency(float freq) { a4Frequency_ = freq; repaint(); }
    float getA4Frequency() const { return a4Frequency_; }
    
    void setLogFrequencyScale(bool useLog) { useLogScale_ = useLog; repaint(); }
    bool getLogFrequencyScale() const { return useLogScale_; }
    
    // Fixed dB range (if autoRange_ is false)
    void setDbRange(float minDb, float maxDb);
    float getMinDb() const { return currentMinDb_; }
    float getMaxDb() const { return currentMaxDb_; }
    
    // Auto-ranging settings
    void setAutoRange(bool autoRange) { autoRange_ = autoRange; }
    bool getAutoRange() const { return autoRange_; }
    
    void setFrequencyRange(float minFreq, float maxFreq);
    float getMinFreq() const { return minFreq_; }
    float getMaxFreq() const { return maxFreq_; }
    
    // Target refresh rate (Hz), -1 for unlimited
    void setTargetRefreshRate(int fps);
    int getTargetRefreshRate() const { return targetFPS_; }
    
    // Set ML enabled preview state (used before Start to show expected mode)
    void setMLPreviewEnabled(bool enabled) { mlPreviewEnabled_ = enabled; repaint(); }
    bool isMLPreviewEnabled() const { return mlPreviewEnabled_; }

private:
    SpectrumData currentData_;
    juce::CriticalSection dataLock_;
    float currentSampleRate_ = 44100.0f;
    
    // Settings
    float a4Frequency_ = 440.0f;
    bool useLogScale_ = true;
    
    // dB range - fixed values (used as fallback)
    float fixedMinDb_ = -90.0f;
    float fixedMaxDb_ = -10.0f;
    
    // Current dynamic dB range
    float currentMinDb_ = -90.0f;
    float currentMaxDb_ = -10.0f;
    
    // Auto-ranging
    bool autoRange_ = true;
    float smoothingFactor_ = 0.1f;  // Smooth transitions
    static constexpr float speechLevelDb = -30.0f;  // Typical speech level
    static constexpr float minRangeDb = 40.0f;      // Minimum 40dB range
    static constexpr float maxRangeDb = 100.0f;     // Maximum 100dB range
    
    // Frequency range
    float minFreq_ = 20.0f;
    float maxFreq_ = 8000.0f;
    float minLogFreq_;
    float maxLogFreq_;
    
    // Throttling - use frame count instead of atomic flag for smoother updates
    int frameCount_ = 0;
    
    // Target refresh rate
    int targetFPS_ = 60;
    bool unlimitedFPS_ = false;
    static constexpr int skipFrames_ = 1;  // Update every frame (25fps)
    
    // Layout margins
    static constexpr int leftMargin = 50;
    static constexpr int rightMargin = 50;  // Increased for right Y-axis labels
    static constexpr int topMargin = 20;    // Increased for top labels
    static constexpr int bottomMargin = 35;
    
    // Drawing functions
    void drawBackground(juce::Graphics& g);
    void drawAxes(juce::Graphics& g);
    void drawGrid(juce::Graphics& g);
    void drawSpectrum(juce::Graphics& g);
    void drawMLSpectrum(juce::Graphics& g);  // ML mode with dual Y-axes
    void drawModeLabel(juce::Graphics& g);   // Mode indicator label
    void drawNoteMarkers(juce::Graphics& g);
    
    // Coordinate conversion
    juce::Rectangle<int> getPlotArea() const;
    float freqToX(float freq) const;
    float dbToY(float db) const;
    float magnitudeToY(float magnitude) const;
    
    // Auto-ranging
    void updateDynamicRange();
    
    // Helpers
    void updateLogFreqRange();
    juce::String freqToNoteName(float freq) const;
    juce::String freqToString(float freq) const;
    
    // ML preview state (used before Start to show expected mode from Settings)
    bool mlPreviewEnabled_ = false;
    
    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(SpectrumDisplay)
};

} // namespace spm

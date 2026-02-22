#pragma once

#include <juce_core/juce_core.h>
#include <vector>
#include <atomic>
#include "../Audio/SpectrumData.h"

namespace spm {

/**
 * AutoTracker - Pitch Waterfall Auto-Tracking System
 * 
 * Automatically adjusts the Y-axis (frequency) view to keep spectral points 
 * in the vertical center 1/3 region when user is not interacting.
 * 
 * Note: X-axis (time) auto-scrolls independently, only Y-axis needs tracking.
 * 
 * Design doc: Docs/05_UI/WaterfallPitches自动追踪设计.md
 */
class AutoTracker
{
public:
    struct Config
    {
        // Center zone definition (ratio of screen)
        float centerZoneRatio = 0.333f;
        
        // Priority weights for candidate selection
        float confidenceWeight = 1000.0f;
        float energyWeight = 1.0f;
        
        // Animation parameters
        float approachSpeed = 0.15f;        // Interpolation factor (0-1)
        float minApproachFreq = 5.0f;       // Minimum frequency change per second
        float stopThreshold = 0.01f;        // Stop when within this ratio of center
        
        // Interaction parameters
        int cooldownSeconds = 10;           // Seconds before auto-tracking resumes
        int hysteresisMs = 200;             // Hysteresis zone time
        float maxPanSpeed = 0.5f;           // Max pan speed (screen fraction per second)
        
        // Debug options
        bool showCenterZone = false;
        bool showTrackingTarget = false;
    };

    AutoTracker(const Config& config = Config());
    ~AutoTracker() = default;

    /**
     * Update tracking state and compute new view center frequency
     * @param currentPitches Current frame's pitch candidates (NOT history)
     * @param hasValidDetection Whether current frame has any valid detection
     * @param currentViewCenterFreq Current center frequency of the view
     * @param viewHeightSemitones Total frequency range displayed (in semitones)
     * @param deltaTime Time since last update (seconds)
     * @param outNewViewCenterFreq Output: new center frequency
     * @return true if view needs to be updated
     */
    bool update(const std::vector<PitchCandidate>& currentPitches,
                bool hasValidDetection,
                float currentViewCenterFreq,
                float viewHeightSemitones,
                float deltaTime,
                float& outNewViewCenterFreq);

    // User interaction handlers
    void onUserInteraction();
    void onUserInteractionEnd();
    bool isUserInteracting() const { return userInteracting_; }
    
    // Check if auto-tracking is currently active
    bool isTrackingActive() const { return trackingActive_; }
    
    // Get time until auto-tracking resumes (seconds)
    float getCooldownRemaining() const;
    
    // Reset cooldown to expired state (auto-tracking will activate immediately)
    void resetCooldown();
    
    // Get current target frequency (for debug display)
    float getCurrentTargetFreq() const { return targetFreq_; }

    // Configuration
    void setConfig(const Config& config) { config_ = config; }
    const Config& getConfig() const { return config_; }

private:
    Config config_;
    
    // State
    std::atomic<bool> userInteracting_{false};
    std::atomic<bool> trackingActive_{false};
    juce::Time lastInteractionTime_;
    juce::Time targetSetTime_;
    
    // Detection state - track whether we have valid detections
    bool lastFrameHadDetection_ = false;
    int consecutiveEmptyFrames_ = 0;
    static constexpr int EMPTY_FRAMES_THRESHOLD = 3;  // Pause tracking after 3 empty frames
    
    // Current target for smooth approach
    float targetFreq_ = 0.0f;
    float currentTargetConfidence_ = 0.0f;
    
    // Animation state
    bool isApproaching_ = false;
    float approachStartFreq_ = 0.0f;
    float approachTargetFreq_ = 0.0f;
    double approachProgress_ = 0.0;
    
    // Pending target - store target when no valid detection to resume later
    bool hasPendingTarget_ = false;
    float pendingTargetFreq_ = 0.0f;
    
    // Center zone: middle 1/3 of vertical range
    // Returns min and max MIDI note offset from center
    void getCenterZoneRange(float viewHeightSemitones, 
                            float& outMinOffset, 
                            float& outMaxOffset) const;
    
    // Case 1: Global jump - find best point anywhere
    bool findGlobalBest(const std::vector<PitchCandidate>& pitches,
                        float& outFreq, float& outConfidence) const;
    
    // Case 2: Find point nearest to center zone (in semitone distance)
    bool findNearestToCenter(const std::vector<PitchCandidate>& pitches,
                              float viewCenterFreq,
                              float viewHeightSemitones,
                              float& outFreq, float& outConfidence) const;
    
    // Check if a frequency is in the vertical center zone
    bool isInCenterZone(float freq, float viewCenterFreq, float viewHeightSemitones) const;
    
    // Check if any point is in center zone
    bool hasPointInCenterZone(const std::vector<PitchCandidate>& pitches,
                               float viewCenterFreq,
                               float viewHeightSemitones) const;
    
    // Compute distance from freq to center zone (in semitones, 0 if inside)
    float distanceToCenterZone(float freq, float viewCenterFreq, float viewHeightSemitones) const;
    
    // Smooth approach animation
    float computeApproachPosition(float currentCenter, float targetCenter, float deltaTime);
    
    // Ease-in-out interpolation
    static float easeInOut(float t);
    
    // Calculate score for candidate selection (confidence weighted)
    float calculateScore(const PitchCandidate& pitch) const;
    
    // Convert frequency to MIDI note
    static float freqToMidi(float freq);
    // Convert MIDI note to frequency  
    static float midiToFreq(float midi);
};

} // namespace spm

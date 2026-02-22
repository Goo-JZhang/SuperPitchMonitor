#include "AutoTracker.h"
#include <algorithm>
#include <cmath>

namespace spm {

AutoTracker::AutoTracker(const Config& config)
    : config_(config)
{
    // Initialize with cooldown already expired so tracking starts immediately
    // (User has not interacted yet, so auto-tracking should be active)
    lastInteractionTime_ = juce::Time(0);  // Far in the past
}

void AutoTracker::onUserInteraction()
{
    userInteracting_ = true;
    trackingActive_ = false;
    isApproaching_ = false;
    lastInteractionTime_ = juce::Time::getCurrentTime();
}

void AutoTracker::onUserInteractionEnd()
{
    userInteracting_ = false;
    lastInteractionTime_ = juce::Time::getCurrentTime();
}

float AutoTracker::getCooldownRemaining() const
{
    if (userInteracting_)
        return static_cast<float>(config_.cooldownSeconds);
    
    auto elapsed = juce::Time::getCurrentTime() - lastInteractionTime_;
    float remaining = static_cast<float>(config_.cooldownSeconds) - elapsed.inSeconds();
    return std::max(0.0f, remaining);
}

void AutoTracker::resetCooldown()
{
    // Reset cooldown to expired state so auto-tracking activates immediately
    lastInteractionTime_ = juce::Time(0);  // Far in the past
    trackingActive_ = false;
    isApproaching_ = false;
}

void AutoTracker::getCenterZoneRange(float viewHeightSemitones, 
                                      float& outMinOffset, 
                                      float& outMaxOffset) const
{
    float centerZoneHeight = viewHeightSemitones * config_.centerZoneRatio;
    outMinOffset = -centerZoneHeight * 0.5f;
    outMaxOffset = centerZoneHeight * 0.5f;
}

float AutoTracker::freqToMidi(float freq)
{
    return 69.0f + 12.0f * std::log2(freq / 440.0f);
}

float AutoTracker::midiToFreq(float midi)
{
    return 440.0f * std::pow(2.0f, (midi - 69.0f) / 12.0f);
}

float AutoTracker::calculateScore(const PitchCandidate& pitch) const
{
    return pitch.confidence * config_.confidenceWeight + 
           pitch.amplitude * config_.energyWeight;
}

bool AutoTracker::isInCenterZone(float freq, float viewCenterFreq, float viewHeightSemitones) const
{
    // Check if freq is inside the center zone (middle 1/3 of view)
    float midi = freqToMidi(freq);
    float centerMidi = freqToMidi(viewCenterFreq);
    float offset = midi - centerMidi;
    
    float minOffset, maxOffset;
    getCenterZoneRange(viewHeightSemitones, minOffset, maxOffset);
    
    return (offset >= minOffset && offset <= maxOffset);
}

float AutoTracker::distanceToCenterZone(float freq, float viewCenterFreq, float viewHeightSemitones) const
{
    // Calculate distance from freq to the CENTER ZONE (not to exact center)
    // Returns 0 if inside center zone, otherwise distance to nearest boundary
    float midi = freqToMidi(freq);
    float centerMidi = freqToMidi(viewCenterFreq);
    float offset = midi - centerMidi;
    
    float minOffset, maxOffset;
    getCenterZoneRange(viewHeightSemitones, minOffset, maxOffset);
    
    if (offset < minOffset)
        return minOffset - offset;  // Below center zone
    else if (offset > maxOffset)
        return offset - maxOffset;  // Above center zone
    else
        return 0.0f;  // Inside center zone
}

bool AutoTracker::hasPointInCenterZone(const std::vector<PitchCandidate>& pitches,
                                        float viewCenterFreq,
                                        float viewHeightSemitones) const
{
    // Case 3: Check if any point is inside the center zone (middle 1/3 of view)
    for (const auto& pitch : pitches)
    {
        if (pitch.confidence > 0.1f && isInCenterZone(pitch.frequency, viewCenterFreq, viewHeightSemitones))
            return true;
    }
    return false;
}

bool AutoTracker::findGlobalBest(const std::vector<PitchCandidate>& pitches,
                                  float& outFreq, float& outConfidence) const
{
    const PitchCandidate* best = nullptr;
    float bestScore = -1.0f;
    
    for (const auto& pitch : pitches)
    {
        if (pitch.confidence < 0.1f)  // Filter out low confidence
            continue;
            
        float score = calculateScore(pitch);
        if (score > bestScore)
        {
            bestScore = score;
            best = &pitch;
        }
    }
    
    if (best)
    {
        outFreq = best->frequency;
        outConfidence = best->confidence;
        return true;
    }
    
    return false;
}

bool AutoTracker::findNearestToCenter(const std::vector<PitchCandidate>& pitches,
                                       float viewCenterFreq,
                                       float viewHeightSemitones,
                                       float& outFreq, float& outConfidence) const
{
    const PitchCandidate* nearest = nullptr;
    float minDistance = std::numeric_limits<float>::max();
    float bestConfidence = 0.0f;
    
    for (const auto& pitch : pitches)
    {
        if (pitch.confidence < 0.1f)
            continue;
            
        // Skip points already in center zone (Case 3: hold position)
        if (isInCenterZone(pitch.frequency, viewCenterFreq, viewHeightSemitones))
            continue;
            
        // Calculate distance to center zone boundary (0 if inside)
        float distance = distanceToCenterZone(pitch.frequency, viewCenterFreq, viewHeightSemitones);
        
        // Compare: distance first, then confidence
        bool isBetter = false;
        if (distance < minDistance - 0.01f)
            isBetter = true;
        else if (std::abs(distance - minDistance) <= 0.01f && pitch.confidence > bestConfidence)
            isBetter = true;
            
        if (isBetter)
        {
            minDistance = distance;
            bestConfidence = pitch.confidence;
            nearest = &pitch;
        }
    }
    
    if (nearest)
    {
        outFreq = nearest->frequency;
        outConfidence = nearest->confidence;
        return true;
    }
    
    return false;
}

float AutoTracker::easeInOut(float t)
{
    if (t < 0.5f)
        return 2.0f * t * t;
    else
        return 1.0f - std::pow(-2.0f * t + 2.0f, 2.0f) / 2.0f;
}

float AutoTracker::computeApproachPosition(float currentCenter, float targetCenter, float deltaTime)
{
    float diff = targetCenter - currentCenter;
    float diffMidi = freqToMidi(targetCenter) - freqToMidi(currentCenter);
    
    // Check if already close enough (within center zone boundary)
    if (std::abs(diffMidi) < 0.5f)  // Less than half semitone
        return targetCenter;
    
    // Compute speed-limited movement
    float maxFreqChange = config_.maxPanSpeed * std::abs(diff) * deltaTime;
    maxFreqChange = std::max(maxFreqChange, config_.minApproachFreq * deltaTime);
    
    // Apply ease-in-out
    float t = std::min(1.0f, config_.approachSpeed * deltaTime * 10.0f);
    float easedT = easeInOut(t);
    
    float moveAmount = diff * easedT;
    
    // Clamp to max speed
    if (std::abs(moveAmount) > maxFreqChange)
        moveAmount = (moveAmount > 0) ? maxFreqChange : -maxFreqChange;
    
    return currentCenter + moveAmount;
}

bool AutoTracker::update(const std::vector<PitchCandidate>& currentPitches,
                          bool hasValidDetection,
                          float currentViewCenterFreq,
                          float viewHeightSemitones,
                          float deltaTime,
                          float& outNewViewCenterFreq)
{
    outNewViewCenterFreq = currentViewCenterFreq;
    
    // Check if we should be tracking
    if (userInteracting_)
    {
        trackingActive_ = false;
        return false;
    }
    
    float cooldownRemaining = getCooldownRemaining();
    if (cooldownRemaining > 0)
    {
        trackingActive_ = false;
        return false;
    }
    
    // Track detection state for intelligent pause/resume
    if (hasValidDetection)
    {
        // We have valid detection - reset empty frame counter
        consecutiveEmptyFrames_ = 0;
        lastFrameHadDetection_ = true;
    }
    else
    {
        // No valid detection - increment counter
        consecutiveEmptyFrames_++;
        
        // If we've had too many empty frames, pause tracking
        if (consecutiveEmptyFrames_ >= EMPTY_FRAMES_THRESHOLD)
        {
            lastFrameHadDetection_ = false;
            isApproaching_ = false;
            trackingActive_ = false;
            return false;
        }
        
        // During brief gaps (1-2 frames), continue with pending target if any
        if (!hasPendingTarget_)
        {
            trackingActive_ = false;
            return false;
        }
    }
    
    trackingActive_ = true;
    
    // Filter valid pitches (confidence > 0.1)
    std::vector<const PitchCandidate*> validPitches;
    for (const auto& pitch : currentPitches)
    {
        if (pitch.confidence > 0.1f)
            validPitches.push_back(&pitch);
    }
    
    // Case 3: Already have points in center zone - do nothing
    if (!validPitches.empty() && 
        hasPointInCenterZone(currentPitches, currentViewCenterFreq, viewHeightSemitones))
    {
        isApproaching_ = false;
        hasPendingTarget_ = false;
        return false;
    }
    
    // No valid pitches on this frame - check if we should use pending target
    if (validPitches.empty())
    {
        if (hasPendingTarget_ && isApproaching_)
        {
            // Continue approaching the pending target (brief noise gap)
            outNewViewCenterFreq = computeApproachPosition(currentViewCenterFreq, 
                                                            pendingTargetFreq_, 
                                                            deltaTime);
            
            // Check if we've reached the pending target
            if (isInCenterZone(pendingTargetFreq_, outNewViewCenterFreq, viewHeightSemitones))
            {
                isApproaching_ = false;
                hasPendingTarget_ = false;
            }
            
            return outNewViewCenterFreq != currentViewCenterFreq;
        }
        
        isApproaching_ = false;
        return false;
    }
    
    float targetFreq = 0.0f;
    float targetConf = 0.0f;
    
    // Case 1: No points visible on screen at all - global jump
    // Check if any pitch is within the current view (within ±viewHeightSemitones/2)
    bool hasPointsInView = false;
    float viewHalfHeight = viewHeightSemitones / 2.0f;
    for (const auto* pitch : validPitches)
    {
        float dist = std::abs(freqToMidi(pitch->frequency) - freqToMidi(currentViewCenterFreq));
        if (dist < viewHalfHeight)
        {
            hasPointsInView = true;
            break;
        }
    }
    
    if (!hasPointsInView)
    {
        // Global jump to best point (Case 1: no points in view at all)
        if (findGlobalBest(currentPitches, targetFreq, targetConf))
        {
            outNewViewCenterFreq = targetFreq;
            targetFreq_ = targetFreq;
            pendingTargetFreq_ = targetFreq;
            currentTargetConfidence_ = targetConf;
            hasPendingTarget_ = true;
            isApproaching_ = false;
            return true;
        }
    }
    
    // Case 2: Points visible but not in center - smooth approach
    if (findNearestToCenter(currentPitches, currentViewCenterFreq, viewHeightSemitones, 
                             targetFreq, targetConf))
    {
        // Initialize approach if target changed significantly
        if (!isApproaching_ || std::abs(freqToMidi(targetFreq) - freqToMidi(targetFreq_)) > 1.0f)
        {
            isApproaching_ = true;
            approachStartFreq_ = currentViewCenterFreq;
            approachTargetFreq_ = targetFreq;
            targetFreq_ = targetFreq;
            pendingTargetFreq_ = targetFreq;
            currentTargetConfidence_ = targetConf;
            hasPendingTarget_ = true;
        }
        
        // Continue approach
        outNewViewCenterFreq = computeApproachPosition(currentViewCenterFreq, targetFreq_, deltaTime);
        
        // Check if we've reached the target (inside center zone)
        if (isInCenterZone(targetFreq_, outNewViewCenterFreq, viewHeightSemitones))
        {
            isApproaching_ = false;
        }
        
        return outNewViewCenterFreq != currentViewCenterFreq;
    }
    
    isApproaching_ = false;
    return false;
}

} // namespace spm

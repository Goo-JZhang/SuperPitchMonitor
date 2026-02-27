#include "PitchWaterfallDisplay.h"
#include "../Utils/Logger.h"
#include "../Utils/AutoTracker.h"

namespace spm {

PitchWaterfallDisplay::PitchWaterfallDisplay()
{
    updateLogFreqRange();
    setOpaque(true);
    
    // Start timer for continuous display refresh (30fps)
    // This ensures the time axis keeps scrolling even without pitch detection
    startTimerHz(30);
}

PitchWaterfallDisplay::~PitchWaterfallDisplay()
{
    stopTimer();
}

void PitchWaterfallDisplay::timerCallback()
{
    // Skip updates if paused (e.g., when Settings panel is open)
    if (isPaused_)
        return;
    
    // Auto-tracking update
    if (autoTrackingEnabled_)
    {
        double now = juce::Time::getMillisecondCounterHiRes() / 1000.0;
        double deltaTime = (lastAutoTrackUpdateTime_ > 0.0) ? 
                          static_cast<float>(now - lastAutoTrackUpdateTime_) : 0.033f;
        lastAutoTrackUpdateTime_ = now;
        
        // Get current visible range in semitones
        float viewHeightSemitones = 24.0f;  // Approximate visible range
        float currentCenterFreq = std::sqrt(minFreq_ * maxFreq_);  // Geometric mean
        
        // Apply scroll offset to center frequency
        float centerMidi = 69.0f + scrollOffset_;  // A4 = 69 MIDI + offset
        currentCenterFreq = 440.0f * std::pow(2.0f, (centerMidi - 69.0f) / 12.0f);
        
        float newCenterFreq = currentCenterFreq;
        
        std::vector<PitchCandidate> pitchesCopy;
        bool hasDetection;
        {
            juce::ScopedLock lock(currentPitchesLock_);
            pitchesCopy = currentPitches_;
            hasDetection = hasValidDetection_;
        }
        
        if (autoTracker_.update(pitchesCopy, hasDetection, currentCenterFreq, 
                                viewHeightSemitones, static_cast<float>(deltaTime), newCenterFreq))
        {
            // Update scroll offset based on new center frequency
            float newCenterMidi = 69.0f + 12.0f * std::log2(newCenterFreq / 440.0f);
            scrollOffset_ = newCenterMidi - 69.0f;
        }
    }
    
    // Continuous repaint to keep time axis scrolling
    repaint();
}

void PitchWaterfallDisplay::updateLogFreqRange()
{
    minLogFreq_ = std::log(minFreq_);
    maxLogFreq_ = std::log(maxFreq_);
    logFreqRange_ = maxLogFreq_ - minLogFreq_;
}

void PitchWaterfallDisplay::setFrequencyRange(float minFreq, float maxFreq)
{
    minFreq_ = juce::jlimit(20.0f, 20000.0f, minFreq);
    maxFreq_ = juce::jlimit(20.0f, 20000.0f, maxFreq);
    if (minFreq_ >= maxFreq_) maxFreq_ = minFreq_ * 10;
    updateLogFreqRange();
    repaint();
}

void PitchWaterfallDisplay::setTimeWindow(float seconds)
{
    timeWindow_ = juce::jlimit(1.0f, 30.0f, seconds);
    
    double now = juce::Time::getMillisecondCounterHiRes() / 1000.0;
    while (!pitchHistory_.empty() && 
           (now - pitchHistory_.front().timestamp) > timeWindow_)
    {
        pitchHistory_.pop_front();
    }
    
    repaint();
}

void PitchWaterfallDisplay::setScrollOffset(float offset)
{
    scrollOffset_ = offset;
    repaint();
}

float PitchWaterfallDisplay::freqToLogY(float freq) const
{
    float logFreq = std::log(juce::jlimit(minFreq_, maxFreq_, freq));
    return (logFreq - minLogFreq_) / logFreqRange_;
}

float PitchWaterfallDisplay::midiToLogY(float midiNote) const
{
    float freq = a4Frequency_ * std::pow(2.0f, (midiNote - 69.0f) / 12.0f);
    return freqToLogY(freq);
}

float PitchWaterfallDisplay::logYToFreq(float logY) const
{
    float logFreq = minLogFreq_ + logY * logFreqRange_;
    return std::exp(logFreq);
}

juce::String PitchWaterfallDisplay::midiToNoteName(float midiNote) const
{
    static const char* noteNames[] = { "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B" };
    
    int noteIndex = (int)std::round(midiNote) % 12;
    if (noteIndex < 0) noteIndex += 12;
    
    int octave = (int)std::round(midiNote - 12) / 12;
    
    return juce::String(noteNames[noteIndex]) + juce::String(octave);
}

bool PitchWaterfallDisplay::isMainNote(int noteIndex) const
{
    // CDEFGAB = 0, 2, 4, 5, 7, 9, 11
    return noteIndex == 0 || noteIndex == 2 || noteIndex == 4 || 
           noteIndex == 5 || noteIndex == 7 || noteIndex == 9 || noteIndex == 11;
}

void PitchWaterfallDisplay::getVisibleMidiRange(float& minMidi, float& maxMidi) const
{
    // Base range centered on A4 + scroll offset
    float baseCenter = 69.0f + scrollOffset_;  // A4 + scroll
    
    minMidi = baseCenter - visibleSemitones_ / 2.0f;
    maxMidi = baseCenter + visibleSemitones_ / 2.0f;
}

juce::Rectangle<int> PitchWaterfallDisplay::getPlotArea() const
{
    return getLocalBounds()
        .withTrimmedLeft(leftMargin)
        .withTrimmedRight(rightMargin)
        .withTrimmedTop(topMargin)
        .withTrimmedBottom(bottomMargin);
}

float PitchWaterfallDisplay::timeToX(double timestamp, double now) const
{
    auto plotArea = getPlotArea();
    float age = (float)(now - timestamp);
    float t = 1.0f - (age / timeWindow_);
    t = juce::jlimit(0.0f, 1.0f, t);
    return plotArea.getX() + t * plotArea.getWidth();
}

float PitchWaterfallDisplay::midiToY(float midiNote) const
{
    auto plotArea = getPlotArea();
    float minMidi, maxMidi;
    getVisibleMidiRange(minMidi, maxMidi);
    
    float t = (midiNote - minMidi) / (maxMidi - minMidi);
    return plotArea.getBottom() - t * plotArea.getHeight();
}

float PitchWaterfallDisplay::freqToY(float freq) const
{
    float midiNote = 69.0f + 12.0f * std::log2(freq / a4Frequency_);
    return midiToY(midiNote);
}

void PitchWaterfallDisplay::mouseWheelMove(const juce::MouseEvent& event, 
                                           const juce::MouseWheelDetails& wheel)
{
    // Notify auto-tracker of user interaction
    autoTracker_.onUserInteraction();
    
    // Mouse wheel: Zoom frequency range (adjust visible semitones)
    // Wheel up (deltaY > 0): Zoom in (smaller visible range)
    // Wheel down (deltaY < 0): Zoom out (larger visible range)
    if (std::abs(wheel.deltaY) > 0.01f)
    {
        float zoomFactor = 1.0f - wheel.deltaY * 0.1f;
        visibleSemitones_ *= zoomFactor;
        visibleSemitones_ = juce::jlimit(minVisibleSemitones, maxVisibleSemitones, visibleSemitones_);
        
        repaint();
    }
    
    // End interaction after a short delay
    autoTracker_.onUserInteractionEnd();
}

void PitchWaterfallDisplay::mouseMagnify(const juce::MouseEvent& event, float scaleFactor)
{
    // Notify auto-tracker of user interaction
    autoTracker_.onUserInteraction();
    
    // scaleFactor > 1.0: 放大 (缩小可见范围)
    // scaleFactor < 1.0: 缩小 (扩大可见范围)
    // 注意：这里除以 scaleFactor，因为 scaleFactor > 1 表示放大（显示更少内容）
    visibleSemitones_ /= scaleFactor;
    visibleSemitones_ = juce::jlimit(minVisibleSemitones, maxVisibleSemitones, visibleSemitones_);
    
    repaint();
    
    // End interaction after a short delay
    autoTracker_.onUserInteractionEnd();
}

void PitchWaterfallDisplay::mouseDown(const juce::MouseEvent& event)
{
    // Check for double-click (jump to best or reset)
    if (event.getNumberOfClicks() == 2)
    {
        performJumpToBestOrReset();
        return;
    }
    
    // Notify auto-tracker of user interaction
    autoTracker_.onUserInteraction();
    
    // Start dragging on left mouse button in plot area
    if (event.mods.isLeftButtonDown())
    {
        auto plotArea = getPlotArea();
        if (plotArea.contains(event.getPosition()))
        {
            isDragging_ = true;
            dragStartY_ = event.getPosition().y;
            dragStartOffset_ = scrollOffset_;
            setMouseCursor(juce::MouseCursor::DraggingHandCursor);
        }
    }
}

void PitchWaterfallDisplay::mouseDrag(const juce::MouseEvent& event)
{
    if (isDragging_)
    {
        auto plotArea = getPlotArea();
        float deltaY = event.getPosition().y - dragStartY_;
        
        // Convert pixel delta to semitones
        // Dragging down -> scroll down (see lower frequencies)
        // Dragging up -> scroll up (see higher frequencies)
        float plotHeight = plotArea.getHeight();
        float visibleSemitones = 24.0f;  // Default visible range is about 2 octaves
        float semitoneDelta = (deltaY / plotHeight) * visibleSemitones;
        
        scrollOffset_ = dragStartOffset_ + semitoneDelta;
        
        // Limit scroll range (about +/- 3 octaves from center)
        scrollOffset_ = juce::jlimit(-36.0f, 36.0f, scrollOffset_);
        
        repaint();
    }
}

void PitchWaterfallDisplay::mouseUp(const juce::MouseEvent& event)
{
    if (isDragging_)
    {
        isDragging_ = false;
        setMouseCursor(juce::MouseCursor::NormalCursor);
    }
    
    // Notify auto-tracker that interaction ended
    autoTracker_.onUserInteractionEnd();
}

void PitchWaterfallDisplay::updatePitch(const PitchCandidate& pitch)
{
    PitchVector pitches;
    pitches.push_back(pitch);
    updatePitches(pitches);
}

void PitchWaterfallDisplay::updatePitches(const PitchVector& pitches)
{
    // Store current pitches for auto-tracking (only current frame, not history)
    {
        juce::ScopedLock lock(currentPitchesLock_);
        currentPitches_.clear();
        hasValidDetection_ = false;
        for (const auto& pitch : pitches)
        {
            if (pitch.confidence > 0.1f)  // Lower threshold for auto-tracker
            {
                currentPitches_.push_back(pitch);
                hasValidDetection_ = true;
            }
        }
    }
    
    if (pitches.empty()) return;
    
    {
        juce::ScopedLock lock(dataLock_);
        
        double now = juce::Time::getMillisecondCounterHiRes() / 1000.0;
        
        // Add all pitches with the same timestamp (polyphonic support)
        for (const auto& pitch : pitches)
        {
            if (pitch.confidence > 0.2f)
            {
                PitchHistory hist;
                hist.timestamp = now;
                hist.midiNote = pitch.midiNote;
                hist.confidence = pitch.confidence;
                hist.frequency = pitch.frequency;
                hist.amplitude = pitch.amplitude;
                hist.isMLEnergy = pitch.isMLEnergy;
                hist.hasPitch = true;
                
                pitchHistory_.push_back(hist);
            }
        }
        
        // Clean up old entries
        while (!pitchHistory_.empty() && 
               (now - pitchHistory_.front().timestamp) > timeWindow_)
        {
            pitchHistory_.pop_front();
        }
        
        while (pitchHistory_.size() > maxHistorySize)
        {
            pitchHistory_.pop_front();
        }
    }
    
    juce::MessageManager::callAsync([this]() {
        repaint();
    });
}

void PitchWaterfallDisplay::updateSpectrum(const SpectrumData& data)
{
    {
        juce::ScopedLock lock(dataLock_);
        
        if (std::abs(data.sampleRate - currentSampleRate_) > 1.0f)
        {
            currentSampleRate_ = data.sampleRate;
            pitchHistory_.clear();
        }
        
        currentSpectrum_ = data.magnitudes;
    }
}

void PitchWaterfallDisplay::clear()
{
    {
        juce::ScopedLock lock(dataLock_);
        pitchHistory_.clear();
        currentSpectrum_.clear();
    }
    {
        juce::ScopedLock lock(currentPitchesLock_);
        currentPitches_.clear();
        hasValidDetection_ = false;
    }
    repaint();
}

void PitchWaterfallDisplay::performJumpToBestOrReset()
{
    // Double-click / double-tap: Jump to best pitch or reset to A4
    
    std::vector<PitchCandidate> pitchesCopy;
    {
        juce::ScopedLock lock(currentPitchesLock_);
        pitchesCopy = currentPitches_;
    }
    
    // Find best pitch by confidence
    const PitchCandidate* best = nullptr;
    float bestScore = -1.0f;
    for (const auto& pitch : pitchesCopy)
    {
        if (pitch.confidence > 0.1f)
        {
            float score = pitch.confidence * 1000.0f + pitch.amplitude;
            if (score > bestScore)
            {
                bestScore = score;
                best = &pitch;
            }
        }
    }
    
    if (best)
    {
        // Jump to best pitch
        float targetMidi = best->midiNote;
        scrollOffset_ = targetMidi - 69.0f;  // A4 = 69
        SPM_LOG_INFO("[AutoTrack] Double-click: Jump to best pitch " + 
                    juce::String(best->frequency) + " Hz");
    }
    else
    {
        // Reset to A4 center
        scrollOffset_ = 0.0f;
        SPM_LOG_INFO("[AutoTrack] Double-click: Reset to A4 (440 Hz)");
    }
    
    repaint();
}

void PitchWaterfallDisplay::paint(juce::Graphics& g)
{
    drawBackground(g);
    drawAxes(g);
    drawGrid(g);
    drawPitchHistory(g);
    drawCurrentPosition(g);
}

void PitchWaterfallDisplay::drawBackground(juce::Graphics& g)
{
    juce::ColourGradient gradient(
        juce::Colour(0xFF0D0D12), 0.0f, 0.0f,
        juce::Colour(0xFF1A1A20), 0.0f, (float)getHeight(),
        false
    );
    g.setGradientFill(gradient);
    g.fillAll();
    
    auto plotArea = getPlotArea();
    g.setColour(juce::Colour(0xFF15151A));
    g.fillRect(plotArea);
}

void PitchWaterfallDisplay::drawAxes(juce::Graphics& g)
{
    auto plotArea = getPlotArea();
    
    g.setColour(juce::Colours::white.withAlpha(0.6f));
    g.drawVerticalLine(plotArea.getX(), plotArea.getY(), plotArea.getBottom());
    g.drawHorizontalLine(plotArea.getBottom(), plotArea.getX(), plotArea.getRight());
    
    // === Y Axis Labels - Only CDEFGAB main notes ===
    g.setFont(11.0f);
    
    float minMidi, maxMidi;
    getVisibleMidiRange(minMidi, maxMidi);
    
    int minNote = (int)std::floor(minMidi);
    int maxNote = (int)std::ceil(maxMidi);
    
    for (int note = minNote; note <= maxNote; ++note)
    {
        int noteIndex = note % 12;
        if (noteIndex < 0) noteIndex += 12;
        
        // Only draw main notes (CDEFGAB)
        if (!isMainNote(noteIndex))
            continue;
            
        float y = midiToY((float)note);
        if (y < plotArea.getY() || y > plotArea.getBottom()) 
            continue;
        
        // Tick mark
        g.setColour(juce::Colours::white.withAlpha(0.8f));
        g.drawLine(plotArea.getX() - 8, y, plotArea.getX(), y);
        
        // Label - just note name (no frequency)
        if (showNoteNames_)
        {
            g.setColour(juce::Colours::white.withAlpha(0.9f));
            juce::String label = midiToNoteName((float)note);
            
            // Highlight C in brighter color
            if (noteIndex == 0)
                g.setColour(juce::Colours::cyan.withAlpha(0.9f));
            
            g.drawText(label, 2, (int)y - 8, leftMargin - 6, 16,
                      juce::Justification::centredRight);
        }
    }
    
    // Y axis title
    g.setColour(juce::Colours::white.withAlpha(0.5f));
    g.setFont(9.0f);
    g.drawText("Note", 2, plotArea.getY() - 8, leftMargin - 4, 12,
              juce::Justification::centred);
    
    // === X Axis Labels (Time) ===
    g.setColour(juce::Colours::white.withAlpha(0.6f));
    
    int numTimeTicks = 6;
    for (int i = 0; i < numTimeTicks; ++i)
    {
        float t = (float)i / (numTimeTicks - 1);
        float x = plotArea.getX() + t * plotArea.getWidth();
        
        g.drawLine(x, plotArea.getBottom(), x, plotArea.getBottom() + 5);
        
        float timeSec = -(1.0f - t) * timeWindow_;
        juce::String timeLabel = juce::String(timeSec, 1) + "s";
        
        g.setColour(juce::Colours::white.withAlpha(0.5f));
        g.setFont(9.0f);
        g.drawText(timeLabel, (int)x - 20, plotArea.getBottom() + 8, 40, 12,
                  juce::Justification::centred);
    }
    
    g.setColour(juce::Colours::white.withAlpha(0.5f));
    g.drawText("Time", plotArea.getCentreX() - 20, getHeight() - 14, 40, 12,
              juce::Justification::centred);
}

void PitchWaterfallDisplay::drawGrid(juce::Graphics& g)
{
    auto plotArea = getPlotArea();
    
    float minMidi, maxMidi;
    getVisibleMidiRange(minMidi, maxMidi);
    
    int minNote = (int)std::floor(minMidi);
    int maxNote = (int)std::ceil(maxMidi);
    
    for (int note = minNote; note <= maxNote; ++note)
    {
        int noteIndex = note % 12;
        if (noteIndex < 0) noteIndex += 12;
            
        float y = midiToY((float)note);
        if (y < plotArea.getY() || y > plotArea.getBottom()) 
            continue;
        
        // All notes get grid lines, with different brightness
        if (noteIndex == 0)
            g.setColour(juce::Colours::white.withAlpha(0.20f));  // C - brightest
        else if (isMainNote(noteIndex))
            g.setColour(juce::Colours::white.withAlpha(0.10f));  // Main notes (CDEFGAB) - medium
        else
            g.setColour(juce::Colours::white.withAlpha(0.04f));  // Accidentals (C#, D#, etc.) - dim
        
        g.drawHorizontalLine((int)y, plotArea.getX(), plotArea.getRight());
    }
    
    // Vertical time grid
    int numTimeDivisions = 10;
    g.setColour(juce::Colours::white.withAlpha(0.05f));
    for (int i = 1; i < numTimeDivisions; ++i)
    {
        float t = (float)i / numTimeDivisions;
        float x = plotArea.getX() + t * plotArea.getWidth();
        g.drawVerticalLine((int)x, plotArea.getY(), plotArea.getBottom());
    }
}

void PitchWaterfallDisplay::drawPitchHistory(juce::Graphics& g)
{
    juce::ScopedLock lock(dataLock_);
    
    if (pitchHistory_.empty()) return;
    
    auto plotArea = getPlotArea();
    double now = juce::Time::getMillisecondCounterHiRes() / 1000.0;
    
    float minMidi, maxMidi;
    getVisibleMidiRange(minMidi, maxMidi);
    
    // Draw pitch history as scattered dots (no lines)
    // Brightness represents intensity/confidence
    for (const auto& hist : pitchHistory_)
    {
        // Skip if outside visible range
        if (hist.midiNote < minMidi - 1 || hist.midiNote > maxMidi + 1)
            continue;
        
        float x = timeToX(hist.timestamp, now);
        
        if (x < plotArea.getX()) continue;
        if (x > plotArea.getRight()) break;
        
        if (hist.hasPitch && hist.confidence > 0.2f)
        {
            float y = midiToY(hist.midiNote);
            y = juce::jlimit((float)plotArea.getY(), (float)plotArea.getBottom(), y);
            
            // Normalize confidence and amplitude to 0-1 range
            float confNorm = juce::jlimit(0.0f, 1.0f, hist.confidence);
            float ampNorm;
            
            if (hist.isMLEnergy)
            {
                // ML mode: energy is already in relative scale (0-2 typical range)
                // Use logarithmic scaling for better dynamic range
                ampNorm = juce::jlimit(0.0f, 1.0f, hist.amplitude / 2.0f);
                // Apply log curve: low energy still visible, high energy saturates gracefully
                ampNorm = std::sqrt(ampNorm);  // sqrt for better low-level visibility
            }
            else
            {
                // FFT mode: amplitude is in dB range, normalize to typical max (200)
                ampNorm = juce::jlimit(0.0f, 1.0f, hist.amplitude / 200.0f);
            }
            
            // Four-quadrant display strategy:
            // Size: High conf = small (precise), Low conf = large (uncertain)
            // Brightness: High amp = bright, Low amp = dim
            
            float size, brightness;
            
            if (confNorm > 0.6f && ampNorm > 0.5f)
            {
                // Q1: High confidence + High energy → Small point + High brightness
                size = 3.0f;
                brightness = 0.85f + ampNorm * 0.15f;  // 0.85-1.0
            }
            else if (confNorm <= 0.6f && ampNorm > 0.5f)
            {
                // Q2: Low confidence + High energy → Medium point + Medium brightness
                size = 5.0f;
                brightness = 0.5f + ampNorm * 0.25f;  // 0.5-0.75
            }
            else if (confNorm > 0.6f && ampNorm <= 0.5f)
            {
                // Q3: High confidence + Low energy → Small point + Medium brightness
                size = 3.5f;
                brightness = 0.45f + confNorm * 0.2f;  // 0.45-0.65
            }
            else
            {
                // Q4: Low confidence + Low energy → Large point + Low brightness
                size = 7.0f;
                brightness = 0.25f + (confNorm + ampNorm) * 0.15f;  // 0.25-0.4
            }
            
            // Color based on confidence (hue)
            float hue = 0.05f + confNorm * 0.5f;  // Orange (0.05) to Cyan (0.55)
            
            // Draw radial gradient dot
            // Center: bright, high alpha
            // Edge: same brightness but alpha=0
            drawRadialGradientDot(g, x, y, size, hue, brightness);
        }
    }
    
    // Current pitch info
    if (!pitchHistory_.empty())
    {
        const auto& latest = pitchHistory_.back();
        if (latest.hasPitch && latest.confidence > 0.5f)
        {
            auto infoArea = plotArea.removeFromTop(30).removeFromRight(200);
            
            g.setColour(juce::Colours::black.withAlpha(0.5f));
            g.fillRoundedRectangle(infoArea.toFloat(), 4.0f);
            
            g.setColour(juce::Colours::white);
            g.setFont(12.0f);
            juce::String info = midiToNoteName(latest.midiNote);
            info += " (" + juce::String(latest.frequency, 1) + "Hz)";
            g.drawText(info, infoArea.reduced(5), juce::Justification::centredRight);
        }
    }
}

void PitchWaterfallDisplay::drawCurrentPosition(juce::Graphics& g)
{
    auto plotArea = getPlotArea();
    
    g.setColour(juce::Colours::white.withAlpha(0.3f));
    g.drawVerticalLine(plotArea.getRight(), plotArea.getY(), plotArea.getBottom());
    
    g.setFont(9.0f);
    g.drawText("Now", plotArea.getRight() - 15, plotArea.getBottom() + 8, 30, 12,
              juce::Justification::centred);
}

void PitchWaterfallDisplay::drawRadialGradientDot(juce::Graphics& g, float x, float y, 
                                                   float radius, float hue, float brightness)
{
    // Use JUCE radial gradient with multiple colour stops for sharp peak effect
    float saturation = 0.9f;
    
    // Centre colour (brightest)
    juce::Colour centreColour = juce::Colour::fromHSV(hue, saturation, 1.0f, 1.0f);
    // Mid colour (slightly dimmer)
    juce::Colour midColour = juce::Colour::fromHSV(hue, saturation, brightness, 0.6f);
    // Edge colour (transparent)
    juce::Colour edgeColour = juce::Colour::fromHSV(hue, saturation, brightness, 0.0f);
    
    // Create radial gradient with colour stops
    juce::ColourGradient gradient;
    gradient.isRadial = true;
    gradient.point1 = juce::Point<float>(x, y);  // Centre
    gradient.point2 = juce::Point<float>(x + radius, y);  // Edge point
    
    // Add colour stops for sharp peak:
    // 0.0 = centre (bright)
    // 0.3 = near centre (still fairly bright) 
    // 1.0 = edge (transparent)
    gradient.addColour(0.0, centreColour);
    gradient.addColour(0.25, midColour);
    gradient.addColour(1.0, edgeColour);
    
    g.setGradientFill(gradient);
    g.fillEllipse(x - radius, y - radius, radius * 2, radius * 2);
}

void PitchWaterfallDisplay::resized()
{
    repaint();
}

void PitchWaterfallDisplay::setTargetRefreshRate(int fps)
{
    targetFPS_ = fps;
    unlimitedFPS_ = (fps < 0);
    
    // Restart timer with new rate
    stopTimer();
    
    if (unlimitedFPS_)
    {
        // Use 1000Hz for "unlimited" (effectively vsync limited)
        startTimerHz(1000);
    }
    else
    {
        // Clamp to reasonable range
        fps = juce::jlimit(1, 1000, fps);
        startTimerHz(fps);
    }
}

} // namespace spm

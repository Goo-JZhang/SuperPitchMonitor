#include "SpectrumDisplay.h"
#include "../Utils/Logger.h"

namespace spm {

SpectrumDisplay::SpectrumDisplay()
{
    updateLogFreqRange();
    setOpaque(true);
    
    // Use a slightly longer interval to avoid flickering
    startTimerHz(25);  // 25fps = 40ms interval
}

SpectrumDisplay::~SpectrumDisplay()
{
    stopTimer();
}

void SpectrumDisplay::timerCallback()
{
    frameCount_++;
    if (frameCount_ % skipFrames_ == 0)
    {
        repaint();
    }
}

void SpectrumDisplay::resized()
{
    // Component resized - repaint will be triggered automatically
}

void SpectrumDisplay::updateLogFreqRange()
{
    minLogFreq_ = std::log(std::max(1.0f, minFreq_));
    maxLogFreq_ = std::log(maxFreq_);
}

juce::Rectangle<int> SpectrumDisplay::getPlotArea() const
{
    return getLocalBounds()
        .withTrimmedLeft(leftMargin)
        .withTrimmedRight(rightMargin)
        .withTrimmedTop(topMargin)
        .withTrimmedBottom(bottomMargin);
}

void SpectrumDisplay::setDbRange(float minDb, float maxDb)
{
    fixedMinDb_ = juce::jlimit(-120.0f, 0.0f, minDb);
    fixedMaxDb_ = juce::jlimit(-120.0f, 10.0f, maxDb);
    if (fixedMinDb_ >= fixedMaxDb_) fixedMaxDb_ = fixedMinDb_ + 10;
    
    currentMinDb_ = fixedMinDb_;
    currentMaxDb_ = fixedMaxDb_;
}

void SpectrumDisplay::setFrequencyRange(float minFreq, float maxFreq)
{
    minFreq_ = juce::jlimit(10.0f, 20000.0f, minFreq);
    maxFreq_ = juce::jlimit(10.0f, 20000.0f, maxFreq);
    if (minFreq_ >= maxFreq_) maxFreq_ = minFreq_ * 10;
    updateLogFreqRange();
}

float SpectrumDisplay::freqToX(float freq) const
{
    auto plotArea = getPlotArea();
    
    if (useLogScale_)
    {
        float logFreq = std::log(juce::jlimit(minFreq_, maxFreq_, freq));
        float t = (logFreq - minLogFreq_) / (maxLogFreq_ - minLogFreq_);
        return plotArea.getX() + t * plotArea.getWidth();
    }
    else
    {
        float t = (freq - minFreq_) / (maxFreq_ - minFreq_);
        return plotArea.getX() + t * plotArea.getWidth();
    }
}

float SpectrumDisplay::dbToY(float db) const
{
    auto plotArea = getPlotArea();
    // Allow some overshoot (up to 10% above plot area) to show peaks fully
    float t = (db - currentMinDb_) / (currentMaxDb_ - currentMinDb_);
    t = juce::jlimit(-0.1f, 1.1f, t);  // Allow 10% overshoot at both ends
    return plotArea.getBottom() - t * plotArea.getHeight();
}

float SpectrumDisplay::magnitudeToY(float magnitude) const
{
    float db = 20.0f * std::log10(magnitude + 0.0000001f);
    return dbToY(db);
}

juce::String SpectrumDisplay::freqToNoteName(float freq) const
{
    static const char* noteNames[] = { "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B" };
    
    if (freq <= 0) return "";
    
    float midiNote = 69 + 12 * std::log2(freq / a4Frequency_);
    int noteIndex = (int)std::round(midiNote) % 12;
    if (noteIndex < 0) noteIndex += 12;
    
    int octave = (int)std::round(midiNote - 12) / 12;
    
    return juce::String(noteNames[noteIndex]) + juce::String(octave);
}

juce::String SpectrumDisplay::freqToString(float freq) const
{
    if (freq >= 1000)
        return juce::String(freq/1000, 1) + "k";
    else
        return juce::String((int)freq);
}

void SpectrumDisplay::updateDynamicRange()
{
    if (!autoRange_)
    {
        currentMinDb_ = fixedMinDb_;
        currentMaxDb_ = fixedMaxDb_;
        return;
    }
    
    // Note: updateSpectrum already holds the lock when calling this,
    // so we don't lock here to avoid deadlock
    
    if (currentData_.magnitudes.empty())
        return;
    
    // Find peak dB in current spectrum
    float peakDb = -120.0f;
    for (float mag : currentData_.magnitudes)
    {
        float db = 20.0f * std::log10(mag + 0.0001f);
        peakDb = std::max(peakDb, db);
    }
    
    // Dynamic range logic:
    // - Fixed minimum (floor) at -90dB
    // - Dynamic maximum: follows current peak with headroom
    static constexpr float minDbFloor = -90.0f;  // Fixed floor
    static constexpr float maxDbCeiling = 60.0f;  // Allow up to +60dB (for very strong signals)
    
    // Upper limit follows the actual peak (add headroom for visibility)
    float targetMax = peakDb + 12.0f;  // 12dB headroom above peak
    
    // Clamp to reasonable limits
    targetMax = juce::jlimit(minDbFloor + minRangeDb, maxDbCeiling, targetMax);
    
    float targetMin = minDbFloor;  // Fixed floor at -90dB
    
    // Dynamic range update logic:
    // - If peak exceeds current max: INSTANT attack (immediate jump)
    // - If peak falls below current max: SLOW decay (gradual reduction)
    static constexpr float decayFactor = 0.02f;  // Slow decay rate per frame (~2%)
    
    if (targetMax > currentMaxDb_)
    {
        // INSTANT ATTACK: Peak exceeds current max, immediately adjust
        currentMaxDb_ = targetMax;
    }
    else
    {
        // SLOW DECAY: Peak is lower, gradually reduce max
        currentMaxDb_ += (targetMax - currentMaxDb_) * decayFactor;
    }
    
    // Min dB always follows target directly (or with slow decay if you want)
    currentMinDb_ = targetMin;
}

void SpectrumDisplay::updateSpectrum(const SpectrumData& data)
{
    juce::ScopedLock lock(dataLock_);
    
    if (std::abs(data.sampleRate - currentSampleRate_) > 1.0f)
    {
        currentSampleRate_ = data.sampleRate;
    }
    
    currentData_ = data;
    
    // Update dynamic range
    updateDynamicRange();
    
    // Repaint will happen on next timer callback
}

void SpectrumDisplay::clear()
{
    juce::MessageManager::callAsync([this]() {
        juce::ScopedLock lock(dataLock_);
        currentData_.magnitudes.clear();
        currentData_.mlConfidence.clear();
        currentData_.mlEnergy.clear();
        currentData_.isMLMode = false;
        currentData_.isFFTMode = false;
        repaint();
    });
}

void SpectrumDisplay::paint(juce::Graphics& g)
{
    drawBackground(g);
    drawAxes(g);
    drawGrid(g);
    drawNoteMarkers(g);
    drawSpectrum(g);
    drawModeLabel(g);  // Draw mode indicator on top
}

void SpectrumDisplay::drawBackground(juce::Graphics& g)
{
    juce::ColourGradient gradient(
        juce::Colour(0xFF1A1A20), 0.0f, 0.0f,
        juce::Colour(0xFF0D0D10), 0.0f, (float)getHeight(),
        false
    );
    g.setGradientFill(gradient);
    g.fillAll();
}

void SpectrumDisplay::drawAxes(juce::Graphics& g)
{
    auto plotArea = getPlotArea();
    
    g.setColour(juce::Colours::white.withAlpha(0.6f));
    
    // Determine mode: use preview state if no data yet
    bool isMLMode = currentData_.isMLMode || 
                    (!currentData_.isMLMode && !currentData_.isFFTMode && mlPreviewEnabled_);
    
    // ML Mode: Dual Y-axes
    if (isMLMode)
    {
        // === Left Y Axis: Confidence (0.0, 0.2, 0.4, 0.6, 0.8, 1.0) ===
        g.setColour(juce::Colours::lime.withAlpha(0.6f));
        float confValues[] = {0.0f, 0.2f, 0.4f, 0.6f, 0.8f, 1.0f};
        for (float conf : confValues)
        {
            float y = plotArea.getBottom() - conf * plotArea.getHeight();
            if (y < plotArea.getY() || y > plotArea.getBottom()) continue;
            
            g.drawLine(plotArea.getX() - 5, y, plotArea.getX(), y);
            
            g.setColour(juce::Colours::lime.withAlpha(0.6f));
            // Fixed one decimal place for all values (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
            juce::String label = juce::String(conf, 1);
            g.drawText(label, 2, (int)y - 6, leftMargin - 10, 12, 
                      juce::Justification::centredRight);
            g.setColour(juce::Colours::lime.withAlpha(0.6f));
        }
        
        // Left Y axis title (at top, not overlapping with labels)
        g.setColour(juce::Colours::lime.withAlpha(0.8f));
        g.setFont(10.0f);
        g.drawText("Confidence", 2, plotArea.getY() - 18, leftMargin - 4, 14,
                  juce::Justification::centredRight);
        
        // === Right Y Axis: Energy (0.0, 0.2, 0.4, 0.6, 0.8, 1.0) ===
        g.setColour(juce::Colours::orange.withAlpha(0.6f));
        float energyValues[] = {0.0f, 0.2f, 0.4f, 0.6f, 0.8f, 1.0f};
        for (float e : energyValues)
        {
            float y = plotArea.getBottom() - e * plotArea.getHeight();
            if (y < plotArea.getY() || y > plotArea.getBottom()) continue;
            
            g.drawLine(plotArea.getRight(), y, plotArea.getRight() + 5, y);
            
            g.setColour(juce::Colours::orange.withAlpha(0.6f));
            // Fixed one decimal place for all values
            juce::String label = juce::String(e, 1);
            g.drawText(label, plotArea.getRight() + 8, (int)y - 6, 40, 12, 
                      juce::Justification::centredLeft);
            g.setColour(juce::Colours::orange.withAlpha(0.6f));
        }
        
        // Right Y axis title (at top)
        g.setColour(juce::Colours::orange.withAlpha(0.8f));
        g.setFont(10.0f);
        g.drawText("Energy", plotArea.getRight() + 8, plotArea.getY() - 18, 40, 14,
                  juce::Justification::centredLeft);
    }
    else
    {
        // === FFT Mode: Single Y Axis (dB) ===
        float dbStep = 20.0f;
        for (float db = std::ceil(currentMinDb_ / dbStep) * dbStep; db <= currentMaxDb_; db += dbStep)
        {
            float y = dbToY(db);
            if (y < plotArea.getY() || y > plotArea.getBottom()) continue;
            
            g.drawLine(plotArea.getX() - 5, y, plotArea.getX(), y);
            
            g.setColour(juce::Colours::white.withAlpha(0.5f));
            juce::String label = juce::String((int)db);
            g.drawText(label, 2, (int)y - 6, leftMargin - 8, 12, 
                      juce::Justification::centredRight);
            g.setColour(juce::Colours::white.withAlpha(0.6f));
        }
        
        // Y axis title
        g.setColour(juce::Colours::white.withAlpha(0.5f));
        g.setFont(9.0f);
        g.drawText("dB", 2, plotArea.getY() - 8, leftMargin - 4, 12,
                  juce::Justification::centred);
    }
    
    // === X Axis Labels (Frequency) ===
    if (useLogScale_)
    {
        float freqs[] = { 50, 100, 200, 500, 1000, 2000, 5000 };
        for (float freq : freqs)
        {
            if (freq < minFreq_ || freq > maxFreq_) continue;
            
            float x = freqToX(freq);
            
            g.drawLine(x, plotArea.getBottom(), x, plotArea.getBottom() + 5);
            
            g.setColour(juce::Colours::white.withAlpha(0.5f));
            g.setFont(9.0f);
            juce::String freqText = freqToString(freq);
            g.drawText(freqText, (int)x - 15, plotArea.getBottom() + 8, 30, 12, 
                      juce::Justification::centred);
            g.setColour(juce::Colours::white.withAlpha(0.6f));
        }
    }
    else
    {
        int numDivisions = 5;
        for (int i = 0; i <= numDivisions; ++i)
        {
            float t = (float)i / numDivisions;
            float x = plotArea.getX() + t * plotArea.getWidth();
            float freq = minFreq_ + t * (maxFreq_ - minFreq_);
            
            g.drawLine(x, plotArea.getBottom(), x, plotArea.getBottom() + 5);
            
            g.setColour(juce::Colours::white.withAlpha(0.5f));
            g.setFont(9.0f);
            g.drawText(freqToString(freq), (int)x - 20, plotArea.getBottom() + 8, 40, 12, 
                      juce::Justification::centred);
            g.setColour(juce::Colours::white.withAlpha(0.6f));
        }
    }
    
    // X axis title
    g.setColour(juce::Colours::white.withAlpha(0.5f));
    g.drawText(useLogScale_ ? "Frequency (log)" : "Frequency (linear)", 
              plotArea.getCentreX() - 50, getHeight() - 14, 100, 12,
              juce::Justification::centred);
}

void SpectrumDisplay::drawGrid(juce::Graphics& g)
{
    auto plotArea = getPlotArea();
    
    // Horizontal dB grid lines
    g.setColour(juce::Colours::white.withAlpha(0.08f));
    float dbStep = 20.0f;
    for (float db = std::ceil(currentMinDb_ / dbStep) * dbStep; db <= currentMaxDb_; db += dbStep)
    {
        float y = dbToY(db);
        if (y >= plotArea.getY() && y <= plotArea.getBottom())
        {
            g.drawHorizontalLine((int)y, plotArea.getX(), plotArea.getRight());
        }
    }
    
    // Vertical frequency grid lines
    if (useLogScale_)
    {
        float freqs[] = { 50, 100, 200, 500, 1000, 2000, 5000 };
        for (float freq : freqs)
        {
            if (freq < minFreq_ || freq > maxFreq_) continue;
            float x = freqToX(freq);
            g.drawVerticalLine((int)x, plotArea.getY(), plotArea.getBottom());
        }
    }
    else
    {
        int numDivisions = 5;
        for (int i = 1; i < numDivisions; ++i)
        {
            float t = (float)i / numDivisions;
            float x = plotArea.getX() + t * plotArea.getWidth();
            g.drawVerticalLine((int)x, plotArea.getY(), plotArea.getBottom());
        }
    }
}

void SpectrumDisplay::drawNoteMarkers(juce::Graphics& g)
{
    auto plotArea = getPlotArea();
    
    // Draw note positions on frequency axis
    int minNote = (int)std::floor(12.0f * std::log2(minFreq_ / a4Frequency_) + 69);
    int maxNote = (int)std::ceil(12.0f * std::log2(maxFreq_ / a4Frequency_) + 69);
    
    for (int note = minNote; note <= maxNote; ++note)
    {
        float freq = a4Frequency_ * std::pow(2.0f, (note - 69.0f) / 12.0f);
        if (freq < minFreq_ || freq > maxFreq_) continue;
        
        float x = freqToX(freq);
        bool isC = (note % 12 == 0);
        
        float tickHeight = isC ? 6.0f : 3.0f;
        g.setColour(isC ? juce::Colours::cyan.withAlpha(0.6f) : 
                         juce::Colours::cyan.withAlpha(0.3f));
        g.drawLine(x, plotArea.getBottom() - tickHeight, x, plotArea.getBottom());
        
        if (isC)
        {
            g.setFont(8.0f);
            juce::String noteText = freqToNoteName(freq);
            g.setColour(juce::Colours::cyan.withAlpha(0.5f));
            g.drawText(noteText, (int)x - 10, plotArea.getY() + 2, 20, 10, 
                      juce::Justification::centred);
        }
    }
}

void SpectrumDisplay::drawSpectrum(juce::Graphics& g)
{
    juce::ScopedLock lock(dataLock_);
    
    auto plotArea = getPlotArea();
    
    // ML Mode: Draw confidence and energy with dual Y-axes
    if (currentData_.isMLMode && !currentData_.mlConfidence.empty())
    {
        drawMLSpectrum(g);
        return;
    }
    
    // FFT Mode or legacy mode: Draw single magnitude spectrum
    if (currentData_.magnitudes.empty()) return;
    
    int numBins = (int)currentData_.magnitudes.size();
    float sampleRate = currentData_.sampleRate > 0 ? currentData_.sampleRate : 44100.0f;
    
    // FIX: Correct bin width calculation
    int fftSize = currentData_.fftSize > 0 ? currentData_.fftSize : 2 * (numBins - 1);
    float binWidth = sampleRate / fftSize;
    
    juce::Path spectrumPath;
    bool firstPoint = true;
    
    // Build spectrum path
    for (int i = 0; i < numBins; ++i)
    {
        float freq = (i < (int)currentData_.frequencies.size()) 
                     ? currentData_.frequencies[i] 
                     : i * binWidth;
        if (freq < minFreq_ || freq > maxFreq_) continue;
        
        float x = freqToX(freq);
        float y = magnitudeToY(currentData_.magnitudes[i]);
        
        y = juce::jlimit((float)plotArea.getY(), (float)plotArea.getBottom(), y);
        
        if (firstPoint)
        {
            spectrumPath.startNewSubPath(x, (float)plotArea.getBottom());
            spectrumPath.lineTo(x, y);
            firstPoint = false;
        }
        else
        {
            spectrumPath.lineTo(x, y);
        }
    }
    
    // Close the path at the bottom
    if (!spectrumPath.isEmpty())
    {
        spectrumPath.lineTo(plotArea.getRight(), (float)plotArea.getBottom());
        spectrumPath.closeSubPath();
        
        // Fill with gradient
        juce::ColourGradient fillGradient(
            juce::Colours::cyan.withAlpha(0.7f), plotArea.getX(), plotArea.getY(),
            juce::Colours::cyan.withAlpha(0.1f), plotArea.getX(), plotArea.getBottom(),
            false
        );
        g.setGradientFill(fillGradient);
        g.fillPath(spectrumPath);
        
        // Draw outline
        g.setColour(juce::Colours::cyan.withAlpha(0.9f));
        g.strokePath(spectrumPath, juce::PathStrokeType(1.5f));
    }
}

void SpectrumDisplay::drawMLSpectrum(juce::Graphics& g)
{
    auto plotArea = getPlotArea();
    
    // Leave some padding at top for axis labels
    float topPadding = 20.0f;
    float effectiveHeight = plotArea.getHeight() - topPadding;
    
    int numBins = (int)currentData_.mlConfidence.size();
    if (numBins == 0) return;
    
    // ML model valid frequency range
    const float mlMinFreq = 20.0f;
    const float mlMaxFreq = 5000.0f;
    
    // Left Y-axis: Confidence (0-1, mapped to effective plot area)
    juce::Path confidencePath;
    bool firstPoint = true;
    
    for (int i = 0; i < numBins; ++i)
    {
        if (i >= (int)currentData_.frequencies.size()) continue;
        
        float freq = currentData_.frequencies[i];
        
        // Skip if outside display range
        if (freq < minFreq_ || freq > maxFreq_) continue;
        
        // For ML mode: values outside 20-5000Hz are invalid and should be 0
        float confidence = 0.0f;
        if (freq >= mlMinFreq && freq <= mlMaxFreq && i < (int)currentData_.mlConfidence.size())
        {
            confidence = currentData_.mlConfidence[i];
        }
        
        float x = freqToX(freq);
        float y = plotArea.getBottom() - confidence * effectiveHeight;
        y = juce::jlimit((float)plotArea.getY() + topPadding, (float)plotArea.getBottom(), y);
        
        if (firstPoint)
        {
            confidencePath.startNewSubPath(x, (float)plotArea.getBottom());
            confidencePath.lineTo(x, y);
            firstPoint = false;
        }
        else
        {
            confidencePath.lineTo(x, y);
        }
    }
    
    // Close and fill confidence path
    if (!confidencePath.isEmpty())
    {
        confidencePath.lineTo(plotArea.getRight(), (float)plotArea.getBottom());
        confidencePath.closeSubPath();
        
        // Fill with green gradient (confidence)
        juce::ColourGradient fillGradient(
            juce::Colours::lime.withAlpha(0.5f), plotArea.getX(), plotArea.getY() + topPadding,
            juce::Colours::lime.withAlpha(0.05f), plotArea.getX(), plotArea.getBottom(),
            false
        );
        g.setGradientFill(fillGradient);
        g.fillPath(confidencePath);
        
        // Draw outline
        g.setColour(juce::Colours::lime.withAlpha(0.8f));
        g.strokePath(confidencePath, juce::PathStrokeType(1.5f));
    }
    
    // Right Y-axis: Energy (mapped separately)
    if (!currentData_.mlEnergy.empty())
    {
        juce::Path energyPath;
        firstPoint = true;
        
        // ML Energy is already softmax-normalized (sum=1), display raw values
        // Uniform distribution across 2048 bins: ~0.0005 (will appear as a low line)
        // Single peak can be close to 1.0 (will fill the display)
        // This is the true probability distribution
        
        for (int i = 0; i < numBins && i < (int)currentData_.mlEnergy.size(); ++i)
        {
            if (i >= (int)currentData_.frequencies.size()) continue;
            
            float freq = currentData_.frequencies[i];
            if (freq < minFreq_ || freq > maxFreq_) continue;
            
            // For ML mode: values outside 20-5000Hz are invalid and should be 0
            float energy = 0.0f;
            if (freq >= mlMinFreq && freq <= mlMaxFreq)
            {
                // Use raw softmax output (already in [0,1] range)
                energy = currentData_.mlEnergy[i];
            }
            
            float x = freqToX(freq);
            float y = plotArea.getBottom() - energy * effectiveHeight;
            y = juce::jlimit((float)plotArea.getY() + topPadding, (float)plotArea.getBottom(), y);
            
            if (firstPoint)
            {
                energyPath.startNewSubPath(x, (float)plotArea.getBottom());
                energyPath.lineTo(x, y);
                firstPoint = false;
            }
            else
            {
                energyPath.lineTo(x, y);
            }
        }
        
        // Draw energy outline (no fill to avoid obscuring confidence)
        if (!energyPath.isEmpty())
        {
            g.setColour(juce::Colours::orange.withAlpha(0.9f));
            g.strokePath(energyPath, juce::PathStrokeType(2.0f));
        }
    }
}

void SpectrumDisplay::setTargetRefreshRate(int fps)
{
    targetFPS_ = fps;
    unlimitedFPS_ = (fps < 0);
    
    // Note: SpectrumDisplay doesn't use a timer, it updates on data arrival
    // The FPS limit is effectively applied by the pitch display timer
    // which triggers repaints at the target rate
}

void SpectrumDisplay::drawModeLabel(juce::Graphics& g)
{
    juce::ScopedLock lock(dataLock_);
    
    auto plotArea = getPlotArea();
    
    // Mode indicator in bottom-left corner
    juce::String label;
    juce::Colour color;
    juce::String subtext;
    
    // Determine mode: use preview state if no data yet
    bool hasData = !currentData_.magnitudes.empty() || !currentData_.mlConfidence.empty();
    bool isMLMode = currentData_.isMLMode || (!hasData && mlPreviewEnabled_);
    bool isFFTMode = currentData_.isFFTMode || (!hasData && !mlPreviewEnabled_);
    
    if (isMLMode)
    {
        label = "ML MODE";
        color = juce::Colours::lime;
        subtext = hasData ? "Base frequency distribution only" : "Ready (ML enabled)";
    }
    else if (isFFTMode)
    {
        label = "FFT MODE";
        color = juce::Colours::cyan;
        subtext = hasData ? "Traditional FFT spectrum" : "Ready (FFT Analysis)";
    }
    else
    {
        return; // Unknown mode, don't draw
    }
    
    // Draw semi-transparent background in bottom-left
    int labelWidth = 160;
    int labelHeight = 32;
    int x = plotArea.getX() + 10;
    int y = plotArea.getBottom() - labelHeight - 10;
    
    // More transparent background so spectrum curve shows through
    g.setColour(juce::Colours::black.withAlpha(0.35f));
    g.fillRoundedRectangle(x, y, labelWidth, labelHeight, 6);
    
    // Semi-transparent border
    g.setColour(color.withAlpha(0.5f));
    g.drawRoundedRectangle(x, y, labelWidth, labelHeight, 6, 1.0f);
    
    // Draw main label with slight transparency
    g.setColour(color.withAlpha(0.9f));
    g.setFont(juce::Font(11.0f, juce::Font::bold));
    g.drawText(label, x, y + 3, labelWidth, 14, juce::Justification::centred);
    
    // Draw subtext with more transparency
    g.setColour(juce::Colours::white.withAlpha(0.75f));
    g.setFont(juce::Font(8.0f));
    g.drawText(subtext, x, y + 16, labelWidth, 12, juce::Justification::centred);
}

} // namespace spm

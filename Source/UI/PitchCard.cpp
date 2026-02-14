#include "PitchCard.h"

namespace spm {

PitchCard::PitchCard()
{
    setOpaque(false);
}

void PitchCard::SmoothingState::update(const PitchCandidate& newData)
{
    if (!initialized)
    {
        // First update - initialize with raw values
        smoothedFreq = newData.frequency;
        smoothedCents = newData.centsDeviation;
        smoothedMidi = newData.midiNote;
        smoothedConfidence = newData.confidence;
        initialized = true;
        lastUpdateTime = juce::Time::getMillisecondCounter();
        return;
    }
    
    // Check if pitch jumped significantly (note change)
    float semitoneDiff = std::abs(newData.midiNote - smoothedMidi);
    if (semitoneDiff > resetThresholdSemitones)
    {
        // Large jump - reset smoothing to follow new pitch quickly
        smoothedFreq = newData.frequency;
        smoothedCents = newData.centsDeviation;
        smoothedMidi = newData.midiNote;
        smoothedConfidence = newData.confidence;
    }
    else
    {
        // Small variation - apply EMA smoothing
        smoothedFreq = alpha * newData.frequency + (1.0f - alpha) * smoothedFreq;
        smoothedCents = alpha * newData.centsDeviation + (1.0f - alpha) * smoothedCents;
        smoothedMidi = alpha * newData.midiNote + (1.0f - alpha) * smoothedMidi;
        smoothedConfidence = alpha * newData.confidence + (1.0f - alpha) * smoothedConfidence;
    }
    
    lastUpdateTime = juce::Time::getMillisecondCounter();
}

void PitchCard::SmoothingState::reset()
{
    initialized = false;
    smoothedFreq = 0.0f;
    smoothedCents = 0.0f;
    smoothedMidi = 0.0f;
    smoothedConfidence = 0.0f;
    lastUpdateTime = 0;
}

void PitchCard::setPitchData(const PitchCandidate& data)
{
    {
        juce::ScopedLock lock(lock_);
        
        // Apply temporal smoothing
        smoothing_.update(data);
        
        // Store smoothed data for display
        data_ = data;
        data_.frequency = smoothing_.smoothedFreq;
        data_.centsDeviation = smoothing_.smoothedCents;
        data_.midiNote = smoothing_.smoothedMidi;
        data_.confidence = smoothing_.smoothedConfidence;
    }
    repaint();
}

void PitchCard::clearData()
{
    {
        juce::ScopedLock lock(lock_);
        data_ = PitchCandidate();  // Reset to default
        smoothing_.reset();        // Reset smoothing state
    }
    repaint();
}

void PitchCard::paint(juce::Graphics& g)
{
    juce::ScopedLock lock(lock_);
    
    auto bounds = getLocalBounds().toFloat().reduced(2.0f);
    
    // Background color based on confidence
    float hue = juce::jlimit(0.0f, 0.33f, data_.confidence * 0.33f);
    juce::Colour bgColour = juce::Colour::fromHSV(hue, 0.8f, 0.2f, 1.0f);
    juce::Colour borderColour = juce::Colour::fromHSV(hue, 0.8f, 0.8f, 1.0f);
    
    g.setColour(bgColour);
    g.fillRoundedRectangle(bounds, 6.0f);
    
    g.setColour(borderColour);
    g.drawRoundedRectangle(bounds, 6.0f, 2.0f);
    
    // Note name
    g.setColour(juce::Colours::white);
    g.setFont(20.0f);
    
    int noteIndex = (int)data_.midiNote % 12;
    int octave = (int)data_.midiNote / 12 - 1;
    const char* noteNames[] = {"C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"};
    
    juce::String noteText = juce::String(noteNames[noteIndex]) + juce::String(octave);
    g.drawText(noteText, bounds.removeFromLeft(60).reduced(4), juce::Justification::centredLeft);
    
    // Frequency
    g.setFont(12.0f);
    g.setColour(juce::Colours::lightgrey);
    juce::String freqText = juce::String(data_.frequency, 1) + " Hz";
    g.drawText(freqText, bounds.removeFromTop(20).reduced(4), juce::Justification::centredLeft);
    
    // Cents deviation
    juce::String centsText = (data_.centsDeviation >= 0 ? "+" : "") + 
                              juce::String(data_.centsDeviation, 1) + "c";
    juce::Colour centsColour = std::abs(data_.centsDeviation) < 5.0f ? juce::Colours::green :
                               std::abs(data_.centsDeviation) < 20.0f ? juce::Colours::yellow : 
                               juce::Colours::red;
    g.setColour(centsColour);
    g.drawText(centsText, bounds.removeFromTop(20).reduced(4), juce::Justification::centredLeft);
    
    // dB level (amplitude)
    g.setFont(11.0f);
    float db = 20.0f * std::log10(data_.amplitude + 0.0001f);
    juce::String dbText = juce::String::formatted("%.1f dB", db);
    g.setColour(db > -40.0f ? juce::Colours::lightgreen : 
                db > -60.0f ? juce::Colours::yellow : juce::Colours::grey);
    g.drawText(dbText, bounds.removeFromTop(20).reduced(4), juce::Justification::centredLeft);
    
    // Confidence bar
    float barY = bounds.getCentreY();
    float barWidth = bounds.getWidth() * 0.6f;
    float barHeight = 4.0f;
    float barX = bounds.getX() + 10;
    
    g.setColour(juce::Colours::darkgrey);
    g.fillRoundedRectangle(barX, barY - barHeight/2, barWidth, barHeight, 2.0f);
    
    g.setColour(juce::Colour::fromHSV(data_.confidence * 0.33f, 0.8f, 1.0f, 1.0f));
    g.fillRoundedRectangle(barX, barY - barHeight/2, barWidth * data_.confidence, barHeight, 2.0f);
}

} // namespace spm

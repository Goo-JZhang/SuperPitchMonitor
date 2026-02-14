#include "TunerDisplay.h"

namespace spm {

TunerDisplay::TunerDisplay()
{
    setOpaque(false);
}

TunerDisplay::~TunerDisplay() = default;

void TunerDisplay::paint(juce::Graphics& g)
{
    auto bounds = getLocalBounds().toFloat().reduced(8.0f);
    
    // Background
    g.setColour(juce::Colour(0xFF2A2A35));
    g.fillRoundedRectangle(bounds, 12.0f);
    g.setColour(juce::Colours::white.withAlpha(0.2f));
    g.drawRoundedRectangle(bounds, 12.0f, 1.0f);
    
    drawScale(g);
    drawNeedle(g);
    drawNoteDisplay(g);
}

void TunerDisplay::resized()
{
    repaint();
}

void TunerDisplay::setTargetPitch(float midiNote, float centsDeviation)
{
    targetMidi_ = midiNote;
    centsDeviation_ = centsDeviation;
    repaint();
}

void TunerDisplay::drawScale(juce::Graphics& g)
{
    auto bounds = getLocalBounds().toFloat().reduced(20.0f);
    float centreX = bounds.getCentreX();
    float centreY = bounds.getBottom() - 30;
    float radius = juce::jmin(bounds.getWidth() / 2 - 20, 80.0f);
    
    // Draw scale arc
    juce::Path arc;
    arc.addCentredArc(centreX, centreY, radius, radius * 0.7f, 0, 
                      -2.2f, 2.2f, true);
    g.setColour(juce::Colours::darkgrey);
    g.strokePath(arc, juce::PathStrokeType(4.0f));
    
    // Scale marks
    for (int i = -50; i <= 50; i += 10)
    {
        float angle = i / 50.0f * 2.2f;
        float cosA = std::cos(angle - juce::MathConstants<float>::halfPi);
        float sinA = std::sin(angle - juce::MathConstants<float>::halfPi);
        
        float x1 = centreX + cosA * (radius - 15);
        float y1 = centreY + sinA * (radius - 15) * 0.7f;
        float x2 = centreX + cosA * radius;
        float y2 = centreY + sinA * radius * 0.7f;
        
        bool isCenter = (i == 0);
        g.setColour(isCenter ? juce::Colours::green : juce::Colours::grey);
        g.drawLine(x1, y1, x2, y2, isCenter ? 4.0f : 2.0f);
        
        // Labels
        if (i % 25 == 0)
        {
            float lx = centreX + cosA * (radius + 20);
            float ly = centreY + sinA * (radius + 20) * 0.7f;
            g.setFont(12.0f);
            g.setColour(juce::Colours::lightgrey);
            g.drawText(juce::String(i), (int)lx - 15, (int)ly - 10, 30, 20, 
                       juce::Justification::centred);
        }
    }
}

void TunerDisplay::drawNeedle(juce::Graphics& g)
{
    auto bounds = getLocalBounds().toFloat().reduced(20.0f);
    float centreX = bounds.getCentreX();
    float centreY = bounds.getBottom() - 30;
    float radius = juce::jmin(bounds.getWidth() / 2 - 20, 80.0f);
    
    // Limit deviation range
    float clampedCents = juce::jlimit(-50.0f, 50.0f, centsDeviation_);
    float angle = clampedCents / 50.0f * 2.2f;
    
    float cosA = std::cos(angle - juce::MathConstants<float>::halfPi);
    float sinA = std::sin(angle - juce::MathConstants<float>::halfPi);
    
    float nx = centreX + cosA * (radius - 10);
    float ny = centreY + sinA * (radius - 10) * 0.7f;
    
    // Needle color
    juce::Colour needleColour = std::abs(centsDeviation_) < 5.0f ? juce::Colours::green :
                                std::abs(centsDeviation_) < 20.0f ? juce::Colours::yellow : 
                                juce::Colours::red;
    
    // Draw needle
    g.setColour(needleColour);
    g.drawLine(centreX, centreY, nx, ny, 4.0f);
    
    // Center circle
    g.fillEllipse(centreX - 6, centreY - 6, 12, 12);
    g.setColour(juce::Colours::white);
    g.drawEllipse(centreX - 6, centreY - 6, 12, 12, 2.0f);
}

void TunerDisplay::drawNoteDisplay(juce::Graphics& g)
{
    auto bounds = getLocalBounds().toFloat().reduced(20.0f);
    float centreX = bounds.getCentreX();
    
    // Calculate note name
    int noteIndex = (int)targetMidi_ % 12;
    int octave = (int)targetMidi_ / 12 - 1;
    const char* noteNames[] = {"C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"};
    
    // Note name (large)
    g.setFont(36.0f);
    g.setColour(juce::Colours::white);
    juce::String noteText = juce::String(noteNames[noteIndex]) + juce::String(octave);
    g.drawText(noteText, (int)centreX - 60, 20, 120, 40, juce::Justification::centred);
    
    // Tuning status
    juce::String status;
    juce::Colour statusColour;
    
    if (std::abs(centsDeviation_) < 5.0f)
    {
        status = "IN TUNE";
        statusColour = juce::Colours::green;
    }
    else if (centsDeviation_ > 0)
    {
        status = "SHARP";
        statusColour = juce::Colours::yellow;
    }
    else
    {
        status = "FLAT";
        statusColour = juce::Colours::orange;
    }
    
    g.setFont(14.0f);
    g.setColour(statusColour);
    g.drawText(status, (int)centreX - 50, 60, 100, 20, juce::Justification::centred);
    
    // Precise cents value
    g.setFont(12.0f);
    g.setColour(juce::Colours::lightgrey);
    juce::String centsText = juce::String(centsDeviation_, 1) + " cents";
    g.drawText(centsText, (int)centreX - 50, 78, 100, 16, juce::Justification::centred);
}

} // namespace spm


#include "PitchDisplay.h"

namespace spm {

PitchDisplay::PitchDisplay()
{
    setOpaque(false);
    
    titleLabel_.setText("Detected Pitches", juce::dontSendNotification);
    titleLabel_.setJustificationType(juce::Justification::centred);
    titleLabel_.setColour(juce::Label::textColourId, juce::Colours::white);
    titleLabel_.setFont(16.0f);
    addAndMakeVisible(titleLabel_);
}

PitchDisplay::~PitchDisplay() = default;

void PitchDisplay::paint(juce::Graphics& g)
{
    g.setColour(juce::Colour(0xFF2A2A35));
    g.fillRoundedRectangle(getLocalBounds().toFloat().reduced(4.0f), 8.0f);
    
    g.setColour(juce::Colours::white.withAlpha(0.2f));
    g.drawRoundedRectangle(getLocalBounds().toFloat().reduced(4.0f), 8.0f, 1.0f);
}

void PitchDisplay::resized()
{
    auto bounds = getLocalBounds().reduced(8);
    
    titleLabel_.setBounds(bounds.removeFromTop(30));
    
    int cardHeight = 70;
    int gap = 4;
    
    for (auto& card : pitchCards_)
    {
        if (bounds.getHeight() >= cardHeight)
        {
            card->setBounds(bounds.removeFromTop(cardHeight));
            bounds.removeFromTop(gap);
            card->setVisible(true);
        }
        else
        {
            card->setVisible(false);
        }
    }
}

void PitchDisplay::updatePitches(const PitchVector& pitches)
{
    juce::MessageManager::callAsync([this, pitches]() {
        refreshCards(pitches);
        repaint();
    });
}

void PitchDisplay::refreshCards(const PitchVector& pitches)
{
    // Create enough cards if needed
    while (pitchCards_.size() < pitches.size())
    {
        auto card = std::make_unique<PitchCard>();
        addAndMakeVisible(card.get());
        pitchCards_.push_back(std::move(card));
    }
    
    // Update all cards with new data
    for (size_t i = 0; i < pitches.size(); ++i)
    {
        pitchCards_[i]->setPitchData(pitches[i]);
        pitchCards_[i]->setVisible(true);
    }
    
    // Hide unused cards and clear their data
    for (size_t i = pitches.size(); i < pitchCards_.size(); ++i)
    {
        pitchCards_[i]->setVisible(false);
        pitchCards_[i]->clearData();  // Clear old data
    }
    
    resized();
}

void PitchDisplay::clear()
{
    juce::MessageManager::callAsync([this]() {
        juce::ScopedLock lock(dataLock_);
        for (auto& card : pitchCards_)
        {
            card->setVisible(false);
        }
        repaint();
    });
}

} // namespace spm

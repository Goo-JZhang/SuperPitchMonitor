#include "DebugControlPanel.h"
#include "../Audio/AudioEngine.h"

namespace spm {

DebugControlPanel::DebugControlPanel()
{
    setOpaque(true);
    setVisible(false);
    
    setupUI();
    
    // Start UI update timer
    uiTimer_ = std::make_unique<TimerCallback>(*this);
    uiTimer_->startTimerHz(30);
}

DebugControlPanel::~DebugControlPanel()
{
    simulator_.stop();
    uiTimer_->stopTimer();
}

void DebugControlPanel::setupUI()
{
    // Title
    titleLabel_.setText("Debug Mode", juce::dontSendNotification);
    titleLabel_.setJustificationType(juce::Justification::centred);
    titleLabel_.setFont(24.0f);
    titleLabel_.setColour(juce::Label::textColourId, juce::Colour(0xFFFFD700));  // Gold
    addAndMakeVisible(titleLabel_);
    
    // Close button
    closeButton_.addListener(this);
    addAndMakeVisible(closeButton_);
    
    // Mode selection (File only)
    modeLabel_.setText("Input Mode:", juce::dontSendNotification);
    modeLabel_.setColour(juce::Label::textColourId, juce::Colours::white);
    addAndMakeVisible(modeLabel_);
    
    modeCombo_.addItem("Real Device", 1);
    modeCombo_.addItem("Audio File", 2);
    modeCombo_.setSelectedId(2);  // Default to audio file
    modeCombo_.addListener(this);
    addAndMakeVisible(modeCombo_);
    
    // File chooser
    fileChooser_ = std::make_unique<juce::FilenameComponent>(
        "Audio File",
        juce::File(),
        false, false, false,
        "*.wav;*.mp3;*.aiff;*.flac",
        "*.wav",
        "Select audio file"
    );
    fileChooser_->addListener(this);
    addAndMakeVisible(fileChooser_.get());
    
    // Playback controls
    playButton_.addListener(this);
    playButton_.setColour(juce::TextButton::buttonColourId, juce::Colours::green.withBrightness(0.3f));
    addAndMakeVisible(playButton_);
    
    pauseButton_.addListener(this);
    pauseButton_.setEnabled(false);
    addAndMakeVisible(pauseButton_);
    
    stopButton_.addListener(this);
    stopButton_.setEnabled(false);
    addAndMakeVisible(stopButton_);
    
    loopButton_.setToggleState(true, juce::dontSendNotification);
    loopButton_.setColour(juce::ToggleButton::textColourId, juce::Colours::white);
    addAndMakeVisible(loopButton_);
    
    // Progress bar
    positionSlider_.setRange(0.0, 100.0, 0.1);
    positionSlider_.setValue(0.0);
    positionSlider_.setSliderStyle(juce::Slider::LinearHorizontal);
    positionSlider_.setTextBoxStyle(juce::Slider::NoTextBox, false, 0, 0);
    positionSlider_.addListener(this);
    addAndMakeVisible(positionSlider_);
    
    timeLabel_.setText("0:00 / 0:00", juce::dontSendNotification);
    timeLabel_.setJustificationType(juce::Justification::centred);
    timeLabel_.setColour(juce::Label::textColourId, juce::Colours::white);
    addAndMakeVisible(timeLabel_);
    
    // Playback speed
    speedLabel_.setText("Speed:", juce::dontSendNotification);
    speedLabel_.setColour(juce::Label::textColourId, juce::Colours::white);
    addAndMakeVisible(speedLabel_);
    
    speedSlider_.setRange(0.25, 2.0, 0.25);
    speedSlider_.setValue(1.0);
    speedSlider_.setSliderStyle(juce::Slider::LinearHorizontal);
    speedSlider_.setTextBoxStyle(juce::Slider::TextBoxRight, false, 50, 24);
    speedSlider_.addListener(this);
    addAndMakeVisible(speedSlider_);
    
    // Status label
    statusLabel_.setText("Ready - Select an audio file", juce::dontSendNotification);
    statusLabel_.setJustificationType(juce::Justification::centred);
    statusLabel_.setColour(juce::Label::textColourId, juce::Colours::lightgrey);
    addAndMakeVisible(statusLabel_);
    
    // Setup simulator callbacks
    simulator_.setAudioCallback([this](const juce::AudioBuffer<float>& buffer) {
        // Forward to AudioEngine for processing
        if (audioEngine_)
        {
           #if defined(DEBUG) || defined(_DEBUG)
            DBG("[DebugControlPanel] Audio callback: " << buffer.getNumSamples() << " samples");
           #endif
            audioEngine_->processAudioBlock(buffer);
        }
       #if defined(DEBUG) || defined(_DEBUG)
        else
        {
            DBG("[DebugControlPanel] ERROR: AudioEngine is null in callback!");
        }
       #endif
    });
    
    simulator_.setLevelCallback([this](float level) {
        // Update level display
       #if defined(DEBUG) || defined(_DEBUG)
        static int levelCount = 0;
        if (++levelCount % 50 == 0)
        {
            DBG("[DebugControlPanel] Level callback: " << level);
        }
       #endif
    });
}

void DebugControlPanel::paint(juce::Graphics& g)
{
    // Semi-transparent dark background
    g.fillAll(juce::Colour(0xEE1A1A20));
    
    // Debug panel border
    auto bounds = getLocalBounds().reduced(20).toFloat();
    g.setColour(juce::Colour(0xFFFFD700));  // Gold border
    g.drawRoundedRectangle(bounds, 16.0f, 2.0f);
    
    // Title background
    auto headerBounds = bounds.removeFromTop(50);
    g.setColour(juce::Colour(0x44FFD700));
    g.fillRoundedRectangle(headerBounds.reduced(4.0f), 8.0f);
}

void DebugControlPanel::resized()
{
    auto bounds = getLocalBounds().reduced(30);
    
    // Header
    auto header = bounds.removeFromTop(50);
    closeButton_.setBounds(header.removeFromRight(80).reduced(4));
    titleLabel_.setBounds(header);
    
    bounds.removeFromTop(20);
    
    // Mode selection
    auto modeRow = bounds.removeFromTop(30);
    modeLabel_.setBounds(modeRow.removeFromLeft(100));
    modeCombo_.setBounds(modeRow);
    
    bounds.removeFromTop(15);
    
    // File selection
    if (modeCombo_.getSelectedId() == 2)
    {
        fileChooser_->setBounds(bounds.removeFromTop(30));
        bounds.removeFromTop(15);
    }
    
    // Playback controls
    auto controlRow = bounds.removeFromTop(40);
    playButton_.setBounds(controlRow.removeFromLeft(80).reduced(4));
    pauseButton_.setBounds(controlRow.removeFromLeft(80).reduced(4));
    stopButton_.setBounds(controlRow.removeFromLeft(80).reduced(4));
    loopButton_.setBounds(controlRow.removeFromLeft(80).reduced(4));
    
    bounds.removeFromTop(15);
    
    // Progress bar
    auto progressRow = bounds.removeFromTop(40);
    positionSlider_.setBounds(progressRow.removeFromTop(25));
    timeLabel_.setBounds(progressRow);
    
    bounds.removeFromTop(15);
    
    // Playback speed
    auto speedRow = bounds.removeFromTop(30);
    speedLabel_.setBounds(speedRow.removeFromLeft(80));
    speedSlider_.setBounds(speedRow);
    
    bounds.removeFromTop(20);
    
    // Status
    statusLabel_.setBounds(bounds.removeFromTop(30));
}

void DebugControlPanel::buttonClicked(juce::Button* button)
{
    if (button == &playButton_)
    {
        simulator_.start();
        playButton_.setEnabled(false);
        pauseButton_.setEnabled(true);
        stopButton_.setEnabled(true);
        statusLabel_.setText("Playing...", juce::dontSendNotification);
    }
    else if (button == &pauseButton_)
    {
        simulator_.pause();
        playButton_.setEnabled(true);
        pauseButton_.setEnabled(false);
        statusLabel_.setText("Paused", juce::dontSendNotification);
    }
    else if (button == &stopButton_)
    {
        simulator_.stop();
        playButton_.setEnabled(true);
        pauseButton_.setEnabled(false);
        stopButton_.setEnabled(false);
        statusLabel_.setText("Stopped", juce::dontSendNotification);
    }
    else if (button == &closeButton_)
    {
        simulator_.stop();
        setVisible(false);
    }
}

void DebugControlPanel::comboBoxChanged(juce::ComboBox* comboBox)
{
    if (comboBox == &modeCombo_)
    {
        int mode = modeCombo_.getSelectedId();
        
        // Enable/disable controls
        fileChooser_->setEnabled(mode == 2);
        
        // Stop current playback
        simulator_.stop();
        playButton_.setEnabled(true);
        pauseButton_.setEnabled(false);
        stopButton_.setEnabled(false);
        
        resized();
    }
}

void DebugControlPanel::sliderValueChanged(juce::Slider* slider)
{
    if (slider == &positionSlider_)
    {
        double position = positionSlider_.getValue() / 100.0 * simulator_.getTotalDuration();
        simulator_.setPlayPosition(position);
    }
    else if (slider == &speedSlider_)
    {
        simulator_.setPlaybackSpeed((float)speedSlider_.getValue());
    }
}

void DebugControlPanel::filenameComponentChanged(juce::FilenameComponent* fileComponentThatHasChanged)
{
    if (fileComponentThatHasChanged == fileChooser_.get())
    {
        auto file = fileChooser_->getCurrentFile();
        if (file.existsAsFile())
        {
            if (simulator_.loadAudioFile(file))
            {
                statusLabel_.setText("Loaded: " + file.getFileName(), juce::dontSendNotification);
                updateTimeDisplay();
            }
            else
            {
                statusLabel_.setText("Failed to load: " + file.getFileName(), juce::dontSendNotification);
            }
        }
    }
}

void DebugControlPanel::updateUI()
{
    if (simulator_.isPlaying())
    {
        double progress = 0.0;
        if (simulator_.getTotalDuration() > 0)
        {
            progress = simulator_.getPlayPosition() / simulator_.getTotalDuration() * 100.0;
        }
        positionSlider_.setValue(progress, juce::dontSendNotification);
        
        updateTimeDisplay();
    }
}

void DebugControlPanel::updateTimeDisplay()
{
    auto current = simulator_.getPlayPosition();
    auto total = simulator_.getTotalDuration();
    
    auto formatTime = [](double seconds) -> juce::String {
        int mins = (int)(seconds / 60);
        int secs = (int)(seconds) % 60;
        int ms = (int)((seconds - (int)seconds) * 100);
        return juce::String::formatted("%d:%02d.%02d", mins, secs, ms);
    };
    
    timeLabel_.setText(formatTime(current) + " / " + formatTime(total), 
                       juce::dontSendNotification);
}

} // namespace spm

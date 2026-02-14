#include "InputSourceSelector.h"
#include "../Audio/DeviceAudioInput.h"
#include "../Audio/SystemAudioInput.h"
#include "../Audio/FileAudioInput.h"

namespace spm {

InputSourceSelector::InputSourceSelector()
{
    setOpaque(true);

    // Title
    titleLabel_.setText("Audio Input Source", juce::dontSendNotification);
    titleLabel_.setFont(juce::Font(18.0f, juce::Font::bold));
    titleLabel_.setColour(juce::Label::textColourId, juce::Colours::white);
    addAndMakeVisible(titleLabel_);

    // Source type
    sourceTypeLabel_.setText("Input Type:", juce::dontSendNotification);
    sourceTypeLabel_.setColour(juce::Label::textColourId, juce::Colours::lightgrey);
    addAndMakeVisible(sourceTypeLabel_);

    sourceTypeCombo_.addItem("Audio Device (Microphone)", 1);
    
    // Only add system audio if supported
    if (SystemAudioInput::isSupported())
    {
        sourceTypeCombo_.addItem("System Audio (Loopback)", 2);
    }
    
    sourceTypeCombo_.addItem("File Playback", 3);
    sourceTypeCombo_.setSelectedId(1);
    sourceTypeCombo_.addListener(this);
    addAndMakeVisible(sourceTypeCombo_);

    // Device selector
    deviceLabel_.setText("Device:", juce::dontSendNotification);
    deviceLabel_.setColour(juce::Label::textColourId, juce::Colours::lightgrey);
    addAndMakeVisible(deviceLabel_);

    deviceCombo_.addListener(this);
    addAndMakeVisible(deviceCombo_);

    // File selector
    fileLabel_.setText("Audio File:", juce::dontSendNotification);
    fileLabel_.setColour(juce::Label::textColourId, juce::Colours::lightgrey);
    addAndMakeVisible(fileLabel_);

    fileCombo_.addListener(this);
    addChildComponent(fileCombo_);  // Initially hidden

    refreshButton_.addListener(this);
    addAndMakeVisible(refreshButton_);

    // Status
    statusLabel_.setText("Select an input source", juce::dontSendNotification);
    statusLabel_.setJustificationType(juce::Justification::centred);
    statusLabel_.setColour(juce::Label::textColourId, juce::Colours::grey);
    addAndMakeVisible(statusLabel_);

    // Initial refresh
    refreshSources();
    
    // Create initial source - prefer FilePlayback if test files exist
    if (!availableFiles_.isEmpty())
    {
        sourceTypeCombo_.setSelectedId(3);  // File Playback
        fileLabel_.setVisible(true);
        fileCombo_.setVisible(true);
        deviceLabel_.setVisible(false);
        deviceCombo_.setVisible(false);
        createSource(AudioInputSource::Type::FilePlayback);
    }
    else
    {
        createSource(AudioInputSource::Type::Device);
    }
}

InputSourceSelector::~InputSourceSelector() = default;

void InputSourceSelector::paint(juce::Graphics& g)
{
    g.fillAll(juce::Colour(0xFF2A2A35));
    
    // Border
    g.setColour(juce::Colours::white.withAlpha(0.1f));
    g.drawRect(getLocalBounds(), 1);
}

void InputSourceSelector::resized()
{
    auto bounds = getLocalBounds().reduced(20);
    
    // Title
    titleLabel_.setBounds(bounds.removeFromTop(30));
    bounds.removeFromTop(20);
    
    // Source type
    auto typeRow = bounds.removeFromTop(30);
    sourceTypeLabel_.setBounds(typeRow.removeFromLeft(100));
    sourceTypeCombo_.setBounds(typeRow);
    bounds.removeFromTop(15);
    
    // Device selector
    auto deviceRow = bounds.removeFromTop(30);
    deviceLabel_.setBounds(deviceRow.removeFromLeft(100));
    deviceCombo_.setBounds(deviceRow.removeFromRight(deviceRow.getWidth() - 10));
    bounds.removeFromTop(15);
    
    // File selector
    auto fileRow = bounds.removeFromTop(30);
    fileLabel_.setBounds(fileRow.removeFromLeft(100));
    fileCombo_.setBounds(fileRow.removeFromRight(fileRow.getWidth() - 10));
    bounds.removeFromTop(15);
    
    // Refresh button
    refreshButton_.setBounds(bounds.removeFromTop(30).reduced(4));
    bounds.removeFromTop(20);
    
    // Status
    statusLabel_.setBounds(bounds.removeFromTop(30));
}

void InputSourceSelector::comboBoxChanged(juce::ComboBox* comboBox)
{
    if (comboBox == &sourceTypeCombo_)
    {
        int id = sourceTypeCombo_.getSelectedId();
        
        // Update visibility
        bool isDevice = (id == 1);
        bool isSystem = (id == 2);
        bool isFile = (id == 3);
        
        deviceLabel_.setVisible(isDevice || isSystem);
        deviceCombo_.setVisible(isDevice || isSystem);
        fileLabel_.setVisible(isFile);
        fileCombo_.setVisible(isFile);
        
        // Create new source
        AudioInputSource::Type type;
        switch (id)
        {
            case 1: type = AudioInputSource::Type::Device; break;
            case 2: type = AudioInputSource::Type::SystemAudio; break;
            case 3: type = AudioInputSource::Type::FilePlayback; break;
            default: type = AudioInputSource::Type::Device; break;
        }
        
        createSource(type);
        resized();
    }
    else if (comboBox == &deviceCombo_)
    {
        if (currentSource_ && currentSource_->getType() == AudioInputSource::Type::Device)
        {
            if (auto* deviceInput = dynamic_cast<DeviceAudioInput*>(currentSource_.get()))
            {
                deviceInput->setDevice(deviceCombo_.getText());
            }
        }
    }
    else if (comboBox == &fileCombo_)
    {
        if (currentSource_ && currentSource_->getType() == AudioInputSource::Type::FilePlayback)
        {
            if (auto* fileInput = dynamic_cast<FileAudioInput*>(currentSource_.get()))
            {
                fileInput->loadTestFile(fileCombo_.getText());
                statusLabel_.setText("Loaded: " + fileCombo_.getText(), juce::dontSendNotification);
            }
        }
    }
}

void InputSourceSelector::buttonClicked(juce::Button* button)
{
    if (button == &refreshButton_)
    {
        refreshSources();
    }
}

void InputSourceSelector::refreshSources()
{
    // Refresh devices
    availableDevices_ = AudioInputSource::getAvailableDevices();
    deviceCombo_.clear();
    for (auto& device : availableDevices_)
    {
        deviceCombo_.addItem(device, deviceCombo_.getNumItems() + 1);
    }
    
    // Refresh files
    availableFiles_ = FileAudioInput::getAvailableTestFiles();
    fileCombo_.clear();
    for (auto& file : availableFiles_)
    {
        fileCombo_.addItem(file, fileCombo_.getNumItems() + 1);
    }
    
    // Update status
    if (availableFiles_.isEmpty())
    {
        statusLabel_.setText("No test files found in Resources/TestAudio", juce::dontSendNotification);
        statusLabel_.setColour(juce::Label::textColourId, juce::Colours::orange);
    }
    else
    {
        statusLabel_.setText(juce::String(availableFiles_.size()) + " test files available", juce::dontSendNotification);
        statusLabel_.setColour(juce::Label::textColourId, juce::Colours::lightgreen);
    }
}

void InputSourceSelector::createSource(AudioInputSource::Type type)
{
    std::shared_ptr<AudioInputSource> newSource;
    
    switch (type)
    {
        case AudioInputSource::Type::Device:
            newSource = std::make_shared<DeviceAudioInput>();
            break;
            
        case AudioInputSource::Type::SystemAudio:
            newSource = std::make_shared<SystemAudioInput>();
            break;
            
        case AudioInputSource::Type::FilePlayback:
            newSource = std::make_shared<FileAudioInput>();
            // Load first available file if any
            if (!availableFiles_.isEmpty())
            {
                if (auto* fileInput = dynamic_cast<FileAudioInput*>(newSource.get()))
                {
                    fileInput->loadTestFile(availableFiles_[0]);
                }
            }
            break;
    }
    
    if (newSource)
    {
        currentSource_ = newSource;
        
        if (sourceChangedCallback_)
        {
            // Pass shared_ptr to callback - AudioEngine will share ownership
            sourceChangedCallback_(currentSource_);
        }
        
       #if defined(DEBUG) || defined(_DEBUG)
        DBG("[InputSourceSelector] Created source: " << (int)type);
       #endif
    }
}

} // namespace spm

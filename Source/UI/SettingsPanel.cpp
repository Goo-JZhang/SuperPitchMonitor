#include "SettingsPanel.h"
#include "../Utils/Config.h"
#include "../Utils/Logger.h"
#include "../Audio/DeviceAudioInput.h"
#include "../Audio/SystemAudioInput.h"
#include "../Audio/FileAudioInput.h"

namespace spm {

// =============================================================================
// SettingsLookAndFeel Implementation
// =============================================================================

SettingsLookAndFeel::SettingsLookAndFeel()
{
    // Custom look and feel with smaller menu items for mobile
    
    // Disable menu shadows to avoid VirtualDesktopWatcher crash on Windows
    setColour(juce::PopupMenu::backgroundColourId, juce::Colour(0xFF2A2A35));
    setColour(juce::PopupMenu::highlightedBackgroundColourId, juce::Colour(0xFF3A3A45));
    setColour(juce::PopupMenu::textColourId, juce::Colours::white);
}

void SettingsLookAndFeel::drawPopupMenuBackground(juce::Graphics& g, int width, int height)
{
    // Draw background without shadow
    g.fillAll(findColour(juce::PopupMenu::backgroundColourId));
    
    // Draw border
    g.setColour(juce::Colours::white.withAlpha(0.2f));
    g.drawRect(0, 0, width, height, 1);
}

void SettingsLookAndFeel::getIdealPopupMenuItemSize(const juce::String& text, bool isSeparator,
                                                     int standardMenuItemHeight,
                                                     int& idealWidth, int& idealHeight)
{
    // Call base implementation
    LookAndFeel_V4::getIdealPopupMenuItemSize(text, isSeparator, standardMenuItemHeight,
                                               idealWidth, idealHeight);
    
    // Limit item height for better scrolling on mobile
    if (!isSeparator && idealHeight > 36)
    {
        idealHeight = 36;  // Max 36px per item (~10-11 items visible on screen)
    }
}

bool SettingsLookAndFeel::isPopupMenuNativeActive()
{
    // Force JUCE-drawn menus instead of native Windows menus
    // This avoids the VirtualDesktopWatcher crash
    return false;
}

// =============================================================================
// SettingsContent Implementation
// =============================================================================

SettingsContent::SettingsContent()
{
    // Apply custom look and feel to this component and children
    setLookAndFeel(&lookAndFeel_);
    
    setupComponents();
    loadSettings();
}

SettingsContent::~SettingsContent()
{
    // Remove look and feel before destruction
    setLookAndFeel(nullptr);
}

void SettingsContent::setupComponents()
{
    // Close button
    closeButton_.addListener(this);
    addAndMakeVisible(closeButton_);
    
    // Title
    titleLabel_.setText("Settings", juce::dontSendNotification);
    titleLabel_.setJustificationType(juce::Justification::centred);
    titleLabel_.setColour(juce::Label::textColourId, juce::Colours::white);
    titleLabel_.setFont(24.0f);
    addAndMakeVisible(titleLabel_);
    
    // ===== Input Source Section =====
    inputSourceLabel_.setText("Audio Input Source", juce::dontSendNotification);
    inputSourceLabel_.setFont(16.0f);
    inputSourceLabel_.setColour(juce::Label::textColourId, juce::Colours::cyan);
    addAndMakeVisible(inputSourceLabel_);
    
    sourceTypeLabel_.setText("Input Type:", juce::dontSendNotification);
    sourceTypeLabel_.setColour(juce::Label::textColourId, juce::Colours::lightgrey);
    addAndMakeVisible(sourceTypeLabel_);
    
    sourceTypeCombo_.addItem("Audio Device (Microphone)", 1);
    if (SystemAudioInput::isSupported())
    {
        sourceTypeCombo_.addItem("System Audio (Loopback)", 2);
    }
    sourceTypeCombo_.addItem("File Playback", 3);
    sourceTypeCombo_.setSelectedId(3);
    sourceTypeCombo_.addListener(this);
    addAndMakeVisible(sourceTypeCombo_);
    
    deviceLabel_.setText("Device:", juce::dontSendNotification);
    deviceLabel_.setColour(juce::Label::textColourId, juce::Colours::lightgrey);
    addAndMakeVisible(deviceLabel_);
    deviceCombo_.addListener(this);
    addAndMakeVisible(deviceCombo_);
    
    fileLabel_.setText("Audio File:", juce::dontSendNotification);
    fileLabel_.setColour(juce::Label::textColourId, juce::Colours::lightgrey);
    addAndMakeVisible(fileLabel_);
    fileCombo_.addListener(this);
    addChildComponent(fileCombo_);
    
    refreshButton_.addListener(this);
    addAndMakeVisible(refreshButton_);
    
    // ===== Tuning Section =====
    tuningLabel_.setText("Tuning", juce::dontSendNotification);
    tuningLabel_.setFont(16.0f);
    tuningLabel_.setColour(juce::Label::textColourId, juce::Colours::cyan);
    addAndMakeVisible(tuningLabel_);
    
    a4Slider_.setRange(415.0, 466.0, 0.1);
    a4Slider_.setValue(440.0);
    a4Slider_.setSliderStyle(juce::Slider::LinearHorizontal);
    a4Slider_.setTextBoxStyle(juce::Slider::NoTextBox, false, 0, 0);
    a4Slider_.addListener(this);
    addAndMakeVisible(a4Slider_);
    
    a4Label_.setText("A4 Reference:", juce::dontSendNotification);
    a4Label_.setColour(juce::Label::textColourId, juce::Colours::white);
    addAndMakeVisible(a4Label_);
    
    a4ValueLabel_.setText("440.0 Hz", juce::dontSendNotification);
    a4ValueLabel_.setColour(juce::Label::textColourId, juce::Colours::cyan);
    addAndMakeVisible(a4ValueLabel_);
    
    // ===== Display Section =====
    displayLabel_.setText("Display", juce::dontSendNotification);
    displayLabel_.setFont(16.0f);
    displayLabel_.setColour(juce::Label::textColourId, juce::Colours::cyan);
    addAndMakeVisible(displayLabel_);
    
    logScaleButton_.setButtonText("Log Frequency Scale");
    logScaleButton_.setToggleState(true, juce::dontSendNotification);
    logScaleButton_.addListener(this);
    addAndMakeVisible(logScaleButton_);
    
    showNotesButton_.setButtonText("Show Note Names");
    showNotesButton_.setToggleState(true, juce::dontSendNotification);
    addAndMakeVisible(showNotesButton_);
    
    // ===== Waterfall Display Section =====
    waterfallLabel_.setText("Waterfall Display", juce::dontSendNotification);
    waterfallLabel_.setFont(16.0f);
    waterfallLabel_.setColour(juce::Label::textColourId, juce::Colours::cyan);
    addAndMakeVisible(waterfallLabel_);
    
    timeWindowSlider_.setRange(1.0, 30.0, 0.5);
    timeWindowSlider_.setValue(5.0);
    timeWindowSlider_.setSliderStyle(juce::Slider::LinearHorizontal);
    timeWindowSlider_.setTextBoxStyle(juce::Slider::NoTextBox, false, 0, 0);
    timeWindowSlider_.addListener(this);
    addAndMakeVisible(timeWindowSlider_);
    
    timeWindowLabel_.setText("Time Window:", juce::dontSendNotification);
    timeWindowLabel_.setColour(juce::Label::textColourId, juce::Colours::white);
    addAndMakeVisible(timeWindowLabel_);
    
    timeWindowValueLabel_.setText("5.0 s", juce::dontSendNotification);
    timeWindowValueLabel_.setColour(juce::Label::textColourId, juce::Colours::cyan);
    addAndMakeVisible(timeWindowValueLabel_);
    
    // ===== Spectrum Section =====
    spectrumLabel_.setText("Spectrum", juce::dontSendNotification);
    spectrumLabel_.setFont(16.0f);
    spectrumLabel_.setColour(juce::Label::textColourId, juce::Colours::cyan);
    addAndMakeVisible(spectrumLabel_);
    
    dbMinSlider_.setRange(-120.0, -40.0, 1.0);
    dbMinSlider_.setValue(-90.0);
    dbMinSlider_.setSliderStyle(juce::Slider::LinearHorizontal);
    dbMinSlider_.setTextBoxStyle(juce::Slider::TextBoxRight, false, 50, 24);
    dbMinSlider_.addListener(this);
    addAndMakeVisible(dbMinSlider_);
    
    dbMinLabel_.setText("Min dB:", juce::dontSendNotification);
    dbMinLabel_.setColour(juce::Label::textColourId, juce::Colours::white);
    addAndMakeVisible(dbMinLabel_);
    
    dbMaxSlider_.setRange(-60.0, 0.0, 1.0);
    dbMaxSlider_.setValue(-10.0);
    dbMaxSlider_.setSliderStyle(juce::Slider::LinearHorizontal);
    dbMaxSlider_.setTextBoxStyle(juce::Slider::TextBoxRight, false, 50, 24);
    dbMaxSlider_.addListener(this);
    addAndMakeVisible(dbMaxSlider_);
    
    dbMaxLabel_.setText("Max dB:", juce::dontSendNotification);
    dbMaxLabel_.setColour(juce::Label::textColourId, juce::Colours::white);
    addAndMakeVisible(dbMaxLabel_);
    
    // ===== Pitch Detection Section =====
    pitchLabel_.setText("Pitch Detection", juce::dontSendNotification);
    pitchLabel_.setFont(16.0f);
    pitchLabel_.setColour(juce::Label::textColourId, juce::Colours::cyan);
    addAndMakeVisible(pitchLabel_);
    
    minFreqSlider_.setRange(20, 500, 1);
    minFreqSlider_.setValue(Config::Pitch::MinFrequency);
    minFreqSlider_.setSliderStyle(juce::Slider::LinearHorizontal);
    minFreqSlider_.setTextBoxStyle(juce::Slider::TextBoxRight, false, 60, 24);
    addAndMakeVisible(minFreqSlider_);
    
    minFreqLabel_.setText("Min Freq:", juce::dontSendNotification);
    minFreqLabel_.setColour(juce::Label::textColourId, juce::Colours::white);
    addAndMakeVisible(minFreqLabel_);
    
    maxFreqSlider_.setRange(1000, 8000, 10);
    maxFreqSlider_.setValue(Config::Pitch::MaxFrequency);
    maxFreqSlider_.setSliderStyle(juce::Slider::LinearHorizontal);
    maxFreqSlider_.setTextBoxStyle(juce::Slider::TextBoxRight, false, 60, 24);
    addAndMakeVisible(maxFreqSlider_);
    
    maxFreqLabel_.setText("Max Freq:", juce::dontSendNotification);
    maxFreqLabel_.setColour(juce::Label::textColourId, juce::Colours::white);
    addAndMakeVisible(maxFreqLabel_);
    
    multiToneButton_.setButtonText("Enable Multi-tone Detection");
    multiToneButton_.setToggleState(true, juce::dontSendNotification);
    addAndMakeVisible(multiToneButton_);
    
    multiResButton_.setButtonText("Multi-resolution Analysis (Low/Hi Split)");
    multiResButton_.setToggleState(false, juce::dontSendNotification);
    multiResButton_.addListener(this);
    addAndMakeVisible(multiResButton_);
    
    // ===== Performance Section =====
    performanceLabel_.setText("Performance", juce::dontSendNotification);
    performanceLabel_.setFont(16.0f);
    performanceLabel_.setColour(juce::Label::textColourId, juce::Colours::cyan);
    addAndMakeVisible(performanceLabel_);
    
    fpsLabel_.setText("Display FPS:", juce::dontSendNotification);
    fpsLabel_.setColour(juce::Label::textColourId, juce::Colours::lightgrey);
    addAndMakeVisible(fpsLabel_);
    
    fpsCombo_.addItem("Unlimited", 1);
    fpsCombo_.addItem("144 Hz", 2);
    fpsCombo_.addItem("120 Hz", 3);
    fpsCombo_.addItem("60 Hz", 4);
    fpsCombo_.addItem("30 Hz", 5);
    fpsCombo_.setSelectedId(4);  // Default 60 Hz
    fpsCombo_.addListener(this);
    addAndMakeVisible(fpsCombo_);
    
    // Initial refresh
    refreshSources();
    
    // Set initial visibility based on default selection (File Playback)
    deviceLabel_.setVisible(false);
    deviceCombo_.setVisible(false);
    fileLabel_.setVisible(true);
    fileCombo_.setVisible(true);
}

void SettingsContent::loadSettings()
{
    juce::PropertiesFile::Options options;
    options.applicationName = "SuperPitchMonitor";
    options.filenameSuffix = ".settings";
    options.folderName = "SuperPitchMonitor";
    options.osxLibrarySubFolder = "Application Support";
    
    juce::ApplicationProperties appProps;
    appProps.setStorageParameters(options);
    
    auto* props = appProps.getUserSettings();
    
    float a4 = (float)props->getDoubleValue("a4Frequency", 440.0);
    a4Slider_.setValue(a4, juce::dontSendNotification);
    a4ValueLabel_.setText(juce::String(a4, 1) + " Hz", juce::dontSendNotification);
    
    bool logScale = props->getBoolValue("logScale", true);
    logScaleButton_.setToggleState(logScale, juce::dontSendNotification);
    
    bool showNotes = props->getBoolValue("showNotes", true);
    showNotesButton_.setToggleState(showNotes, juce::dontSendNotification);
    
    float timeWindow = (float)props->getDoubleValue("timeWindow", 5.0);
    timeWindowSlider_.setValue(timeWindow, juce::dontSendNotification);
    timeWindowValueLabel_.setText(juce::String(timeWindow, 1) + " s", juce::dontSendNotification);
    
    float dbMin = (float)props->getDoubleValue("dbMin", -90.0);
    dbMinSlider_.setValue(dbMin, juce::dontSendNotification);
    
    float dbMax = (float)props->getDoubleValue("dbMax", -10.0);
    dbMaxSlider_.setValue(dbMax, juce::dontSendNotification);
}

void SettingsContent::saveSettings()
{
    juce::PropertiesFile::Options options;
    options.applicationName = "SuperPitchMonitor";
    options.filenameSuffix = ".settings";
    options.folderName = "SuperPitchMonitor";
    options.osxLibrarySubFolder = "Application Support";
    
    juce::ApplicationProperties appProps;
    appProps.setStorageParameters(options);
    
    auto* props = appProps.getUserSettings();
    
    props->setValue("a4Frequency", a4Slider_.getValue());
    props->setValue("logScale", logScaleButton_.getToggleState());
    props->setValue("showNotes", showNotesButton_.getToggleState());
    props->setValue("timeWindow", timeWindowSlider_.getValue());
    props->setValue("dbMin", dbMinSlider_.getValue());
    props->setValue("dbMax", dbMaxSlider_.getValue());
    
    appProps.saveIfNeeded();
}

void SettingsContent::refreshSources()
{
    // Refresh devices
    availableDevices_ = AudioInputSource::getAvailableDevices();
    deviceCombo_.clear();
    
    for (auto& device : availableDevices_)
    {
        // Sanitize device name to ensure valid Unicode
        juce::String sanitizedName = sanitizeDeviceName(device);
        deviceCombo_.addItem(sanitizedName, deviceCombo_.getNumItems() + 1);
    }
    
    // Refresh files
    availableFiles_ = FileAudioInput::getAvailableTestFiles();
    fileCombo_.clear();
    for (auto& file : availableFiles_)
    {
        fileCombo_.addItem(file, fileCombo_.getNumItems() + 1);
    }
    
    // Limit dropdown visible rows for better scrolling on mobile
    deviceCombo_.setScrollWheelEnabled(true);
    fileCombo_.setScrollWheelEnabled(true);
}

juce::String SettingsContent::sanitizeDeviceName(const juce::String& name)
{
    // Remove or replace invalid Unicode characters
    juce::String result;
    result.preallocateBytes(name.getNumBytesAsUTF8());
    
    for (int i = 0; i < name.length(); ++i)
    {
        juce::juce_wchar c = name[i];
        
        // Replace control characters except tab/newline
        if (c < 32 && c != '\t' && c != '\n' && c != '\r')
        {
            continue;  // Skip control characters
        }
        
        // Replace high private use surrogate areas that might cause issues
        if (c >= 0xE000 && c <= 0xF8FF)
        {
            continue;  // Skip private use characters
        }
        
        // Keep valid characters
        result += c;
    }
    
    // If result is empty after sanitization, return original to avoid data loss
    if (result.isEmpty() && !name.isEmpty())
    {
        return "Unknown Device";
    }
    
    return result;
}

void SettingsContent::createSource(AudioInputSource::Type type)
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
            sourceChangedCallback_(currentSource_);
        }
        
       SPM_LOG_INFO("[SettingsPanel] Created source: " + newSource->getName());
    }
}

void SettingsContent::paint(juce::Graphics& g)
{
    // Background
    g.fillAll(juce::Colour(0xFF2A2A35));
}

void SettingsContent::resized()
{
    // Calculate total height needed
    int y = 20;
    int sectionSpacing = 15;
    int rowHeight = 40;
    int labelWidth = 100;
    int margin = 30;
    int contentWidth = 540;
    
    // Title and close button
    closeButton_.setBounds(contentWidth - 80, y, 70, 30);
    titleLabel_.setBounds(margin, y, contentWidth - 100, 30);
    y += 50;
    
    // Input Source Section
    inputSourceLabel_.setBounds(margin, y, 200, 25);
    y += 30;
    
    auto typeRow = juce::Rectangle<int>(margin, y, contentWidth - 2*margin, 30);
    sourceTypeLabel_.setBounds(typeRow.removeFromLeft(100));
    sourceTypeCombo_.setBounds(typeRow);
    y += 40;
    
    auto deviceRow = juce::Rectangle<int>(margin, y, contentWidth - 2*margin, 30);
    deviceLabel_.setBounds(deviceRow.removeFromLeft(100));
    deviceCombo_.setBounds(deviceRow.removeFromRight(deviceRow.getWidth() - 10));
    y += 40;
    
    auto fileRow = juce::Rectangle<int>(margin, y, contentWidth - 2*margin, 30);
    fileLabel_.setBounds(fileRow.removeFromLeft(100));
    fileCombo_.setBounds(fileRow.removeFromRight(fileRow.getWidth() - 10));
    y += 40;
    
    refreshButton_.setBounds(margin, y, 100, 30);
    y += 50;
    
    // Tuning Section
    tuningLabel_.setBounds(margin, y, 200, 25);
    y += 30;
    
    auto a4Row = juce::Rectangle<int>(margin, y, contentWidth - 2*margin, 30);
    a4Label_.setBounds(a4Row.removeFromLeft(100));
    a4ValueLabel_.setBounds(a4Row.removeFromRight(60));
    a4Slider_.setBounds(a4Row.reduced(5, 0));
    y += 50;
    
    // Display Section
    displayLabel_.setBounds(margin, y, 200, 25);
    y += 30;
    
    logScaleButton_.setBounds(margin, y, 200, 30);
    y += 35;
    
    showNotesButton_.setBounds(margin, y, 200, 30);
    y += 50;
    
    // Waterfall Section
    waterfallLabel_.setBounds(margin, y, 200, 25);
    y += 30;
    
    auto timeRow = juce::Rectangle<int>(margin, y, contentWidth - 2*margin, 30);
    timeWindowLabel_.setBounds(timeRow.removeFromLeft(100));
    timeWindowValueLabel_.setBounds(timeRow.removeFromRight(60));
    timeWindowSlider_.setBounds(timeRow.reduced(5, 0));
    y += 50;
    
    // Spectrum Section
    spectrumLabel_.setBounds(margin, y, 200, 25);
    y += 30;
    
    auto dbMinRow = juce::Rectangle<int>(margin, y, contentWidth - 2*margin, 30);
    dbMinLabel_.setBounds(dbMinRow.removeFromLeft(70));
    dbMinSlider_.setBounds(dbMinRow.reduced(5, 0));
    y += 40;
    
    auto dbMaxRow = juce::Rectangle<int>(margin, y, contentWidth - 2*margin, 30);
    dbMaxLabel_.setBounds(dbMaxRow.removeFromLeft(70));
    dbMaxSlider_.setBounds(dbMaxRow.reduced(5, 0));
    y += 50;
    
    // Pitch Detection Section
    pitchLabel_.setBounds(margin, y, 200, 25);
    y += 30;
    
    auto minFreqRow = juce::Rectangle<int>(margin, y, contentWidth - 2*margin, 30);
    minFreqLabel_.setBounds(minFreqRow.removeFromLeft(80));
    minFreqSlider_.setBounds(minFreqRow.reduced(5, 0));
    y += 40;
    
    auto maxFreqRow = juce::Rectangle<int>(margin, y, contentWidth - 2*margin, 30);
    maxFreqLabel_.setBounds(maxFreqRow.removeFromLeft(80));
    maxFreqSlider_.setBounds(maxFreqRow.reduced(5, 0));
    y += 40;
    
    multiToneButton_.setBounds(margin, y, 250, 30);
    y += 35;
    
    multiResButton_.setBounds(margin, y, 350, 30);
    y += 50;
    
    // Performance Section
    performanceLabel_.setBounds(margin, y, 200, 25);
    y += 30;
    
    auto fpsRow = juce::Rectangle<int>(margin, y, contentWidth - 2*margin, 30);
    fpsLabel_.setBounds(fpsRow.removeFromLeft(100));
    fpsCombo_.setBounds(fpsRow.removeFromLeft(150));
    y += 50;
    
    // Set total height for scrolling
    setBounds(getX(), getY(), contentWidth, y + 20);
}

void SettingsContent::buttonClicked(juce::Button* button)
{
    if (button == &closeButton_)
    {
        saveSettings();
        // Use callback to notify parent
        if (closeCallback_)
        {
            closeCallback_();
        }
    }
    else if (button == &refreshButton_)
    {
        refreshSources();
    }
    else if (button == &logScaleButton_)
    {
        if (scaleCallback_)
            scaleCallback_(logScaleButton_.getToggleState());
    }
    else if (button == &multiResButton_)
    {
        bool enabled = multiResButton_.getToggleState();
        SPM_LOG_INFO("[Settings] Multi-resolution analysis " + juce::String(enabled ? "enabled" : "disabled"));
        
        if (multiResCallback_)
            multiResCallback_(enabled);
    }
}

void SettingsContent::sliderValueChanged(juce::Slider* slider)
{
    if (slider == &a4Slider_)
    {
        float freq = (float)a4Slider_.getValue();
        a4ValueLabel_.setText(juce::String(freq, 1) + " Hz", juce::dontSendNotification);
        
        if (a4Callback_)
            a4Callback_(freq);
    }
    else if (slider == &timeWindowSlider_)
    {
        float timeWindow = (float)timeWindowSlider_.getValue();
        timeWindowValueLabel_.setText(juce::String(timeWindow, 1) + " s", juce::dontSendNotification);
        
        if (timeWindowCallback_)
            timeWindowCallback_(timeWindow);
    }
    else if (slider == &dbMinSlider_)
    {
        if (dbMinSlider_.getValue() >= dbMaxSlider_.getValue())
            dbMinSlider_.setValue(dbMaxSlider_.getValue() - 5, juce::dontSendNotification);
    }
    else if (slider == &dbMaxSlider_)
    {
        if (dbMaxSlider_.getValue() <= dbMinSlider_.getValue())
            dbMaxSlider_.setValue(dbMinSlider_.getValue() + 5, juce::dontSendNotification);
    }
}

void SettingsContent::comboBoxChanged(juce::ComboBox* comboBox)
{
    if (comboBox == &fpsCombo_)
    {
        int id = fpsCombo_.getSelectedId();
        int fps = 60;  // default
        
        switch (id)
        {
            case 1: fps = -1; break;  // Unlimited
            case 2: fps = 144; break;
            case 3: fps = 120; break;
            case 4: fps = 60; break;
            case 5: fps = 30; break;
        }
        
        if (fpsCallback_)
            fpsCallback_(fps);
    }
    else if (comboBox == &sourceTypeCombo_)
    {
        int id = sourceTypeCombo_.getSelectedId();
        
        // Update visibility
        bool isDevice = (id == 1);
        bool isSystem = (id == 2);
        bool isFile = (id == 3);
        
        // Device dropdown only for Audio Device input
        // System Audio uses default output device (loopback), no selection needed
        deviceLabel_.setVisible(isDevice);
        deviceCombo_.setVisible(isDevice);
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
                juce::String fileName = fileCombo_.getText();
                SPM_LOG_INFO("[SettingsPanel] User selected file: " + fileName);
                fileInput->loadTestFile(fileName);
            }
        }
    }
}

// =============================================================================
// SettingsPanel Implementation (Viewport wrapper)
// =============================================================================

SettingsPanel::SettingsPanel()
{
    setOpaque(true);
    setVisible(false);
    
    // Create content
    content_ = std::make_unique<SettingsContent>();
    
    // Create viewport with scrollbar
    viewport_ = std::make_unique<juce::Viewport>("SettingsViewport");
    viewport_->setViewedComponent(content_.get(), false);
    viewport_->setScrollBarsShown(true, false);  // Vertical only
    viewport_->setScrollOnDragEnabled(true);      // Touch/drag support
    addAndMakeVisible(viewport_.get());
}

SettingsPanel::~SettingsPanel() = default;

void SettingsPanel::paint(juce::Graphics& g)
{
    // Semi-transparent overlay
    g.fillAll(juce::Colour(0xDD1A1A20));
    
    // Panel border
    auto bounds = getLocalBounds().withSizeKeepingCentre(600, 700);
    g.setColour(juce::Colour(0xFF2A2A35));
    g.fillRoundedRectangle(bounds.toFloat(), 16.0f);
    
    g.setColour(juce::Colours::white.withAlpha(0.3f));
    g.drawRoundedRectangle(bounds.toFloat(), 16.0f, 2.0f);
}

void SettingsPanel::resized()
{
    auto bounds = getLocalBounds().withSizeKeepingCentre(600, 700).reduced(10);
    viewport_->setBounds(bounds);
}

void SettingsPanel::setVisible(bool shouldBeVisible)
{
    Component::setVisible(shouldBeVisible);
    if (shouldBeVisible && viewport_)
    {
        viewport_->setViewPosition(0, 0);  // Scroll to top when shown
    }
}

//=============================================================================
// SettingsContent getTargetFPS Implementation
//=============================================================================

int SettingsContent::getTargetFPS() const
{
    int id = fpsCombo_.getSelectedId();
    switch (id)
    {
        case 1: return -1;   // Unlimited
        case 2: return 144;
        case 3: return 120;
        case 4: return 60;   // Default
        case 5: return 30;
        default: return 60;
    }
}

void SettingsContent::setMultiResolutionEnabled(bool enabled)
{
    multiResButton_.setToggleState(enabled, juce::dontSendNotification);
    if (multiResCallback_)
        multiResCallback_(enabled);
}

} // namespace spm

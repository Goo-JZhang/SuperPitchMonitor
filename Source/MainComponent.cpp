#include "MainComponent.h"
#include "Utils/Logger.h"
#include "Audio/FileAudioInput.h"

namespace spm {

MainComponent::MainComponent()
{
    // Initialize file logger - it will auto-detect project root and use Saved/Logs
    
    // Install assertion handler for crash logging
    installAssertionHandler();
    
    SPM_LOG_INFO("========================================");
    SPM_LOG_INFO("Application Starting");
    SPM_LOG_INFO("[MainComponent] Constructor start");
    
    setupUI();
    setupAudio();
    connectSettingsCallbacks();
    
    // Start TCP test server for cross-platform testing
    fprintf(stderr, "[MainComponent] Starting TestServer on port 9999...\n");
    testServer_ = std::make_unique<TestServer>();
    testServer_->setAudioEngine(audioEngine_.get());
    testServer_->setMainComponent(this);
    if (testServer_->start(9999))
    {
        fprintf(stderr, "[MainComponent] TestServer started successfully on port 9999\n");
        SPM_LOG_INFO("[MainComponent] TestServer started on port 9999");
    }
    else
    {
        fprintf(stderr, "[MainComponent] Failed to start TestServer on port 9999\n");
        SPM_LOG_ERROR("[MainComponent] Failed to start TestServer on port 9999");
    }
    
    setSize(1400, 900);
    
    // Request audio permission
    Platform::requestPermission(Platform::Permission::AudioInput,
        [this](bool granted)
        {
            if (granted)
            {
                setupAudio();
                statusLabel_.setText("Ready", juce::dontSendNotification);
            }
            else
            {
                handlePermissionDenied();
            }
        }
    );
    
    SPM_LOG_INFO("[MainComponent] Constructor complete");
}

MainComponent::~MainComponent()
{
    // Stop test server
    if (testServer_)
    {
        testServer_->stop();
        testServer_.reset();
    }
}

void MainComponent::timerCallback()
{
    // Timer callback - FPS counter and UI updates only
    // TestServer runs in its own thread
}

void MainComponent::setupUI()
{
    setOpaque(true);
    
    // Dark theme
    getLookAndFeel().setColour(juce::ResizableWindow::backgroundColourId, 
                               juce::Colour(0xFF1A1A20));
    
    // Main pitch waterfall display (top-left area)
    pitchWaterfall_ = std::make_unique<PitchWaterfallDisplay>();
    addAndMakeVisible(pitchWaterfall_.get());
    
    // Right panel - detected pitches list (same height as waterfall)
    pitchDisplay_ = std::make_unique<PitchDisplay>();
    addAndMakeVisible(pitchDisplay_.get());
    
    // Bottom - real-time spectrum (full width below waterfall+pitch list)
    spectrumDisplay_ = std::make_unique<SpectrumDisplay>();
    addAndMakeVisible(spectrumDisplay_.get());
    
    // Settings panel (initially hidden)
    settingsPanel_ = std::make_unique<SettingsPanel>();
    addChildComponent(settingsPanel_.get());
    
    // Control buttons
    settingsButton_.addListener(this);
    addAndMakeVisible(settingsButton_);
    
    startStopButton_.addListener(this);
    startStopButton_.setColour(juce::TextButton::buttonColourId, 
                               juce::Colours::green.withBrightness(0.4f));
    addAndMakeVisible(startStopButton_);
    
    // Status label
    statusLabel_.setText("Initializing...", juce::dontSendNotification);
    statusLabel_.setJustificationType(juce::Justification::left);
    statusLabel_.setColour(juce::Label::textColourId, juce::Colours::white);
    addAndMakeVisible(statusLabel_);
    
    // Input level label
    inputLevelLabel_.setText("Input: -inf dB", juce::dontSendNotification);
    inputLevelLabel_.setJustificationType(juce::Justification::right);
    inputLevelLabel_.setColour(juce::Label::textColourId, juce::Colours::lightgrey);
    addAndMakeVisible(inputLevelLabel_);
    
    // FPS label
    fpsLabel_.setText("FPS: --", juce::dontSendNotification);
    fpsLabel_.setJustificationType(juce::Justification::right);
    fpsLabel_.setColour(juce::Label::textColourId, juce::Colours::cyan);
    addAndMakeVisible(fpsLabel_);
}

void MainComponent::connectSettingsCallbacks()
{
    // Close button handler
    settingsPanel_->onClose([this]() {
        settingsPanel_->setVisible(false);
        
        // Restore all background components visibility
        if (pitchWaterfall_) pitchWaterfall_->setVisible(true);
        if (spectrumDisplay_) spectrumDisplay_->setVisible(true);
        if (pitchDisplay_) pitchDisplay_->setVisible(true);
        statusLabel_.setVisible(true);
        inputLevelLabel_.setVisible(true);
        fpsLabel_.setVisible(true);
        settingsButton_.setVisible(true);
        startStopButton_.setVisible(true);
        
        // Force full repaint to clear any artifacts
        repaint();
    });
    
    // Visibility change handler - hide/show background components completely
    settingsPanel_->onVisibilityChanged([this](bool isVisible) {
        // Hide/show all background components to prevent ghosting
        if (pitchWaterfall_) pitchWaterfall_->setVisible(!isVisible);
        if (spectrumDisplay_) spectrumDisplay_->setVisible(!isVisible);
        if (pitchDisplay_) pitchDisplay_->setVisible(!isVisible);
        if (isVisible)
        {
            statusLabel_.setVisible(false);
            inputLevelLabel_.setVisible(false);
            fpsLabel_.setVisible(false);
            settingsButton_.setVisible(false);
            startStopButton_.setVisible(false);
        }
        else
        {
            statusLabel_.setVisible(true);
            inputLevelLabel_.setVisible(true);
            fpsLabel_.setVisible(true);
            settingsButton_.setVisible(true);
            startStopButton_.setVisible(true);
            // Force full repaint after restoring visibility
            repaint();
        }
    });
    
    auto* content = settingsPanel_->getContent();
    
    // A4 frequency changes
    content->onA4FrequencyChanged([this](float freq) {
        if (pitchWaterfall_)
            pitchWaterfall_->setA4Frequency(freq);
        if (spectrumDisplay_)
            spectrumDisplay_->setA4Frequency(freq);
        // A4 frequency changed - no log needed
    });
    
    // Scale mode changes
    content->onScaleModeChanged([this](bool useLog) {
        if (spectrumDisplay_)
            spectrumDisplay_->setLogFrequencyScale(useLog);
        // Log scale changed - no log needed
    });
    
    // Time window changes
    content->onTimeWindowChanged([this](float seconds) {
        if (pitchWaterfall_)
            pitchWaterfall_->setTimeWindow(seconds);
        // Time window changed - no log needed
    });
    
    // Input source changes
    content->onSourceChanged([this](std::shared_ptr<AudioInputSource> newSource) {
        if (newSource)
        {
            audioEngine_->setInputSource(newSource);
            SPM_LOG_INFO("[Settings] Input source changed to: " + newSource->getName());
            statusLabel_.setText("Source: " + newSource->getName() + " - Press Start", 
                                juce::dontSendNotification);
        }
    });
    
    // FPS changes
    content->onFPSChanged([this](int fps) {
        targetRefreshRate_ = fps;
        unlimitedFPS_ = (fps < 0);
        
        if (pitchWaterfall_)
        {
            pitchWaterfall_->setTargetRefreshRate(fps);
        }
        if (spectrumDisplay_)
        {
            spectrumDisplay_->setTargetRefreshRate(fps);
        }
        
        // Target FPS changed - no log needed
    });
    
    // Buffer Size changes
    content->onBufferSizeChanged([this](int bufferSize) {
        if (audioEngine_)
        {
            audioEngine_->setBufferSize(bufferSize);
        }
        float latencyMs = (bufferSize / 44100.0f) * 1000.0f;
        // Buffer size changed - no log needed
    });
    
    // ML Analysis toggle
    content->onMLAnalyzeChanged([this](bool enabled) {
        if (audioEngine_)
        {
            audioEngine_->setMLAnalysisEnabled(enabled);
        }
        if (spectrumDisplay_)
        {
            spectrumDisplay_->setMLPreviewEnabled(enabled);
        }
        SPM_LOG_INFO("[Settings] ML Analysis: " + juce::String(enabled ? "enabled" : "disabled"));
    });
    
    // ML GPU/CPU mode toggle
    content->onMLModeChanged([this](bool useGPU) {
        if (audioEngine_)
        {
            audioEngine_->setMLGPUEnabled(useGPU);
        }
        SPM_LOG_INFO("[Settings] ML Mode: " + juce::String(useGPU ? "GPU" : "CPU"));
    });
    
    // ML Model selection
    content->onMLModelChanged([this](const juce::String& modelPath) {
        SPM_LOG_INFO("[Settings] ML Model selected: " + modelPath);
        if (audioEngine_)
        {
            // Reload ML detector with new model
            audioEngine_->setMLModelPath(modelPath);
        }
    });
    
}

void MainComponent::setupAudio()
{
   #if defined(DEBUG) || defined(_DEBUG)
    DBG("[MainComponent] setupAudio start");
   #endif
    SPM_LOG_INFO("[MainComponent] setupAudio start");
    
    audioEngine_ = std::make_unique<AudioEngine>();
    
    // Set callbacks
    audioEngine_->setSpectrumCallback(
        [this](const SpectrumData& data) { onSpectrumData(data); });
    
    audioEngine_->setPitchCallback(
        [this](const PitchVector& pitches) { onPitchDetected(pitches); });
    
    audioEngine_->setInputLevelCallback(
        [this](float level) { onInputLevel(level); });
    
    // Initialize with default file source - C Major 7 piano chord for testing
    auto defaultSource = std::make_shared<FileAudioInput>();
    juce::String targetFile = "chord_c_major_7_piano.wav";
    
    if (defaultSource->loadTestFile(targetFile))
    {
        audioEngine_->setInputSource(defaultSource);
        statusLabel_.setText("Loaded: " + targetFile + " - Press Start", 
                            juce::dontSendNotification);
        SPM_LOG_INFO("[MainComponent] Loaded test file: " + targetFile);
    }
    else
    {
        // Fallback to first available file
        auto testFiles = FileAudioInput::getAvailableTestFiles();
        if (!testFiles.isEmpty())
        {
            if (defaultSource->loadTestFile(testFiles[0]))
            {
                audioEngine_->setInputSource(defaultSource);
                statusLabel_.setText("Loaded: " + testFiles[0] + " - Press Start", 
                                    juce::dontSendNotification);
                SPM_LOG_INFO("[MainComponent] Loaded fallback test file: " + testFiles[0]);
            }
            else
            {
                statusLabel_.setText("Failed to load test file - Press Start", juce::dontSendNotification);
            }
        }
        else
        {
            statusLabel_.setText("No test files found - Press Start", juce::dontSendNotification);
        }
    }
    
    // Sync initial settings
    if (settingsPanel_)
    {
        auto* content = settingsPanel_->getContent();
        float a4 = content->getA4Frequency();
        if (pitchWaterfall_) pitchWaterfall_->setA4Frequency(a4);
        if (spectrumDisplay_) spectrumDisplay_->setA4Frequency(a4);
        if (spectrumDisplay_) spectrumDisplay_->setLogFrequencyScale(content->getUseLogScale());
        if (pitchWaterfall_) pitchWaterfall_->setTimeWindow(content->getTimeWindow());
        
        // Sync FPS settings
        int fps = content->getTargetFPS();
        targetRefreshRate_ = fps;
        unlimitedFPS_ = (fps < 0);
        if (pitchWaterfall_) pitchWaterfall_->setTargetRefreshRate(fps);
        if (spectrumDisplay_) spectrumDisplay_->setTargetRefreshRate(fps);
        
        // Auto-enable multi-resolution for testing
        content->setMLAnalysisEnabled(true);
        audioEngine_->setMLAnalysisEnabled(true);
        if (spectrumDisplay_) spectrumDisplay_->setMLPreviewEnabled(true);
        SPM_LOG_INFO("[MainComponent] Auto-enabled multi-resolution analysis");
    }
    
    SPM_LOG_INFO("[MainComponent] setupAudio complete");
    
    // TestServer is already started in constructor
    
    // Start timer for processing test commands (100Hz)
    startTimer(10);
}

void MainComponent::paint(juce::Graphics& g)
{
    // If settings panel is visible, don't draw background (let panel handle it)
    if (settingsPanel_ && settingsPanel_->isVisible())
        return;
    
    // Dark gradient background
    juce::ColourGradient gradient(
        juce::Colour(0xFF1A1A20), 0.0f, 0.0f,
        juce::Colour(0xFF0D0D12), 0.0f, (float)getHeight(),
        false
    );
    g.setGradientFill(gradient);
    g.fillAll();
    
    // Draw separator lines
    g.setColour(juce::Colours::white.withAlpha(0.1f));
    
    auto bounds = getLocalBounds();
    bounds.removeFromTop(statusBarHeight);
    
    // Line between waterfall and pitch list (vertical)
    int waterfallRight = bounds.getWidth() - pitchListWidth;
    g.drawVerticalLine(waterfallRight, statusBarHeight, 
                       (float)(getHeight() - spectrumHeight));
    
    // Line between top section and spectrum (horizontal)
    int spectrumTop = getHeight() - spectrumHeight;
    g.drawHorizontalLine(spectrumTop, 0.0f, (float)getWidth());
}

void MainComponent::resized()
{
    auto bounds = getLocalBounds();
    
    // Settings panel (fullscreen overlay) - MUST use original bounds, not modified
    settingsPanel_->setBounds(getLocalBounds());
    
    // Status bar at top
    auto statusBar = bounds.removeFromTop(statusBarHeight);
    
    int buttonWidth = 80;
    startStopButton_.setBounds(statusBar.removeFromLeft(buttonWidth).reduced(4));
    settingsButton_.setBounds(statusBar.removeFromLeft(buttonWidth).reduced(4));
    
    statusLabel_.setBounds(statusBar.removeFromLeft(350).reduced(4));
    fpsLabel_.setBounds(statusBar.removeFromRight(80).reduced(4));
    inputLevelLabel_.setBounds(statusBar.removeFromRight(150).reduced(4));
    
    // Calculate areas
    int mainContentHeight = bounds.getHeight() - spectrumHeight;
    
    // Right side - pitch list (top section only)
    auto pitchListArea = bounds.removeFromRight(pitchListWidth);
    pitchListArea.setHeight(mainContentHeight);
    pitchDisplay_->setBounds(pitchListArea);
    
    // Bottom - spectrum (full width)
    auto spectrumArea = bounds.removeFromBottom(spectrumHeight);
    // Extend spectrum to full width
    spectrumArea.setX(0);
    spectrumArea.setWidth(getWidth());
    spectrumDisplay_->setBounds(spectrumArea);
    
    // Main area - pitch waterfall (remaining top section)
    pitchWaterfall_->setBounds(bounds.removeFromTop(mainContentHeight));
}

void MainComponent::buttonClicked(juce::Button* button)
{
    if (button == &startStopButton_)
    {
        if (audioEngine_->isRunning())
        {
            audioEngine_->stop();
            startStopButton_.setButtonText("Start");
            startStopButton_.setColour(juce::TextButton::buttonColourId, 
                                       juce::Colours::green.withBrightness(0.4f));
            // Status label shows input source info only (no Running/Stopped)
            updateStatusLabel();
            
            // Clear displays when stopped
            if (pitchWaterfall_)
                pitchWaterfall_->clear();
            if (pitchDisplay_)
                pitchDisplay_->clear();
            if (spectrumDisplay_)
                spectrumDisplay_->clear();
            
            // Reset input level and FPS display
            inputLevelLabel_.setText("Input: -", juce::dontSendNotification);
            fpsLabel_.setText("FPS: -", juce::dontSendNotification);
        }
        else
        {
            // ===== Output current Settings configuration =====
            SPM_LOG_INFO("========================================");
            SPM_LOG_INFO("[START] Input: " + audioEngine_->getInputSourceName() + 
                        " | ML: " + juce::String(audioEngine_->isMLAnalysisEnabled() ? "ON" : "OFF"));
            
            audioEngine_->start();
            startStopButton_.setButtonText("Stop");
            startStopButton_.setColour(juce::TextButton::buttonColourId, 
                                       juce::Colours::red.withBrightness(0.4f));
            
            // Reset auto-tracker cooldown so tracking starts immediately
            if (pitchWaterfall_)
                pitchWaterfall_->resetAutoTrackerCooldown();
            
            // Status label shows input source info only (no Running/Stopped)
            updateStatusLabel();
        }
    }
    else if (button == &settingsButton_)
    {
        // Show settings panel as overlay
        settingsPanel_->setVisible(true);
        settingsPanel_->toFront(true);
    }
}

void MainComponent::updateStatusLabel()
{
    if (audioEngine_)
    {
        juce::String sourceName = audioEngine_->getInputSourceName();
        statusLabel_.setText("Source: " + sourceName, juce::dontSendNotification);
    }
}

void MainComponent::onSpectrumData(const SpectrumData& data)
{
    static int count = 0;
    if (++count % 30 == 0)
    {
        // Spectrum data received - no detailed log
    }
    
    // Spectrum data is used by UI only, TestServer gets data through pitch detection

    // Record frame for FPS calculation
    fpsCounter_.recordFrame();
    
    // Update FPS display every 30 frames (~0.5s at 60fps)
    static int fpsUpdateCount = 0;
    if (++fpsUpdateCount % 30 == 0)
    {
        float fps = fpsCounter_.getAverageFPS();
        juce::MessageManager::callAsync([this, fps]() {
            fpsLabel_.setText(juce::String::formatted("FPS: %.0f", fps), juce::dontSendNotification);
        });
    }
    
    // Send to both displays
    if (pitchWaterfall_)
        pitchWaterfall_->updateSpectrum(data);
    if (spectrumDisplay_)
        spectrumDisplay_->updateSpectrum(data);
}

void MainComponent::onPitchDetected(const PitchVector& pitches)
{
    // Update test server with results
    if (testServer_)
    {
        std::vector<TestServer::DetectionResult> results;
        for (const auto& p : pitches)
        {
            TestServer::DetectionResult r;
            r.frequency = p.frequency;
            r.midiNote = p.midiNote;
            r.confidence = p.confidence;
            r.centsDeviation = p.centsDeviation;
            r.harmonicCount = p.harmonicCount;
            r.timestamp = juce::Time::getCurrentTime().toMilliseconds();
            results.push_back(r);
        }
        testServer_->updateResults(results);
        testServer_->incrementFrameCount();
    }
    
    if (pitches.empty()) return;
    
    // Find strongest pitch for logging
    auto strongest = *std::max_element(pitches.begin(), pitches.end(),
        [](const PitchCandidate& a, const PitchCandidate& b) {
            return a.confidence < b.confidence;
        });
    
    SPM_LOG_INFO("[UI] " + juce::String(pitches.size()) + " pitches detected, strongest=" 
                + juce::String(strongest.frequency, 1) + "Hz, "
                + "midi=" + juce::String(strongest.midiNote, 1) + ", "
                + "conf=" + juce::String(strongest.confidence, 2));
    juce::MessageManager::callAsync([this, pitches]() {
        // Update pitch list
        if (pitchDisplay_)
            pitchDisplay_->updatePitches(pitches);
        
        // Update waterfall with ALL detected pitches (for polyphonic display)
        if (pitchWaterfall_)
        {
            pitchWaterfall_->updatePitches(pitches);
        }
    });
}

void MainComponent::onInputLevel(float level)
{
    juce::MessageManager::callAsync([this, level]() {
        juce::String text;
        if (level > 0.0001f)
        {
            float dB = 20.0f * std::log10(level);
            text = juce::String::formatted("Input: %.1f dB", dB);
        }
        else
        {
            text = "Input: -inf dB";
        }
        inputLevelLabel_.setText(text, juce::dontSendNotification);
    });
}

void MainComponent::handlePermissionDenied()
{
    statusLabel_.setText("Microphone permission denied - Check Settings", juce::dontSendNotification);
}

//=============================================================================
// FPS Counter Implementation
//=============================================================================

void MainComponent::FPSCounter::recordFrame()
{
    double now = juce::Time::getMillisecondCounterHiRes();
    
    if (lastFrameTime > 0)
    {
        float delta = static_cast<float>(now - lastFrameTime);  // in ms
        frameTimes[index] = delta;
        index = (index + 1) % historySize;
        if (count < historySize) count++;
    }
    
    lastFrameTime = now;
}

float MainComponent::FPSCounter::getAverageFPS() const
{
    if (count == 0) return 0.0f;
    
    float sum = 0.0f;
    for (int i = 0; i < count; ++i)
    {
        sum += frameTimes[i];
    }
    
    float avgDelta = sum / count;  // average ms per frame
    if (avgDelta < 0.1f) return 0.0f;
    
    return 1000.0f / avgDelta;  // convert to FPS
}

void MainComponent::FPSCounter::reset()
{
    index = 0;
    count = 0;
    lastFrameTime = 0;
    for (int i = 0; i < historySize; ++i) frameTimes[i] = 0;
}

} // namespace spm

#include <juce_core/juce_core.h>
#include <juce_data_structures/juce_data_structures.h>
#include <juce_events/juce_events.h>
#include <juce_graphics/juce_graphics.h>
#include <juce_gui_basics/juce_gui_basics.h>
#include <juce_gui_extra/juce_gui_extra.h>
#include <juce_audio_basics/juce_audio_basics.h>
#include <juce_audio_devices/juce_audio_devices.h>
#include <juce_audio_formats/juce_audio_formats.h>
#include <juce_audio_processors/juce_audio_processors.h>
#include <juce_audio_utils/juce_audio_utils.h>
#include <juce_dsp/juce_dsp.h>
#include "MainComponent.h"
#include "Utils/PlatformUtils.h"
#include "Utils/Logger.h"
#include "Test/TestServer.h"
#include "Audio/AudioEngine.h"

#include <exception>
#include <csignal>

// Crash handler
void crashHandler(int sig)
{
    fprintf(stderr, "[CRASH] Signal %d received\n", sig);
    
    // Try to log to file
    juce::File logDir = juce::File::getSpecialLocation(juce::File::tempDirectory).getChildFile("SPM_Crash");
    logDir.createDirectory();
    
    juce::File crashLog = logDir.getChildFile("crash_" + juce::Time::getCurrentTime().formatted("%Y%m%d_%H%M%S") + ".log");
    juce::String crashInfo = "Crash signal: " + juce::String(sig) + "\n";
    crashInfo += "Time: " + juce::Time::getCurrentTime().toString(true, true) + "\n";
    
    crashLog.replaceWithText(crashInfo);
    fprintf(stderr, "[CRASH] Crash log written to: %s\n", crashLog.getFullPathName().toRawUTF8());
    
    exit(1);
}

void terminateHandler()
{
    fprintf(stderr, "[CRASH] std::terminate called\n");
    
    juce::File logDir = juce::File::getSpecialLocation(juce::File::tempDirectory).getChildFile("SPM_Crash");
    logDir.createDirectory();
    
    juce::File crashLog = logDir.getChildFile("terminate_" + juce::Time::getCurrentTime().formatted("%Y%m%d_%H%M%S") + ".log");
    crashLog.replaceWithText("std::terminate called at " + juce::Time::getCurrentTime().toString(true, true));
    
    exit(1);
}

/**
 * SuperPitchMonitor Application Entry Point
 * 
 * Unified cross-platform testing architecture:
 * - Normal mode: GUI with TestServer enabled (for external testing)
 * - Test mode (-TestMode): Headless, only TestServer + AudioEngine
 */

class SuperPitchMonitorApp : public juce::JUCEApplication
{
public:
    SuperPitchMonitorApp() {}

    const juce::String getApplicationName() override
    {
        return "SuperPitchMonitor";
    }

    const juce::String getApplicationVersion() override
    {
        return "1.0.0";
    }

    bool moreThanOneInstanceAllowed() override
    {
        return false;
    }

    void initialise(const juce::String& commandLine) override
    {
        fprintf(stderr, "[SPM] initialise() called\n");
        fprintf(stderr, "[SPM] Command line: %s\n", commandLine.toRawUTF8());
        
        // Initialize logger - it will auto-detect project root and use Saved/Logs
        // If no explicit directory is set, Logger will find project root by looking for CMakeLists.txt
        auto exeDir = juce::File::getSpecialLocation(juce::File::currentExecutableFile)
                          .getParentDirectory();
        auto parentDir = exeDir.getParentDirectory();
        auto grandParentDir = parentDir.getParentDirectory();
        
        // Try to find project root (contains CMakeLists.txt)
        juce::File projectRoot = exeDir;
        for (int i = 0; i < 5; ++i) {
            if (projectRoot.getChildFile("CMakeLists.txt").existsAsFile())
                break;
            projectRoot = projectRoot.getParentDirectory();
        }
        
        // Use Saved/Logs relative to project root
        juce::File logDir;
        if (projectRoot.getChildFile("CMakeLists.txt").existsAsFile())
            logDir = projectRoot.getChildFile("Saved/Logs");
        else
            logDir = exeDir.getChildFile("Saved/Logs");
        
        spm::FileLogger::getInstance().initialize(logDir);
        fprintf(stderr, "[SPM] Logger initialized, logs at: %s\n", logDir.getFullPathName().toRawUTF8());
        
        // Check for test mode
        testMode_ = commandLine.contains("-TestMode") || commandLine.contains("--test-mode");
        testPort_ = 9999; // Default port
        
        // Parse port if specified: -TestPort 9999
        if (commandLine.contains("-TestPort"))
        {
            auto portStr = commandLine.fromFirstOccurrenceOf("-TestPort", false, false)
                                     .trimStart().upToFirstOccurrenceOf(" ", false, false);
            testPort_ = portStr.getIntValue();
            if (testPort_ <= 0 || testPort_ > 65535)
                testPort_ = 9999;
        }
        
        fprintf(stderr, "[SPM] Test mode: %s (port: %d)\n", testMode_ ? "YES" : "NO", testPort_);
        
        if (testMode_)
        {
            // Test mode: run headless with TestServer only
            fprintf(stderr, "[SPM] Starting test mode...\n");
            runTestMode();
        }
        else
        {
            // Normal mode: create main window (TestServer runs inside MainComponent)
            fprintf(stderr, "[SPM] Creating main window...\n");
            mainWindow.reset(new MainWindow(getApplicationName()));
            fprintf(stderr, "[SPM] Main window created\n");
        }
    }
    
    void runTestMode()
    {
        fprintf(stderr, "[SPM] runTestMode() starting\n");
        
        // Create audio engine
        fprintf(stderr, "[SPM] Creating AudioEngine...\n");
        audioEngine_ = std::make_unique<spm::AudioEngine>();
        fprintf(stderr, "[SPM] AudioEngine created\n");
        
        // Set up callbacks for test mode
        fprintf(stderr, "[SPM] Setting up callbacks...\n");
        audioEngine_->setPitchCallback([this](const spm::PitchVector& pitches) {
            // Update test server with results
            if (testServer_)
            {
                std::vector<spm::TestServer::DetectionResult> results;
                for (const auto& p : pitches)
                {
                    spm::TestServer::DetectionResult r;
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
        });
        
        audioEngine_->setSpectrumCallback([this](const spm::SpectrumData& data) {
            // Spectrum data is processed but not used in basic tests
            juce::ignoreUnused(data);
        });
        fprintf(stderr, "[SPM] Callbacks set up\n");
        
        // Create and start TestServer
        fprintf(stderr, "[SPM] Creating TestServer on port %d...\n", testPort_);
        testServer_ = std::make_unique<spm::TestServer>();
        testServer_->setAudioEngine(audioEngine_.get());
        testServer_->setMainComponent(nullptr); // No GUI in test mode
        
        if (!testServer_->start(testPort_))
        {
            fprintf(stderr, "[SPM] Failed to start TestServer on port %d!\n", testPort_);
            quit();
            return;
        }
        fprintf(stderr, "[SPM] TestServer started successfully on port %d\n", testPort_);
        
        // Keep the app running - TestServer runs in its own thread
        fprintf(stderr, "[SPM] Test mode ready. Waiting for test commands...\n");
    }

    void shutdown() override
    {
        fprintf(stderr, "[SPM] shutdown() called\n");
        
        // Stop test server
        if (testServer_)
        {
            fprintf(stderr, "[SPM] Stopping TestServer...\n");
            testServer_->stop();
            testServer_.reset();
        }
        
        // Stop audio engine
        if (audioEngine_)
        {
            fprintf(stderr, "[SPM] Stopping AudioEngine...\n");
            if (audioEngine_->isRunning())
            {
                audioEngine_->stop();
            }
            audioEngine_.reset();
        }
        
        mainWindow = nullptr;
        fprintf(stderr, "[SPM] shutdown() complete\n");
    }

    void systemRequestedQuit() override
    {
        fprintf(stderr, "[SPM] systemRequestedQuit() called\n");
        quit();
    }

    void anotherInstanceStarted(const juce::String& commandLine) override
    {
        juce::ignoreUnused(commandLine);
        // Bring current window to front when another instance starts
        if (mainWindow != nullptr)
        {
            mainWindow->toFront(true);
        }
    }

    /**
     * Main Application Window (Normal Mode)
     */
    class MainWindow : public juce::DocumentWindow
    {
    public:
        MainWindow(juce::String name)
            : DocumentWindow(
                name,
                juce::Colours::black,
                DocumentWindow::allButtons)
        {
            setUsingNativeTitleBar(true);
            
            // Create main component (which includes TestServer)
            auto* mainComponent = new spm::MainComponent();
            setContentOwned(mainComponent, true);
            
            // Platform-specific window setup
            spm::Platform::configureMainWindow(this);
            
            setVisible(true);
        }

        void closeButtonPressed() override
        {
            // Exit application when close button pressed
            JUCEApplication::getInstance()->systemRequestedQuit();
        }

    private:
        JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(MainWindow)
    };

private:
    std::unique_ptr<MainWindow> mainWindow;
    std::unique_ptr<spm::TestServer> testServer_;
    std::unique_ptr<spm::AudioEngine> audioEngine_;
    bool testMode_ = false;
    int testPort_ = 9999;
};

//==============================================================================
// Application Entry Point
//==============================================================================

// Install crash handlers before JUCE takes over
struct CrashHandlerInstaller {
    CrashHandlerInstaller() {
        signal(SIGSEGV, crashHandler);
        signal(SIGABRT, crashHandler);
        signal(SIGFPE, crashHandler);
        signal(SIGILL, crashHandler);
        std::set_terminate(terminateHandler);
    }
} g_crashHandlerInstaller;

START_JUCE_APPLICATION(SuperPitchMonitorApp)

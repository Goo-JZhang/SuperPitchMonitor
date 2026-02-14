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
#include "Test/AutoTestManager.h"

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
 * A JUCE-based Android real-time spectrum analysis and pitch detection tool
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
        
        // Initialize logger first (before anything else)
        auto exeDir = juce::File::getSpecialLocation(juce::File::currentExecutableFile)
                          .getParentDirectory();
        auto logDir = exeDir.getChildFile("build_logs");
        spm::FileLogger::getInstance().initialize(logDir);
        fprintf(stderr, "[SPM] Logger initialized\n");
        
        // Parse command line for auto-test mode
        autoTestManager_.parseCommandLine(commandLine);
        
        fprintf(stderr, "[SPM] AutoTest mode: %s\n", autoTestManager_.isAutoTestMode() ? "YES" : "NO");
        
        if (autoTestManager_.isAutoTestMode())
        {
            // Test mode: run without UI
            fprintf(stderr, "[SPM] Starting test mode...\n");
            runTestMode();
            fprintf(stderr, "[SPM] Test mode initialized\n");
        }
        else
        {
            // Normal mode: create main window
            fprintf(stderr, "[SPM] Creating main window...\n");
            mainWindow.reset(new MainWindow(getApplicationName()));
            fprintf(stderr, "[SPM] Main window created\n");
        }
    }
    
    void runTestMode()
    {
        fprintf(stderr, "[SPM] runTestMode() - Step 1\n");
        
        // Create audio engine for test mode (headless)
        fprintf(stderr, "[SPM] Creating AudioEngine...\n");
        audioEngine_ = std::make_unique<spm::AudioEngine>();
        fprintf(stderr, "[SPM] AudioEngine created\n");
        
        // Set up callbacks for test mode
        fprintf(stderr, "[SPM] Setting up callbacks...\n");
        audioEngine_->setPitchCallback([this](const spm::PitchVector& pitches) {
            autoTestManager_.updatePitchResults(pitches);
        });
        audioEngine_->setSpectrumCallback([this](const spm::SpectrumData& data) {
            autoTestManager_.updateSpectrumData(data);
        });
        fprintf(stderr, "[SPM] Callbacks set up\n");
        
        // Start auto test manager
        fprintf(stderr, "[SPM] Starting AutoTestManager...\n");
        if (!autoTestManager_.startTestMode(audioEngine_.get(), nullptr))
        {
            fprintf(stderr, "[SPM] Failed to start AutoTestManager!\n");
            return;
        }
        fprintf(stderr, "[SPM] AutoTestManager started\n");
        
        // Just use a simple timer to check exit condition
        fprintf(stderr, "[SPM] Starting exit check timer...\n");
        juce::Timer::callAfterDelay(100, [this]() { checkTestExit(); });
        fprintf(stderr, "[SPM] Test mode setup complete\n");
    }
    
    void checkTestExit()
    {
        if (autoTestManager_.shouldExit())
        {
            fprintf(stderr, "[SPM] Exit requested, quitting...\n");
            systemRequestedQuit();
        }
        else
        {
            // Check again in 100ms
            juce::Timer::callAfterDelay(100, [this]() { checkTestExit(); });
        }
    }

    void shutdown() override
    {
        mainWindow = nullptr;
    }

    void systemRequestedQuit() override
    {
        quit();
    }

    void anotherInstanceStarted(const juce::String& commandLine) override
    {
        // Bring current window to front when another instance starts
        if (mainWindow != nullptr)
        {
            mainWindow->toFront(true);
        }
    }

    /**
     * Main Application Window
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
            
            // Create main component
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
    spm::AutoTestManager autoTestManager_;
    std::unique_ptr<spm::AudioEngine> audioEngine_;
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


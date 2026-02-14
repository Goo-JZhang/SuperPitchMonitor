#pragma once

#include <juce_core/juce_core.h>
#include <fstream>
#include <mutex>
#include <queue>
#include <thread>
#include <atomic>
#include <functional>

// Platform-specific includes for crash handling
#ifdef _WIN32
    #include <windows.h>
    #include <dbghelp.h>
    #pragma comment(lib, "dbghelp.lib")
#endif

namespace spm {

/**
 * File Logger with Crash Support
 * - Normal logging
 * - Crash/assertion capture
 * - Stack trace on crash
 */
class FileLogger
{
public:
    static FileLogger& getInstance();
    
    // Initialize with log directory
    void initialize(const juce::File& logDirectory);
    
    // Log levels
    enum class Level {
        Debug,
        Info,
        Warning,
        Error,
        Fatal  // For crashes
    };
    
    // Main log function (thread-safe)
    void log(Level level, const juce::String& message);
    
    // Convenience functions
    void debug(const juce::String& message) { log(Level::Debug, message); }
    void info(const juce::String& message) { log(Level::Info, message); }
    void warning(const juce::String& message) { log(Level::Warning, message); }
    void error(const juce::String& message) { log(Level::Error, message); }
    void fatal(const juce::String& message) { log(Level::Fatal, message); }
    
    // Flush pending logs
    void flush();
    
    // Get current log file path
    juce::String getCurrentLogFile() const { return currentLogFile_.getFullPathName(); }
    
    // Install crash handlers
    void installCrashHandlers();
    
    // Capture stack trace
    static juce::String getStackTrace(int skipFrames = 1);
    
    // Write crash dump with full context
    void writeCrashDump(const juce::String& reason);
    
    ~FileLogger();

private:
    FileLogger() = default;
    FileLogger(const FileLogger&) = delete;
    FileLogger& operator=(const FileLogger&) = delete;
    
    void writeLogEntry(const juce::String& entry);
    void rotateLogIfNeeded();
    juce::String levelToString(Level level);
    void writeHeader();
    
    // Crash handlers
    static void installSEHFilter();
    static void installTerminateHandler();
    // Note: std::set_unexpected is removed in C++17
    // static void installUnexpectedHandler();
    static void installPureCallHandler();
    static void installInvalidParameterHandler();
    
#ifdef _WIN32
    static LONG WINAPI sehHandler(EXCEPTION_POINTERS* pExceptionInfo);
    static juce::String getSEHDescription(DWORD code);
    static juce::String getStackTraceFromContext(const CONTEXT* context);
#endif
    
    static void terminateHandler();
    // static void unexpectedHandler();  // Removed in C++17
    static void pureCallHandler();
    static void invalidParameterHandler(const wchar_t* expression,
                                         const wchar_t* function,
                                         const wchar_t* file,
                                         unsigned int line,
                                         uintptr_t pReserved);
    
    std::mutex mutex_;
    juce::File currentLogFile_;
    std::ofstream fileStream_;
    
    std::atomic<bool> initialized_{false};
    std::atomic<bool> inCrash_{false};  // Prevent recursive logging during crash
    juce::File logDirectory_;
    
    juce::int64 currentFileSize_ = 0;
    static constexpr juce::int64 maxFileSize_ = 10 * 1024 * 1024;  // 10 MB
};

// Macro for convenient logging
#if defined(DEBUG) || defined(_DEBUG)
    #define SPM_LOG_DEBUG(msg) spm::FileLogger::getInstance().debug(msg)
#else
    #define SPM_LOG_DEBUG(msg)
#endif

#define SPM_LOG_INFO(msg) spm::FileLogger::getInstance().info(msg)
#define SPM_LOG_WARNING(msg) spm::FileLogger::getInstance().warning(msg)
#define SPM_LOG_ERROR(msg) spm::FileLogger::getInstance().error(msg)
#define SPM_LOG_FATAL(msg) spm::FileLogger::getInstance().fatal(msg)

// Assertion handler that logs to file
void installAssertionHandler();

} // namespace spm

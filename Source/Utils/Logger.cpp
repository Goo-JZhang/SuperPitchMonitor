#include "Logger.h"

#ifdef _WIN32
    #include <windows.h>
    #include <dbghelp.h>
    #include <psapi.h>
    #include <intrin.h>
#endif

#include <csignal>
#include <cstdlib>
#include <sstream>

namespace spm {

// Global pointer for crash handlers to access logger
static FileLogger* g_loggerInstance = nullptr;

FileLogger& FileLogger::getInstance()
{
    static FileLogger instance;
    g_loggerInstance = &instance;
    return instance;
}

FileLogger::~FileLogger()
{
    if (!inCrash_)
    {
        flush();
        if (fileStream_.is_open())
        {
            fileStream_ << "\n=== Logger Shutdown ===\n";
            fileStream_.flush();
            fileStream_.close();
        }
    }
}

void FileLogger::initialize(const juce::File& logDirectory)
{
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (initialized_)
        return;
    
    logDirectory_ = logDirectory;
    
    // Create log directory if it doesn't exist
    if (!logDirectory.exists())
    {
        logDirectory.createDirectory();
    }
    
    // Create log filename with timestamp
    auto now = juce::Time::getCurrentTime();
    juce::String filename = "app_" + now.formatted("%Y-%m-%d_%H-%M-%S") + ".log";
    currentLogFile_ = logDirectory.getChildFile(filename);
    
    // Open file
    fileStream_.open(currentLogFile_.getFullPathName().toStdString(), 
                     std::ios::out | std::ios::app);
    
    if (fileStream_.is_open())
    {
        initialized_ = true;
        writeHeader();
        
        currentFileSize_ = currentLogFile_.getSize();
        
        // Install crash handlers
        installCrashHandlers();
    }
}

void FileLogger::writeHeader()
{
    if (fileStream_.is_open())
    {
        auto now = juce::Time::getCurrentTime();
        juce::String header;
        header += "========================================\n";
        header += "SuperPitchMonitor Log Started\n";
        header += "Time: " + now.toString(true, true) + "\n";
        header += "Log File: " + currentLogFile_.getFullPathName() + "\n";
        header += "Build: " + juce::String(__DATE__) + " " + juce::String(__TIME__) + "\n";
        
#ifdef _WIN32
        header += "Platform: Windows\n";
#ifdef _WIN64
        header += "Architecture: x64\n";
#else
        header += "Architecture: x86\n";
#endif
#endif
        
        header += "========================================\n";
        
        fileStream_ << header.toStdString();
        fileStream_.flush();
    }
}

void FileLogger::log(Level level, const juce::String& message)
{
    // Prevent recursive logging during crash
    if (inCrash_ && level != Level::Fatal)
        return;
    
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (!initialized_)
    {
        // Try to initialize with default location
        if (!logDirectory_.exists())
        {
            // Use Saved/Logs directory for runtime logs
            // This directory is persistent across builds and platform-independent
            auto exeDir = juce::File::getSpecialLocation(juce::File::currentExecutableFile)
                              .getParentDirectory();
            
            // Check if we're in a build directory (has build-* pattern)
            auto parentDir = exeDir.getParentDirectory();
            auto grandParentDir = parentDir.getParentDirectory();
            
            // Try to find project root (contains CMakeLists.txt)
            juce::File projectRoot = exeDir;
            for (int i = 0; i < 5; ++i) {
                if (projectRoot.getChildFile("CMakeLists.txt").existsAsFile())
                    break;
                projectRoot = projectRoot.getParentDirectory();
            }
            
            // Use Saved/Logs relative to project root, or fallback to exe directory
            if (projectRoot.getChildFile("CMakeLists.txt").existsAsFile())
                logDirectory_ = projectRoot.getChildFile("Saved/Logs");
            else
                logDirectory_ = exeDir.getChildFile("Saved/Logs");
        }
        initialize(logDirectory_);
    }
    
    if (!fileStream_.is_open())
        return;
    
    // Format: [YYYY-MM-DD HH:MM:SS.mmm] [LEVEL] Message
    auto now = juce::Time::getCurrentTime();
    juce::String timestamp = now.formatted("%Y-%m-%d %H:%M:%S") + 
                            juce::String::formatted(".%03d", now.getMilliseconds());
    
    juce::String entry = "[" + timestamp + "] [" + levelToString(level) + "] " + message;
    
    writeLogEntry(entry);
    
    // Also output to debugger in debug builds
#if defined(DEBUG) || defined(_DEBUG)
    DBG(entry);
#endif
}

void FileLogger::writeLogEntry(const juce::String& entry)
{
    if (fileStream_.is_open())
    {
        fileStream_ << entry.toStdString() << std::endl;
        
        // Flush immediately for errors and crashes
        if (inCrash_)
        {
            fileStream_.flush();
        }
        
        currentFileSize_ += entry.getNumBytesAsUTF8() + 1;
        
        if (currentFileSize_ > maxFileSize_)
        {
            rotateLogIfNeeded();
        }
    }
}

void FileLogger::rotateLogIfNeeded()
{
    fileStream_.close();
    
    auto now = juce::Time::getCurrentTime();
    juce::String filename = "app_" + now.formatted("%Y-%m-%d_%H-%M-%S") + "_rotated.log";
    
    currentLogFile_ = logDirectory_.getChildFile(filename);
    
    fileStream_.open(currentLogFile_.getFullPathName().toStdString(),
                     std::ios::out | std::ios::app);
    
    if (fileStream_.is_open())
    {
        currentFileSize_ = 0;
        fileStream_ << "\n=== Log Rotated ===\n";
        fileStream_.flush();
    }
}

void FileLogger::flush()
{
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (fileStream_.is_open())
    {
        fileStream_.flush();
    }
}

juce::String FileLogger::levelToString(Level level)
{
    switch (level)
    {
        case Level::Debug:   return "DEBUG";
        case Level::Info:    return "INFO ";
        case Level::Warning: return "WARN ";
        case Level::Error:   return "ERROR";
        case Level::Fatal:   return "FATAL";
        default:             return "UNKNOWN";
    }
}

// =============================================================================
// Stack Trace Implementation
// =============================================================================

juce::String FileLogger::getStackTrace(int skipFrames)
{
    juce::String result;
    result += "Stack Trace:\n";
    
#ifdef _WIN32
    const int maxFrames = 64;
    void* stack[maxFrames];
    HANDLE process = GetCurrentProcess();
    
    // Initialize symbol handler
    SymInitialize(process, NULL, TRUE);
    
    // Capture stack
    WORD frames = CaptureStackBackTrace(skipFrames, maxFrames, stack, NULL);
    
    // Symbol info buffer
    char buffer[sizeof(SYMBOL_INFO) + MAX_SYM_NAME * sizeof(TCHAR)];
    PSYMBOL_INFO symbol = (PSYMBOL_INFO)buffer;
    symbol->SizeOfStruct = sizeof(SYMBOL_INFO);
    symbol->MaxNameLen = MAX_SYM_NAME;
    
    IMAGEHLP_LINE64 line;
    line.SizeOfStruct = sizeof(IMAGEHLP_LINE64);
    DWORD displacement;
    
    for (WORD i = 0; i < frames; i++)
    {
        DWORD64 address = (DWORD64)stack[i];
        
        // Get symbol name
        if (SymFromAddr(process, address, 0, symbol))
        {
            result += "  [" + juce::String(i) + "] " + juce::String(symbol->Name);
            
            // Get line info
            if (SymGetLineFromAddr64(process, address, &displacement, &line))
            {
                result += " (" + juce::String(line.FileName) + ":" + juce::String((int)line.LineNumber) + ")";
            }
            
            result += "\n";
        }
        else
        {
            result += "  [" + juce::String(i) + "] 0x" + juce::String::toHexString((juce::int64)address).toUpperCase() + "\n";
        }
    }
    
    SymCleanup(process);
#else
    result += "  (Stack trace not implemented for this platform)\n";
#endif
    
    return result;
}

// =============================================================================
// Crash Handler Implementation
// =============================================================================

void FileLogger::installCrashHandlers()
{
#ifdef _WIN32
    // Set unhandled exception filter
    SetUnhandledExceptionFilter(&FileLogger::sehHandler);
    
    // Install C++ exception handlers
    installTerminateHandler();
    // installUnexpectedHandler();  // Removed in C++17
    installPureCallHandler();
    installInvalidParameterHandler();
#endif
    
    // Signal handlers (cross-platform)
    signal(SIGSEGV, [](int sig) {
        if (g_loggerInstance)
        {
            g_loggerInstance->writeCrashDump("SIGSEGV - Segmentation fault");
        }
        exit(1);
    });
    
    signal(SIGABRT, [](int sig) {
        if (g_loggerInstance)
        {
            g_loggerInstance->writeCrashDump("SIGABRT - Abort");
        }
        exit(1);
    });
    
    signal(SIGFPE, [](int sig) {
        if (g_loggerInstance)
        {
            g_loggerInstance->writeCrashDump("SIGFPE - Floating point exception");
        }
        exit(1);
    });
    
    signal(SIGILL, [](int sig) {
        if (g_loggerInstance)
        {
            g_loggerInstance->writeCrashDump("SIGILL - Illegal instruction");
        }
        exit(1);
    });
}

void FileLogger::writeCrashDump(const juce::String& reason)
{
    inCrash_ = true;
    
    juce::String crashInfo;
    crashInfo += "\n";
    crashInfo += "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n";
    crashInfo += "!!! CRASH DETECTED\n";
    crashInfo += "!!! Reason: " + reason + "\n";
    crashInfo += "!!! Time: " + juce::Time::getCurrentTime().toString(true, true) + "\n";
    crashInfo += "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n";
    
    // Stack trace
    crashInfo += getStackTrace(2);
    
    crashInfo += "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n";
    
    // Write directly without locking (we're in crash state)
    if (fileStream_.is_open())
    {
        fileStream_ << crashInfo.toStdString() << std::endl;
        fileStream_.flush();
    }
    
    // Also try to write to a separate crash file
    juce::File crashFile = logDirectory_.getChildFile("crash_" + 
        juce::Time::getCurrentTime().formatted("%Y%m%d_%H%M%S") + ".txt");
    crashFile.replaceWithText(crashInfo);
}

#ifdef _WIN32

LONG WINAPI FileLogger::sehHandler(EXCEPTION_POINTERS* pExceptionInfo)
{
    if (g_loggerInstance)
    {
        DWORD code = pExceptionInfo->ExceptionRecord->ExceptionCode;
        juce::String reason = getSEHDescription(code);
        
        juce::String crashInfo;
        crashInfo += "\n";
        crashInfo += "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n";
        crashInfo += "!!! EXCEPTION DETECTED (SEH)\n";
        crashInfo += "!!! Code: 0x" + juce::String::toHexString((juce::int64)code).toUpperCase() + "\n";
        crashInfo += "!!! Description: " + reason + "\n";
        crashInfo += "!!! Address: 0x" + juce::String::toHexString((juce::int64)pExceptionInfo->ExceptionRecord->ExceptionAddress).toUpperCase() + "\n";
        crashInfo += "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n";
        
        // Stack trace from exception context
        crashInfo += "\nException Stack Trace:\n";
        crashInfo += getStackTraceFromContext(pExceptionInfo->ContextRecord);
        
        crashInfo += "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n";
        
        g_loggerInstance->fatal(crashInfo);
        g_loggerInstance->flush();
        
        // Write crash dump file
        juce::File crashFile = g_loggerInstance->logDirectory_.getChildFile("crash_" + 
            juce::Time::getCurrentTime().formatted("%Y%m%d_%H%M%S") + ".txt");
        crashFile.replaceWithText(crashInfo);
    }
    
    // Let the OS handle it (will show crash dialog)
    return EXCEPTION_EXECUTE_HANDLER;
}

juce::String FileLogger::getSEHDescription(DWORD code)
{
    switch (code)
    {
        case EXCEPTION_ACCESS_VIOLATION:         return "Access violation";
        case EXCEPTION_ARRAY_BOUNDS_EXCEEDED:    return "Array bounds exceeded";
        case EXCEPTION_BREAKPOINT:               return "Breakpoint";
        case EXCEPTION_DATATYPE_MISALIGNMENT:    return "Data type misalignment";
        case EXCEPTION_FLT_DIVIDE_BY_ZERO:       return "Float divide by zero";
        case EXCEPTION_FLT_INVALID_OPERATION:    return "Float invalid operation";
        case EXCEPTION_FLT_OVERFLOW:             return "Float overflow";
        case EXCEPTION_FLT_UNDERFLOW:            return "Float underflow";
        case EXCEPTION_ILLEGAL_INSTRUCTION:      return "Illegal instruction";
        case EXCEPTION_IN_PAGE_ERROR:            return "In-page error";
        case EXCEPTION_INT_DIVIDE_BY_ZERO:       return "Integer divide by zero";
        case EXCEPTION_INT_OVERFLOW:             return "Integer overflow";
        case EXCEPTION_INVALID_DISPOSITION:      return "Invalid disposition";
        case EXCEPTION_NONCONTINUABLE_EXCEPTION: return "Non-continuable exception";
        case EXCEPTION_PRIV_INSTRUCTION:         return "Privileged instruction";
        case EXCEPTION_SINGLE_STEP:              return "Single step";
        case EXCEPTION_STACK_OVERFLOW:           return "Stack overflow";
        default:                                 return "Unknown exception";
    }
}

juce::String FileLogger::getStackTraceFromContext(const CONTEXT* context)
{
    juce::String result;
    
    HANDLE process = GetCurrentProcess();
    HANDLE thread = GetCurrentThread();
    
    STACKFRAME64 stackFrame;
    memset(&stackFrame, 0, sizeof(STACKFRAME64));
    
#ifdef _WIN64
    DWORD machineType = IMAGE_FILE_MACHINE_AMD64;
    stackFrame.AddrPC.Offset = context->Rip;
    stackFrame.AddrPC.Mode = AddrModeFlat;
    stackFrame.AddrFrame.Offset = context->Rbp;
    stackFrame.AddrFrame.Mode = AddrModeFlat;
    stackFrame.AddrStack.Offset = context->Rsp;
    stackFrame.AddrStack.Mode = AddrModeFlat;
#else
    DWORD machineType = IMAGE_FILE_MACHINE_I386;
    stackFrame.AddrPC.Offset = context->Eip;
    stackFrame.AddrPC.Mode = AddrModeFlat;
    stackFrame.AddrFrame.Offset = context->Ebp;
    stackFrame.AddrFrame.Mode = AddrModeFlat;
    stackFrame.AddrStack.Offset = context->Esp;
    stackFrame.AddrStack.Mode = AddrModeFlat;
#endif
    
    SymInitialize(process, NULL, TRUE);
    
    char buffer[sizeof(SYMBOL_INFO) + MAX_SYM_NAME * sizeof(TCHAR)];
    PSYMBOL_INFO symbol = (PSYMBOL_INFO)buffer;
    symbol->SizeOfStruct = sizeof(SYMBOL_INFO);
    symbol->MaxNameLen = MAX_SYM_NAME;
    
    int frameCount = 0;
    while (StackWalk64(machineType, process, thread, &stackFrame, 
                       (LPVOID)context, NULL, SymFunctionTableAccess64, 
                       SymGetModuleBase64, NULL) && frameCount < 64)
    {
        if (stackFrame.AddrPC.Offset == 0)
            break;
        
        if (SymFromAddr(process, stackFrame.AddrPC.Offset, 0, symbol))
        {
            result += "  [" + juce::String(frameCount) + "] " + juce::String(symbol->Name);
            
            IMAGEHLP_LINE64 line;
            line.SizeOfStruct = sizeof(IMAGEHLP_LINE64);
            DWORD displacement;
            if (SymGetLineFromAddr64(process, stackFrame.AddrPC.Offset, &displacement, &line))
            {
                result += " (" + juce::String(line.FileName) + ":" + juce::String((int)line.LineNumber) + ")";
            }
            result += "\n";
        }
        else
        {
            result += "  [" + juce::String(frameCount) + "] 0x" + 
                     juce::String::toHexString((juce::int64)stackFrame.AddrPC.Offset).toUpperCase() + "\n";
        }
        
        frameCount++;
    }
    
    SymCleanup(process);
    
    return result;
}

#endif // _WIN32

void FileLogger::installTerminateHandler()
{
    std::set_terminate([]() {
        if (g_loggerInstance)
        {
            g_loggerInstance->writeCrashDump("std::terminate called (uncaught exception)");
        }
        abort();
    });
}

#ifdef _WIN32
void FileLogger::installPureCallHandler()
{
    _set_purecall_handler([]() {
        if (g_loggerInstance)
        {
            g_loggerInstance->writeCrashDump("Pure virtual function called");
        }
        abort();
    });
}

void FileLogger::installInvalidParameterHandler()
{
    _set_invalid_parameter_handler([](const wchar_t* expression,
                                       const wchar_t* function,
                                       const wchar_t* file,
                                       unsigned int line,
                                       uintptr_t pReserved) {
        if (g_loggerInstance)
        {
            juce::String msg = "Invalid parameter handler called";
            if (expression)
                msg += "\n  Expression: " + juce::String(expression);
            if (function)
                msg += "\n  Function: " + juce::String(function);
            if (file)
                msg += "\n  File: " + juce::String(file) + ":" + juce::String((int)line);
            
            g_loggerInstance->writeCrashDump(msg);
        }
        abort();
    });
}
#endif // _WIN32

void FileLogger::terminateHandler()
{
    // Static handler that forwards to instance
    if (g_loggerInstance)
    {
        g_loggerInstance->writeCrashDump("std::terminate called");
    }
    abort();
}

void FileLogger::pureCallHandler()
{
    if (g_loggerInstance)
    {
        g_loggerInstance->writeCrashDump("Pure virtual function called");
    }
    abort();
}

void FileLogger::invalidParameterHandler(const wchar_t* expression,
                                          const wchar_t* function,
                                          const wchar_t* file,
                                          unsigned int line,
                                          uintptr_t pReserved)
{
    if (g_loggerInstance)
    {
        juce::String msg = "Invalid parameter";
        if (expression)
            msg += " - Expression: " + juce::String(expression);
        g_loggerInstance->writeCrashDump(msg);
    }
    abort();
}

// =============================================================================
// JUCE Assertion Handler
// =============================================================================

void installAssertionHandler()
{
    // Note: JUCE assertion handling is done through JUCE_APPLICATION_BASE_CLASS
    // or by overriding juce::LeakedObjectDetector behaviour.
    // For now, we rely on the crash handlers to catch assertion failures.
    // 
    // To capture assertions, we would need to:
    // 1. Define JUCE_LOG_ASSERTIONS macro
    // 2. Implement a custom assertion handler
    // 
    // For now, standard crash handling is sufficient.
}

} // namespace spm

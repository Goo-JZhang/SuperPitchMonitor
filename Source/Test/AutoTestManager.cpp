#include "AutoTestManager.h"
#include "../Audio/AudioEngine.h"
#include "../MainComponent.h"
#include "../Utils/Logger.h"
#include "../Audio/FileAudioInput.h"

// Windows 命名管道 API
#ifdef _WIN32
#include <windows.h>
#endif

namespace spm {

AutoTestManager::AutoTestManager()
{
}

AutoTestManager::~AutoTestManager()
{
    stopTestMode();
}

bool AutoTestManager::parseCommandLine(const juce::String& commandLine)
{
    fprintf(stderr, "[AutoTest] parseCommandLine: %s\n", commandLine.toRawUTF8());
    
    // Simple string-based parsing
    if (commandLine.contains("-AutoTest") || commandLine.contains("--autotest"))
    {
        autoTestMode_ = true;
        fprintf(stderr, "[AutoTest] Test mode enabled\n");
    }
    
    return autoTestMode_;
}

bool AutoTestManager::startTestMode(AudioEngine* engine, MainComponent* main)
{
    if (!autoTestMode_ || running_)
        return false;
    
    audioEngine_ = engine;
    mainComponent_ = main;
    running_ = true;
    
    // 启动命名管道监听线程
    class PipeThread : public juce::Thread {
    public:
        PipeThread(AutoTestManager* mgr) : Thread("TestPipeThread"), manager_(mgr) {}
        void run() override { if (manager_) manager_->runPipeServer(); }
    private:
        AutoTestManager* manager_;
    };
    pipeThread_ = std::make_unique<PipeThread>(this);
    pipeThread_->startThread();
    
    SPM_LOG_INFO("[AutoTest] Test mode started, pipe: \\\\.\\pipe\\" + pipeName_);
    return true;
}

void AutoTestManager::stopTestMode()
{
    running_ = false;
    frameEvent_.signal();
    
    // 关闭管道
#ifdef _WIN32
    if (pipeHandle_ != nullptr)
    {
        CloseHandle(pipeHandle_);
        pipeHandle_ = nullptr;
    }
#endif
    
    if (pipeThread_)
    {
        pipeThread_->stopThread(2000);
        pipeThread_.reset();
    }
}

void AutoTestManager::runPipeServer()
{
#ifdef _WIN32
    while (running_)
    {
        // 创建命名管道
        HANDLE hPipe = CreateNamedPipeA(
            ("\\\\.\\pipe\\" + pipeName_).toStdString().c_str(),
            PIPE_ACCESS_DUPLEX,
            PIPE_TYPE_MESSAGE | PIPE_READMODE_MESSAGE | PIPE_WAIT,
            1,  // 最大实例数
            4096,  // 输出缓冲区
            4096,  // 输入缓冲区
            5000,  // 超时
            nullptr
        );
        
        if (hPipe == INVALID_HANDLE_VALUE)
        {
            SPM_LOG_ERROR("[AutoTest] Failed to create named pipe");
            juce::Thread::sleep(1000);
            continue;
        }
        
        pipeHandle_ = hPipe;
        SPM_LOG_INFO("[AutoTest] Waiting for client connection...");
        
        // 等待客户端连接
        BOOL connected = ConnectNamedPipe(hPipe, nullptr) ? TRUE : (GetLastError() == ERROR_PIPE_CONNECTED);
        
        if (connected)
        {
            SPM_LOG_INFO("[AutoTest] Client connected");
            
            // 处理客户端命令
            char buffer[4096];
            DWORD bytesRead, bytesWritten;
            
            while (running_ && pipeHandle_ != nullptr)
            {
                // 读取命令长度（4字节，大端序）
                unsigned char lenBuf[4];
                BOOL success = ReadFile(hPipe, lenBuf, 4, &bytesRead, nullptr);
                
                if (!success || bytesRead != 4)
                {
                    SPM_LOG_INFO("[AutoTest] Client disconnected (read error)");
                    break;
                }
                
                // 转换大端序到本地字节序
                DWORD lenBytes = ((DWORD)lenBuf[0] << 24) | ((DWORD)lenBuf[1] << 16) | 
                                 ((DWORD)lenBuf[2] << 8) | (DWORD)lenBuf[3];
                
                // 读取命令数据
                if (lenBytes > 0 && lenBytes < 4096)
                {
                    success = ReadFile(hPipe, buffer, lenBytes, &bytesRead, nullptr);
                    if (!success || bytesRead != lenBytes)
                    {
                        SPM_LOG_ERROR("[AutoTest] Failed to read command data");
                        break;
                    }
                    
                    buffer[lenBytes] = '\0';
                    juce::String jsonCmd(buffer);
                    
                    // 解析命令
                    auto json = juce::JSON::parse(jsonCmd);
                    if (json.isObject())
                    {
                        TestCommand cmd;
                        cmd.cmd = json.getProperty("cmd", "").toString();
                        cmd.params = json.getDynamicObject();
                        
                        // 添加到队列或直接处理
                        if (cmd.cmd == "exit")
                        {
                            shouldExit_ = true;
                            juce::String response = R"({"status":"ok","message":"exiting"})";
                            DWORD respLen = response.getNumBytesAsUTF8();
                            unsigned char respLenBuf[4] = {(unsigned char)(respLen >> 24), (unsigned char)(respLen >> 16), 
                                                           (unsigned char)(respLen >> 8), (unsigned char)respLen};
                            WriteFile(hPipe, respLenBuf, 4, &bytesWritten, nullptr);
                            WriteFile(hPipe, response.toRawUTF8(), respLen, &bytesWritten, nullptr);
                            break;
                        }
                        else
                        {
                            // 添加到队列，由主线程处理
                            {
                                juce::ScopedLock lock(commandLock_);
                                commandQueue_.push(cmd);
                            }
                            
                            // 等待处理完成
                            juce::Thread::sleep(50);
                            
                            // 发送响应（大端序长度）
                            juce::String response = processCommand(cmd);
                            DWORD respLen = response.getNumBytesAsUTF8();
                            unsigned char respLenBuf[4] = {(unsigned char)(respLen >> 24), (unsigned char)(respLen >> 16), 
                                                           (unsigned char)(respLen >> 8), (unsigned char)respLen};
                            WriteFile(hPipe, respLenBuf, 4, &bytesWritten, nullptr);
                            WriteFile(hPipe, response.toRawUTF8(), respLen, &bytesWritten, nullptr);
                        }
                    }
                }
            }
        }
        
        DisconnectNamedPipe(hPipe);
        CloseHandle(hPipe);
        pipeHandle_ = nullptr;
    }
#endif
}

void AutoTestManager::processPendingCommands()
{
    juce::Array<TestCommand> pending;
    
    {
        juce::ScopedLock lock(commandLock_);
        while (!commandQueue_.empty())
        {
            pending.add(commandQueue_.front());
            commandQueue_.pop();
        }
    }
    
    for (auto& cmd : pending)
    {
        processCommand(cmd);
    }
}

juce::String AutoTestManager::processCommand(const TestCommand& command)
{
    juce::DynamicObject::Ptr response = new juce::DynamicObject();
    
    if (command.cmd == "setMultiRes")
    {
        bool enabled = command.params->getProperty("enabled").operator bool();
        if (audioEngine_)
        {
            audioEngine_->setMultiResolutionEnabled(enabled);
            response->setProperty("status", "ok");
            response->setProperty("multiRes", enabled);
            SPM_LOG_INFO("[AutoTest] setMultiRes: " + juce::String(enabled ? "ON" : "OFF"));
        }
        else
        {
            response->setProperty("status", "error");
            response->setProperty("message", "AudioEngine not available");
        }
    }
    else if (command.cmd == "loadFile")
    {
        juce::String filename = command.params->getProperty("filename").toString();
        SPM_LOG_INFO("[AutoTest] loadFile request: " + filename);
        
        if (audioEngine_ && !filename.isEmpty())
        {
            auto newSource = std::make_shared<FileAudioInput>();
            
            // Try to find file in test directory
            auto testDir = FileAudioInput::getTestAudioDirectory();
            auto fullPath = testDir.getChildFile(filename);
            
            SPM_LOG_INFO("[AutoTest] Looking for file at: " + fullPath.getFullPathName());
            SPM_LOG_INFO("[AutoTest] Test directory: " + testDir.getFullPathName());
            SPM_LOG_INFO("[AutoTest] File exists: " + juce::String(fullPath.existsAsFile() ? "YES" : "NO"));
            
            if (newSource->loadTestFile(filename))
            {
                audioEngine_->setInputSource(newSource);
                response->setProperty("status", "ok");
                response->setProperty("filename", filename);
                response->setProperty("path", fullPath.getFullPathName());
                SPM_LOG_INFO("[AutoTest] loadFile success: " + filename);
            }
            else
            {
                response->setProperty("status", "error");
                response->setProperty("message", "Failed to load file: " + filename);
                response->setProperty("attempted_path", fullPath.getFullPathName());
            }
        }
        else
        {
            response->setProperty("status", "error");
            response->setProperty("message", "Invalid filename or AudioEngine");
        }
    }
    else if (command.cmd == "start")
    {
        if (audioEngine_)
        {
            audioEngine_->start();
            response->setProperty("status", "ok");
            response->setProperty("running", true);
            SPM_LOG_INFO("[AutoTest] start playback");
        }
        else
        {
            response->setProperty("status", "error");
        }
    }
    else if (command.cmd == "stop")
    {
        if (audioEngine_)
        {
            audioEngine_->stop();
            response->setProperty("status", "ok");
            response->setProperty("running", false);
            SPM_LOG_INFO("[AutoTest] stop playback");
        }
        else
        {
            response->setProperty("status", "error");
        }
    }
    else if (command.cmd == "getPitches")
    {
        juce::Array<juce::var> pitchesArray;
        int pitchCount = 0;
        {
            juce::ScopedLock lock(resultsLock_);
            pitchCount = (int)latestPitches_.size();
            for (const auto& p : latestPitches_)
            {
                juce::DynamicObject::Ptr pitchObj = new juce::DynamicObject();
                pitchObj->setProperty("frequency", p.frequency);
                pitchObj->setProperty("midiNote", p.midiNote);
                pitchObj->setProperty("confidence", p.confidence);
                pitchObj->setProperty("centsDeviation", p.centsDeviation);
                pitchObj->setProperty("harmonicCount", p.harmonicCount);
                pitchObj->setProperty("amplitude", p.amplitude);  // Include amplitude for energy analysis
                pitchesArray.add(pitchObj.get());
            }
        }
        
        response->setProperty("status", "ok");
        response->setProperty("frameCount", (int)frameCount_.load());
        response->setProperty("pitchCount", pitchCount);
        response->setProperty("pitches", pitchesArray);
        
        SPM_LOG_INFO("[AutoTest] getPitches: " + juce::String(pitchCount) + " pitches");
    }
    else if (command.cmd == "getSpectrumPeaks")
    {
        float freqMin = (float)(double)command.params->getProperty("freqMin");
        float freqMax = (float)(double)command.params->getProperty("freqMax");
        
        response->setProperty("status", "ok");
        response->setProperty("freqMin", freqMin);
        response->setProperty("freqMax", freqMax);
        
        juce::Array<juce::var> peaksArray;
        {
            juce::ScopedLock lock(resultsLock_);
            if (!latestSpectrum_.magnitudes.empty())
            {
                float binWidth = latestSpectrum_.sampleRate / (float)latestSpectrum_.magnitudes.size() / 2.0f;
                for (size_t i = 0; i < latestSpectrum_.magnitudes.size(); ++i)
                {
                    float freq = i * binWidth;
                    if (freq >= freqMin && freq <= freqMax && latestSpectrum_.magnitudes[i] > 0.001f)
                    {
                        juce::DynamicObject::Ptr peakObj = new juce::DynamicObject();
                        peakObj->setProperty("frequency", freq);
                        peakObj->setProperty("magnitude", latestSpectrum_.magnitudes[i]);
                        peaksArray.add(peakObj.get());
                    }
                }
            }
        }
        response->setProperty("peaks", peaksArray);
    }
    else if (command.cmd == "wait")
    {
        int frames = (int)(double)command.params->getProperty("frames");
        int timeout = (int)(double)command.params->getProperty("timeout");
        
        bool success = waitForFrames(frames, timeout);
        response->setProperty("status", success ? "ok" : "timeout");
        response->setProperty("frameCount", frameCount_.load());
    }
    else if (command.cmd == "getStatus")
    {
        response->setProperty("status", "ok");
        response->setProperty("running", audioEngine_ ? audioEngine_->isRunning() : false);
        response->setProperty("multiRes", audioEngine_ ? audioEngine_->isMultiResolutionEnabled() : false);
        response->setProperty("frameCount", frameCount_.load());
    }
    else
    {
        response->setProperty("status", "error");
        response->setProperty("message", "Unknown command: " + command.cmd);
    }
    
    return juce::JSON::toString(response.get());
}

void AutoTestManager::updatePitchResults(const std::vector<PitchCandidate>& pitches)
{
    juce::ScopedLock lock(resultsLock_);
    latestPitches_ = pitches;
}

void AutoTestManager::updateSpectrumData(const SpectrumData& data)
{
    juce::ScopedLock lock(resultsLock_);
    latestSpectrum_ = data;
}

void AutoTestManager::onFrameProcessed()
{
    ++frameCount_;
    
    if (targetFrame_ > 0 && frameCount_ >= targetFrame_)
    {
        frameEvent_.signal();
    }
}

bool AutoTestManager::waitForFrames(int frameCount, int timeoutMs)
{
    targetFrame_ = frameCount_.load() + frameCount;
    frameEvent_.reset();
    
    bool success = frameEvent_.wait(timeoutMs);
    targetFrame_ = 0;
    
    return success;
}

} // namespace spm

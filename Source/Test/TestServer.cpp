#include "TestServer.h"
#include "../Audio/AudioEngine.h"
#include "../MainComponent.h"
#include "../Utils/Logger.h"
#include "../Audio/FileAudioInput.h"

namespace spm {

TestServer::TestServer()
    : juce::Thread("TestServerThread")
{
}

TestServer::~TestServer()
{
    stop();
}

bool TestServer::start(int port)
{
    port_ = port;
    
    serverSocket_ = std::make_unique<juce::StreamingSocket>();
    
    if (!serverSocket_->createListener(port_, "127.0.0.1"))
    {
        SPM_LOG_ERROR("[TestServer] Failed to create listener on port " + juce::String(port_));
        return false;
    }
    
    running_ = true;
    startThread();
    
    SPM_LOG_INFO("[TestServer] Started on port " + juce::String(port_));
    return true;
}

void TestServer::stop()
{
    running_ = false;
    
    if (serverSocket_)
    {
        serverSocket_->close();
    }
    
    stopThread(2000);
    
    SPM_LOG_INFO("[TestServer] Stopped");
}

void TestServer::run()
{
    while (running_ && !threadShouldExit())
    {
        // Wait for connection with timeout
        auto* clientSocket = serverSocket_->waitForNextConnection();
        
        if (clientSocket != nullptr)
        {
            SPM_LOG_INFO("[TestServer] Client connected");
            handleClient(clientSocket);
            delete clientSocket;
            SPM_LOG_INFO("[TestServer] Client disconnected");
        }
    }
}

void TestServer::handleClient(juce::StreamingSocket* clientSocket)
{
    juce::MemoryBlock buffer;
    buffer.setSize(4096);
    
    while (running_ && clientSocket->isConnected())
    {
        // Read command length (4 bytes, big-endian)
        char lengthBytes[4];
        int bytesRead = clientSocket->read(lengthBytes, 4, true);
        
        if (bytesRead != 4)
        {
            break; // Connection closed or error
        }
        
        int messageLength = juce::ByteOrder::bigEndianInt(lengthBytes);
        
        if (messageLength <= 0 || messageLength > 65536)
        {
            SPM_LOG_ERROR("[TestServer] Invalid message length: " + juce::String(messageLength));
            break;
        }
        
        // Read command data
        juce::MemoryBlock messageData;
        messageData.setSize(messageLength);
        
        bytesRead = clientSocket->read(messageData.getData(), messageLength, true);
        if (bytesRead != messageLength)
        {
            SPM_LOG_ERROR("[TestServer] Failed to read complete message");
            break;
        }
        
        // Parse and process command
        juce::String jsonCommand = juce::String::fromUTF8(
            static_cast<const char*>(messageData.getData()), messageLength);
        
        juce::String response = processCommand(jsonCommand);
        
        // Send response length
        char responseLengthBytes[4];
        int responseLen = response.getNumBytesAsUTF8();
        responseLengthBytes[0] = (responseLen >> 24) & 0xFF;
        responseLengthBytes[1] = (responseLen >> 16) & 0xFF;
        responseLengthBytes[2] = (responseLen >> 8) & 0xFF;
        responseLengthBytes[3] = responseLen & 0xFF;
        clientSocket->write(responseLengthBytes, 4);
        
        // Send response data
        clientSocket->write(response.toRawUTF8(), response.getNumBytesAsUTF8());
    }
}

juce::String TestServer::processCommand(const juce::String& jsonCommand)
{
    auto json = juce::JSON::parse(jsonCommand);
    
    if (!json.isObject())
    {
        return R"({"error": "Invalid JSON command"})";
    }
    
    auto* obj = json.getDynamicObject();
    if (!obj)
    {
        return R"({"error": "Invalid JSON object"})";
    }
    
    juce::String cmd = obj->getProperty("cmd").toString();
    
    SPM_LOG_INFO("[TestServer] Command: " + cmd);
    
    if (cmd == "getStatus")
    {
        return handleGetStatus();
    }
    else if (cmd == "setMultiRes")
    {
        bool enabled = obj->getProperty("enabled");
        return handleSetMultiRes(enabled);
    }
    else if (cmd == "loadFile")
    {
        juce::String filename = obj->getProperty("filename").toString();
        return handleLoadFile(filename);
    }
    else if (cmd == "startPlayback")
    {
        return handleStartPlayback();
    }
    else if (cmd == "stopPlayback")
    {
        return handleStopPlayback();
    }
    else if (cmd == "getPitches")
    {
        return handleGetPitches();
    }
    else if (cmd == "waitForFrames")
    {
        int frames = obj->getProperty("count");
        return handleWaitForFrames(frames);
    }
    else if (cmd == "getSpectrumPeaks")
    {
        return handleGetSpectrumPeaks();
    }
    
    return R"({"error": "Unknown command"})";
}

juce::String TestServer::handleGetStatus()
{
    juce::DynamicObject::Ptr obj = new juce::DynamicObject();
    
    obj->setProperty("status", "ok");
    obj->setProperty("running", audioEngine_ ? audioEngine_->isRunning() : false);
    obj->setProperty("multiRes", audioEngine_ ? audioEngine_->isMultiResolutionEnabled() : false);
    obj->setProperty("frameCount", (juce::int64)frameCount_);
    
    return juce::JSON::toString(obj.get());
}

juce::String TestServer::handleSetMultiRes(bool enabled)
{
    if (audioEngine_)
    {
        // Direct call - AudioEngine::setMultiResolutionEnabled is thread-safe
        audioEngine_->setMultiResolutionEnabled(enabled);
        SPM_LOG_INFO("[TestServer] Multi-resolution set to: " + juce::String(enabled ? "ON" : "OFF"));
    }
    
    juce::DynamicObject::Ptr obj = new juce::DynamicObject();
    obj->setProperty("status", "ok");
    obj->setProperty("multiRes", enabled);
    return juce::JSON::toString(obj.get());
}

juce::String TestServer::handleLoadFile(const juce::String& filename)
{
    fprintf(stderr, "[TestServer] handleLoadFile called: %s, audioEngine=%p\n", 
            filename.toRawUTF8(), (void*)audioEngine_);
    bool success = false;
    
    if (audioEngine_)
    {
        auto newSource = std::make_shared<FileAudioInput>();
        if (newSource->loadTestFile(filename))
        {
            audioEngine_->setInputSource(newSource);
            success = true;
            SPM_LOG_INFO("[TestServer] Loaded file: " + filename);
        }
        else
        {
            fprintf(stderr, "[TestServer] loadTestFile failed\n");
        }
    }
    else
    {
        fprintf(stderr, "[TestServer] audioEngine is null\n");
    }
    
    juce::DynamicObject::Ptr obj = new juce::DynamicObject();
    obj->setProperty("status", success ? "ok" : "error");
    obj->setProperty("filename", filename);
    return juce::JSON::toString(obj.get());
}

juce::String TestServer::handleStartPlayback()
{
    bool success = false;
    
    if (audioEngine_)
    {
        // Direct call - check and start audio engine
        if (!audioEngine_->isRunning())
        {
            audioEngine_->start();
        }
        success = true;
        SPM_LOG_INFO("[TestServer] Playback started");
    }
    
    juce::DynamicObject::Ptr obj = new juce::DynamicObject();
    obj->setProperty("status", success ? "ok" : "error");
    obj->setProperty("running", audioEngine_ ? audioEngine_->isRunning() : false);
    return juce::JSON::toString(obj.get());
}

juce::String TestServer::handleStopPlayback()
{
    if (audioEngine_ && audioEngine_->isRunning())
    {
        audioEngine_->stop();
        SPM_LOG_INFO("[TestServer] Playback stopped");
    }
    
    juce::DynamicObject::Ptr obj = new juce::DynamicObject();
    obj->setProperty("status", "ok");
    obj->setProperty("running", false);
    return juce::JSON::toString(obj.get());
}

juce::String TestServer::handleGetPitches()
{
    juce::DynamicObject::Ptr obj = new juce::DynamicObject();
    obj->setProperty("status", "ok");
    
    juce::Array<juce::var> pitchesArray;
    
    {
        juce::ScopedLock lock(resultsLock_);
        for (const auto& result : latestResults_)
        {
            juce::DynamicObject::Ptr pitchObj = new juce::DynamicObject();
            pitchObj->setProperty("frequency", result.frequency);
            pitchObj->setProperty("midiNote", result.midiNote);
            pitchObj->setProperty("confidence", result.confidence);
            pitchObj->setProperty("centsDeviation", result.centsDeviation);
            pitchObj->setProperty("harmonicCount", result.harmonicCount);
            pitchesArray.add(pitchObj.get());
        }
    }
    
    obj->setProperty("pitches", pitchesArray);
    obj->setProperty("frameCount", (juce::int64)frameCount_);
    
    return juce::JSON::toString(obj.get());
}

juce::String TestServer::handleWaitForFrames(int count)
{
    juce::int64 targetFrame = frameCount_ + count;
    
    int timeoutMs = 5000; // 5 second timeout
    int waited = 0;
    
    while (frameCount_ < targetFrame && waited < timeoutMs)
    {
        juce::Thread::sleep(10);
        waited += 10;
    }
    
    juce::DynamicObject::Ptr obj = new juce::DynamicObject();
    obj->setProperty("status", frameCount_ >= targetFrame ? "ok" : "timeout");
    obj->setProperty("frameCount", (juce::int64)frameCount_);
    return juce::JSON::toString(obj.get());
}

juce::String TestServer::handleGetSpectrumPeaks()
{
    // This would need access to spectrum data - simplified for now
    juce::DynamicObject::Ptr obj = new juce::DynamicObject();
    obj->setProperty("status", "ok");
    obj->setProperty("peaks", juce::Array<juce::var>());
    return juce::JSON::toString(obj.get());
}

void TestServer::updateResults(const std::vector<DetectionResult>& results)
{
    juce::ScopedLock lock(resultsLock_);
    latestResults_ = results;
}

std::vector<TestServer::DetectionResult> TestServer::getLatestResults() const
{
    juce::ScopedLock lock(resultsLock_);
    return latestResults_;
}

void TestServer::waitForFrames(int count, int timeoutMs)
{
    juce::int64 targetFrame = frameCount_ + count;
    int waited = 0;
    
    while (frameCount_ < targetFrame && waited < timeoutMs)
    {
        juce::Thread::sleep(10);
        waited += 10;
    }
}

} // namespace spm

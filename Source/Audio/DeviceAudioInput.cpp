#include "DeviceAudioInput.h"
#include "../Utils/Logger.h"

namespace spm {

DeviceAudioInput::DeviceAudioInput()
{
    // Initialize device manager
    deviceManager_.initialise(1, 0, nullptr, false);
}

DeviceAudioInput::~DeviceAudioInput()
{
    stop();
    deviceManager_.closeAudioDevice();
}

bool DeviceAudioInput::prepare(double sampleRate, int bufferSize)
{
    sampleRate_ = sampleRate;
    bufferSize_ = bufferSize;
    
    juce::AudioDeviceManager::AudioDeviceSetup setup;
    setup.sampleRate = sampleRate;
    setup.bufferSize = bufferSize;
    setup.inputChannels = 1;
    setup.outputChannels = 0;
    setup.useDefaultInputChannels = true;
    
    if (!selectedDevice_.isEmpty())
    {
        setup.inputDeviceName = selectedDevice_;
    }
    
    auto err = deviceManager_.initialise(1, 0, nullptr, true, {}, &setup);
    
    if (err.isNotEmpty())
    {
       #if defined(DEBUG) || defined(_DEBUG)
        DBG("[DeviceAudioInput] Failed to prepare: " << err);
       #endif
        return false;
    }
    
    inputBuffer_.setSize(1, bufferSize);
    
   #if defined(DEBUG) || defined(_DEBUG)
    DBG("[DeviceAudioInput] Prepared: " << sampleRate << "Hz, " << bufferSize << " samples");
   #endif
    
    return true;
}

void DeviceAudioInput::start()
{
    if (isActive_)
        return;
    
    deviceManager_.addAudioCallback(this);
    isActive_ = true;
    
   #if defined(DEBUG) || defined(_DEBUG)
    DBG("[DeviceAudioInput] Started");
   #endif
}

void DeviceAudioInput::stop()
{
    if (!isActive_)
        return;
    
    deviceManager_.removeAudioCallback(this);
    isActive_ = false;
    
   #if defined(DEBUG) || defined(_DEBUG)
    DBG("[DeviceAudioInput] Stopped");
   #endif
}

bool DeviceAudioInput::isActive() const
{
    return isActive_;
}

double DeviceAudioInput::getSampleRate() const
{
    return sampleRate_;
}

int DeviceAudioInput::getBufferSize() const
{
    return bufferSize_;
}

void DeviceAudioInput::setDevice(const juce::String& deviceName)
{
    selectedDevice_ = deviceName;
}

juce::String DeviceAudioInput::getCurrentDevice() const
{
    if (auto* device = deviceManager_.getCurrentAudioDevice())
    {
        return device->getName();
    }
    return {};
}

void DeviceAudioInput::audioDeviceAboutToStart(juce::AudioIODevice* device)
{
    sampleRate_ = device->getCurrentSampleRate();
    bufferSize_ = device->getCurrentBufferSizeSamples();
    inputBuffer_.setSize(1, bufferSize_);
    
   #if defined(DEBUG) || defined(_DEBUG)
    DBG("[DeviceAudioInput] Device started: " << device->getName());
   #endif
}

void DeviceAudioInput::audioDeviceIOCallbackWithContext(
    const float* const* inputChannelData,
    int numInputChannels,
    float* const* outputChannelData,
    int numOutputChannels,
    int numSamples,
    const juce::AudioIODeviceCallbackContext& context)
{
    juce::ignoreUnused(outputChannelData, numOutputChannels, context);
    
    if (numInputChannels > 0 && inputChannelData[0] != nullptr)
    {
        juce::ScopedLock lock(callbackLock_);
        
        // Copy input data
        inputBuffer_.copyFrom(0, 0, inputChannelData[0], numSamples);
        
        // Calculate level
        if (levelCallback_)
        {
            float rms = calculateRMS(inputBuffer_);
            levelCallback_(rms);
        }
        
        // Send to callback
        if (audioCallback_)
        {
            audioCallback_(inputBuffer_);
        }
    }
}

void DeviceAudioInput::audioDeviceStopped()
{
   #if defined(DEBUG) || defined(_DEBUG)
    DBG("[DeviceAudioInput] Device stopped");
   #endif
}

juce::StringArray AudioInputSource::getAvailableDevices()
{
    juce::AudioDeviceManager tempManager;
    tempManager.initialise(1, 0, nullptr, false);
    
    auto* currentDevice = tempManager.getCurrentAudioDevice();
    auto deviceType = tempManager.getCurrentDeviceTypeObject();
    
    if (deviceType != nullptr)
    {
        return deviceType->getDeviceNames(true);  // true = input devices
    }
    
    return {};
}

juce::StringArray AudioInputSource::getAvailableTestFiles()
{
    juce::StringArray files;
    
    // Use same logic as FileAudioInput::getTestAudioDirectory() to find the directory
    juce::Array<juce::File> possiblePaths;
    
    // 1. Current working directory
    possiblePaths.add(juce::File::getCurrentWorkingDirectory()
                      .getChildFile("Resources")
                      .getChildFile("TestAudio"));
    
    // 2. Executable directory (Windows)
    possiblePaths.add(juce::File::getSpecialLocation(juce::File::currentExecutableFile)
                      .getParentDirectory()
                      .getChildFile("Resources")
                      .getChildFile("TestAudio"));
    
    // 3. Executable directory parent (for development)
    possiblePaths.add(juce::File::getSpecialLocation(juce::File::currentExecutableFile)
                      .getParentDirectory()
                      .getParentDirectory()
                      .getChildFile("Resources")
                      .getChildFile("TestAudio"));
    
    // 4. Project root (development)
    possiblePaths.add(juce::File::getCurrentWorkingDirectory()
                      .getParentDirectory()
                      .getChildFile("Resources")
                      .getChildFile("TestAudio"));
    
    // Find the first existing directory
    juce::File testDir;
    for (auto& path : possiblePaths)
    {
        if (path.exists() && path.isDirectory())
        {
            testDir = path;
            break;
        }
    }
    
    // If not found, use the first path anyway (will return empty)
    if (testDir.getFullPathName().isEmpty())
        testDir = possiblePaths[0];
    
    if (testDir.exists() && testDir.isDirectory())
    {
        auto fileArray = testDir.findChildFiles(
            juce::File::findFiles, 
            false, 
            "*.wav;*.mp3;*.flac;*.aiff;*.ogg"
        );
        
        for (auto& f : fileArray)
        {
            files.add(f.getFileName());
        }
    }
    
    return files;
}

float AudioInputSource::calculateRMS(const juce::AudioBuffer<float>& buffer)
{
    if (buffer.getNumSamples() == 0)
        return 0.0f;
    
    float sum = 0.0f;
    auto* data = buffer.getReadPointer(0);
    
    for (int i = 0; i < buffer.getNumSamples(); ++i)
    {
        sum += data[i] * data[i];
    }
    
    return std::sqrt(sum / buffer.getNumSamples());
}

} // namespace spm

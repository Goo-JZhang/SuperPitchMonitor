#include "FileAudioInput.h"
#include "../Utils/Logger.h"

namespace spm {

FileAudioInput::FileAudioInput()
{
    outputBuffer_.setSize(1, bufferSize_);
}

FileAudioInput::~FileAudioInput()
{
    stopTimer();
}

bool FileAudioInput::prepare(double sampleRate, int bufferSize)
{
    sampleRate_ = sampleRate;
    bufferSize_ = bufferSize;
    outputBuffer_.setSize(1, bufferSize_);
    
   #if defined(DEBUG) || defined(_DEBUG)
    DBG("[FileAudioInput] Prepared: " << sampleRate << "Hz, " << bufferSize << " samples");
   #endif
    
    return true;
}

juce::File FileAudioInput::getTestAudioDirectory()
{
    // Try different locations to find Resources/TestAudio
    juce::Array<juce::File> possiblePaths;
    
    // 1. Current working directory
    possiblePaths.add(juce::File::getCurrentWorkingDirectory()
                      .getChildFile("Resources")
                      .getChildFile("TestAudio"));
    
    // 2. Executable directory (Windows) - for deployed app
    possiblePaths.add(juce::File::getSpecialLocation(juce::File::currentExecutableFile)
                      .getParentDirectory()
                      .getChildFile("Resources")
                      .getChildFile("TestAudio"));
    
    // 3. Executable directory parent - one level up from build output
    possiblePaths.add(juce::File::getSpecialLocation(juce::File::currentExecutableFile)
                      .getParentDirectory()
                      .getParentDirectory()
                      .getChildFile("Resources")
                      .getChildFile("TestAudio"));
    
    // 4. Executable great-grandparent - for CMake build structure: build-windows/.../Debug/SPM.exe
    possiblePaths.add(juce::File::getSpecialLocation(juce::File::currentExecutableFile)
                      .getParentDirectory()  // Debug
                      .getParentDirectory()  // artefacts
                      .getParentDirectory()  // build-windows
                      .getChildFile("Resources")
                      .getChildFile("TestAudio"));
    
    // 5. Project root (development) - parent of CWD
    possiblePaths.add(juce::File::getCurrentWorkingDirectory()
                      .getParentDirectory()
                      .getChildFile("Resources")
                      .getChildFile("TestAudio"));
    
    // 6. macOS app bundle - from Contents/MacOS executable to project root
    auto exeFile = juce::File::getSpecialLocation(juce::File::currentExecutableFile);
    possiblePaths.add(exeFile
                      .getParentDirectory()  // MacOS
                      .getParentDirectory()  // Contents
                      .getParentDirectory()  // SuperPitchMonitor.app
                      .getParentDirectory()  // Project root
                      .getChildFile("Resources")
                      .getChildFile("TestAudio"));
    
    // 7. Direct from executable location (for development builds)
    possiblePaths.add(exeFile.getParentDirectory()  // MacOS or build dir
                      .getParentDirectory()  // Contents or build root
                      .getChildFile("Resources")
                      .getChildFile("TestAudio"));
    
    // 8. Hardcoded project path (fallback)
    possiblePaths.add(juce::File("C:\\SuperPitchMonitor")
                      .getChildFile("Resources")
                      .getChildFile("TestAudio"));

    fprintf(stderr, "[FileAudioInput] Searching for test audio directory...\n");
    for (auto& path : possiblePaths)
    {
        fprintf(stderr, "[FileAudioInput] Checking: %s - exists=%d, isDir=%d\n", 
                path.getFullPathName().toRawUTF8(), 
                (int)path.exists(), 
                (int)path.isDirectory());
        if (path.exists() && path.isDirectory())
        {
            fprintf(stderr, "[FileAudioInput] Found test audio directory: %s\n", 
                    path.getFullPathName().toRawUTF8());
            return path;
        }
    }

    // Return first path even if it doesn't exist (for error reporting)
    return possiblePaths[0];
}

juce::StringArray FileAudioInput::getAvailableTestFiles()
{
    juce::StringArray files;
    
    auto testDir = getTestAudioDirectory();
    
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
    
   #if defined(DEBUG) || defined(_DEBUG)
    DBG("[FileAudioInput] Found " << files.size() << " test files in " << testDir.getFullPathName());
   #endif
    
    return files;
}

bool FileAudioInput::loadFile(const juce::File& file)
{
    if (!file.existsAsFile())
    {
        SPM_LOG_ERROR("[FileAudioInput] File not found: " + file.getFullPathName());
        return false;
    }
    
    SPM_LOG_INFO("[FileAudioInput] Loading file: " + file.getFileName());
    
    juce::AudioFormatManager formatManager;
    formatManager.registerBasicFormats();
    
    std::unique_ptr<juce::AudioFormatReader> reader(formatManager.createReaderFor(file));
    
    if (reader == nullptr)
    {
        SPM_LOG_ERROR("[FileAudioInput] Failed to read file: " + file.getFileName());
        return false;
    }
    
    // Read entire file
    int numSamples = (int)reader->lengthInSamples;
    audioBuffer_.setSize((int)reader->numChannels, numSamples);
    
    reader->read(&audioBuffer_, 0, numSamples, 0, true, true);
    
    // Mix to mono if stereo
    if (audioBuffer_.getNumChannels() > 1)
    {
        auto* monoData = audioBuffer_.getWritePointer(0);
        for (int i = 0; i < numSamples; ++i)
        {
            float sum = 0.0f;
            for (int ch = 0; ch < audioBuffer_.getNumChannels(); ++ch)
            {
                sum += audioBuffer_.getSample(ch, i);
            }
            monoData[i] = sum / audioBuffer_.getNumChannels();
        }
        audioBuffer_.setSize(1, numSamples, true, true, false);
    }
    
    sampleRate_ = reader->sampleRate;
    numChannels_ = (int)reader->numChannels;
    currentPosition_ = 0.0;
    currentFileName_ = file.getFileName();
    
    // Detailed log: file loaded successfully
    SPM_LOG_INFO("[FileAudioInput] ====== FILE LOADED ======");
    SPM_LOG_INFO("[FileAudioInput] Filename: " + file.getFileName());
    SPM_LOG_INFO("[FileAudioInput] Full path: " + file.getFullPathName());
    SPM_LOG_INFO("[FileAudioInput] Duration: " + juce::String(getTotalDuration(), 3) + " sec");
    SPM_LOG_INFO("[FileAudioInput] Sample rate: " + juce::String(sampleRate_, 0) + " Hz");
    SPM_LOG_INFO("[FileAudioInput] Channels: " + juce::String(numChannels_));
    SPM_LOG_INFO("[FileAudioInput] Total samples: " + juce::String(audioBuffer_.getNumSamples()));
    
    sendChangeMessage();
    return true;
}

bool FileAudioInput::loadTestFile(const juce::String& fileName)
{
    fprintf(stderr, "[FileAudioInput] loadTestFile called: %s\n", fileName.toRawUTF8());
    auto testDir = getTestAudioDirectory();
    fprintf(stderr, "[FileAudioInput] Test directory: %s\n", testDir.getFullPathName().toRawUTF8());
    auto file = testDir.getChildFile(fileName);
    fprintf(stderr, "[FileAudioInput] Full file path: %s, exists=%d\n", 
            file.getFullPathName().toRawUTF8(), (int)file.existsAsFile());
    return loadFile(file);
}

void FileAudioInput::start()
{
    SPM_LOG_INFO("[FileAudioInput] ====== PLAYBACK START ======");
    SPM_LOG_INFO("[FileAudioInput] File: " + currentFileName_);
    SPM_LOG_INFO("[FileAudioInput] Audio info: " + juce::String(audioBuffer_.getNumSamples()) + " samples, " + 
                 juce::String(sampleRate_, 0) + "Hz, " + juce::String(numChannels_) + " ch");
    SPM_LOG_INFO("[FileAudioInput] Duration: " + juce::String(audioBuffer_.getNumSamples() / sampleRate_, 2) + " sec");
    
    if (audioBuffer_.getNumSamples() == 0)
    {
        SPM_LOG_ERROR("[FileAudioInput] Cannot start: no audio loaded");
        return;
    }
    
    if (isPlaying_)
    {
        SPM_LOG_WARNING("[FileAudioInput] Already playing");
        return;
    }
    
    isPlaying_ = true;
    startTimerHz(timerHz);
    
    SPM_LOG_INFO("[FileAudioInput] Started playback: " + currentFileName_ + " timerHz=" + juce::String(timerHz));
   #if defined(DEBUG) || defined(_DEBUG)
    DBG("[FileAudioInput] Started playback: " << currentFileName_);
   #endif
}

void FileAudioInput::stop()
{
    if (!isPlaying_)
        return;
    
    isPlaying_ = false;
    stopTimer();
    currentPosition_ = 0.0;
    
   #if defined(DEBUG) || defined(_DEBUG)
    DBG("[FileAudioInput] Stopped");
   #endif
}

void FileAudioInput::pause()
{
    isPlaying_ = false;
    stopTimer();
    
   #if defined(DEBUG) || defined(_DEBUG)
    DBG("[FileAudioInput] Paused at " << currentPosition_.load() << "s");
   #endif
}

void FileAudioInput::setPlayPosition(double positionSeconds)
{
    currentPosition_ = juce::jlimit(0.0, getTotalDuration(), positionSeconds);
}

double FileAudioInput::getPlayPosition() const
{
    return currentPosition_.load();
}

double FileAudioInput::getTotalDuration() const
{
    return audioBuffer_.getNumSamples() / sampleRate_;
}

void FileAudioInput::timerCallback()
{
    if (!isPlaying_)
        return;
    
    processAudioBlock(bufferSize_);
}

void FileAudioInput::processAudioBlock(int numSamples)
{
    if (audioBuffer_.getNumSamples() == 0)
        return;
    
    outputBuffer_.setSize(1, numSamples);
    outputBuffer_.clear();
    
    auto* outputData = outputBuffer_.getWritePointer(0);
    auto* inputData = audioBuffer_.getReadPointer(0);
    int totalSamples = audioBuffer_.getNumSamples();
    
    // Calculate read position
    double position = currentPosition_.load();
    int readPos = (int)(position * sampleRate_);
    
    // Read samples
    for (int i = 0; i < numSamples; ++i)
    {
        if (readPos + i >= totalSamples)
        {
            if (isLooping_)
            {
                readPos = 0;
                position = 0.0;
            }
            else
            {
                stop();
                return;
            }
        }
        
        outputData[i] = inputData[readPos + i];
    }
    
    // Update position
    double increment = (numSamples / sampleRate_) * playbackSpeed_;
    position += increment;
    
    if (position >= getTotalDuration())
    {
        if (isLooping_)
            position = 0.0;
        else
            position = getTotalDuration();
    }
    
    currentPosition_.store(position);
    
    // Calculate level
    if (levelCallback_)
    {
        float rms = calculateRMS(outputBuffer_);
        levelCallback_(rms);
    }
    
    // Send to callback
    if (audioCallback_)
    {
        audioCallback_(outputBuffer_);
    }
}

} // namespace spm

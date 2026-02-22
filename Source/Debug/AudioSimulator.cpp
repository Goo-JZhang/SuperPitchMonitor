#include "AudioSimulator.h"
#include "../Utils/Logger.h"

namespace spm {

AudioSimulator::AudioSimulator()
{
    outputBuffer_.setSize(1, bufferSize_);
    outputBuffer_.clear();
}

AudioSimulator::~AudioSimulator()
{
    stopTimer();
}

bool AudioSimulator::loadAudioFile(const juce::File& file)
{
    if (!file.existsAsFile())
        return false;
    
    juce::AudioFormatManager formatManager;
    formatManager.registerBasicFormats();
    
    std::unique_ptr<juce::AudioFormatReader> reader(formatManager.createReaderFor(file));
    
    if (reader == nullptr)
        return false;
    
    // Read entire file
    int numSamples = (int)reader->lengthInSamples;
    audioBuffer_.setSize((int)reader->numChannels, numSamples);
    
    reader->read(&audioBuffer_, 0, numSamples, 0, true, true);
    
    sampleRate_ = reader->sampleRate;
    currentPosition_ = 0.0;
    
    DBG("Loaded audio file: " << file.getFileName());
    DBG("  Duration: " << getTotalDuration() << " seconds");
    DBG("  Sample rate: " << sampleRate_ << " Hz");
    DBG("  Channels: " << audioBuffer_.getNumChannels());
    
    return true;
}

bool AudioSimulator::generateTestSignal(TestSignal type, double durationSeconds)
{
    audioBuffer_.clear();
    currentPosition_ = 0.0;
    
    switch (type)
    {
        case TestSignal::SineWave:
            generateSineWave(durationSeconds, 440.0f);  // A4
            break;
        case TestSignal::Chord:
            generateChord(durationSeconds);
            break;
        case TestSignal::Sweep:
            generateSweep(durationSeconds);
            break;
        case TestSignal::WhiteNoise:
            generateWhiteNoise(durationSeconds);
            break;
        case TestSignal::PinkNoise:
            generatePinkNoise(durationSeconds);
            break;
        default:
            return false;
    }
    
    return true;
}

void AudioSimulator::generateSineWave(double duration, float frequency)
{
    int numSamples = (int)(duration * sampleRate_);
    audioBuffer_.setSize(1, numSamples);
    
    auto* data = audioBuffer_.getWritePointer(0);
    
    for (int i = 0; i < numSamples; ++i)
    {
        float t = i / (float)sampleRate_;
        data[i] = std::sin(2.0f * juce::MathConstants<float>::pi * frequency * t);
    }
    
    // Add fade in/out
    int fadeSamples = (int)(0.01 * sampleRate_);  // 10ms
    for (int i = 0; i < fadeSamples && i < numSamples; ++i)
    {
        float fade = i / (float)fadeSamples;
        data[i] *= fade;
        data[numSamples - 1 - i] *= fade;
    }
}

void AudioSimulator::generateChord(double duration)
{
    // C Major chord: C4 (261.63), E4 (329.63), G4 (392.00)
    int numSamples = (int)(duration * sampleRate_);
    audioBuffer_.setSize(1, numSamples);
    
    auto* data = audioBuffer_.getWritePointer(0);
    
    float frequencies[] = {261.63f, 329.63f, 392.00f};
    float amplitudes[] = {0.33f, 0.33f, 0.34f};
    
    for (int i = 0; i < numSamples; ++i)
    {
        float t = i / (float)sampleRate_;
        float sample = 0.0f;
        
        for (int j = 0; j < 3; ++j)
        {
            sample += amplitudes[j] * std::sin(2.0f * juce::MathConstants<float>::pi * frequencies[j] * t);
        }
        
        data[i] = sample;
    }
    
    // Add fade in/out
    int fadeSamples = (int)(0.01 * sampleRate_);
    for (int i = 0; i < fadeSamples && i < numSamples; ++i)
    {
        float fade = i / (float)fadeSamples;
        data[i] *= fade;
        data[numSamples - 1 - i] *= fade;
    }
}

void AudioSimulator::generateSweep(double duration)
{
    int numSamples = (int)(duration * sampleRate_);
    audioBuffer_.setSize(1, numSamples);
    
    auto* data = audioBuffer_.getWritePointer(0);
    
    float startFreq = 100.0f;
    float endFreq = 2000.0f;
    
    for (int i = 0; i < numSamples; ++i)
    {
        float t = i / (float)sampleRate_;
        float progress = i / (float)numSamples;
        float freq = startFreq + (endFreq - startFreq) * progress;
        
        // Logarithmic sweep
        float phase = 2.0f * juce::MathConstants<float>::pi * 
                      (startFreq * t + (endFreq - startFreq) * t * t / (2.0f * duration));
        data[i] = std::sin(phase) * 0.5f;
    }
}

void AudioSimulator::generateWhiteNoise(double duration)
{
    int numSamples = (int)(duration * sampleRate_);
    audioBuffer_.setSize(1, numSamples);
    
    auto* data = audioBuffer_.getWritePointer(0);
    juce::Random random;
    
    for (int i = 0; i < numSamples; ++i)
    {
        data[i] = random.nextFloat() * 2.0f - 1.0f;
        data[i] *= 0.3f;  // Reduce amplitude
    }
}

void AudioSimulator::generatePinkNoise(double duration)
{
    int numSamples = (int)(duration * sampleRate_);
    audioBuffer_.setSize(1, numSamples);
    
    auto* data = audioBuffer_.getWritePointer(0);
    juce::Random random;
    
    // Simple pink noise generation (Voss-McCartney algorithm simplified)
    float b0 = 0.0f, b1 = 0.0f, b2 = 0.0f, b3 = 0.0f, b4 = 0.0f, b5 = 0.0f, b6 = 0.0f;
    
    for (int i = 0; i < numSamples; ++i)
    {
        float white = random.nextFloat() * 2.0f - 1.0f;
        
        b0 = 0.99886f * b0 + white * 0.0555179f;
        b1 = 0.99332f * b1 + white * 0.0750759f;
        b2 = 0.96900f * b2 + white * 0.1538520f;
        b3 = 0.86650f * b3 + white * 0.3104856f;
        b4 = 0.55000f * b4 + white * 0.5329522f;
        b5 = -0.7616f * b5 - white * 0.0168980f;
        
        data[i] = (b0 + b1 + b2 + b3 + b4 + b5 + b6 + white * 0.5362f) * 0.11f;
        b6 = white * 0.115926f;
    }
}

void AudioSimulator::start()
{
    if (audioBuffer_.getNumSamples() == 0)
    {
       #if defined(DEBUG) || defined(_DEBUG)
        DBG("[AudioSimulator] Cannot start: audio buffer is empty!");
       #endif
        return;
    }
    
    isPlaying_ = true;
    
   #if defined(DEBUG) || defined(_DEBUG)
    DBG("[AudioSimulator] Starting playback, buffer=" << audioBuffer_.getNumSamples() 
        << " samples, duration=" << getTotalDuration() << "s");
   #endif
    
    // Start timer (~23ms interval, simulating 44.1kHz/2048 callback rate)
    startTimerHz(60);
}

void AudioSimulator::stop()
{
    isPlaying_ = false;
    stopTimer();
    currentPosition_ = 0.0;
}

void AudioSimulator::pause()
{
    isPlaying_ = false;
    stopTimer();
}

void AudioSimulator::setPlayPosition(double positionSeconds)
{
    currentPosition_ = juce::jlimit(0.0, getTotalDuration(), positionSeconds);
}

double AudioSimulator::getPlayPosition() const
{
    return currentPosition_.load();
}

double AudioSimulator::getTotalDuration() const
{
    return audioBuffer_.getNumSamples() / sampleRate_;
}

void AudioSimulator::timerCallback()
{
    if (!isPlaying_)
        return;
    
   #if defined(DEBUG) || defined(_DEBUG)
    static int callbackCount = 0;
    if (++callbackCount % 60 == 0)  // Log every ~1 second
    {
        DBG("[AudioSimulator] Timer callback #" << callbackCount 
            << " position=" << currentPosition_.load());
    }
   #endif
    
    processAudioBlock(bufferSize_);
}

void AudioSimulator::processAudioBlock(int numSamples)
{
    if (audioBuffer_.getNumSamples() == 0)
        return;
    
    outputBuffer_.setSize(1, numSamples);
    outputBuffer_.clear();
    
    auto* outputData = outputBuffer_.getWritePointer(0);
    auto* inputData = audioBuffer_.getReadPointer(0);
    int totalSamples = audioBuffer_.getNumSamples();
    
    // Calculate read position (considering playback speed)
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
    float rms = 0.0f;
    for (int i = 0; i < numSamples; ++i)
    {
        rms += outputData[i] * outputData[i];
    }
    rms = std::sqrt(rms / numSamples);
    
    // Callbacks
   #if defined(DEBUG) || defined(_DEBUG)
    static int callbackCount = 0;
    bool shouldLog = (++callbackCount % 60 == 0);
    if (shouldLog)
    {
        DBG("[AudioSimulator] Processed " << numSamples << " samples, RMS=" << rms);
    }
   #endif
    
    if (levelCallback_)
    {
        levelCallback_(rms);
       #if defined(DEBUG) || defined(_DEBUG)
        if (shouldLog) DBG("[AudioSimulator] Level callback triggered");
       #endif
    }
    
    if (audioCallback_)
    {
        audioCallback_(outputBuffer_);
       #if defined(DEBUG) || defined(_DEBUG)
        if (shouldLog) DBG("[AudioSimulator] Audio callback triggered");
       #endif
    }
    else
    {
       #if defined(DEBUG) || defined(_DEBUG)
        if (shouldLog) DBG("[AudioSimulator] WARNING: No audio callback set!");
       #endif
    }
}

} // namespace spm


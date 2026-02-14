#include "SystemAudioInput.h"
#include "../Utils/Logger.h"

#if JUCE_WINDOWS
    #include <windows.h>
    #include <mmdeviceapi.h>
    #include <audioclient.h>
    #include <functiondiscoverykeys_devpkey.h>
    
    #pragma comment(lib, "ole32.lib")
    #pragma comment(lib, "mmdevapi.lib")
#endif

namespace spm {

#if JUCE_WINDOWS

class WASAPILoopbackCapture
{
public:
    WASAPILoopbackCapture() = default;
    ~WASAPILoopbackCapture() { stop(); }

    bool start(int sampleRate, int bufferSize, 
               std::function<void(const juce::AudioBuffer<float>&)> callback);
    void stop();
    bool isRunning() const { return running_; }

private:
    static DWORD WINAPI captureThreadProc(LPVOID param);
    void captureLoop();

    IMMDeviceEnumerator* enumerator_ = nullptr;
    IMMDevice* device_ = nullptr;
    IAudioClient* audioClient_ = nullptr;
    IAudioCaptureClient* captureClient_ = nullptr;
    
    WAVEFORMATEX* format_ = nullptr;
    HANDLE thread_ = nullptr;
    HANDLE stopEvent_ = nullptr;
    
    std::function<void(const juce::AudioBuffer<float>&)> callback_;
    int bufferSize_ = 512;
    int sampleRate_ = 44100;
    std::atomic<bool> running_{false};
    bool comInitialized_ = false;
};

bool WASAPILoopbackCapture::start(int sampleRate, int bufferSize,
                                   std::function<void(const juce::AudioBuffer<float>&)> callback)
{
    sampleRate_ = sampleRate;
    bufferSize_ = bufferSize;
    callback_ = callback;

    SPM_LOG_INFO("[WASAPILoopback] Starting capture at " + juce::String(sampleRate) + "Hz");

    // Initialize COM for this thread
    HRESULT hr = CoInitializeEx(nullptr, COINIT_APARTMENTTHREADED);
    if (FAILED(hr) && hr != RPC_E_CHANGED_MODE)
    {
        SPM_LOG_ERROR("[WASAPILoopback] CoInitializeEx failed: " + juce::String((int)hr));
        return false;
    }
    comInitialized_ = SUCCEEDED(hr);

    // Create device enumerator
    hr = CoCreateInstance(__uuidof(MMDeviceEnumerator), nullptr, 
                          CLSCTX_ALL, IID_PPV_ARGS(&enumerator_));
    if (FAILED(hr))
    {
        SPM_LOG_ERROR("[WASAPILoopback] Failed to create device enumerator: " + juce::String((int)hr));
        return false;
    }

    // Get default output device (eRender = rendering/playback device)
    hr = enumerator_->GetDefaultAudioEndpoint(eRender, eConsole, &device_);
    if (FAILED(hr))
    {
        SPM_LOG_ERROR("[WASAPILoopback] Failed to get default audio endpoint: " + juce::String((int)hr));
        return false;
    }

    // Get device name for logging
    IPropertyStore* props = nullptr;
    hr = device_->OpenPropertyStore(STGM_READ, &props);
    if (SUCCEEDED(hr))
    {
        PROPVARIANT varName;
        PropVariantInit(&varName);
        hr = props->GetValue(PKEY_Device_FriendlyName, &varName);
        if (SUCCEEDED(hr) && varName.vt == VT_LPWSTR)
        {
            // Properly convert wide string to UTF-8
            int utf8Size = WideCharToMultiByte(CP_UTF8, 0, varName.pwszVal, -1, nullptr, 0, nullptr, nullptr);
            if (utf8Size > 0)
            {
                juce::HeapBlock<char> utf8Buffer(utf8Size);
                WideCharToMultiByte(CP_UTF8, 0, varName.pwszVal, -1, utf8Buffer, utf8Size, nullptr, nullptr);
                juce::String deviceName(juce::String::fromUTF8(utf8Buffer, utf8Size - 1));
                SPM_LOG_INFO("[WASAPILoopback] Using device: " + deviceName);
            }
        }
        PropVariantClear(&varName);
        props->Release();
    }

    // Activate audio client
    hr = device_->Activate(__uuidof(IAudioClient), CLSCTX_ALL, nullptr, (void**)&audioClient_);
    if (FAILED(hr))
    {
        SPM_LOG_ERROR("[WASAPILoopback] Failed to activate audio client: " + juce::String((int)hr));
        return false;
    }

    // Get mix format
    hr = audioClient_->GetMixFormat(&format_);
    if (FAILED(hr))
    {
        SPM_LOG_ERROR("[WASAPILoopback] Failed to get mix format: " + juce::String((int)hr));
        return false;
    }

    SPM_LOG_INFO("[WASAPILoopback] Device format: " + juce::String((int)format_->nSamplesPerSec) + "Hz, " +
                 juce::String((int)format_->nChannels) + " channels, " +
                 juce::String((int)format_->wBitsPerSample) + " bits");

    // Adjust format to match requested sample rate if possible
    if (format_->nSamplesPerSec != (DWORD)sampleRate_)
    {
        SPM_LOG_INFO("[WASAPILoopback] Adjusting sample rate from " + juce::String((int)format_->nSamplesPerSec) +
                     " to " + juce::String(sampleRate_));
        format_->nSamplesPerSec = sampleRate_;
        format_->nAvgBytesPerSec = format_->nSamplesPerSec * format_->nBlockAlign;
    }

    // Initialize audio client in loopback mode
    // AUDCLNT_STREAMFLAGS_LOOPBACK = capture output
    // AUDCLNT_STREAMFLAGS_AUTOCONVERTPCM = allow format conversion
    REFERENCE_TIME bufferDuration = (REFERENCE_TIME)(bufferSize_ * 10000000.0 / sampleRate_);
    
    hr = audioClient_->Initialize(
        AUDCLNT_SHAREMODE_SHARED,
        AUDCLNT_STREAMFLAGS_LOOPBACK,
        bufferDuration,
        0,
        format_,
        nullptr);

    if (FAILED(hr))
    {
        SPM_LOG_ERROR("[WASAPILoopback] Failed to initialize audio client: " + juce::String((int)hr));
        
        // Try with device native format
        CoTaskMemFree(format_);
        hr = audioClient_->GetMixFormat(&format_);
        if (SUCCEEDED(hr))
        {
            SPM_LOG_INFO("[WASAPILoopback] Retrying with native format: " + juce::String((int)format_->nSamplesPerSec) + "Hz");
            hr = audioClient_->Initialize(
                AUDCLNT_SHAREMODE_SHARED,
                AUDCLNT_STREAMFLAGS_LOOPBACK,
                bufferDuration,
                0,
                format_,
                nullptr);
        }
        
        if (FAILED(hr))
        {
            SPM_LOG_ERROR("[WASAPILoopback] Failed to initialize audio client (2nd attempt): " + juce::String((int)hr));
            return false;
        }
    }

    // Get capture client
    hr = audioClient_->GetService(IID_PPV_ARGS(&captureClient_));
    if (FAILED(hr))
    {
        SPM_LOG_ERROR("[WASAPILoopback] Failed to get capture client: " + juce::String((int)hr));
        return false;
    }

    // Create stop event
    stopEvent_ = CreateEvent(nullptr, FALSE, FALSE, nullptr);
    if (!stopEvent_)
    {
        SPM_LOG_ERROR("[WASAPILoopback] Failed to create stop event");
        return false;
    }

    // Start audio client
    hr = audioClient_->Start();
    if (FAILED(hr))
    {
        SPM_LOG_ERROR("[WASAPILoopback] Failed to start audio client: " + juce::String((int)hr));
        return false;
    }

    SPM_LOG_INFO("[WASAPILoopback] Audio client started successfully");

    // Start capture thread
    running_ = true;
    thread_ = CreateThread(nullptr, 0, captureThreadProc, this, 0, nullptr);
    if (!thread_)
    {
        SPM_LOG_ERROR("[WASAPILoopback] Failed to create capture thread");
        stop();
        return false;
    }

    SPM_LOG_INFO("[WASAPILoopback] Capture thread started");
    return true;
}

void WASAPILoopbackCapture::stop()
{
    SPM_LOG_INFO("[WASAPILoopback] Stopping capture...");
    running_ = false;

    if (stopEvent_)
    {
        SetEvent(stopEvent_);
    }

    if (thread_)
    {
        WaitForSingleObject(thread_, 3000);
        CloseHandle(thread_);
        thread_ = nullptr;
    }

    if (audioClient_)
    {
        audioClient_->Stop();
    }

    if (stopEvent_)
    {
        CloseHandle(stopEvent_);
        stopEvent_ = nullptr;
    }

    if (captureClient_)
    {
        captureClient_->Release();
        captureClient_ = nullptr;
    }

    if (audioClient_)
    {
        audioClient_->Release();
        audioClient_ = nullptr;
    }

    if (format_)
    {
        CoTaskMemFree(format_);
        format_ = nullptr;
    }

    if (device_)
    {
        device_->Release();
        device_ = nullptr;
    }

    if (enumerator_)
    {
        enumerator_->Release();
        enumerator_ = nullptr;
    }

    if (comInitialized_)
    {
        CoUninitialize();
        comInitialized_ = false;
    }

    SPM_LOG_INFO("[WASAPILoopback] Stopped");
}

DWORD WINAPI WASAPILoopbackCapture::captureThreadProc(LPVOID param)
{
    auto* capture = static_cast<WASAPILoopbackCapture*>(param);
    capture->captureLoop();
    return 0;
}

void WASAPILoopbackCapture::captureLoop()
{
    // Initialize COM for this thread
    HRESULT hr = CoInitializeEx(nullptr, COINIT_APARTMENTTHREADED);
    bool comInit = SUCCEEDED(hr);

    juce::AudioBuffer<float> buffer(2, bufferSize_);
    int channels = juce::jmin(2, (int)format_->nChannels);
    
    SPM_LOG_INFO("[WASAPILoopback] Capture loop started, channels=" + juce::String(channels));

    UINT32 packetLength = 0;
    int emptyPacketCount = 0;

    while (running_)
    {
        // Check for stop event with timeout
        if (WaitForSingleObject(stopEvent_, 1) == WAIT_OBJECT_0)
            break;

        // Get next packet size
        HRESULT hr = captureClient_->GetNextPacketSize(&packetLength);
        if (FAILED(hr))
        {
            continue;
        }

        if (packetLength == 0)
        {
            // No data available, wait a bit
            emptyPacketCount++;
            if (emptyPacketCount > 1000)
            {
                // Log if we've been waiting too long
                SPM_LOG_INFO("[WASAPILoopback] No audio data for extended period");
                emptyPacketCount = 0;
            }
            Sleep(1);
            continue;
        }

        emptyPacketCount = 0;

        // Get buffer from capture client
        BYTE* data = nullptr;
        UINT32 numFramesAvailable = 0;
        DWORD flags = 0;

        hr = captureClient_->GetBuffer(&data, &numFramesAvailable, &flags, nullptr, nullptr);
        if (FAILED(hr) || !data)
        {
            continue;
        }

        // Check if data is silent
        if (flags & AUDCLNT_BUFFERFLAGS_SILENT)
        {
            // Fill with silence
            buffer.clear();
        }
        else
        {
            int numChannels = format_->nChannels;
            int bitsPerSample = format_->wBitsPerSample;
            
            // Convert to float based on format
            if (bitsPerSample == 32 && format_->wFormatTag == WAVE_FORMAT_IEEE_FLOAT)
            {
                // Float format
                float* floatData = reinterpret_cast<float*>(data);
                for (int ch = 0; ch < channels; ++ch)
                {
                    for (UINT32 i = 0; i < numFramesAvailable && i < (UINT32)bufferSize_; ++i)
                    {
                        buffer.setSample(ch, i, floatData[i * numChannels + ch]);
                    }
                }
            }
            else if (bitsPerSample == 16)
            {
                // 16-bit integer
                int16_t* intData = reinterpret_cast<int16_t*>(data);
                for (int ch = 0; ch < channels; ++ch)
                {
                    for (UINT32 i = 0; i < numFramesAvailable && i < (UINT32)bufferSize_; ++i)
                    {
                        float sample = intData[i * numChannels + ch] / 32768.0f;
                        buffer.setSample(ch, i, sample);
                    }
                }
            }
            else if (bitsPerSample == 32)
            {
                // 32-bit integer
                int32_t* intData = reinterpret_cast<int32_t*>(data);
                for (int ch = 0; ch < channels; ++ch)
                {
                    for (UINT32 i = 0; i < numFramesAvailable && i < (UINT32)bufferSize_; ++i)
                    {
                        float sample = intData[i * numChannels + ch] / 2147483648.0f;
                        buffer.setSample(ch, i, sample);
                    }
                }
            }
            else
            {
                // Unsupported format, clear buffer
                buffer.clear();
            }
        }

        // Release buffer
        captureClient_->ReleaseBuffer(numFramesAvailable);

        // Send to callback
        if (callback_ && numFramesAvailable > 0)
        {
            callback_(buffer);
        }
    }

    if (comInit)
    {
        CoUninitialize();
    }

    SPM_LOG_INFO("[WASAPILoopback] Capture loop ended");
}

#endif  // JUCE_WINDOWS

//=============================================================================
// SystemAudioInput Implementation
//=============================================================================

SystemAudioInput::SystemAudioInput()
{
#if JUCE_WINDOWS
    loopbackCapture_ = std::make_unique<WASAPILoopbackCapture>();
#endif
}

SystemAudioInput::~SystemAudioInput()
{
    stop();
}

bool SystemAudioInput::isSupported()
{
#if JUCE_WINDOWS
    return true;
#else
    return false;
#endif
}

bool SystemAudioInput::prepare(double sampleRate, int bufferSize)
{
    if (!isSupported())
    {
        SPM_LOG_ERROR("[SystemAudioInput] Not supported on this platform");
        return false;
    }

    sampleRate_ = sampleRate;
    bufferSize_ = bufferSize;
    
    outputBuffer_.setSize(1, bufferSize);
    
    isPrepared_ = true;
    SPM_LOG_INFO("[SystemAudioInput] Prepared: " + juce::String(sampleRate, 0) + "Hz, " + 
                 juce::String(bufferSize) + " samples");
    return true;
}

void SystemAudioInput::start()
{
    if (isActive_ || !isPrepared_)
    {
        SPM_LOG_WARNING("[SystemAudioInput] Cannot start: active=" + juce::String((int)isActive_) + 
                        " prepared=" + juce::String((int)isPrepared_));
        return;
    }

#if JUCE_WINDOWS
    if (loopbackCapture_)
    {
        bool started = loopbackCapture_->start((int)sampleRate_, bufferSize_,
            [this](const juce::AudioBuffer<float>& buffer)
        {
            // Mix to mono
            int numSamples = juce::jmin(bufferSize_, buffer.getNumSamples());
            
            if (buffer.getNumChannels() >= 2)
            {
                auto* out = outputBuffer_.getWritePointer(0);
                auto* inL = buffer.getReadPointer(0);
                auto* inR = buffer.getReadPointer(1);
                
                for (int i = 0; i < numSamples; ++i)
                {
                    out[i] = (inL[i] + inR[i]) * 0.5f;
                }
            }
            else
            {
                outputBuffer_.copyFrom(0, 0, buffer.getReadPointer(0), numSamples);
            }

            // Calculate level
            if (levelCallback_)
            {
                float rms = 0.0f;
                auto* data = outputBuffer_.getReadPointer(0);
                for (int i = 0; i < numSamples; ++i)
                {
                    rms += data[i] * data[i];
                }
                rms = std::sqrt(rms / numSamples);
                levelCallback_(rms);
            }

            // Send to audio callback
            if (audioCallback_)
            {
                audioCallback_(outputBuffer_);
            }
        });

        if (!started)
        {
            SPM_LOG_ERROR("[SystemAudioInput] Failed to start loopback capture");
            return;
        }
    }
#endif

    isActive_ = true;
    SPM_LOG_INFO("[SystemAudioInput] Started");
}

void SystemAudioInput::stop()
{
    if (!isActive_)
        return;

#if JUCE_WINDOWS
    if (loopbackCapture_)
    {
        loopbackCapture_->stop();
    }
#endif

    isActive_ = false;
    SPM_LOG_INFO("[SystemAudioInput] Stopped");
}

bool SystemAudioInput::isActive() const
{
    return isActive_;
}

double SystemAudioInput::getSampleRate() const
{
    return sampleRate_;
}

int SystemAudioInput::getBufferSize() const
{
    return bufferSize_;
}

} // namespace spm

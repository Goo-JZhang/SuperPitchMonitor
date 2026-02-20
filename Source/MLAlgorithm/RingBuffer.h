/*
  =============================================================================-
    RingBuffer.h
    Thread-safe ring buffer for audio streaming
  =============================================================================-
*/

#pragma once

#include <JuceHeader.h>
#include <vector>
#include <atomic>
#include <cstring>

namespace spm
{

/**
 * Single-producer single-consumer ring buffer
 * Optimized for the case: producer = audio callback, consumer = inference thread
 */
template <typename T>
class RingBuffer
{
public:
    explicit RingBuffer(size_t capacity)
        : buffer_(capacity), capacity_(capacity), writePos_(0)
    {
        jassert(capacity > 0);
        std::fill(buffer_.begin(), buffer_.end(), T(0));
    }
    
    /** Get total capacity */
    size_t getCapacity() const { return capacity_; }
    
    /** 
     * Write data to ring buffer (producer thread only)
     * @return number of samples actually written
     */
    size_t write(const T* data, size_t numSamples)
    {
        const size_t writePos = writePos_.load(std::memory_order_relaxed);
        
        for (size_t i = 0; i < numSamples; ++i)
        {
            buffer_[(writePos + i) % capacity_] = data[i];
        }
        
        writePos_.store((writePos + numSamples) % capacity_, std::memory_order_release);
        return numSamples;
    }
    
    /**
     * Read the latest N samples from the buffer
     * This reads backwards from the most recent write position
     * @param numSamples: number of samples to read (must be <= capacity)
     * @param output: pre-allocated buffer of size numSamples
     * @return true if successful
     */
    bool readLatest(size_t numSamples, T* output) const
    {
        if (numSamples > capacity_)
            return false;
        
        const size_t writePos = writePos_.load(std::memory_order_acquire);
        
        // Read backwards from write position
        // output[0] will be the oldest sample in the window
        // output[numSamples-1] will be the newest sample
        for (size_t i = 0; i < numSamples; ++i)
        {
            // Go back numSamples-1-i samples from writePos
            size_t readIndex = (writePos + capacity_ - (numSamples - i)) % capacity_;
            output[i] = buffer_[readIndex];
        }
        
        return true;
    }
    
    /** Clear the buffer (fill with zeros) */
    void clear()
    {
        std::fill(buffer_.begin(), buffer_.end(), T(0));
        writePos_.store(0, std::memory_order_release);
    }

private:
    std::vector<T> buffer_;
    const size_t capacity_;
    std::atomic<size_t> writePos_{0};
    
    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(RingBuffer)
};

using AudioRingBuffer = RingBuffer<float>;

} // namespace spm

#pragma once

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

namespace spm {

/**
 * Audio Buffer FIFO
 * Used for data transfer between audio callback and processing thread
 */
template<size_t BufferSize>
class AudioBufferFifo
{
public:
    AudioBufferFifo() : fifo_(BufferSize) {}

    bool push(const juce::AudioBuffer<float>& buffer)
    {
        int start1, size1, start2, size2;
        fifo_.prepareToWrite(1, start1, size1, start2, size2);

        if (size1 > 0)
        {
            buffers_[start1] = buffer;
            fifo_.finishedWrite(1);
            return true;
        }
        return false;
    }

    bool pop(juce::AudioBuffer<float>& buffer)
    {
        int start1, size1, start2, size2;
        fifo_.prepareToRead(1, start1, size1, start2, size2);

        if (size1 > 0)
        {
            buffer = buffers_[start1];
            fifo_.finishedRead(1);
            return true;
        }
        return false;
    }

private:
    juce::AbstractFIFO fifo_;
    std::array<juce::AudioBuffer<float>, BufferSize> buffers_;
};

} // namespace spm


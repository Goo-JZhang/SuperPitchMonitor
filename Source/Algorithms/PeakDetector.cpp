#include "PeakDetector.h"

namespace spm {

std::vector<PeakDetector::Peak> PeakDetector::detect(
    const std::vector<float>& magnitudes,
    const std::vector<float>& frequencies,
    float threshold)
{
    std::vector<Peak> peaks;
    
    // Find maximum value
    float maxMag = *std::max_element(magnitudes.begin(), magnitudes.end());
    float thresholdValue = maxMag * threshold;
    
    // Local maximum detection
    for (size_t i = 2; i < magnitudes.size() - 2; ++i)
    {
        float mag = magnitudes[i];
        
        if (mag < thresholdValue)
            continue;
        
        // 5-point comparison
        if (mag > magnitudes[i-1] && mag > magnitudes[i-2] &&
            mag > magnitudes[i+1] && mag > magnitudes[i+2])
        {
            Peak peak;
            peak.binIndex = (int)i;
            peak.magnitude = mag;
            peak.frequency = frequencies[i];
            peaks.push_back(peak);
        }
    }
    
    return peaks;
}

} // namespace spm


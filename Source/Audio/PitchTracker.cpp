#include "PitchTracker.h"

namespace spm {

void PitchTracker::update(const PitchVector& currentCandidates)
{
    // TODO: Implement pitch tracking logic
}

const std::vector<PitchTracker::TrackedPitch>& PitchTracker::getActivePitches() const
{
    return activePitches_;
}

} // namespace spm


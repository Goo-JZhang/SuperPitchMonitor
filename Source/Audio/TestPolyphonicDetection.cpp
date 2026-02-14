// Quick test for polyphonic detection algorithm
#include <iostream>
#include <vector>
#include <cmath>

struct Peak {
    float frequency;
    float magnitude;
};

// Simulate C4-E4-G4-C5 chord (261.63, 329.63, 392.00, 523.25 Hz)
// With realistic harmonics
std::vector<Peak> createCMajor7Chord() {
    std::vector<Peak> peaks;
    
    // C4 fundamental
    peaks.push_back({261.63f, 1.0f});
    // C4 harmonics
    peaks.push_back({523.26f, 0.45f});  // C5 (2nd harmonic) 
    peaks.push_back({784.89f, 0.28f});  // G5 (3rd harmonic)
    peaks.push_back({1046.52f, 0.18f}); // C6 (4th harmonic)
    peaks.push_back({1308.15f, 0.12f}); // E6 (5th harmonic)
    
    // E4 fundamental
    peaks.push_back({329.63f, 0.85f});
    // E4 harmonics
    peaks.push_back({659.26f, 0.35f});  // E5
    peaks.push_back({988.89f, 0.20f});  // B5
    
    // G4 fundamental
    peaks.push_back({392.00f, 0.75f});
    // G4 harmonics  
    peaks.push_back({784.00f, 0.30f});  // G5 (overlaps with C4's 3rd harmonic)
    peaks.push_back({1176.0f, 0.15f});  // D6
    
    // C5 fundamental (already as C4's 2nd harmonic)
    
    // Some noise peaks
    peaks.push_back({220.0f, 0.4f});    // A3 - false candidate
    peaks.push_back({130.0f, 0.3f});    // C3 - false candidate (subharmonic)
    peaks.push_back({87.0f, 0.35f});    // F2 - false candidate (subharmonic of C4)
    
    // Sort by magnitude
    std::sort(peaks.begin(), peaks.end(), 
              [](const Peak& a, const Peak& b) { return a.magnitude > b.magnitude; });
    
    return peaks;
}

float freqToMidi(float freq) {
    return 69.0f + 12.0f * std::log2(freq / 440.0f);
}

// Simplified version of the new algorithm logic
bool evaluateIfFundamental(const std::vector<Peak>& allPeaks, size_t peakIndex, 
                           float& outConfidence, int& outOwnHarmonics) {
    const Peak& peak = allPeaks[peakIndex];
    float fundFreq = peak.frequency;
    
    // Find possible fundamentals (higher peaks that are integer multiples)
    std::vector<size_t> possibleFundamentals;
    for (size_t i = 0; i < allPeaks.size(); ++i) {
        if (i == peakIndex) continue;
        
        float otherFreq = allPeaks[i].frequency;
        if (otherFreq > fundFreq * 1.2f) {
            float ratio = otherFreq / fundFreq;
            int nearestInt = (int)(ratio + 0.5f);
            
            if (nearestInt >= 2 && nearestInt <= 8) {
                float deviation = std::abs(ratio - nearestInt) / nearestInt;
                if (deviation < 0.03f) {
                    possibleFundamentals.push_back(i);
                }
            }
        }
    }
    
    // Check which harmonics are "own" vs "borrowed"
    int ownHarmonics = 0;
    int borrowedHarmonics = 0;
    
    for (int h = 2; h <= 8; ++h) {
        float expectedFreq = fundFreq * h;
        
        for (size_t i = 0; i < allPeaks.size(); ++i) {
            const auto& p = allPeaks[i];
            float deviation = std::abs(p.frequency - expectedFreq) / expectedFreq;
            
            if (deviation < 0.03f) {
                // Check if this harmonic belongs to another fundamental
                bool isBorrowed = false;
                for (size_t fundIdx : possibleFundamentals) {
                    float fundFreq2 = allPeaks[fundIdx].frequency;
                    float ratio2 = p.frequency / fundFreq2;
                    int nearestInt2 = (int)(ratio2 + 0.5f);
                    
                    if (nearestInt2 >= 1 && nearestInt2 <= 8) {
                        float dev2 = std::abs(ratio2 - nearestInt2) / nearestInt2;
                        if (dev2 < 0.03f) {
                            if (allPeaks[fundIdx].magnitude >= peak.magnitude * 0.8f) {
                                isBorrowed = true;
                                borrowedHarmonics++;
                                break;
                            }
                        }
                    }
                }
                
                if (!isBorrowed) {
                    ownHarmonics++;
                }
                break;
            }
        }
    }
    
    outOwnHarmonics = ownHarmonics;
    
    // Confidence calculation
    bool isSubHarmonic = !possibleFundamentals.empty() && borrowedHarmonics >= ownHarmonics;
    float baseScore = peak.magnitude * 0.3f;
    float harmonicScore = ownHarmonics * 0.1f;
    float penalty = isSubHarmonic ? (0.3f + 0.5f * ((float)borrowedHarmonics / (ownHarmonics + borrowedHarmonics))) : 0.0f;
    
    if (ownHarmonics <= 2 && isSubHarmonic) {
        penalty += 0.2f;
    }
    
    outConfidence = baseScore + harmonicScore - penalty;
    
    return !isSubHarmonic;
}

int main() {
    auto peaks = createCMajor7Chord();
    
    std::cout << "=== Simulated C Major 7 Chord Analysis ===\n";
    std::cout << "Expected: C4(261.6), E4(329.6), G4(392.0), C5(523.3)\n\n";
    
    std::cout << "All peaks (sorted by magnitude):\n";
    for (size_t i = 0; i < peaks.size(); ++i) {
        std::cout << "  [" << i << "] " << peaks[i].frequency << " Hz (" 
                  << peaks[i].magnitude * 100 << "%)\n";
    }
    
    std::cout << "\n=== Evaluation Results ===\n";
    
    for (size_t i = 0; i < peaks.size(); ++i) {
        float confidence;
        int ownHarmonics;
        bool isFundamental = evaluateIfFundamental(peaks, i, confidence, ownHarmonics);
        
        std::cout << "  " << peaks[i].frequency << " Hz: ";
        std::cout << (isFundamental ? "[FUNDAMENTAL]" : "[subharmonic]");
        std::cout << " conf=" << confidence;
        std::cout << " ownHarmonics=" << ownHarmonics;
        std::cout << "\n";
    }
    
    return 0;
}

#!/usr/bin/env python3
"""Test for polyphonic detection algorithm - C Major 7 chord"""

import math

class Peak:
    def __init__(self, freq, mag):
        self.frequency = freq
        self.magnitude = mag

def create_c_major_7_chord():
    """Simulate C4-E4-G4-C5 chord with realistic harmonics"""
    peaks = []
    
    # C4 fundamental
    peaks.append(Peak(261.63, 1.0))
    # C4 harmonics
    peaks.append(Peak(523.26, 0.45))   # C5 (2nd harmonic) 
    peaks.append(Peak(784.89, 0.28))   # G5 (3rd harmonic)
    peaks.append(Peak(1046.52, 0.18))  # C6 (4th harmonic)
    peaks.append(Peak(1308.15, 0.12))  # E6 (5th harmonic)
    
    # E4 fundamental
    peaks.append(Peak(329.63, 0.85))
    # E4 harmonics
    peaks.append(Peak(659.26, 0.35))   # E5
    peaks.append(Peak(988.89, 0.20))   # B5
    
    # G4 fundamental
    peaks.append(Peak(392.00, 0.75))
    # G4 harmonics  
    peaks.append(Peak(784.00, 0.30))   # G5 (overlaps with C4's 3rd harmonic)
    peaks.append(Peak(1176.0, 0.15))   # D6
    
    # Some noise/false peaks
    peaks.append(Peak(220.0, 0.4))     # A3 - false candidate
    peaks.append(Peak(130.0, 0.3))     # C3 - false candidate
    peaks.append(Peak(87.0, 0.35))     # F2 - false candidate (subharmonic of C4)
    
    # Sort by magnitude descending
    peaks.sort(key=lambda p: p.magnitude, reverse=True)
    return peaks

def evaluate_as_fundamental(all_peaks, peak_index):
    """New algorithm with 'borrowed harmonic' detection"""
    peak = all_peaks[peak_index]
    fund_freq = peak.frequency
    
    # 1. Find possible fundamentals (higher peaks that are integer multiples)
    possible_fundamentals = []
    for i, other in enumerate(all_peaks):
        if i == peak_index:
            continue
        other_freq = other.frequency
        if other_freq > fund_freq * 1.2:
            ratio = other_freq / fund_freq
            nearest_int = round(ratio)
            if 2 <= nearest_int <= 8:
                deviation = abs(ratio - nearest_int) / nearest_int
                if deviation < 0.03:
                    possible_fundamentals.append(i)
    
    # 2. Check which harmonics are "own" vs "borrowed"
    own_harmonics = []
    borrowed_harmonics = []
    
    for h in range(2, 9):
        expected_freq = fund_freq * h
        
        for i, p in enumerate(all_peaks):
            deviation = abs(p.frequency - expected_freq) / expected_freq
            if deviation < 0.03:
                # Check if this harmonic belongs to another fundamental
                is_borrowed = False
                for fund_idx in possible_fundamentals:
                    fund_freq2 = all_peaks[fund_idx].frequency
                    ratio2 = p.frequency / fund_freq2
                    nearest_int2 = round(ratio2)
                    if 1 <= nearest_int2 <= 8:
                        dev2 = abs(ratio2 - nearest_int2) / nearest_int2
                        if dev2 < 0.03:
                            if all_peaks[fund_idx].magnitude >= peak.magnitude * 0.8:
                                is_borrowed = True
                                borrowed_harmonics.append((h, p.frequency, p.magnitude))
                                break
                
                if not is_borrowed:
                    own_harmonics.append((h, p.frequency, p.magnitude))
                break
    
    # 3. Calculate confidence
    is_subharmonic = len(possible_fundamentals) > 0 and len(borrowed_harmonics) >= len(own_harmonics)
    
    base_score = peak.magnitude * 0.3
    harmonic_score = len(own_harmonics) * 0.12
    
    penalty = 0.0
    if is_subharmonic:
        total = len(own_harmonics) + len(borrowed_harmonics)
        borrowed_ratio = len(borrowed_harmonics) / total if total > 0 else 0
        penalty = 0.3 + 0.5 * borrowed_ratio
        
        if len(own_harmonics) <= 2:
            penalty += 0.2
    
    confidence = base_score + harmonic_score - penalty
    
    return {
        'is_fundamental': not is_subharmonic,
        'confidence': confidence,
        'own_harmonics': own_harmonics,
        'borrowed_harmonics': borrowed_harmonics,
        'possible_fundamentals': [all_peaks[i].frequency for i in possible_fundamentals]
    }

def freq_to_note(freq):
    """Convert frequency to note name"""
    notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    midi = 69 + 12 * math.log2(freq / 440.0)
    note_idx = int(round(midi)) % 12
    octave = (int(round(midi)) // 12) - 1
    cents = (midi - round(midi)) * 100
    return f"{notes[note_idx]}{octave}", midi, cents

def main():
    peaks = create_c_major_7_chord()
    
    print("=" * 60)
    print("C Major 7 Chord Polyphonic Detection Test")
    print("=" * 60)
    print("\nExpected fundamentals: C4(261.6), E4(329.6), G4(392.0)")
    print("\nAll peaks (sorted by magnitude):")
    
    for i, p in enumerate(peaks):
        note, midi, cents = freq_to_note(p.frequency)
        print(f"  [{i:2}] {p.frequency:7.1f} Hz | {note:4} | {p.magnitude*100:5.1f}%")
    
    print("\n" + "=" * 60)
    print("Evaluation Results:")
    print("=" * 60)
    
    results = []
    for i, p in enumerate(peaks):
        result = evaluate_as_fundamental(peaks, i)
        note, midi, cents = freq_to_note(p.frequency)
        results.append((p.frequency, note, result, p.magnitude))
    
    # Sort by confidence
    results.sort(key=lambda x: x[2]['confidence'], reverse=True)
    
    for freq, note, result, mag in results:
        status = "[FUNDAMENTAL]" if result['is_fundamental'] else "[subharmonic]"
        own = len(result['own_harmonics'])
        borrowed = len(result['borrowed_harmonics'])
        
        print(f"\n{freq:7.1f} Hz ({note:4}): {status}")
        print(f"       Confidence: {result['confidence']:.3f} (mag={mag:.2f})")
        print(f"       Own harmonics: {own}, Borrowed: {borrowed}")
        if result['possible_fundamentals']:
            print(f"       Could be subharmonic of: {[f'{f:.1f}' for f in result['possible_fundamentals']]}")
        if result['own_harmonics']:
            print(f"       Own H: {[f'H{h[0]}={h[1]:.1f}' for h in result['own_harmonics']]}")
        if result['borrowed_harmonics']:
            print(f"       Borrowed H: {[f'H{h[0]}={h[1]:.1f}' for h in result['borrowed_harmonics']]}")
    
    # Summary
    print("\n" + "=" * 60)
    print("Summary: Detected Fundamentals")
    print("=" * 60)
    fundamentals = [(f, n, r['confidence']) for f, n, r, m in results if r['is_fundamental']]
    for freq, note, conf in sorted(fundamentals, key=lambda x: x[0]):
        deviation = abs(freq - round(freq, -1))  # rough deviation
        print(f"  {freq:7.1f} Hz ({note:4}) - confidence: {conf:.3f}")

if __name__ == "__main__":
    main()

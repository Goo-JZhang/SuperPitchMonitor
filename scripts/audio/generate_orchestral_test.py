"""
Generate orchestral test audio with complex harmonic content.
Tests the algorithm's ability to distinguish fundamentals from harmonics.
"""

import numpy as np
import wave
import struct

SAMPLE_RATE = 44100

def apply_brass_envelope(signal, attack=0.05, decay=0.15, sustain=0.8, release=0.3):
    """Apply realistic brass envelope (sharp attack, slow decay)."""
    total_samples = len(signal)
    envelope = np.zeros(total_samples)
    
    attack_samples = int(SAMPLE_RATE * attack)
    decay_samples = int(SAMPLE_RATE * decay)
    release_samples = int(SAMPLE_RATE * release)
    sustain_samples = total_samples - attack_samples - decay_samples - release_samples
    
    # Attack (fast)
    i = 0
    for t in range(attack_samples):
        envelope[i] = t / attack_samples
        i += 1
    
    # Decay to sustain
    for t in range(decay_samples):
        envelope[i] = 1.0 - (1.0 - sustain) * (t / decay_samples)
        i += 1
    
    # Sustain
    for t in range(max(0, sustain_samples)):
        envelope[i] = sustain
        i += 1
    
    # Release
    for t in range(release_samples):
        if i < total_samples:
            envelope[i] = sustain * (1.0 - t / release_samples)
            i += 1
    
    return signal * envelope

def apply_string_envelope(signal, attack=0.1, decay=0.2, sustain=0.6, release=0.5):
    """Apply string envelope (smooth attack, moderate decay)."""
    total_samples = len(signal)
    envelope = np.zeros(total_samples)
    
    attack_samples = int(SAMPLE_RATE * attack)
    decay_samples = int(SAMPLE_RATE * decay)
    release_samples = int(SAMPLE_RATE * release)
    sustain_samples = total_samples - attack_samples - decay_samples - release_samples
    
    i = 0
    for t in range(attack_samples):
        envelope[i] = (t / attack_samples) ** 2  # Quadratic attack
        i += 1
    
    for t in range(decay_samples):
        envelope[i] = 1.0 - (1.0 - sustain) * (t / decay_samples)
        i += 1
    
    for t in range(max(0, sustain_samples)):
        envelope[i] = sustain
        i += 1
    
    for t in range(release_samples):
        if i < total_samples:
            envelope[i] = sustain * (1.0 - t / release_samples)
            i += 1
    
    return signal * envelope

def generate_brass_tone(frequency, duration, vibrato_rate=5.0, vibrato_depth=0.03):
    """Generate brass tone with rich harmonics and vibrato."""
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), False)
    
    # Vibrato (pitch modulation)
    vibrato = 1.0 + vibrato_depth * np.sin(2 * np.pi * vibrato_rate * t)
    freq = frequency * vibrato
    
    # Brass timbre: strong odd and even harmonics
    harmonics = [1.0, 0.7, 0.8, 0.6, 0.5, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15, 0.12]
    
    signal = np.zeros_like(t)
    phase = 0.0
    for i, amp in enumerate(harmonics, 1):
        # Add some randomness to phase for realism
        phase_offset = np.random.uniform(0, 2*np.pi)
        signal += amp * np.sin(2 * np.pi * frequency * i * t + phase_offset)
    
    signal = apply_brass_envelope(signal)
    return signal / np.max(np.abs(signal)) * 0.8

def generate_string_tone(frequency, duration):
    """Generate string tone with rich but softer harmonics."""
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), False)
    
    # String timbre: moderate harmonics, rolled off high frequencies
    harmonics = [1.0, 0.5, 0.3, 0.2, 0.15, 0.1, 0.08, 0.06]
    
    signal = np.zeros_like(t)
    for i, amp in enumerate(harmonics, 1):
        phase_offset = np.random.uniform(0, 2*np.pi)
        signal += amp * np.sin(2 * np.pi * frequency * i * t + phase_offset)
    
    # Add slight detune for string "chorus" effect
    detune = 0.998
    signal += 0.5 * np.sin(2 * np.pi * frequency * detune * t)
    
    signal = apply_string_envelope(signal)
    return signal / np.max(np.abs(signal)) * 0.7

def generate_woodwind_tone(frequency, duration):
    """Generate woodwind tone with selective harmonics."""
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), False)
    
    # Woodwind: mostly odd harmonics
    harmonics = [1.0, 0.0, 0.4, 0.0, 0.2, 0.0, 0.1, 0.0, 0.05]
    
    signal = np.zeros_like(t)
    for i, amp in enumerate(harmonics, 1):
        if amp > 0:
            phase_offset = np.random.uniform(0, 2*np.pi)
            signal += amp * np.sin(2 * np.pi * frequency * i * t + phase_offset)
    
    # Attack/decay
    envelope = np.exp(-t * 3)  # Fast exponential decay
    envelope = np.maximum(envelope, 0.2)  # Sustain level
    
    return signal * envelope / np.max(np.abs(signal)) * 0.75

def generate_orchestral_chord(frequencies, durations, instrument_type='brass'):
    """Generate orchestral chord with multiple instrument tones."""
    max_duration = max(durations)
    total_samples = int(SAMPLE_RATE * max_duration)
    mix = np.zeros(total_samples)
    
    gen_func = {
        'brass': generate_brass_tone,
        'string': generate_string_tone,
        'woodwind': generate_woodwind_tone
    }.get(instrument_type, generate_brass_tone)
    
    for freq, duration in zip(frequencies, durations):
        tone = gen_func(freq, duration)
        samples = len(tone)
        mix[:samples] += tone
    
    # Normalize
    peak = np.max(np.abs(mix))
    if peak > 0:
        mix = mix / peak * 0.95
    
    return mix

def save_wav(filename, signal):
    """Save signal to WAV file."""
    signal = np.int16(signal * 32767)
    with wave.open(filename, 'w') as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(SAMPLE_RATE)
        f.writeframes(signal.tobytes())
    print(f"Generated: {filename}")

# Test frequencies
C4, D4, E4, F4, G4, A4, B4 = 261.63, 293.66, 329.63, 349.23, 392.00, 440.00, 493.88
D5, F5, A5 = 587.33, 698.46, 880.00
G3 = 196.00

# Generate challenging orchestral tests
print("="*60)
print("Generating Orchestral Test Audio")
print("="*60)

# 1. Brass chord with strong harmonics (common false positive scenario)
print("\n--- Brass Chords (Strong Harmonics) ---")
brass_chord = generate_orchestral_chord([C4, E4, G4], [2.0, 2.0, 2.0], 'brass')
save_wav("orchestral_brass_c_major.wav", brass_chord)

brass_chord7 = generate_orchestral_chord([G3, B4, D5, F5], [2.5, 2.5, 2.5, 2.5], 'brass')
save_wav("orchestral_brass_g7.wav", brass_chord7)

# 2. String section with rich but softer harmonics
print("\n--- String Section Chords ---")
string_chord = generate_orchestral_chord([C4, E4, G4, C4*2], [3.0, 3.0, 3.0, 3.0], 'string')
save_wav("orchestral_strings_c_major.wav", string_chord)

string_am = generate_orchestral_chord([A5/2, C4, E4], [2.5, 2.5, 2.5], 'string')
save_wav("orchestral_strings_a_minor.wav", string_am)

# 3. Woodwind choir (selective harmonics)
print("\n--- Woodwind Choir ---")
woodwind_chord = generate_orchestral_chord([E4, G4, B4, E4*2], [2.0, 2.0, 2.0, 2.0], 'woodwind')
save_wav("orchestral_woodwind_emajor.wav", woodwind_chord)

# 4. Orchestral tutti (full mix)
print("\n--- Orchestral Tutti ---")
tutti = (
    generate_orchestral_chord([C4, E4], [2.5, 2.5], 'brass') * 0.4 +
    generate_orchestral_chord([C4, G4], [2.5, 2.5], 'string') * 0.4 +
    generate_orchestral_chord([E4, G4], [2.5, 2.5], 'woodwind') * 0.2
)
# Ensure all arrays same length
min_len = min(len(tutti), len(generate_orchestral_chord([C4, E4], [2.5, 2.5], 'brass')))
tutti = tutti[:min_len]
save_wav("orchestral_tutti_c_major.wav", tutti)

# 5. Dense chord with overlapping harmonics
print("\n--- Dense Chords (Overlapping Harmonics) ---")
# C7 chord with 13th - very dense
c13 = generate_orchestral_chord([G3/2, B4/2, D5/2, F5/2, A5/2, D5], [3.0]*6, 'brass')
save_wav("orchestral_brass_c13.wav", c13)

# 6. Close-spaced chord (major 2nd interval - harmonics almost overlap)
close_chord = generate_orchestral_chord([C4, D4], [2.0, 2.0], 'brass')
save_wav("orchestral_brass_major2nd.wav", close_chord)

print("\n" + "="*60)
print("Orchestral test files generated!")
print("="*60)

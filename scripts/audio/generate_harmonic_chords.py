#!/usr/bin/env python3
"""
Generate harmonic-rich test chords for polyphonic detection testing.
Simulates realistic instruments with rich overtone series.
"""

import wave
import math
import struct
import os

SAMPLE_RATE = 44100
BITS_PER_SAMPLE = 16
MAX_AMPLITUDE = 32767

def midi_to_freq(midi_note):
    """Convert MIDI note number to frequency (Hz)."""
    return 440.0 * (2.0 ** ((midi_note - 69) / 12.0))

def generate_harmonic_tone(freq, duration, harmonics, amplitudes, sample_rate=44100):
    """
    Generate a tone with specified harmonics.
    
    Args:
        freq: Fundamental frequency
        duration: Duration in seconds
        harmonics: List of harmonic numbers (1=fundamental, 2=2nd harmonic, etc.)
        amplitudes: List of amplitudes for each harmonic (0-1)
    """
    num_samples = int(duration * sample_rate)
    samples = [0.0] * num_samples
    
    # ADSR envelope
    attack = min(0.05, duration * 0.1)
    decay = min(0.1, duration * 0.2)
    sustain_level = 0.7
    release = min(0.05, duration * 0.1)
    
    attack_samples = int(attack * sample_rate)
    decay_samples = int(decay * sample_rate)
    release_samples = int(release * sample_rate)
    sustain_samples = num_samples - attack_samples - decay_samples - release_samples
    
    for i, (h, amp) in enumerate(zip(harmonics, amplitudes)):
        harmonic_freq = freq * h
        if harmonic_freq > sample_rate / 2:  # Nyquist limit
            continue
            
        phase = 0.0
        phase_increment = 2.0 * math.pi * harmonic_freq / sample_rate
        
        for n in range(num_samples):
            # ADSR envelope
            if n < attack_samples:
                env = n / attack_samples
            elif n < attack_samples + decay_samples:
                progress = (n - attack_samples) / decay_samples
                env = 1.0 - (1.0 - sustain_level) * progress
            elif n < num_samples - release_samples:
                env = sustain_level
            else:
                progress = (n - (num_samples - release_samples)) / release_samples
                env = sustain_level * (1.0 - progress)
            
            samples[n] += amp * env * math.sin(phase)
            phase += phase_increment
            if phase > 2.0 * math.pi:
                phase -= 2.0 * math.pi
    
    # Normalize
    max_val = max(abs(s) for s in samples)
    if max_val > 0:
        samples = [s / max_val * 0.8 for s in samples]  # Leave headroom
    
    return samples

def mix_samples(*sample_lists):
    """Mix multiple sample lists together."""
    max_len = max(len(s) for s in sample_lists)
    result = [0.0] * max_len
    
    for samples in sample_lists:
        for i, s in enumerate(samples):
            result[i] += s / len(sample_lists)  # Normalize by voice count
    
    return result

def samples_to_wav(samples, filename):
    """Save samples to WAV file."""
    filepath = os.path.join(os.path.dirname(__file__), filename)
    with wave.open(filepath, 'w') as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(SAMPLE_RATE)
        
        for sample in samples:
            clipped = max(-1.0, min(1.0, sample))
            int_sample = int(clipped * MAX_AMPLITUDE)
            wav_file.writeframes(struct.pack('h', int_sample))
    
    print(f"Generated: {filename} ({len(samples)/SAMPLE_RATE:.2f}s)")

# Define instrument timbres with different harmonic profiles

# Piano-like: Strong fundamental, harmonics decay quickly
PIANO_HARMONICS = [1, 2, 3, 4, 5, 6, 7, 8]
PIANO_AMPS = [1.0, 0.5, 0.25, 0.125, 0.06, 0.03, 0.015, 0.008]

# String-like: Rich odd harmonics
STRING_HARMONICS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
STRING_AMPS = [1.0, 0.7, 0.5, 0.4, 0.3, 0.25, 0.2, 0.15, 0.1, 0.08, 0.06, 0.04]

# Brass-like: Strong mid harmonics
BRASS_HARMONICS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
BRASS_AMPS = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]

# Organ-like: All harmonics equal
ORGAN_HARMONICS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
ORGAN_AMPS = [1.0, 0.8, 0.7, 0.6, 0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15]

def generate_chord(notes, duration, harmonics, amps, name):
    """Generate a chord with specified notes and timbre."""
    all_samples = []
    
    for note in notes:
        freq = midi_to_freq(note)
        samples = generate_harmonic_tone(freq, duration, harmonics, amps)
        all_samples.append(samples)
    
    # Mix all voices
    mixed = mix_samples(*all_samples)
    samples_to_wav(mixed, name)

def main():
    print("=" * 60)
    print("Generating Harmonic-Rich Test Chords")
    print("=" * 60)
    
    # 1. Major triad with piano timbre (C Major)
    print("\n--- Piano Timbre Chords ---")
    generate_chord([60, 64, 67], 10.0, PIANO_HARMONICS, PIANO_AMPS, 
                   "chord_c_major_piano.wav")  # C4-E4-G4, 10s for FFT resolution
    generate_chord([60, 64, 67, 72], 10.0, PIANO_HARMONICS, PIANO_AMPS,
                   "chord_c_major_7_piano.wav")  # C-E-G-C5, 10s
    
    # 2. Minor triad with string timbre (A Minor)
    print("\n--- String Timbre Chords ---")
    generate_chord([57, 60, 64], 10.0, STRING_HARMONICS, STRING_AMPS,
                   "chord_a_minor_string.wav")  # A3-C4-E4, 10s
    generate_chord([57, 60, 64, 69], 10.0, STRING_HARMONICS, STRING_AMPS,
                   "chord_a_minor_7_string.wav")  # A-C-E-G, 10s
    
    # 3. Dominant 7th with brass timbre
    print("\n--- Brass Timbre Chords ---")
    generate_chord([55, 59, 62, 65], 10.0, BRASS_HARMONICS, BRASS_AMPS,
                   "chord_g7_brass.wav")  # G3-B3-D4-F4, 10s
    generate_chord([48, 52, 55, 59], 10.0, BRASS_HARMONICS, BRASS_AMPS,
                   "chord_c7_brass.wav")  # C3-E3-G3-Bb3, 10s
    
    # 4. Dense chords with organ timbre (many notes, rich harmonics)
    print("\n--- Organ Timbre (Dense) ---")
    generate_chord([60, 64, 67, 71, 74], 10.0, ORGAN_HARMONICS, ORGAN_AMPS,
                   "chord_c_major_9_organ.wav")  # C-E-G-B-D5, 10s
    generate_chord([48, 55, 60, 64, 67, 72], 10.0, ORGAN_HARMONICS, ORGAN_AMPS,
                   "chord_c_major_6voices_organ.wav")  # 6-note chord, 10s
    
    # 5. Challenging: close-spaced notes with rich harmonics
    print("\n--- Close-Spaced Chords (Challenging) ---")
    # Major 2nd interval (very close, harmonics overlap heavily)
    generate_chord([60, 62], 10.0, STRING_HARMONICS, STRING_AMPS,
                   "chord_major2nd_string.wav")  # C-D, 10s
    
    # Tritone (dissonant, 6 semitones)
    generate_chord([60, 66], 10.0, BRASS_HARMONICS, BRASS_AMPS,
                   "chord_tritone_brass.wav")  # C-F#, 10s
    
    # Cluster chord (many adjacent notes)
    generate_chord([60, 61, 62, 63, 64], 10.0, ORGAN_HARMONICS, ORGAN_AMPS,
                   "chord_cluster_organ.wav")  # C-C#-D-D#-E, 10s
    
    # 6. Arpeggiated chords (simulating real playing)
    print("\n--- Arpeggiated Chords ---")
    # C major arpeggio
    duration_per_note = 0.5
    arpeg_samples = []
    notes = [60, 64, 67, 72, 67, 64]  # C-E-G-C5-G-E
    for note in notes:
        freq = midi_to_freq(note)
        samples = generate_harmonic_tone(freq, duration_per_note, PIANO_HARMONICS, PIANO_AMPS)
        arpeg_samples.extend(samples)
    samples_to_wav(arpeg_samples, "arpeggio_c_major.wav")
    
    print("\n" + "=" * 60)
    print("All harmonic-rich test files generated!")
    print("=" * 60)
    print("\nFiles generated:")
    print("  Piano: chord_c_major_piano.wav, chord_c_major_7_piano.wav")
    print("  String: chord_a_minor_string.wav, chord_a_minor_7_string.wav")
    print("  Brass: chord_g7_brass.wav, chord_c7_brass.wav")
    print("  Organ (dense): chord_c_major_9_organ.wav, chord_c_major_6voices_organ.wav")
    print("  Challenging: chord_major2nd_string.wav, chord_tritone_brass.wav, chord_cluster_organ.wav")
    print("  Arpeggio: arpeggio_c_major.wav")

if __name__ == "__main__":
    main()

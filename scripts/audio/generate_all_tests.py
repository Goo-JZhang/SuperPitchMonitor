#!/usr/bin/env python3
"""
SuperPitchMonitor Test Audio Generator

Generates all test audio files (10-second duration for 0.1Hz FFT resolution)
Output directory: {PROJECT_ROOT}/Resources/TestAudio/

Usage:
    python scripts/audio/generate_all_tests.py
"""

import sys
import os

# Add project root to path for imports
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
sys.path.insert(0, PROJECT_ROOT)

# Output directory (workspace/Resources/TestAudio)
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'Resources', 'TestAudio')

import numpy as np
import wave
import math
import struct
import json

# Constants
SAMPLE_RATE = 44100
DURATION = 10.0  # 10 seconds for 0.1Hz FFT resolution
MAX_AMPLITUDE = 32767

# Note frequencies (12-tone equal temperament, A4=440Hz)
NOTE_FREQS = {
    'C2': 65.41, 'C#2': 69.30, 'D2': 73.42, 'D#2': 77.78, 'E2': 82.41, 'F2': 87.31,
    'F#2': 92.50, 'G2': 98.00, 'G#2': 103.83, 'A2': 110.00, 'A#2': 116.54, 'B2': 123.47,
    'C3': 130.81, 'C#3': 138.59, 'D3': 146.83, 'D#3': 155.56, 'E3': 164.81, 'F3': 174.61,
    'F#3': 185.00, 'G3': 196.00, 'G#3': 207.65, 'A3': 220.00, 'A#3': 233.08, 'B3': 246.94,
    'C4': 261.63, 'C#4': 277.18, 'D4': 293.66, 'D#4': 311.13, 'E4': 329.63, 'F4': 349.23,
    'F#4': 369.99, 'G4': 392.00, 'G#4': 415.30, 'A4': 440.00, 'A#4': 466.16, 'B4': 493.88,
    'C5': 523.25, 'C#5': 554.37, 'D5': 587.33, 'D#5': 622.25, 'E5': 659.25, 'F5': 698.46,
    'F#5': 739.99, 'G5': 783.99, 'G#5': 830.61, 'A5': 880.00, 'A#5': 932.33, 'B5': 987.77,
    'C6': 1046.50, 'C#6': 1108.73, 'D6': 1174.66, 'D#6': 1244.51, 'E6': 1318.51, 'F6': 1396.91,
}

def midi_to_freq(midi_note):
    """Convert MIDI note to frequency"""
    return 440.0 * (2.0 ** ((midi_note - 69) / 12.0))

def save_wav(filename, samples):
    """Save samples as WAV file to Resources/TestAudio/"""
    filepath = os.path.join(OUTPUT_DIR, filename)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    samples = np.array(samples)
    max_val = np.max(np.abs(samples))
    if max_val > 0:
        samples = (samples / max_val * 0.9 * MAX_AMPLITUDE).astype(np.int16)
    else:
        samples = samples.astype(np.int16)
    
    with wave.open(filepath, 'w') as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(SAMPLE_RATE)
        wav.writeframes(samples.tobytes())
    
    print(f"  Generated: {filename} ({DURATION}s)")
    return filepath

def generate_sine_wave(freq, duration=DURATION):
    """Generate pure sine wave"""
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), False)
    samples = np.sin(2 * np.pi * freq * t) * 0.5
    
    fade_samples = int(0.01 * SAMPLE_RATE)
    if fade_samples > 0 and len(samples) > 2 * fade_samples:
        fade_in = np.linspace(0, 1, fade_samples)
        fade_out = np.linspace(1, 0, fade_samples)
        samples[:fade_samples] *= fade_in
        samples[-fade_samples:] *= fade_out
    
    return samples

def generate_piano_tone(freq, duration=DURATION):
    """Generate piano-like tone with harmonics"""
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), False)
    
    harmonics = [1.0, 0.5, 0.25, 0.125, 0.06, 0.03, 0.015, 0.008]
    
    samples = np.zeros_like(t)
    for i, amp in enumerate(harmonics):
        harmonic_freq = freq * (i + 1)
        if harmonic_freq < SAMPLE_RATE / 2:
            samples += amp * np.sin(2 * np.pi * harmonic_freq * t)
    
    # ADSR envelope
    attack = int(0.01 * SAMPLE_RATE)
    decay = int(0.1 * SAMPLE_RATE)
    release = int(0.2 * SAMPLE_RATE)
    sustain_len = len(samples) - attack - decay - release
    
    envelope = np.ones_like(samples)
    if attack > 0:
        envelope[:attack] = np.linspace(0, 1, attack)
    if decay > 0 and sustain_len > 0:
        envelope[attack:attack+decay] = np.linspace(1, 0.7, decay)
        envelope[attack+decay:attack+decay+sustain_len] = 0.7
    if release > 0:
        envelope[-release:] = np.linspace(0.7 if sustain_len > 0 else 1, 0, release)
    
    return samples * envelope * 0.5

def generate_string_tone(freq, duration=DURATION):
    """Generate string tone"""
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), False)
    
    harmonics = [1.0, 0.7, 0.5, 0.4, 0.3, 0.25, 0.2, 0.15, 0.1, 0.08, 0.06, 0.04]
    
    samples = np.zeros_like(t)
    for i, amp in enumerate(harmonics):
        harmonic_freq = freq * (i + 1)
        if harmonic_freq < SAMPLE_RATE / 2:
            detune = 1.0 + (np.random.random() - 0.5) * 0.001
            samples += amp * np.sin(2 * np.pi * harmonic_freq * detune * t)
    
    attack = int(0.1 * SAMPLE_RATE)
    release = int(0.3 * SAMPLE_RATE)
    
    envelope = np.ones_like(samples)
    if attack > 0:
        envelope[:attack] = np.linspace(0, 1, attack)
    if release > 0:
        envelope[-release:] = np.linspace(1, 0, release)
    
    return samples * envelope * 0.5

def generate_brass_tone(freq, duration=DURATION):
    """Generate brass tone"""
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), False)
    
    harmonics = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
    
    samples = np.zeros_like(t)
    for i, amp in enumerate(harmonics):
        harmonic_freq = freq * (i + 1)
        if harmonic_freq < SAMPLE_RATE / 2:
            samples += amp * np.sin(2 * np.pi * harmonic_freq * t)
    
    attack = int(0.05 * SAMPLE_RATE)
    release = int(0.2 * SAMPLE_RATE)
    
    envelope = np.ones_like(samples)
    if attack > 0:
        envelope[:attack] = np.linspace(0, 1, attack)
    if release > 0:
        envelope[-release:] = np.linspace(1, 0, release)
    
    return samples * envelope * 0.5

def generate_organ_tone(freq, duration=DURATION):
    """Generate organ tone"""
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), False)
    
    harmonics = [1.0, 0.8, 0.7, 0.6, 0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15]
    
    samples = np.zeros_like(t)
    for i, amp in enumerate(harmonics):
        harmonic_freq = freq * (i + 1)
        if harmonic_freq < SAMPLE_RATE / 2:
            samples += amp * np.sin(2 * np.pi * harmonic_freq * t)
    
    attack = int(0.02 * SAMPLE_RATE)
    release = int(0.05 * SAMPLE_RATE)
    
    envelope = np.ones_like(samples)
    if attack > 0:
        envelope[:attack] = np.linspace(0, 1, attack)
    if release > 0:
        envelope[-release:] = np.linspace(1, 0, release)
    
    return samples * envelope * 0.5

def mix_tracks(tracks):
    """Mix multiple tracks"""
    max_len = max(len(t) for t in tracks)
    result = np.zeros(max_len)
    for track in tracks:
        result[:len(track)] += track / len(tracks)
    return result

def generate_chord_file(filename, notes, tone_gen, description=""):
    """Generate chord file"""
    tracks = []
    fundamentals = []
    for note_name in notes:
        freq = NOTE_FREQS.get(note_name)
        if freq:
            tracks.append(tone_gen(freq))
            fundamentals.append({'freq': freq, 'note': note_name})
    
    if tracks:
        mixed = mix_tracks(tracks)
        save_wav(filename, mixed)
    
    return fundamentals

def main():
    print("="*70)
    print("SuperPitchMonitor Test Audio Generator")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Duration: {DURATION}s | FFT Resolution: {1/DURATION:.2f}Hz")
    print("="*70)
    
    test_files = {}
    
    # 1. Single tone tests
    print("\n[1/5] Single Tone Tests (Pure Sine Waves)")
    sine_tests = [
        ('sine_220hz.wav', 220.0, 'A3'),
        ('sine_440hz.wav', 440.0, 'A4'),
        ('sine_880hz.wav', 880.0, 'A5'),
    ]
    for filename, freq, note in sine_tests:
        samples = generate_sine_wave(freq)
        save_wav(filename, samples)
        test_files[filename] = {
            'type': 'single_tone',
            'description': f'Pure sine wave at {freq} Hz ({note})',
            'duration_sec': DURATION,
            'fundamentals': [{'freq': freq, 'note': note, 'midi': int(69 + 12 * np.log2(freq/440))}]
        }
    
    # 2. Single note tests
    print("\n[2/5] Single Note Tests (Piano-like)")
    note_tests = [
        ('piano_like_c.wav', 'C4', 261.63),
    ]
    for filename, note, freq in note_tests:
        samples = generate_piano_tone(freq)
        save_wav(filename, samples)
        test_files[filename] = {
            'type': 'single_note',
            'description': f'Piano-like {note} note with harmonics',
            'duration_sec': DURATION,
            'fundamentals': [{'freq': freq, 'note': note, 'midi': int(69 + 12 * np.log2(freq/440))}]
        }
    
    # 3. Simple chords
    print("\n[3/5] Simple Chord Tests")
    simple_chords = [
        ('chord_c_major_piano.wav', ['C4', 'E4', 'G4'], generate_piano_tone, 'C Major (C4-E4-G4)'),
        ('chord_c_major_7_piano.wav', ['C4', 'E4', 'G4', 'C5'], generate_piano_tone, 'C Major 7 (C4-E4-G4-C5)'),
        ('chord_a_minor.wav', ['A3', 'C4', 'E4'], generate_piano_tone, 'A Minor (A3-C4-E4)'),
        ('chord_g_major.wav', ['G3', 'B3', 'D4'], generate_piano_tone, 'G Major (G3-B3-D4)'),
    ]
    for filename, notes, tone_gen, desc in simple_chords:
        fundamentals = generate_chord_file(filename, notes, tone_gen)
        test_files[filename] = {
            'type': 'simple_chord',
            'description': desc,
            'duration_sec': DURATION,
            'fundamentals': [{'freq': f['freq'], 'note': f['note'], 
                            'midi': int(69 + 12 * np.log2(f['freq']/440))} for f in fundamentals]
        }
    
    # 4. Brass and organ
    print("\n[4/5] Brass and Organ Timbre Tests")
    brass_chords = [
        ('chord_c7_brass.wav', ['C4', 'E4', 'G4', 'A#4'], generate_brass_tone, 'C7 Brass (C4-E4-G4-Bb4)'),
    ]
    for filename, notes, tone_gen, desc in brass_chords:
        fundamentals = generate_chord_file(filename, notes, tone_gen)
        test_files[filename] = {
            'type': 'chord',
            'description': desc,
            'duration_sec': DURATION,
            'fundamentals': [{'freq': f['freq'], 'note': f['note'],
                            'midi': int(69 + 12 * np.log2(f['freq']/440))} for f in fundamentals]
        }
    
    # 5. Special signals
    print("\n[5/5] Special Test Signals")
    noise = np.random.uniform(-0.3, 0.3, int(SAMPLE_RATE * DURATION))
    save_wav('white_noise.wav', noise)
    test_files['white_noise.wav'] = {
        'type': 'noise',
        'description': 'White noise',
        'duration_sec': DURATION,
        'fundamentals': []
    }
    
    # Save ground truth
    ground_truth = {
        'description': 'Ground truth for SuperPitchMonitor test files',
        'version': '2.0',
        'sample_rate': SAMPLE_RATE,
        'duration_sec': DURATION,
        'fft_resolution_hz': 1.0 / DURATION,
        'files': test_files
    }
    
    gt_path = os.path.join(OUTPUT_DIR, 'test_ground_truth.json')
    with open(gt_path, 'w') as f:
        json.dump(ground_truth, f, indent=2)
    
    print(f"\n{'='*70}")
    print(f"Generated {len(test_files)} test files")
    print(f"Ground truth: {gt_path}")
    print(f"{'='*70}")

if __name__ == '__main__':
    main()

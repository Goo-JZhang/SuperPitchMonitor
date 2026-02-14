#!/usr/bin/env python3
"""
SuperPitchMonitor Test Client with High-Precision FFT Reference Analysis

This tool performs full-window FFT on test audio files to provide
a "gold standard" reference for validating SPM's real-time detection.

Usage:
    python test_client_with_fft.py --file chord_c_major_7_piano.wav
    python test_client_with_fft.py --compare --file chord_c_major_7_piano.wav
    python test_client_with_fft.py --fft-only --file chord_c_major_7_piano.wav
"""

import subprocess
import time
import json
import struct
import sys
import os
import argparse
import numpy as np
from typing import List, Dict, Any, Tuple
from scipy.io import wavfile
from scipy.fft import fft, fftfreq
from scipy.signal import find_peaks, windows

try:
    import win32file
    import win32pipe
    import pywintypes
except ImportError:
    print("Error: pywin32 not installed. Run: pip install pywin32")
    sys.exit(1)

PIPE_NAME = r'\\.\pipe\SPM_TestPipe'
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
GROUND_TRUTH_FILE = os.path.join(SCRIPT_DIR, 'Resources', 'TestAudio', 'test_ground_truth.json')
TEST_AUDIO_DIR = os.path.join(SCRIPT_DIR, 'Resources', 'TestAudio')


class HighPrecisionFFTAnalyzer:
    """
    High-precision FFT analyzer for reference analysis.
    
    Uses full-window FFT on long audio (10s recommended) to achieve
    0.1Hz frequency resolution for validating pitch detection.
    """
    
    def __init__(self, sample_rate: int = 44100, duration_sec: float = 10.0):
        self.sample_rate = sample_rate
        self.duration_sec = duration_sec
        self.n_samples = int(sample_rate * duration_sec)
        self.freq_resolution = sample_rate / self.n_samples
        
    def analyze_file(self, filepath: str, 
                     freq_min: float = 50.0, 
                     freq_max: float = 2000.0,
                     prominence_db: float = 20.0) -> Dict[str, Any]:
        """
        Perform high-precision FFT analysis on audio file.
        
        Returns:
            Dictionary with detected peaks and spectrum data
        """
        # Load audio
        sr, data = wavfile.read(filepath)
        
        if len(data.shape) > 1:
            data = data.mean(axis=1)  # Convert to mono
            
        # Normalize to float64
        if data.dtype == np.int16:
            data = data.astype(np.float64) / 32768.0
        elif data.dtype == np.int32:
            data = data.astype(np.float64) / 2147483648.0
        else:
            data = data.astype(np.float64)
            
        # Resample if necessary
        if sr != self.sample_rate:
            from scipy import signal
            data = signal.resample(data, int(len(data) * self.sample_rate / sr))
            
        # Pad or truncate to desired length
        if len(data) < self.n_samples:
            data = np.pad(data, (0, self.n_samples - len(data)), mode='constant')
        else:
            data = data[:self.n_samples]
            
        # Apply window to reduce spectral leakage
        window = windows.hann(len(data))
        windowed_data = data * window
        
        # Full-window FFT
        fft_result = fft(windowed_data)
        fft_magnitude = np.abs(fft_result[:len(fft_result)//2])
        
        # Convert to dB
        fft_db = 20 * np.log10(fft_magnitude + 1e-12)
        
        # Frequency axis
        freqs = fftfreq(len(data), 1.0/self.sample_rate)[:len(fft_result)//2]
        
        # Find peaks in frequency range
        freq_mask = (freqs >= freq_min) & (freqs <= freq_max)
        freqs_range = freqs[freq_mask]
        db_range = fft_db[freq_mask]
        
        # Peak detection
        prominence_linear = 10 ** (prominence_db / 20)
        peaks, properties = find_peaks(
            fft_magnitude[freq_mask], 
            prominence=prominence_linear,
            distance=int(5 / self.freq_resolution)  # Min 5Hz separation
        )
        
        # Refine peaks with parabolic interpolation
        detected_peaks = []
        for peak_idx in peaks:
            if peak_idx <= 0 or peak_idx >= len(freqs_range) - 1:
                continue
                
            # Parabolic interpolation for better frequency estimate
            alpha = fft_magnitude[freq_mask][peak_idx - 1]
            beta = fft_magnitude[freq_mask][peak_idx]
            gamma = fft_magnitude[freq_mask][peak_idx + 1]
            
            p = 0.5 * (alpha - gamma) / (alpha - 2*beta + gamma)
            interpolated_freq = freqs_range[peak_idx] + p * self.freq_resolution
            
            # Linear interpolation for magnitude
            interpolated_mag = beta - 0.25 * (alpha - gamma) * p
            interpolated_db = 20 * np.log10(interpolated_mag + 1e-12)
            
            detected_peaks.append({
                'frequency': float(interpolated_freq),
                'magnitude_db': float(interpolated_db),
                'magnitude_linear': float(interpolated_mag),
                'bin': int(np.where(freq_mask)[0][peak_idx])
            })
            
        # Sort by magnitude (descending)
        detected_peaks.sort(key=lambda x: x['magnitude_db'], reverse=True)
        
        return {
            'filepath': filepath,
            'sample_rate': self.sample_rate,
            'fft_size': len(data),
            'frequency_resolution_hz': self.freq_resolution,
            'duration_sec': len(data) / self.sample_rate,
            'freq_min': freq_min,
            'freq_max': freq_max,
            'detected_peaks': detected_peaks,
            'spectrum_db': fft_db[freq_mask].tolist(),
            'spectrum_freqs': freqs_range.tolist(),
            'max_db': float(np.max(db_range)),
            'mean_db': float(np.mean(db_range)),
            'noise_floor_db': float(np.percentile(db_range, 10))
        }
    
    def identify_fundamentals(self, fft_result: Dict, 
                              expected_fundamentals: List[Dict],
                              harmonic_tolerance: float = 0.02) -> Dict:
        """
        Identify which expected fundamentals are present in FFT result.
        """
        peaks = fft_result['detected_peaks']
        
        identified = []
        missing = []
        
        for expected in expected_fundamentals:
            exp_freq = expected['freq']
            exp_note = expected.get('note', '')
            
            # Check if fundamental is detected
            fundamental_match = None
            for peak in peaks:
                if abs(peak['frequency'] - exp_freq) / exp_freq < harmonic_tolerance:
                    fundamental_match = peak
                    break
                    
            # Check for harmonics
            harmonics_found = []
            for h in range(2, 8):
                harmonic_freq = exp_freq * h
                for peak in peaks:
                    if abs(peak['frequency'] - harmonic_freq) / harmonic_freq < harmonic_tolerance:
                        harmonics_found.append({
                            'order': h,
                            'frequency': peak['frequency'],
                            'magnitude_db': peak['magnitude_db']
                        })
                        break
                        
            if fundamental_match:
                identified.append({
                    'expected_freq': exp_freq,
                    'note': exp_note,
                    'detected_freq': fundamental_match['frequency'],
                    'magnitude_db': fundamental_match['magnitude_db'],
                    'harmonics_found': harmonics_found
                })
            else:
                missing.append({
                    'expected_freq': exp_freq,
                    'note': exp_note,
                    'harmonics_found': harmonics_found
                })
                
        return {
            'identified_fundamentals': identified,
            'missing_fundamentals': missing,
            'total_expected': len(expected_fundamentals),
            'total_identified': len(identified),
            'identification_rate': len(identified) / len(expected_fundamentals) if expected_fundamentals else 0
        }


class FFTComparisonReport:
    """Generate comparison report between FFT reference and SPM detection"""
    
    @staticmethod
    def generate(fft_result: Dict, spm_result: Dict, ground_truth: Dict) -> str:
        """Generate detailed comparison report"""
        lines = []
        lines.append("="*80)
        lines.append("SPM vs High-Precision FFT Reference Comparison")
        lines.append("="*80)
        lines.append(f"File: {fft_result['filepath']}")
        lines.append(f"FFT Resolution: {fft_result['frequency_resolution_hz']:.3f} Hz")
        lines.append(f"Analysis Duration: {fft_result['duration_sec']:.2f}s")
        lines.append("")
        
        # Ground Truth
        lines.append("-"*40)
        lines.append("GROUND TRUTH (Expected Fundamentals)")
        lines.append("-"*40)
        for f in ground_truth.get('fundamentals', []):
            lines.append(f"  {f['note']:6s} {f['freq']:7.2f} Hz")
        lines.append("")
        
        # FFT Reference Results
        lines.append("-"*40)
        lines.append("HIGH-PRECISION FFT REFERENCE")
        lines.append("-"*40)
        lines.append(f"Top 10 Peaks (by magnitude):")
        for i, peak in enumerate(fft_result['detected_peaks'][:10], 1):
            lines.append(f"  {i:2d}. {peak['frequency']:8.3f} Hz | {peak['magnitude_db']:6.1f} dB")
        lines.append("")
        
        # Check which fundamentals FFT detected
        analyzer = HighPrecisionFFTAnalyzer()
        fundamental_analysis = analyzer.identify_fundamentals(
            fft_result, 
            ground_truth.get('fundamentals', [])
        )
        
        lines.append("FFT Fundamental Identification:")
        lines.append(f"  Identified: {fundamental_analysis['total_identified']}/{fundamental_analysis['total_expected']}")
        for fund in fundamental_analysis['identified_fundamentals']:
            lines.append(f"  [OK] {fund['note']:6s} {fund['detected_freq']:8.3f} Hz ({fund['magnitude_db']:5.1f} dB)")
        for miss in fundamental_analysis['missing_fundamentals']:
            lines.append(f"  [MISS] {miss['note']:6s} {miss['expected_freq']:8.2f} Hz - MISSING")
            if miss['harmonics_found']:
                lines.append(f"      (but found {len(miss['harmonics_found'])} harmonics)")
        lines.append("")
        
        # SPM Results
        lines.append("-"*40)
        lines.append("SPM REAL-TIME DETECTION")
        lines.append("-"*40)
        
        spm_pitches = spm_result.get('detected_pitches', [])
        if spm_pitches:
            lines.append(f"Detected {len(spm_pitches)} pitches:")
            for p in sorted(spm_pitches, key=lambda x: x['frequency']):
                lines.append(f"  {p['frequency']:8.2f} Hz | conf: {p['confidence']:.2f} | H: {p['harmonicCount']}")
        else:
            lines.append("  No pitches detected")
        lines.append("")
        
        # Comparison
        lines.append("-"*40)
        lines.append("COMPARISON ANALYSIS")
        lines.append("-"*40)
        
        # Match FFT peaks to SPM detections
        fft_peaks = fft_result['detected_peaks']
        matches = []
        missed_by_spm = []
        false_positives = []
        
        tolerance_hz = 10.0
        
        # Check what FFT found but SPM missed
        for peak in fft_peaks[:15]:  # Top 15 FFT peaks
            fft_freq = peak['frequency']
            matched = False
            for spm_p in spm_pitches:
                if abs(spm_p['frequency'] - fft_freq) <= tolerance_hz:
                    matches.append({
                        'fft_freq': fft_freq,
                        'fft_db': peak['magnitude_db'],
                        'spm_freq': spm_p['frequency'],
                        'spm_conf': spm_p['confidence']
                    })
                    matched = True
                    break
            if not matched:
                # Check if it's a fundamental or strong harmonic
                is_significant = peak['magnitude_db'] > fft_result['max_db'] - 30
                if is_significant:
                    missed_by_spm.append(peak)
                    
        # Check what SPM found but FFT didn't
        for spm_p in spm_pitches:
            spm_freq = spm_p['frequency']
            matched = False
            for peak in fft_peaks:
                if abs(peak['frequency'] - spm_freq) <= tolerance_hz:
                    matched = True
                    break
            if not matched:
                false_positives.append(spm_p)
                
        lines.append(f"FFT-SPM Matches: {len(matches)}")
        for m in matches[:10]:
            lines.append(f"  [MATCH] {m['fft_freq']:8.2f} Hz (FFT: {m['fft_db']:5.1f}dB) | SPM: {m['spm_freq']:8.2f}Hz conf={m['spm_conf']:.2f}")
            
        lines.append("")
        lines.append(f"Detected by FFT but MISSED by SPM ({len(missed_by_spm)}):")
        if missed_by_spm:
            for peak in missed_by_spm[:10]:
                lines.append(f"  [FFT_ONLY] {peak['frequency']:8.3f} Hz | {peak['magnitude_db']:6.1f} dB")
        else:
            lines.append("  (None)")
            
        lines.append("")
        lines.append(f"Detected by SPM but NOT in FFT ({len(false_positives)}):")
        if false_positives:
            for p in false_positives:
                lines.append(f"  [SPM_ONLY] {p['frequency']:8.2f} Hz | conf: {p['confidence']:.2f}")
        else:
            lines.append("  (None)")
            
        lines.append("")
        lines.append("="*80)
        
        return "\n".join(lines)


class SPMTestClient:
    """SuperPitchMonitor test client (simplified)"""
    
    def __init__(self, pipe_name=PIPE_NAME):
        self.pipe_name = pipe_name
        self.pipe = None
        
    def connect(self, timeout=30):
        """Connect to SPM test pipe"""
        print(f"Connecting to {self.pipe_name}...")
        start = time.time()
        
        while time.time() - start < timeout:
            try:
                self.pipe = win32file.CreateFile(
                    self.pipe_name,
                    win32file.GENERIC_READ | win32file.GENERIC_WRITE,
                    0, None,
                    win32file.OPEN_EXISTING,
                    0, None
                )
                win32pipe.SetNamedPipeHandleState(self.pipe, win32pipe.PIPE_READMODE_MESSAGE, None, None)
                print("Connected!")
                return True
            except pywintypes.error as e:
                if e.winerror == 2:
                    time.sleep(0.5)
                    continue
                raise
        
        print("Connection timeout!")
        return False
    
    def disconnect(self):
        if self.pipe:
            win32file.CloseHandle(self.pipe)
            self.pipe = None
    
    def send_command(self, cmd_dict):
        """Send command and receive response"""
        if not self.pipe:
            raise RuntimeError("Not connected")
        
        json_data = json.dumps(cmd_dict).encode('utf-8')
        length = len(json_data)
        
        win32file.WriteFile(self.pipe, struct.pack('>I', length))
        win32file.WriteFile(self.pipe, json_data)
        
        data = win32file.ReadFile(self.pipe, 4)
        resp_len = struct.unpack('>I', data[1])[0]
        
        resp_data = b''
        while len(resp_data) < resp_len:
            chunk = win32file.ReadFile(self.pipe, resp_len - len(resp_data))
            resp_data += chunk[1]
        
        return json.loads(resp_data.decode('utf-8'))
    
    def test_file(self, filename: str, multi_res: bool = False) -> Dict:
        """Test a file and return detected pitches"""
        self.send_command({"cmd": "setMultiRes", "enabled": multi_res})
        self.send_command({"cmd": "loadFile", "filename": filename})
        self.send_command({"cmd": "start"})
        
        # Warmup
        self.send_command({"cmd": "wait", "frames": 60, "timeout": 5000})
        
        # Collect pitches
        all_pitches = []
        for _ in range(5):
            result = self.send_command({"cmd": "getPitches"})
            pitches = result.get('pitches', [])
            all_pitches.extend(pitches)
            self.send_command({"cmd": "wait", "frames": 12, "timeout": 2000})
        
        self.send_command({"cmd": "stop"})
        
        # Deduplicate
        merged = {}
        for p in all_pitches:
            freq = p['frequency']
            key = round(freq / 5) * 5
            if key not in merged or p['confidence'] > merged[key]['confidence']:
                merged[key] = p
        
        return {
            'detected_pitches': list(merged.values()),
            'multi_res': multi_res
        }


def find_spm_exe():
    """Find SPM executable"""
    possible_paths = [
        r"build-windows\SuperPitchMonitor_artefacts\Debug\SuperPitchMonitor.exe",
        r"build-windows\SuperPitchMonitor_artefacts\Release\SuperPitchMonitor.exe",
    ]
    
    for path in possible_paths:
        full = os.path.join(SCRIPT_DIR, path)
        if os.path.exists(full):
            return os.path.abspath(full)
    
    return None


def main():
    parser = argparse.ArgumentParser(description='SPM Test with FFT Reference')
    parser.add_argument('--file', type=str, required=True, help='Test file name')
    parser.add_argument('--fft-only', action='store_true', help='Only do FFT analysis')
    parser.add_argument('--compare', action='store_true', help='Compare FFT with SPM')
    parser.add_argument('--duration', type=float, default=10.0, help='FFT window duration (default 10s)')
    parser.add_argument('--freq-min', type=float, default=50.0, help='Min frequency for analysis')
    parser.add_argument('--freq-max', type=float, default=2000.0, help='Max frequency for analysis')
    args = parser.parse_args()
    
    # Load ground truth
    with open(GROUND_TRUTH_FILE, 'r') as f:
        ground_truth_data = json.load(f)
    
    file_truth = ground_truth_data['files'].get(args.file)
    if not file_truth:
        print(f"ERROR: No ground truth for {args.file}")
        sys.exit(1)
    
    filepath = os.path.join(TEST_AUDIO_DIR, args.file)
    if not os.path.exists(filepath):
        print(f"ERROR: File not found: {filepath}")
        sys.exit(1)
    
    # FFT Analysis
    print(f"\nPerforming high-precision FFT analysis...")
    print(f"  Window duration: {args.duration}s")
    print(f"  Frequency resolution: {44100 / (44100 * args.duration):.3f} Hz")
    
    analyzer = HighPrecisionFFTAnalyzer(duration_sec=args.duration)
    fft_result = analyzer.analyze_file(
        filepath,
        freq_min=args.freq_min,
        freq_max=args.freq_max
    )
    
    # Identify fundamentals
    fundamental_analysis = analyzer.identify_fundamentals(
        fft_result,
        file_truth.get('fundamentals', [])
    )
    
    print(f"\nFFT detected {len(fft_result['detected_peaks'])} peaks")
    print(f"Fundamentals identified: {fundamental_analysis['total_identified']}/{fundamental_analysis['total_expected']}")
    
    if args.fft_only:
        print("\nTop 20 FFT Peaks:")
        for i, p in enumerate(fft_result['detected_peaks'][:20], 1):
            print(f"  {i:2d}. {p['frequency']:8.3f} Hz | {p['magnitude_db']:6.1f} dB")
        sys.exit(0)
    
    # SPM Test
    if args.compare:
        exe = find_spm_exe()
        if not exe:
            print("ERROR: SPM executable not found!")
            sys.exit(1)
        
        print(f"\nStarting SPM: {exe}")
        proc = subprocess.Popen(
            [exe, "-AutoTest"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=os.path.dirname(exe)
        )
        
        try:
            time.sleep(2)
            
            client = SPMTestClient()
            if not client.connect(timeout=30):
                print("Failed to connect to SPM!")
                sys.exit(1)
            
            # Test with multi-res mode
            spm_result = client.test_file(args.file, multi_res=True)
            client.send_command({"cmd": "exit"})
            client.disconnect()
            
            # Generate comparison report
            report = FFTComparisonReport.generate(fft_result, spm_result, file_truth)
            print("\n" + report)
            
            # Save report to file
            report_path = os.path.join(SCRIPT_DIR, f"comparison_report_{args.file.replace('.wav', '')}.txt")
            with open(report_path, 'w') as f:
                f.write(report)
            print(f"\nReport saved to: {report_path}")
            
        finally:
            proc.terminate()
            proc.wait()


if __name__ == '__main__':
    main()

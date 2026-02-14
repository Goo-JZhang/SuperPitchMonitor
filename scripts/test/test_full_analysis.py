#!/usr/bin/env python3
"""
SuperPitchMonitor Full Test Suite with FFT Reference Analysis

This script performs comprehensive testing:
1. High-precision FFT analysis (0.1Hz resolution) as ground truth
2. SPM real-time pitch detection via named pipes
3. Comparison and detailed reporting

Usage:
    python test_full_analysis.py --all              # Test all files
    python test_full_analysis.py --category single  # Test specific category
    python test_full_analysis.py --file chord_c_major_7_10s.wav  # Test specific file
    python test_full_analysis.py --report           # Generate HTML report
"""

import subprocess
import time
import json
import struct
import sys
import os
import argparse
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime
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

# Constants
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_AUDIO_DIR = os.path.join(SCRIPT_DIR, 'Resources', 'TestAudio')
GROUND_TRUTH_FILE = os.path.join(TEST_AUDIO_DIR, 'test_ground_truth.json')
PIPE_NAME = r'\\.\pipe\SPM_TestPipe'
SAMPLE_RATE = 44100
FFT_DURATION = 10.0  # Full 10s window for 0.1Hz resolution


class HighPrecisionFFTAnalyzer:
    """High-precision FFT analyzer for reference analysis"""
    
    def __init__(self, sample_rate: int = SAMPLE_RATE, duration_sec: float = FFT_DURATION):
        self.sample_rate = sample_rate
        self.duration_sec = duration_sec
        self.n_samples = int(sample_rate * duration_sec)
        self.freq_resolution = sample_rate / self.n_samples
        
    def analyze_file(self, filepath: str, 
                     freq_min: float = 50.0, 
                     freq_max: float = 2000.0,
                     prominence_db: float = 15.0) -> Dict[str, Any]:
        """Perform high-precision FFT analysis on audio file"""
        # Load audio
        sr, data = wavfile.read(filepath)
        
        if len(data.shape) > 1:
            data = data.mean(axis=1)
            
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
            
        # Apply window
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
            prominence=prominence_linear * np.max(fft_magnitude[freq_mask]),
            distance=int(5 / self.freq_resolution)
        )
        
        # Refine peaks with parabolic interpolation
        detected_peaks = []
        for peak_idx in peaks:
            if peak_idx <= 0 or peak_idx >= len(freqs_range) - 1:
                continue
                
            # Parabolic interpolation
            alpha = fft_magnitude[freq_mask][peak_idx - 1]
            beta = fft_magnitude[freq_mask][peak_idx]
            gamma = fft_magnitude[freq_mask][peak_idx + 1]
            
            p = 0.5 * (alpha - gamma) / (alpha - 2*beta + gamma)
            interpolated_freq = freqs_range[peak_idx] + p * self.freq_resolution
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
            'max_db': float(np.max(db_range)),
            'mean_db': float(np.mean(db_range)),
            'noise_floor_db': float(np.percentile(db_range, 10))
        }
    
    def identify_fundamentals(self, fft_result: Dict, 
                              expected_fundamentals: List[Dict],
                              tolerance_hz: float = 5.0) -> Dict:
        """Identify which expected fundamentals are present in FFT"""
        peaks = fft_result['detected_peaks']
        
        identified = []
        missing = []
        
        for expected in expected_fundamentals:
            exp_freq = expected['freq']
            exp_note = expected.get('note', '')
            
            # Check if fundamental is detected
            fundamental_match = None
            for peak in peaks:
                if abs(peak['frequency'] - exp_freq) <= tolerance_hz:
                    fundamental_match = peak
                    break
                    
            # Check for harmonics
            harmonics_found = []
            for h in range(2, 8):
                harmonic_freq = exp_freq * h
                for peak in peaks:
                    if abs(peak['frequency'] - harmonic_freq) <= tolerance_hz:
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


class SPMTestClient:
    """SuperPitchMonitor test client"""
    
    def __init__(self, pipe_name=PIPE_NAME):
        self.pipe_name = pipe_name
        self.pipe = None
        
    def connect(self, timeout=30):
        """Connect to SPM test pipe"""
        print(f"  Connecting to SPM...", end=' ')
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
                print("OK")
                return True
            except pywintypes.error as e:
                if e.winerror == 2:
                    time.sleep(0.5)
                    continue
                raise
        
        print("TIMEOUT")
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
    
    def test_file(self, filename: str, multi_res: bool = True, 
                  warmup_frames: int = 60, test_frames: int = 60) -> Dict:
        """Test a file and return detected pitches"""
        self.send_command({"cmd": "setMultiRes", "enabled": multi_res})
        
        # Check if file exists in original or generated folder
        filepath = os.path.join(TEST_AUDIO_DIR, filename)
        if not os.path.exists(filepath):
            # Try original folder
            alt_path = os.path.join(os.path.dirname(TEST_AUDIO_DIR), filename)
            if os.path.exists(alt_path):
                filepath = alt_path
        
        # Use filename only - SPM will resolve path
        result = self.send_command({"cmd": "loadFile", "filename": filename})
        if result.get('status') != 'ok':
            return {'error': 'Failed to load file', 'detected_pitches': []}
        
        self.send_command({"cmd": "start"})
        self.send_command({"cmd": "wait", "frames": warmup_frames, "timeout": 5000})
        
        # Collect pitches
        all_pitches = []
        for _ in range(5):
            result = self.send_command({"cmd": "getPitches"})
            pitches = result.get('pitches', [])
            all_pitches.extend(pitches)
            self.send_command({"cmd": "wait", "frames": test_frames // 5, "timeout": 2000})
        
        self.send_command({"cmd": "stop"})
        
        # Deduplicate by frequency (within 10Hz)
        merged = {}
        for p in all_pitches:
            freq = p['frequency']
            key = round(freq / 10) * 10
            if key not in merged or p['confidence'] > merged[key]['confidence']:
                merged[key] = p
        
        return {
            'detected_pitches': list(merged.values()),
            'multi_res': multi_res,
            'total_frames': len(all_pitches)
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


def compare_fft_vs_spm(fft_result: Dict, spm_result: Dict, 
                       ground_truth: Dict) -> Dict:
    """Compare FFT reference with SPM detection"""
    fft_peaks = fft_result['detected_peaks']
    spm_pitches = spm_result.get('detected_pitches', [])
    fundamentals = ground_truth.get('fundamentals', [])
    
    tolerance_hz = 10.0
    
    # Match FFT peaks to SPM detections
    matches = []
    fft_only = []
    
    for peak in fft_peaks[:20]:  # Top 20 FFT peaks
        fft_freq = peak['frequency']
        matched = False
        for spm_p in spm_pitches:
            if abs(spm_p['frequency'] - fft_freq) <= tolerance_hz:
                matches.append({
                    'fft_freq': fft_freq,
                    'fft_db': peak['magnitude_db'],
                    'spm_freq': spm_p['frequency'],
                    'spm_conf': spm_p['confidence'],
                    'spm_harmonics': spm_p.get('harmonicCount', 0)
                })
                matched = True
                break
        if not matched:
            # Check if it's significant (within 30dB of max)
            is_significant = peak['magnitude_db'] > fft_result['max_db'] - 30
            fft_only.append({
                'freq': peak['frequency'],
                'db': peak['magnitude_db'],
                'significant': is_significant
            })
    
    # SPM false positives
    spm_only = []
    for spm_p in spm_pitches:
        spm_freq = spm_p['frequency']
        matched = False
        for peak in fft_peaks:
            if abs(peak['frequency'] - spm_freq) <= tolerance_hz:
                matched = True
                break
        if not matched:
            spm_only.append({
                'freq': spm_freq,
                'conf': spm_p['confidence'],
                'harmonics': spm_p.get('harmonicCount', 0)
            })
    
    # Check fundamentals
    fundamental_results = []
    for fund in fundamentals:
        exp_freq = fund['freq']
        note = fund.get('note', '')
        
        fft_match = None
        spm_match = None
        
        for peak in fft_peaks:
            if abs(peak['frequency'] - exp_freq) <= tolerance_hz:
                fft_match = peak
                break
        
        for spm_p in spm_pitches:
            if abs(spm_p['frequency'] - exp_freq) <= tolerance_hz:
                spm_match = spm_p
                break
        
        fundamental_results.append({
            'note': note,
            'expected_freq': exp_freq,
            'fft_detected': fft_match['frequency'] if fft_match else None,
            'fft_db': fft_match['magnitude_db'] if fft_match else None,
            'spm_detected': spm_match['frequency'] if spm_match else None,
            'spm_conf': spm_match['confidence'] if spm_match else None,
            'status': 'OK' if spm_match else 'MISSED'
        })
    
    return {
        'matches': matches,
        'fft_only': fft_only,
        'spm_only': spm_only,
        'fundamentals': fundamental_results,
        'match_count': len(matches),
        'fft_only_count': len(fft_only),
        'spm_only_count': len(spm_only),
        'missed_count': sum(1 for f in fundamental_results if f['status'] == 'MISSED')
    }


def run_single_test(filename: str, ground_truth: Dict, 
                    analyzer: HighPrecisionFFTAnalyzer,
                    client: SPMTestClient) -> Dict:
    """Run complete test for a single file"""
    print(f"\n{'='*70}")
    print(f"Testing: {filename}")
    print(f"{'='*70}")
    
    filepath = os.path.join(TEST_AUDIO_DIR, filename)
    if not os.path.exists(filepath):
        # Try original test audio folder
        alt_dir = os.path.dirname(TEST_AUDIO_DIR)
        filepath = os.path.join(alt_dir, filename)
    
    if not os.path.exists(filepath):
        return {'filename': filename, 'error': f'File not found: {filepath}'}
    
    # FFT Analysis
    print("  FFT Analysis...", end=' ')
    fft_result = analyzer.analyze_file(filepath)
    print(f"Found {len(fft_result['detected_peaks'])} peaks")
    
    # Identify fundamentals in FFT
    fundamentals = ground_truth.get('fundamentals', [])
    fft_fund = analyzer.identify_fundamentals(fft_result, fundamentals)
    
    # SPM Detection
    print("  SPM Detection...")
    spm_result = client.test_file(filename, multi_res=True)
    
    if 'error' in spm_result:
        return {'error': spm_result['error']}
    
    # Compare
    comparison = compare_fft_vs_spm(fft_result, spm_result, ground_truth)
    
    # Print summary
    print(f"\n  Results:")
    print(f"    FFT-SPM Matches: {comparison['match_count']}")
    print(f"    FFT Only: {comparison['fft_only_count']}")
    print(f"    SPM Only (False Positives): {comparison['spm_only_count']}")
    print(f"    Fundamentals: {len(fundamentals) - comparison['missed_count']}/{len(fundamentals)} detected")
    
    if comparison['fundamentals']:
        print(f"\n  Fundamental Detection:")
        for f in comparison['fundamentals']:
            status_str = "[OK]" if f['status'] == 'OK' else "[MISS]"
            fft_info = f"FFT:{f['fft_db']:.1f}dB" if f['fft_db'] else "FFT:N/A"
            spm_info = f"SPM:{f['spm_detected']:.1f}Hz" if f['spm_detected'] else "SPM:N/A"
            print(f"    {status_str} {f['note']:6s} {f['expected_freq']:.1f}Hz | {fft_info} | {spm_info}")
    
    # Identify critical issues
    critical_missed = []
    critical_false_positives = []
    
    for f in comparison['fundamentals']:
        if f['status'] == 'MISSED' and f['fft_db'] and f['fft_db'] > fft_result['max_db'] - 20:
            # Strong fundamental missed by SPM
            critical_missed.append(f)
    
    for fp in comparison['spm_only']:
        if fp['conf'] > 0.6:
            # High confidence false positive
            critical_false_positives.append(fp)
    
    if critical_missed:
        print(f"\n  [CRITICAL] Strong fundamentals missed by SPM:")
        for m in critical_missed:
            print(f"    - {m['note']} ({m['expected_freq']:.1f}Hz, {m['fft_db']:.1f}dB in FFT)")
    
    if critical_false_positives:
        print(f"\n  [CRITICAL] High confidence false positives:")
        for fp in critical_false_positives:
            print(f"    - {fp['freq']:.1f}Hz (conf={fp['conf']:.2f}, not in FFT)")
    
    return {
        'filename': filename,
        'fft': fft_result,
        'spm': spm_result,
        'comparison': comparison,
        'critical_missed': critical_missed,
        'critical_false_positives': critical_false_positives
    }


def generate_report(results: List[Dict], output_file: str):
    """Generate HTML test report"""
    
    html = """<!DOCTYPE html>
<html>
<head>
    <title>SuperPitchMonitor Test Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }
        h1 { color: #333; }
        h2 { color: #555; border-bottom: 2px solid #ddd; padding-bottom: 10px; }
        .summary { background: white; padding: 20px; border-radius: 8px; margin: 20px 0; }
        .test-case { background: white; padding: 20px; border-radius: 8px; margin: 20px 0; }
        .ok { color: green; }
        .warning { color: orange; }
        .error { color: red; }
        table { border-collapse: collapse; width: 100%; margin: 10px 0; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background: #f0f0f0; }
        .status-ok { background: #d4edda; }
        .status-miss { background: #f8d7da; }
        .critical { background: #ffebee; border: 2px solid #f44336; padding: 10px; margin: 10px 0; }
    </style>
</head>
<body>
    <h1>SuperPitchMonitor Test Report</h1>
    <p>Generated: """ + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + """</p>
"""
    
    # Summary statistics
    total_tests = len(results)
    total_fundamentals = sum(len(r['comparison']['fundamentals']) for r in results if 'comparison' in r)
    total_detected = sum(
        sum(1 for f in r['comparison']['fundamentals'] if f['status'] == 'OK')
        for r in results if 'comparison' in r
    )
    total_critical_missed = sum(len(r.get('critical_missed', [])) for r in results)
    total_critical_fp = sum(len(r.get('critical_false_positives', [])) for r in results)
    
    html += f"""
    <div class="summary">
        <h2>Summary</h2>
        <p>Total Tests: {total_tests}</p>
        <p>Fundamentals Detection Rate: {total_detected}/{total_fundamentals} ({100*total_detected/total_fundamentals:.1f}%)</p>
        <p>Critical Missed (strong fundamentals): {total_critical_missed}</p>
        <p>Critical False Positives (high conf): {total_critical_fp}</p>
    </div>
"""
    
    # Per-test details
    for result in results:
        if 'error' in result:
            html += f"""
    <div class="test-case">
        <h3 class="error">{result['filename']} - ERROR</h3>
        <p>{result['error']}</p>
    </div>
"""
            continue
        
        comp = result['comparison']
        missed_count = comp['missed_count']
        total_funds = len(comp['fundamentals'])
        
        status_class = 'ok' if missed_count == 0 else 'warning' if missed_count <= 1 else 'error'
        
        html += f"""
    <div class="test-case">
        <h3 class="{status_class}">{result['filename']} - {total_funds - missed_count}/{total_funds} detected</h3>
        
        <h4>Fundamental Detection</h4>
        <table>
            <tr>
                <th>Note</th>
                <th>Expected (Hz)</th>
                <th>FFT Level (dB)</th>
                <th>SPM Detected (Hz)</th>
                <th>SPM Confidence</th>
                <th>Status</th>
            </tr>
"""
        
        for f in comp['fundamentals']:
            row_class = 'status-ok' if f['status'] == 'OK' else 'status-miss'
            fft_level = f"{f['fft_db']:.1f}" if f['fft_db'] else "N/A"
            spm_freq = f"{f['spm_detected']:.1f}" if f['spm_detected'] else "N/A"
            spm_conf = f"{f['spm_conf']:.2f}" if f['spm_conf'] else "N/A"
            
            html += f"""
            <tr class="{row_class}">
                <td>{f['note']}</td>
                <td>{f['expected_freq']:.2f}</td>
                <td>{fft_level}</td>
                <td>{spm_freq}</td>
                <td>{spm_conf}</td>
                <td>{f['status']}</td>
            </tr>
"""
        
        html += """
        </table>
"""
        
        if result.get('critical_missed'):
            html += """
        <div class="critical">
            <strong>CRITICAL MISSED:</strong> Strong fundamentals present in FFT but not detected by SPM
        </div>
"""
        
        if result.get('critical_false_positives'):
            html += """
        <div class="critical">
            <strong>CRITICAL FALSE POSITIVES:</strong> High confidence detections not present in FFT
        </div>
"""
        
        html += """
    </div>
"""
    
    html += """
</body>
</html>
"""
    
    with open(output_file, 'w') as f:
        f.write(html)
    
    print(f"\n{'='*70}")
    print(f"Report saved to: {output_file}")
    print(f"{'='*70}")


def main():
    parser = argparse.ArgumentParser(description='SPM Full Test Suite')
    parser.add_argument('--all', action='store_true', help='Test all files')
    parser.add_argument('--file', type=str, help='Test specific file')
    parser.add_argument('--category', type=str, help='Test category (single_tone, single_note, etc.)')
    parser.add_argument('--report', action='store_true', help='Generate HTML report')
    parser.add_argument('--output', type=str, default='test_report.html', help='Output report filename')
    args = parser.parse_args()
    
    # Load ground truth
    if not os.path.exists(GROUND_TRUTH_FILE):
        print(f"Error: Ground truth file not found: {GROUND_TRUTH_FILE}")
        print("Run: python scripts/generate_test_audio.py")
        sys.exit(1)
    
    with open(GROUND_TRUTH_FILE, 'r') as f:
        ground_truth_data = json.load(f)
    
    files_data = ground_truth_data.get('files', {})
    
    # Determine which files to test
    if args.file:
        if args.file in files_data:
            files_to_test = {args.file: files_data[args.file]}
        else:
            print(f"Error: File not in ground truth: {args.file}")
            sys.exit(1)
    elif args.category:
        files_to_test = {k: v for k, v in files_data.items() 
                        if v.get('type') == args.category}
        if not files_to_test:
            print(f"No files found for category: {args.category}")
            sys.exit(1)
    else:
        files_to_test = files_data
    
    # Start SPM
    exe = find_spm_exe()
    if not exe:
        print("Error: SuperPitchMonitor.exe not found!")
        sys.exit(1)
    
    print(f"SPM Executable: {exe}")
    print(f"Starting SPM in AutoTest mode...")
    
    proc = subprocess.Popen(
        [exe, "-AutoTest"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=os.path.dirname(exe)
    )
    
    results = []
    
    try:
        time.sleep(2)
        
        client = SPMTestClient()
        if not client.connect(timeout=30):
            print("Failed to connect to SPM!")
            sys.exit(1)
        
        analyzer = HighPrecisionFFTAnalyzer()
        
        print(f"\n{'='*70}")
        print(f"Running {len(files_to_test)} tests...")
        print(f"{'='*70}")
        
        for filename, gt in files_to_test.items():
            result = run_single_test(filename, gt, analyzer, client)
            results.append(result)
        
        client.send_command({"cmd": "exit"})
        client.disconnect()
        
        # Summary
        print(f"\n{'='*70}")
        print("TEST SUMMARY")
        print(f"{'='*70}")
        
        total_funds = 0
        total_detected = 0
        total_critical_missed = 0
        total_critical_fp = 0
        
        for r in results:
            if 'comparison' in r:
                funds = r['comparison']['fundamentals']
                total_funds += len(funds)
                total_detected += sum(1 for f in funds if f['status'] == 'OK')
                total_critical_missed += len(r.get('critical_missed', []))
                total_critical_fp += len(r.get('critical_false_positives', []))
        
        print(f"Total Fundamentals: {total_funds}")
        print(f"Detected: {total_detected} ({100*total_detected/total_funds:.1f}%)")
        print(f"Critical Missed (strong fundamentals): {total_critical_missed}")
        print(f"Critical False Positives (high conf): {total_critical_fp}")
        
        # Generate report if requested
        if args.report or args.all:
            report_path = os.path.join(SCRIPT_DIR, 'Docs', 'test_reports', args.output)
            generate_report(results, report_path)
        
        sys.exit(0 if total_critical_missed == 0 else 1)
        
    finally:
        proc.terminate()
        proc.wait()


if __name__ == '__main__':
    main()

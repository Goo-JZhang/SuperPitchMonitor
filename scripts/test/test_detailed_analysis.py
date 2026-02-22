#!/usr/bin/env python3
"""
SuperPitchMonitor Detailed Analysis Test Suite

Comprehensive testing with:
- High-precision FFT (10s window, 0.1Hz resolution)
- Per-frame SPM data collection (frequency, confidence, magnitude)
- Statistical analysis (mean & std for each metric)
- Both Multi-Resolution ON and OFF modes
- Detailed comparison tables for each test file

Usage:
    python test_detailed_analysis.py --all
    python test_detailed_analysis.py --file chord_c_major_7_piano.wav
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
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
TEST_AUDIO_DIR = os.path.join(PROJECT_ROOT, 'Resources', 'TestAudio')
GROUND_TRUTH_FILE = os.path.join(TEST_AUDIO_DIR, 'test_ground_truth.json')
PIPE_NAME = r'\\.\pipe\SPM_TestPipe'
SAMPLE_RATE = 44100
FFT_DURATION = 10.0
SAMPLES = int(SAMPLE_RATE * FFT_DURATION)
FREQ_RESOLUTION = SAMPLE_RATE / SAMPLES  # 0.1 Hz


class HighPrecisionFFTAnalyzer:
    """High-precision FFT analyzer for reference analysis"""
    
    def __init__(self, sample_rate: int = SAMPLE_RATE, duration: float = FFT_DURATION):
        self.sample_rate = sample_rate
        self.duration = duration
        self.n_samples = int(sample_rate * duration)
        self.freq_resolution = sample_rate / self.n_samples
        
    def analyze_full_window(self, filepath: str, 
                           freq_min: float = 50.0, 
                           freq_max: float = 2000.0) -> Dict:
        """
        Perform full-window FFT analysis (0.1Hz resolution)
        
        Returns complete spectrum with magnitude in dB
        """
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
            
        # Use exactly 10s from the middle of the file
        if len(data) > self.n_samples:
            start = (len(data) - self.n_samples) // 2
            data = data[start:start + self.n_samples]
        else:
            # Pad if too short
            data = np.pad(data, (0, self.n_samples - len(data)), mode='constant')
        
        # Apply Hann window
        window = windows.hann(len(data))
        windowed_data = data * window
        
        # Full-window FFT
        fft_result = fft(windowed_data)
        fft_magnitude = np.abs(fft_result[:len(fft_result)//2])
        
        # Convert to dB (full scale)
        fft_db = 20 * np.log10(fft_magnitude + 1e-12)
        
        # Frequency axis
        freqs = fftfreq(len(data), 1.0/self.sample_rate)[:len(fft_result)//2]
        
        # Find peaks
        freq_mask = (freqs >= freq_min) & (freqs <= freq_max)
        peaks_indices, properties = find_peaks(
            fft_magnitude[freq_mask],
            height=np.max(fft_magnitude) * 0.001,  # 0.1% of max
            distance=int(5 / self.freq_resolution)  # 5Hz minimum separation
        )
        
        # Refine peaks
        detected_peaks = []
        for idx in peaks_indices:
            actual_idx = np.where(freq_mask)[0][idx]
            if actual_idx <= 0 or actual_idx >= len(freqs) - 1:
                continue
                
            # Parabolic interpolation
            alpha = fft_magnitude[actual_idx - 1]
            beta = fft_magnitude[actual_idx]
            gamma = fft_magnitude[actual_idx + 1]
            
            p = 0.5 * (alpha - gamma) / (alpha - 2*beta + gamma)
            interpolated_freq = freqs[actual_idx] + p * self.freq_resolution
            interpolated_mag = beta - 0.25 * (alpha - gamma) * p
            interpolated_db = 20 * np.log10(interpolated_mag + 1e-12)
            
            detected_peaks.append({
                'frequency': float(interpolated_freq),
                'magnitude_db': float(interpolated_db),
                'magnitude_linear': float(interpolated_mag),
                'bin': int(actual_idx)
            })
        
        # Sort by magnitude (descending)
        detected_peaks.sort(key=lambda x: x['magnitude_db'], reverse=True)
        
        return {
            'filepath': filepath,
            'sample_rate': self.sample_rate,
            'fft_size': len(data),
            'freq_resolution_hz': self.freq_resolution,
            'duration': self.duration,
            'frequencies': freqs.tolist(),
            'magnitude_db': fft_db.tolist(),
            'detected_peaks': detected_peaks,
            'max_db': float(np.max(fft_db[freq_mask])),
            'noise_floor_db': float(np.percentile(fft_db[freq_mask], 10))
        }
    
    def get_magnitude_at_freq(self, fft_result: Dict, target_freq: float) -> float:
        """Get magnitude (dB) at specific frequency"""
        freqs = np.array(fft_result['frequencies'])
        mags = np.array(fft_result['magnitude_db'])
        
        # Find closest bin
        idx = np.argmin(np.abs(freqs - target_freq))
        return float(mags[idx])


class SPMTestClient:
    """SuperPitchMonitor test client with per-frame data collection"""
    
    def __init__(self, pipe_name=PIPE_NAME):
        self.pipe_name = pipe_name
        self.pipe = None
        
    def connect(self, timeout=30):
        """Connect to SPM test pipe"""
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
                return True
            except pywintypes.error as e:
                if e.winerror == 2:
                    time.sleep(0.5)
                    continue
                raise
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
    
    def collect_per_frame_data(self, filename: str, multi_res: bool = True, 
                               num_frames: int = 30, warmup_frames: int = 20) -> Dict:
        """
        Collect per-frame pitch data from SPM
        
        Returns detailed statistics for each detected pitch
        """
        self.send_command({"cmd": "setMultiRes", "enabled": multi_res})
        
        result = self.send_command({"cmd": "loadFile", "filename": filename})
        if result.get('status') != 'ok':
            return {'error': f'Failed to load file: {result.get("message", "unknown")}'}
        
        self.send_command({"cmd": "start"})
        
        # Warmup
        self.send_command({"cmd": "wait", "frames": warmup_frames, "timeout": 10000})
        
        # Collect per-frame data
        all_frame_data = []
        for frame_idx in range(num_frames):
            result = self.send_command({"cmd": "getPitches"})
            pitches = result.get('pitches', [])
            
            frame_data = {
                'frame': frame_idx,
                'pitches': []
            }
            
            for p in pitches:
                frame_data['pitches'].append({
                    'frequency': p.get('frequency', 0),
                    'confidence': p.get('confidence', 0),
                    'midiNote': p.get('midiNote', 0),
                    'harmonicCount': p.get('harmonicCount', 0),
                    'amplitude': p.get('amplitude', 0)
                })
            
            all_frame_data.append(frame_data)
            
            # Wait for next frame
            self.send_command({"cmd": "wait", "frames": 1, "timeout": 1000})
        
        self.send_command({"cmd": "stop"})
        
        # Get spectrum data for the last frame
        spectrum_result = self.send_command({
            "cmd": "getSpectrumPeaks",
            "freqMin": 50,
            "freqMax": 2000
        })
        
        return {
            'multi_res': multi_res,
            'frame_data': all_frame_data,
            'spectrum_peaks': spectrum_result.get('peaks', []),
            'num_frames': num_frames
        }


def analyze_pitch_statistics(frame_data: List[Dict]) -> List[Dict]:
    """
    Analyze per-frame pitch data to extract statistics
    
    Groups pitches by frequency and calculates mean/std for each metric
    """
    # Group pitches by approximate frequency (within 10Hz)
    pitch_groups = {}
    
    for frame in frame_data:
        for pitch in frame['pitches']:
            freq = pitch['frequency']
            # Use 10Hz bins for grouping
            key = round(freq / 10) * 10
            
            if key not in pitch_groups:
                pitch_groups[key] = {
                    'frequencies': [],
                    'confidences': [],
                    'amplitudes': [],
                    'harmonic_counts': []
                }
            
            pitch_groups[key]['frequencies'].append(freq)
            pitch_groups[key]['confidences'].append(pitch['confidence'])
            pitch_groups[key]['amplitudes'].append(pitch['amplitude'])
            pitch_groups[key]['harmonic_counts'].append(pitch['harmonicCount'])
    
    # Calculate statistics
    results = []
    for key, data in pitch_groups.items():
        n = len(data['frequencies'])
        if n < 3:  # Skip if too few samples
            continue
            
        stats = {
            'freq_key': key,
            'detection_count': n,
            'detection_rate': n / len(frame_data),
            'frequency': {
                'mean': np.mean(data['frequencies']),
                'std': np.std(data['frequencies'])
            },
            'confidence': {
                'mean': np.mean(data['confidences']),
                'std': np.std(data['confidences'])
            },
            'amplitude': {
                'mean': np.mean(data['amplitudes']),
                'std': np.std(data['amplitudes'])
            },
            'harmonic_count': {
                'mean': np.mean(data['harmonic_counts']),
                'std': np.std(data['harmonic_counts'])
            }
        }
        
        results.append(stats)
    
    # Sort by frequency
    results.sort(key=lambda x: x['frequency']['mean'])
    return results


def compare_with_ground_truth(spm_stats: List[Dict], 
                              fft_result: Dict,
                              ground_truth: Dict) -> Dict:
    """
    Compare SPM results with ground truth and FFT reference
    
    For each expected fundamental:
    - Check if detected by SPM
    - Get FFT magnitude at that frequency
    - If missed, record the spectrum magnitude
    """
    fundamentals = ground_truth.get('fundamentals', [])
    comparison = []
    
    for fund in fundamentals:
        exp_freq = fund['freq']
        note = fund.get('note', '')
        
        # Get FFT magnitude at expected frequency
        fft_mag = fft_result.get('magnitude_at_freq', {}).get(str(round(exp_freq, 1)), None)
        if fft_mag is None:
            # Calculate on the fly
            freqs = np.array(fft_result['frequencies'])
            mags = np.array(fft_result['magnitude_db'])
            idx = np.argmin(np.abs(freqs - exp_freq))
            fft_mag = float(mags[idx])
        
        # Find matching SPM detection (within 15Hz tolerance)
        spm_match = None
        for stats in spm_stats:
            if abs(stats['frequency']['mean'] - exp_freq) <= 15:
                spm_match = stats
                break
        
        entry = {
            'note': note,
            'expected_freq': exp_freq,
            'fft_magnitude_db': fft_mag,
            'detected': spm_match is not None
        }
        
        if spm_match:
            entry['spm'] = {
                'frequency_mean': spm_match['frequency']['mean'],
                'frequency_std': spm_match['frequency']['std'],
                'confidence_mean': spm_match['confidence']['mean'],
                'confidence_std': spm_match['confidence']['std'],
                'amplitude_mean': spm_match['amplitude']['mean'],
                'amplitude_std': spm_match['amplitude']['std'],
                'detection_rate': spm_match['detection_rate'],
                'harmonic_count_mean': spm_match['harmonic_count']['mean']
            }
        else:
            entry['spm'] = None  # Missed - will show spectrum magnitude
        
        comparison.append(entry)
    
    return comparison


def generate_detailed_report(test_results: List[Dict], output_file: str):
    """Generate detailed HTML report with tables for each test file"""
    
    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>SuperPitchMonitor Detailed Test Report</title>
    <style>
        body {{ font-family: 'Segoe UI', Arial, sans-serif; margin: 20px; background: #f8f9fa; }}
        h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
        h2 {{ color: #34495e; margin-top: 30px; }}
        h3 {{ color: #7f8c8d; }}
        .test-section {{ background: white; padding: 20px; margin: 20px 0; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .mode-section {{ margin: 15px 0; padding: 15px; background: #f8f9fa; border-radius: 5px; }}
        table {{ border-collapse: collapse; width: 100%; margin: 15px 0; font-size: 13px; }}
        th {{ background: #3498db; color: white; padding: 10px; text-align: center; font-weight: 600; }}
        td {{ border: 1px solid #ddd; padding: 8px; text-align: center; }}
        tr:nth-child(even) {{ background: #f8f9fa; }}
        .ok {{ background: #d4edda !important; color: #155724; font-weight: 600; }}
        .miss {{ background: #f8d7da !important; color: #721c24; font-weight: 600; }}
        .false-pos {{ background: #fff3cd !important; color: #856404; }}
        .fft-peak {{ background: #e3f2fd; margin: 5px 0; padding: 5px 10px; border-radius: 4px; font-size: 12px; }}
        .summary-stats {{ display: flex; gap: 20px; margin: 20px 0; }}
        .stat-box {{ background: white; padding: 15px; border-radius: 8px; flex: 1; text-align: center; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .stat-value {{ font-size: 24px; font-weight: bold; color: #3498db; }}
        .stat-label {{ color: #7f8c8d; font-size: 12px; }}
        .metric {{ font-family: 'Consolas', monospace; }}
        .header-info {{ background: #e3f2fd; padding: 15px; border-radius: 8px; margin: 20px 0; }}
    </style>
</head>
<body>
    <h1>SuperPitchMonitor Detailed Test Report</h1>
    <div class="header-info">
        <strong>Test Configuration:</strong><br>
        FFT Window: 10s (0.1Hz resolution)<br>
        SPM Frames Collected: 100 per test<br>
        Warmup Frames: 50<br>
        Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    </div>
"""
    
    # Overall statistics
    total_funds = sum(len(r['comparison_mr_on']) for r in test_results if 'comparison_mr_on' in r)
    detected_mr_on = sum(sum(1 for e in r['comparison_mr_on'] if e['detected']) for r in test_results if 'comparison_mr_on' in r)
    detected_mr_off = sum(sum(1 for e in r['comparison_mr_off'] if e['detected']) for r in test_results if 'comparison_mr_off' in r)
    
    html += f"""
    <div class="summary-stats">
        <div class="stat-box">
            <div class="stat-value">{len(test_results)}</div>
            <div class="stat-label">Test Files</div>
        </div>
        <div class="stat-box">
            <div class="stat-value">{detected_mr_on}/{total_funds}</div>
            <div class="stat-label">Detected (MR ON)</div>
        </div>
        <div class="stat-box">
            <div class="stat-value">{detected_mr_off}/{total_funds}</div>
            <div class="stat-label">Detected (MR OFF)</div>
        </div>
        <div class="stat-box">
            <div class="stat-value">{FREQ_RESOLUTION:.2f}Hz</div>
            <div class="stat-label">FFT Resolution</div>
        </div>
    </div>
"""
    
    # Per-file detailed results
    for result in test_results:
        if 'error' in result:
            html += f"""
    <div class="test-section">
        <h2>{result['filename']} - ERROR</h2>
        <p style="color: red;">{result['error']}</p>
    </div>
"""
            continue
        
        html += f"""
    <div class="test-section">
        <h2>{result['filename']}</h2>
        <p><strong>Type:</strong> {result.get('type', 'unknown')} | 
           <strong>Description:</strong> {result.get('description', 'N/A')}</p>
        
        <h3>FFT Reference Peaks (Top 10)</h3>
        <div style="display: flex; flex-wrap: wrap; gap: 10px;">
"""
        
        # FFT peaks
        for peak in result['fft_peaks'][:10]:
            html += f'<div class="fft-peak">{peak["frequency"]:.1f}Hz ({peak["magnitude_db"]:.1f}dB)</div>'
        
        html += """
        </div>
        
        <h3>SPM Results - Multi-Resolution OFF</h3>
        <div class="mode-section">
"""
        html += generate_comparison_table(result['comparison_mr_off'])
        html += """
        </div>
        
        <h3>SPM Results - Multi-Resolution ON</h3>
        <div class="mode-section">
"""
        html += generate_comparison_table(result['comparison_mr_on'])
        html += """
        </div>
    </div>
"""
    
    html += """
</body>
</html>
"""
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html)
    
    print(f"\nDetailed report saved to: {output_file}")


def generate_comparison_table(comparison: List[Dict]) -> str:
    """Generate HTML table for comparison data"""
    
    html = """
    <table>
        <thead>
            <tr>
                <th>Note</th>
                <th>Expected Freq<br>(Hz)</th>
                <th>FFT Mag<br>(dB)</th>
                <th>Status</th>
                <th>SPM Freq<br>Mean卤Std (Hz)</th>
                <th>Confidence<br>Mean卤Std</th>
                <th>Amplitude<br>Mean卤Std</th>
                <th>Detection<br>Rate</th>
                <th>H. Count<br>Mean</th>
            </tr>
        </thead>
        <tbody>
"""
    
    for entry in comparison:
        note = entry['note']
        exp_freq = entry['expected_freq']
        fft_mag = entry['fft_magnitude_db']
        detected = entry['detected']
        
        row_class = 'ok' if detected else 'miss'
        status_text = 'DETECTED' if detected else 'MISSED'
        
        if detected and entry['spm']:
            spm = entry['spm']
            freq_str = f"{spm['frequency_mean']:.1f}卤{spm['frequency_std']:.1f}"
            conf_str = f"{spm['confidence_mean']:.2f}卤{spm['confidence_std']:.2f}"
            amp_str = f"{spm['amplitude_mean']:.4f}卤{spm['amplitude_std']:.4f}" if spm.get('amplitude_mean') else "N/A"
            rate_str = f"{spm['detection_rate']*100:.0f}%"
            h_count_str = f"{spm['harmonic_count_mean']:.1f}"
        else:
            freq_str = "N/A"
            conf_str = "N/A"
            amp_str = "N/A"
            rate_str = "0%"
            h_count_str = "N/A"
        
        html += f"""
            <tr class="{row_class}">
                <td><strong>{note}</strong></td>
                <td>{exp_freq:.2f}</td>
                <td class="metric">{fft_mag:.1f}</td>
                <td>{status_text}</td>
                <td class="metric">{freq_str}</td>
                <td class="metric">{conf_str}</td>
                <td class="metric">{amp_str}</td>
                <td>{rate_str}</td>
                <td>{h_count_str}</td>
            </tr>
"""
    
    html += """
        </tbody>
    </table>
"""
    return html


def run_single_test(filename: str, ground_truth: Dict,
                    analyzer: HighPrecisionFFTAnalyzer,
                    client: SPMTestClient) -> Dict:
    """Run complete test for a single file with both MR modes"""
    
    print(f"\n{'='*70}")
    print(f"Testing: {filename}")
    print(f"{'='*70}")
    
    filepath = os.path.join(TEST_AUDIO_DIR, filename)
    if not os.path.exists(filepath):
        return {'filename': filename, 'error': f'File not found: {filepath}'}
    
    # FFT Analysis
    print("  Running FFT analysis (10s window, 0.1Hz resolution)...")
    fft_result = analyzer.analyze_full_window(filepath)
    print(f"    Found {len(fft_result['detected_peaks'])} peaks in FFT")
    
    # Test with MR OFF
    print("  Testing with Multi-Resolution OFF...")
    spm_data_off = client.collect_per_frame_data(filename, multi_res=False)
    if 'error' in spm_data_off:
        return {'filename': filename, 'error': spm_data_off['error']}
    spm_stats_off = analyze_pitch_statistics(spm_data_off['frame_data'])
    comparison_off = compare_with_ground_truth(spm_stats_off, fft_result, ground_truth)
    
    # Test with MR ON
    print("  Testing with Multi-Resolution ON...")
    spm_data_on = client.collect_per_frame_data(filename, multi_res=True)
    if 'error' in spm_data_on:
        return {'filename': filename, 'error': spm_data_on['error']}
    spm_stats_on = analyze_pitch_statistics(spm_data_on['frame_data'])
    comparison_on = compare_with_ground_truth(spm_stats_on, fft_result, ground_truth)
    
    # Summary
    detected_off = sum(1 for e in comparison_off if e['detected'])
    detected_on = sum(1 for e in comparison_on if e['detected'])
    total = len(ground_truth.get('fundamentals', []))
    
    print(f"    MR OFF: {detected_off}/{total} detected")
    print(f"    MR ON:  {detected_on}/{total} detected")
    
    return {
        'filename': filename,
        'type': ground_truth.get('type', 'unknown'),
        'description': ground_truth.get('description', ''),
        'fft_peaks': fft_result['detected_peaks'],
        'comparison_mr_off': comparison_off,
        'comparison_mr_on': comparison_on
    }


def find_spm_exe():
    """Find SPM executable"""
    possible_paths = [
        r"build-windows\SuperPitchMonitor_artefacts\Debug\SuperPitchMonitor.exe",
        r"build-windows\SuperPitchMonitor_artefacts\Release\SuperPitchMonitor.exe",
        r"build-macos\SuperPitchMonitor_artefacts\SuperPitchMonitor.app\Contents\MacOS\SuperPitchMonitor",
        r"build-linux\SuperPitchMonitor",
    ]
    for path in possible_paths:
        full = os.path.join(PROJECT_ROOT, path)
        if os.path.exists(full):
            return os.path.abspath(full)
    return None


def main():
    parser = argparse.ArgumentParser(description='SPM Detailed Analysis Test Suite')
    parser.add_argument('--all', action='store_true', help='Test all files')
    parser.add_argument('--file', type=str, help='Test specific file')
    args = parser.parse_args()
    
    # Load ground truth
    if not os.path.exists(GROUND_TRUTH_FILE):
        print(f"Error: Ground truth file not found: {GROUND_TRUTH_FILE}")
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
    else:
        # Test first 5 files for now (can be extended)
        files_to_test = dict(list(files_data.items())[:5])
    
    # Start SPM
    exe = find_spm_exe()
    if not exe:
        print("Error: SuperPitchMonitor.exe not found!")
        sys.exit(1)
    
    print(f"SPM Executable: {exe}")
    print("Starting SPM in AutoTest mode...")
    
    proc = subprocess.Popen(
        [exe, "-AutoTest"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=PROJECT_ROOT
    )
    
    results = []
    
    try:
        time.sleep(2)
        
        client = SPMTestClient()
        if not client.connect(timeout=30):
            print("Failed to connect to SPM!")
            sys.exit(1)
        
        analyzer = HighPrecisionFFTAnalyzer()
        
        print(f"\nRunning {len(files_to_test)} tests...")
        
        for filename, gt in files_to_test.items():
            result = run_single_test(filename, gt, analyzer, client)
            results.append(result)
        
        client.send_command({"cmd": "exit"})
        client.disconnect()
        
        # Generate report
        report_path = os.path.join(PROJECT_ROOT, 'Docs', 'test_reports', 'detailed_test_report.html')
        generate_detailed_report(results, report_path)
        
    finally:
        proc.terminate()
        proc.wait()


if __name__ == '__main__':
    main()

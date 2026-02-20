#!/usr/bin/env python3
"""
SuperPitchMonitor Test Report Generator

Generates a comprehensive test report comparing:
1. Ground Truth (expected frequencies)
2. Full-window High-precision FFT results (Python reference)
3. SPM results (multi-resolution OFF)
4. SPM results (multi-resolution ON)

Usage:
    python generate_test_report.py
    python generate_test_report.py --output report.md
    python generate_test_report.py --exe /path/to/SuperPitchMonitor
"""

import subprocess
import time
import json
import struct
import sys
import os
import argparse
import socket
import platform
import signal
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime

import numpy as np
from numpy.fft import rfft, rfftfreq
import wave

# Configuration
DEFAULT_TCP_PORT = 9999
DEFAULT_WAIT_FRAMES = 60
FFT_SIZE = 65536  # Full window high-precision FFT

SCRIPT_DIR = Path(__file__).parent.absolute()
PROJECT_ROOT = SCRIPT_DIR.parent.parent
TEST_AUDIO_DIR = PROJECT_ROOT / "Resources" / "TestAudio"
GROUND_TRUTH_FILE = TEST_AUDIO_DIR / "test_ground_truth.json"
REPORTS_DIR = PROJECT_ROOT / "TestReports"


@dataclass
class TestResult:
    """Single test result"""
    filename: str
    ground_truth: List[Dict[str, Any]]
    fft_results: List[Dict[str, Any]]
    spm_standard: List[Dict[str, Any]]
    spm_multires: List[Dict[str, Any]]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "filename": self.filename,
            "ground_truth": self.ground_truth,
            "fft_results": self.fft_results,
            "spm_standard": self.spm_standard,
            "spm_multires": self.spm_multires,
        }


def find_spm_executable() -> Optional[Path]:
    """Find SPM executable for current platform"""
    system = platform.system()
    candidates = []
    
    if system == "Darwin":
        candidates = [
            PROJECT_ROOT / "SuperPitchMonitor.app" / "Contents" / "MacOS" / "SuperPitchMonitor",
            PROJECT_ROOT / "build-macos" / "SuperPitchMonitor.app" / "Contents" / "MacOS" / "SuperPitchMonitor",
        ]
    elif system == "Windows":
        candidates = [
            PROJECT_ROOT / "SuperPitchMonitor.exe",
            PROJECT_ROOT / "build-windows" / "SuperPitchMonitor_artefacts" / "SuperPitchMonitor.exe",
        ]
    else:
        candidates = [
            PROJECT_ROOT / "SuperPitchMonitor",
            PROJECT_ROOT / "build-linux" / "SuperPitchMonitor",
        ]
    
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def load_ground_truth() -> Dict[str, Any]:
    """Load ground truth data"""
    with open(GROUND_TRUTH_FILE, 'r', encoding='utf-8') as f:
        return json.load(f)


def analyze_audio_file_fft(filepath: Path) -> List[Dict[str, Any]]:
    """
    Perform full-window high-precision FFT analysis on audio file
    
    Returns list of detected peaks with frequency and magnitude
    """
    try:
        # Read WAV file
        with wave.open(str(filepath), 'rb') as wav_file:
            n_channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            sample_rate = wav_file.getframerate()
            n_frames = wav_file.getnframes()
            
            # Read raw data
            raw_data = wav_file.readframes(n_frames)
            
            # Convert to numpy array
            if sample_width == 2:
                data = np.frombuffer(raw_data, dtype=np.int16)
            elif sample_width == 4:
                data = np.frombuffer(raw_data, dtype=np.int32)
            else:
                data = np.frombuffer(raw_data, dtype=np.int8)
            
            # Convert to float and normalize
            data = data.astype(np.float32) / (2 ** (sample_width * 8 - 1))
            
            # Convert to mono if stereo
            if n_channels == 2:
                data = data.reshape(-1, 2).mean(axis=1)
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return []
    
    # Use full file for FFT (or maximum FFT_SIZE samples from middle)
    if len(data) > FFT_SIZE:
        # Take middle section for stationary analysis
        start = (len(data) - FFT_SIZE) // 2
        data = data[start:start + FFT_SIZE]
    
    # Apply Hann window
    window = np.hanning(len(data))
    data_windowed = data * window
    
    # Perform FFT
    fft_result = rfft(data_windowed)
    fft_magnitude = np.abs(fft_result)
    fft_magnitude[0] = 0  # Remove DC component
    
    # Frequency bins
    freqs = rfftfreq(len(data), 1.0 / sample_rate)
    
    # Find peaks
    peaks = find_peaks_simple(freqs, fft_magnitude, sample_rate)
    
    return peaks


def find_peaks_simple(freqs: np.ndarray, magnitudes: np.ndarray, 
                      sample_rate: int, 
                      threshold_db: float = -60,
                      min_freq: float = 50,
                      max_freq: float = 5000) -> List[Dict[str, Any]]:
    """Simple peak detection for reference FFT"""
    
    # Convert to dB
    mag_db = 20 * np.log10(magnitudes + 1e-10)
    max_db = np.max(mag_db)
    threshold = max_db + threshold_db
    
    # Find local maxima
    peaks = []
    for i in range(2, len(magnitudes) - 2):
        # Check if local maximum
        if (magnitudes[i] > magnitudes[i-1] and 
            magnitudes[i] > magnitudes[i-2] and
            magnitudes[i] > magnitudes[i+1] and 
            magnitudes[i] > magnitudes[i+2]):
            
            freq = freqs[i]
            mag = magnitudes[i]
            mag_db_val = mag_db[i]
            
            # Apply filters
            if freq < min_freq or freq > max_freq:
                continue
            if mag_db_val < threshold:
                continue
            
            # Parabolic interpolation for better frequency accuracy
            if i > 0 and i < len(magnitudes) - 1:
                alpha = np.log(magnitudes[i-1] + 1e-10)
                beta = np.log(magnitudes[i] + 1e-10)
                gamma = np.log(magnitudes[i+1] + 1e-10)
                p = 0.5 * (alpha - gamma) / (alpha - 2*beta + gamma)
                freq = freqs[i] + p * (freqs[1] - freqs[0])
            
            peaks.append({
                "frequency": round(float(freq), 2),
                "amplitude": round(float(mag_db_val - max_db), 2),
                "magnitude": round(float(mag), 6)
            })
    
    # Sort by amplitude (descending)
    peaks.sort(key=lambda x: x["amplitude"], reverse=True)
    
    # Return top 8 peaks
    return peaks[:8]


class SPMTestClient:
    """SPM TCP test client"""
    
    def __init__(self, host: str = "127.0.0.1", port: int = DEFAULT_TCP_PORT):
        self.host = host
        self.port = port
        self.socket: Optional[socket.socket] = None
    
    def connect(self, timeout: int = 30) -> bool:
        """Connect to SPM test server"""
        print(f"Connecting to {self.host}:{self.port}...")
        start = time.time()
        
        while time.time() - start < timeout:
            try:
                self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.socket.settimeout(5)
                self.socket.connect((self.host, self.port))
                self.socket.settimeout(None)
                print("Connected to SPM!")
                return True
            except (socket.error, ConnectionRefusedError):
                if self.socket:
                    self.socket.close()
                    self.socket = None
                time.sleep(0.5)
        
        print("Connection timeout!")
        return False
    
    def disconnect(self):
        """Disconnect from server"""
        if self.socket:
            try:
                self.socket.close()
            except:
                pass
            self.socket = None
    
    def send_command(self, cmd_dict: Dict) -> Dict:
        """Send command and receive response"""
        if not self.socket:
            raise RuntimeError("Not connected")
        
        json_data = json.dumps(cmd_dict).encode('utf-8')
        length = len(json_data)
        
        # Send length (4 bytes, big-endian)
        self.socket.sendall(struct.pack('>I', length))
        self.socket.sendall(json_data)
        
        # Receive response
        resp_len_bytes = self._recv_all(4)
        resp_len = struct.unpack('>I', resp_len_bytes)[0]
        resp_data = self._recv_all(resp_len)
        
        return json.loads(resp_data.decode('utf-8'))
    
    def _recv_all(self, n: int) -> bytes:
        """Receive exactly n bytes"""
        data = b''
        while len(data) < n:
            chunk = self.socket.recv(n - len(data))
            if not chunk:
                raise RuntimeError("Connection closed")
            data += chunk
        return data
    
    def set_multi_res(self, enabled: bool) -> Dict:
        return self.send_command({"cmd": "setMultiRes", "enabled": enabled})
    
    def load_file(self, filename: str) -> Dict:
        return self.send_command({"cmd": "loadFile", "filename": filename})
    
    def start(self) -> Dict:
        return self.send_command({"cmd": "startPlayback"})
    
    def stop(self) -> Dict:
        return self.send_command({"cmd": "stopPlayback"})
    
    def get_pitches(self) -> Dict:
        return self.send_command({"cmd": "getPitches"})
    
    def wait_frames(self, count: int, timeout_ms: int = 5000) -> Dict:
        return self.send_command({"cmd": "waitForFrames", "count": count})


def run_spm_test(filename: str, multi_res: bool, client: SPMTestClient, 
                 wait_frames: int = DEFAULT_WAIT_FRAMES) -> List[Dict[str, Any]]:
    """Run SPM test and return detected pitches"""
    
    print(f"  SPM test (multi-res {'ON' if multi_res else 'OFF'})...")
    
    # Set mode
    client.set_multi_res(multi_res)
    
    # Load file
    result = client.load_file(filename)
    if result.get('status') != 'ok':
        print(f"    Failed to load: {result}")
        return []
    
    # Start and wait
    client.start()
    client.wait_frames(wait_frames)
    
    # Get results
    pitches_result = client.get_pitches()
    pitches = pitches_result.get('pitches', [])
    
    # Stop
    client.stop()
    
    # Format results
    formatted = []
    for p in pitches:
        formatted.append({
            "frequency": round(float(p.get('frequency', 0)), 2),
            "amplitude": round(float(p.get('amplitude', 0)), 3),
            "confidence": round(float(p.get('confidence', 0)), 2),
            "midi_note": round(float(p.get('midiNote', 0)), 1)
        })
    
    return formatted


def generate_markdown_report(results: List[TestResult], output_path: Path):
    """Generate markdown test report"""
    
    md = []
    md.append("# SuperPitchMonitor Test Report")
    md.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    md.append(f"Platform: {platform.system()} {platform.machine()}")
    md.append(f"Python: {platform.python_version()}")
    md.append(f"NumPy: {np.__version__}")
    md.append("")
    
    # Summary table
    md.append("## Summary")
    md.append("")
    md.append("| File | Type | Ground Truth | FFT Peaks | SPM Std | SPM Multi |")
    md.append("|------|------|--------------|-----------|---------|-----------|")
    
    for result in results:
        gt_data = load_ground_truth()['files'].get(result.filename, {})
        file_type = gt_data.get('type', 'unknown')
        gt_count = len(result.ground_truth)
        fft_count = len(result.fft_results)
        std_count = len(result.spm_standard)
        multi_count = len(result.spm_multires)
        
        md.append(f"| {result.filename} | {file_type} | {gt_count} | {fft_count} | {std_count} | {multi_count} |")
    
    md.append("")
    md.append("---")
    md.append("")
    
    # Detailed results for each file
    for result in results:
        gt_data = load_ground_truth()['files'].get(result.filename, {})
        file_type = gt_data.get('type', 'unknown')
        description = gt_data.get('description', '')
        
        md.append(f"## {result.filename}")
        md.append(f"\n**Type:** {file_type}")
        md.append(f"\n**Description:** {description}")
        md.append("")
        
        # Ground Truth
        md.append("### Ground Truth")
        md.append("")
        if result.ground_truth:
            md.append("| Note | Frequency (Hz) | MIDI |")
            md.append("|------|----------------|------|")
            for note in result.ground_truth:
                md.append(f"| {note.get('note', 'N/A')} | {note.get('freq', 'N/A')} | {note.get('midi', 'N/A')} |")
        else:
            md.append("*No fundamentals (noise or silent)*")
        md.append("")
        
        # FFT Results
        md.append("### Full-Window High-Precision FFT (Python Reference)")
        md.append("")
        md.append(f"FFT Size: {FFT_SIZE} samples")
        md.append("")
        if result.fft_results:
            md.append("| # | Frequency (Hz) | Amplitude (dB) | Magnitude |")
            md.append("|---|----------------|----------------|-----------|")
            for i, peak in enumerate(result.fft_results[:8], 1):
                md.append(f"| {i} | {peak['frequency']:.2f} | {peak['amplitude']:.2f} | {peak['magnitude']:.6f} |")
        else:
            md.append("*No peaks detected*")
        md.append("")
        
        # SPM Standard
        md.append("### SPM Result (Multi-Resolution OFF)")
        md.append("")
        if result.spm_standard:
            md.append("| # | Frequency (Hz) | Amplitude | Confidence | MIDI Note |")
            md.append("|---|----------------|-----------|------------|-----------|")
            for i, p in enumerate(result.spm_standard, 1):
                md.append(f"| {i} | {p['frequency']:.2f} | {p['amplitude']:.3f} | {p['confidence']:.2f} | {p['midi_note']:.1f} |")
        else:
            md.append("*No pitches detected*")
        md.append("")
        
        # SPM Multi
        md.append("### SPM Result (Multi-Resolution ON)")
        md.append("")
        if result.spm_multires:
            md.append("| # | Frequency (Hz) | Amplitude | Confidence | MIDI Note |")
            md.append("|---|----------------|-----------|------------|-----------|")
            for i, p in enumerate(result.spm_multires, 1):
                md.append(f"| {i} | {p['frequency']:.2f} | {p['amplitude']:.3f} | {p['confidence']:.2f} | {p['midi_note']:.1f} |")
        else:
            md.append("*No pitches detected*")
        md.append("")
        
        # Comparison notes
        md.append("### Comparison Notes")
        md.append("")
        
        # Compare with ground truth
        gt_freqs = [n['freq'] for n in result.ground_truth]
        fft_freqs = [p['frequency'] for p in result.fft_results[:len(gt_freqs)]]
        std_freqs = [p['frequency'] for p in result.spm_standard]
        multi_freqs = [p['frequency'] for p in result.spm_multires]
        
        notes = []
        
        # Check FFT vs Ground Truth
        if gt_freqs and fft_freqs:
            fft_matches = sum(1 for gt in gt_freqs 
                            for det in fft_freqs 
                            if abs(gt - det) < 5.0)
            notes.append(f"- FFT matches GT: {fft_matches}/{len(gt_freqs)}")
        
        # Check SPM Standard vs Ground Truth
        if gt_freqs and std_freqs:
            std_matches = sum(1 for gt in gt_freqs 
                            for det in std_freqs 
                            if abs(gt - det) < 10.0)
            notes.append(f"- SPM Std matches GT: {std_matches}/{len(gt_freqs)}")
        
        # Check SPM Multi vs Ground Truth
        if gt_freqs and multi_freqs:
            multi_matches = sum(1 for gt in gt_freqs 
                              for det in multi_freqs 
                              if abs(gt - det) < 10.0)
            notes.append(f"- SPM Multi matches GT: {multi_matches}/{len(gt_freqs)}")
        
        # Check Std vs Multi consistency
        if std_freqs and multi_freqs:
            common = set(round(f, 0) for f in std_freqs) & set(round(f, 0) for f in multi_freqs)
            notes.append(f"- Std/Multi common detections: {len(common)}")
        
        if notes:
            md.extend(notes)
        else:
            md.append("- No comparison data available")
        
        md.append("")
        md.append("---")
        md.append("")
    
    # Write to file
    REPORTS_DIR.mkdir(exist_ok=True)
    output_path.write_text('\n'.join(md), encoding='utf-8')
    print(f"\nReport saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="SuperPitchMonitor Test Report Generator"
    )
    parser.add_argument("--output", type=str, 
                       default=REPORTS_DIR / f"test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                       help="Output markdown file path")
    parser.add_argument("--exe", type=str, default=None,
                       help="Path to SPM executable (auto-detect if not specified)")
    parser.add_argument("--port", type=int, default=DEFAULT_TCP_PORT,
                       help=f"TCP port (default: {DEFAULT_TCP_PORT})")
    parser.add_argument("--wait-frames", type=int, default=DEFAULT_WAIT_FRAMES,
                       help=f"Frames to wait for SPM analysis (default: {DEFAULT_WAIT_FRAMES})")
    
    args = parser.parse_args()
    
    # Ensure reports directory exists
    REPORTS_DIR.mkdir(exist_ok=True)
    
    # Load ground truth
    ground_truth = load_ground_truth()
    test_files = list(ground_truth['files'].keys())
    
    print("=" * 70)
    print("SuperPitchMonitor Test Report Generator")
    print("=" * 70)
    print(f"\nTest files: {len(test_files)}")
    print(f"Output: {args.output}")
    
    # Step 1: Run FFT analysis on all files
    print("\n" + "=" * 70)
    print("STEP 1: Full-Window High-Precision FFT Analysis")
    print("=" * 70)
    
    fft_results = {}
    for filename in test_files:
        filepath = TEST_AUDIO_DIR / filename
        if filepath.exists():
            print(f"\nAnalyzing: {filename}")
            peaks = analyze_audio_file_fft(filepath)
            fft_results[filename] = peaks
            print(f"  Detected {len(peaks)} peaks")
            for i, p in enumerate(peaks[:5], 1):
                print(f"    {i}. {p['frequency']:.2f} Hz ({p['amplitude']:.1f} dB)")
        else:
            print(f"Warning: File not found: {filepath}")
            fft_results[filename] = []
    
    # Step 2: Run SPM tests
    print("\n" + "=" * 70)
    print("STEP 2: SPM Testing")
    print("=" * 70)
    
    # Find SPM executable
    exe_path = args.exe
    if exe_path:
        exe_path = Path(exe_path)
    else:
        exe_path = find_spm_executable()
    
    if not exe_path or not exe_path.exists():
        print("Error: SPM executable not found")
        print("Use --exe to specify the path")
        return 1
    
    print(f"SPM executable: {exe_path}")
    
    # Launch SPM
    cmd = [str(exe_path), "-TestMode", "-TestPort", str(args.port)]
    print(f"Launching SPM: {' '.join(cmd)}")
    
    if platform.system() == "Windows":
        spm_process = subprocess.Popen(cmd, creationflags=subprocess.CREATE_NEW_PROCESS_GROUP, cwd=PROJECT_ROOT)
    else:
        spm_process = subprocess.Popen(cmd, cwd=PROJECT_ROOT)
    
    # Wait for SPM to start
    print("Waiting for SPM to start...")
    time.sleep(3)
    
    # Connect to SPM
    client = SPMTestClient(port=args.port)
    if not client.connect(timeout=30):
        print("Failed to connect to SPM")
        spm_process.terminate()
        return 1
    
    spm_results_standard = {}
    spm_results_multires = {}
    
    try:
        # Test with multi-resolution OFF
        print("\n--- Testing with Multi-Resolution OFF ---")
        for filename in test_files:
            pitches = run_spm_test(filename, False, client, args.wait_frames)
            spm_results_standard[filename] = pitches
        
        # Test with multi-resolution ON
        print("\n--- Testing with Multi-Resolution ON ---")
        for filename in test_files:
            pitches = run_spm_test(filename, True, client, args.wait_frames)
            spm_results_multires[filename] = pitches
    
    finally:
        client.disconnect()
        print("\nStopping SPM...")
        try:
            if platform.system() == "Windows":
                spm_process.send_signal(signal.CTRL_BREAK_EVENT)
            else:
                spm_process.terminate()
            spm_process.wait(timeout=5)
        except:
            spm_process.kill()
    
    # Step 3: Generate report
    print("\n" + "=" * 70)
    print("STEP 3: Generating Report")
    print("=" * 70)
    
    # Compile results
    results = []
    for filename in test_files:
        gt_fundamentals = ground_truth['files'][filename].get('fundamentals', [])
        
        result = TestResult(
            filename=filename,
            ground_truth=gt_fundamentals,
            fft_results=fft_results.get(filename, []),
            spm_standard=spm_results_standard.get(filename, []),
            spm_multires=spm_results_multires.get(filename, [])
        )
        results.append(result)
    
    # Generate markdown
    generate_markdown_report(results, Path(args.output))
    
    print("\n" + "=" * 70)
    print("Test Report Generation Complete!")
    print("=" * 70)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

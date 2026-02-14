#!/usr/bin/env python3
"""
SuperPitchMonitor Unified Test Client (Cross-Platform)
Supports Windows, macOS, and Linux

Replaces the original Windows-only test client (which used named pipes)
with a unified TCP socket-based implementation.

Usage:
    # Run all tests (auto-detect SPM executable)
    python test_client.py
    
    # Run specific test category
    python test_client.py --test single_tone
    
    # Use custom SPM executable
    python test_client.py --exe /path/to/SuperPitchMonitor
    
    # Use custom port
    python test_client.py --port 9999
    
    # Keep SPM running after tests (for debugging)
    python test_client.py --keep-alive
    
Architecture:
    1. Launches SPM in TestMode (headless) with TestServer enabled
    2. Connects via TCP socket to port 9999 (default)
    3. Sends JSON commands and receives responses
    4. Validates results against ground truth
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

# Configuration
DEFAULT_TCP_PORT = 9999
DEFAULT_WAIT_FRAMES = 60

SCRIPT_DIR = Path(__file__).parent.absolute()
PROJECT_ROOT = SCRIPT_DIR.parent.parent
TEST_AUDIO_DIR = PROJECT_ROOT / "Resources" / "TestAudio"
GROUND_TRUTH_FILE = TEST_AUDIO_DIR / "test_ground_truth.json"


def find_spm_executable() -> Optional[Path]:
    """Find SPM executable for current platform"""
    system = platform.system()
    
    # Possible executable names and locations
    candidates = []
    
    if system == "Darwin":  # macOS
        candidates = [
            PROJECT_ROOT / "SuperPitchMonitor.app" / "Contents" / "MacOS" / "SuperPitchMonitor",
            PROJECT_ROOT / "build-macos" / "SuperPitchMonitor.app" / "Contents" / "MacOS" / "SuperPitchMonitor",
        ]
    elif system == "Windows":
        candidates = [
            PROJECT_ROOT / "SuperPitchMonitor.exe",
            PROJECT_ROOT / "build-windows" / "SuperPitchMonitor_artefacts" / "SuperPitchMonitor.exe",
        ]
    else:  # Linux
        candidates = [
            PROJECT_ROOT / "SuperPitchMonitor",
            PROJECT_ROOT / "build-linux" / "SuperPitchMonitor",
        ]
    
    for candidate in candidates:
        if candidate.exists():
            return candidate
    
    return None


class GroundTruth:
    """Ground truth data manager"""
    
    def __init__(self, filepath: Path):
        self.filepath = filepath
        self.data = None
        self.load()
    
    def load(self):
        """Load ground truth data from JSON"""
        if not self.filepath.exists():
            print(f"Error: Ground truth file not found: {self.filepath}")
            sys.exit(1)
            
        with open(self.filepath, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        print(f"Loaded ground truth for {len(self.data['files'])} test files")
    
    def get_file_truth(self, filename: str) -> Dict[str, Any]:
        """Get ground truth for a specific file"""
        return self.data['files'].get(filename)
    
    def get_test_categories(self) -> Dict[str, Any]:
        """Get test categories"""
        return self.data.get('test_categories', {})
    
    def list_files(self) -> List[str]:
        """List all files with ground truth"""
        return list(self.data['files'].keys())
    
    def validate_detection(self, filename: str, detected_pitches: List[Dict], 
                          mode: str = "standard") -> Tuple[bool, Dict]:
        """
        Validate detected pitches against ground truth
        
        Returns: (passed, details)
        """
        truth = self.get_file_truth(filename)
        if not truth:
            return False, {"error": f"No ground truth for {filename}"}
        
        fundamentals = truth.get('fundamentals', [])
        expected_peaks = truth.get('expected_peaks', [])
        
        results = {
            "file": filename,
            "mode": mode,
            "total_expected": len(fundamentals),
            "total_detected": len(detected_pitches),
            "matches": [],
            "missed": [],
            "false_positives": [],
            "score": 0.0
        }
        
        # Check each expected fundamental
        matched_detected = set()
        for expected in fundamentals:
            exp_freq = expected['freq']
            tolerance = expected.get('tolerance_hz', 10.0)
            note = expected.get('note', '')
            
            # Find matching detected pitch
            match = None
            for i, det in enumerate(detected_pitches):
                if i in matched_detected:
                    continue
                det_freq = det.get('frequency', 0)
                if abs(det_freq - exp_freq) <= tolerance:
                    match = det
                    matched_detected.add(i)
                    break
            
            if match:
                results['matches'].append({
                    "expected": {"freq": exp_freq, "note": note},
                    "detected": {
                        "freq": match.get('frequency'),
                        "midi": match.get('midiNote'),
                        "confidence": match.get('confidence'),
                        "harmonics": match.get('harmonicCount')
                    },
                    "error_hz": match.get('frequency', 0) - exp_freq
                })
            else:
                results['missed'].append({
                    "freq": exp_freq,
                    "note": note,
                    "tolerance": tolerance
                })
        
        # Check for false positives
        for i, det in enumerate(detected_pitches):
            if i in matched_detected:
                continue
            
            det_freq = det.get('frequency', 0)
            # Check if it's a harmonic of any expected fundamental
            is_harmonic = False
            for exp in fundamentals:
                ratio = det_freq / exp['freq']
                nearest_harmonic = round(ratio)
                if nearest_harmonic >= 2 and nearest_harmonic <= 8:
                    if abs(ratio - nearest_harmonic) < 0.05:
                        is_harmonic = True
                        break
            
            if not is_harmonic:
                results['false_positives'].append({
                    "freq": det_freq,
                    "midi": det.get('midiNote'),
                    "confidence": det.get('confidence')
                })
        
        # Calculate score
        if results['total_expected'] > 0:
            recall = len(results['matches']) / results['total_expected']
            precision = len(results['matches']) / max(len(results['matches']) + len(results['false_positives']), 1)
            results['score'] = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            results['recall'] = recall
            results['precision'] = precision
        
        # Determine pass/fail
        passed = len(results['missed']) <= 1 and len(results['false_positives']) <= 1
        if truth.get('type') == 'single_tone':
            passed = len(results['matches']) >= 1
        
        return passed, results


class SPMTestClient:
    """SuperPitchMonitor TCP test client (cross-platform)"""
    
    def __init__(self, host: str = "127.0.0.1", port: int = DEFAULT_TCP_PORT):
        self.host = host
        self.port = port
        self.socket: Optional[socket.socket] = None
        self.ground_truth = GroundTruth(GROUND_TRUTH_FILE)
    
    def connect(self, timeout: int = 30) -> bool:
        """Connect to SPM test server via TCP"""
        print(f"Connecting to {self.host}:{self.port}...")
        start = time.time()
        
        while time.time() - start < timeout:
            try:
                self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.socket.settimeout(5)
                self.socket.connect((self.host, self.port))
                self.socket.settimeout(None)
                print("Connected!")
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
        # Send data
        self.socket.sendall(json_data)
        
        # Receive response length
        resp_len_bytes = self._recv_all(4)
        resp_len = struct.unpack('>I', resp_len_bytes)[0]
        
        # Receive response data
        resp_data = self._recv_all(resp_len)
        
        return json.loads(resp_data.decode('utf-8'))
    
    def _recv_all(self, n: int) -> bytes:
        """Receive exactly n bytes"""
        data = b''
        while len(data) < n:
            chunk = self.socket.recv(n - len(data))
            if not chunk:
                raise RuntimeError("Connection closed unexpectedly")
            data += chunk
        return data
    
    # Convenience methods
    def get_status(self) -> Dict:
        return self.send_command({"cmd": "getStatus"})
    
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
    
    def get_spectrum_peaks(self, freq_min: float = 50, freq_max: float = 5000) -> Dict:
        return self.send_command({"cmd": "getSpectrumPeaks", "freqMin": freq_min, "freqMax": freq_max})
    
    def run_test_file(self, filename: str, multi_res: bool = True, 
                      wait_frames: int = DEFAULT_WAIT_FRAMES) -> Tuple[bool, Dict]:
        """Run a complete test on a single file"""
        print(f"\n{'='*60}")
        print(f"Testing: {filename}")
        print(f"Multi-resolution: {'ON' if multi_res else 'OFF'}")
        print(f"{'='*60}")
        
        # Set multi-res mode
        self.set_multi_res(multi_res)
        
        # Load file
        result = self.load_file(filename)
        if result.get('status') != 'ok':
            print(f"Failed to load file: {result}")
            return False, {"error": "Failed to load file"}
        print(f"Loaded: {filename}")
        
        # Start playback
        self.start()
        print("Playback started")
        
        # Wait for analysis
        print(f"Waiting for {wait_frames} frames...")
        self.wait_frames(wait_frames)
        
        # Get results
        pitches_result = self.get_pitches()
        detected = pitches_result.get('pitches', [])
        
        # Stop playback
        self.stop()
        print("Playback stopped")
        
        # Validate against ground truth
        passed, details = self.ground_truth.validate_detection(
            filename, detected, 
            mode="multi_res" if multi_res else "single_res"
        )
        
        # Print results
        print(f"\nResults:")
        print(f"  Expected: {details['total_expected']}")
        print(f"  Detected: {details['total_detected']}")
        print(f"  Matches: {len(details['matches'])}")
        print(f"  Missed: {len(details['missed'])}")
        print(f"  False Positives: {len(details['false_positives'])}")
        print(f"  Score: {details['score']:.2%}")
        print(f"  Status: {'PASS' if passed else 'FAIL'}")
        
        if details['matches']:
            print(f"\n  Matched pitches:")
            for m in details['matches']:
                print(f"    {m['expected']['note']}: {m['detected']['freq']:.1f} Hz "
                      f"(expected {m['expected']['freq']:.1f} Hz, "
                      f"error {m['error_hz']:+.1f} Hz)")
        
        return passed, details


def run_all_tests(args) -> int:
    """Run all tests"""
    client = SPMTestClient(port=args.port)
    
    # Connect to server
    if not client.connect(timeout=args.timeout):
        print("Failed to connect to SPM. Is it running?")
        return 1
    
    try:
        # Get list of test files
        files = client.ground_truth.list_files()
        if args.test:
            files = [f for f in files if args.test in f]
        
        results = []
        
        # Run tests with multi-resolution ON
        print("\n" + "="*60)
        print("TESTING WITH MULTI-RESOLUTION ON")
        print("="*60)
        
        for filename in files:
            passed, details = client.run_test_file(filename, multi_res=True, 
                                                   wait_frames=args.wait_frames)
            results.append({
                "file": filename,
                "multi_res": True,
                "passed": passed,
                "details": details
            })
        
        # Print summary
        print("\n" + "="*60)
        print("TEST SUMMARY")
        print("="*60)
        
        passed_count = sum(1 for r in results if r['passed'])
        total_count = len(results)
        
        for r in results:
            status = "PASS" if r['passed'] else "FAIL"
            score = r['details'].get('score', 0)
            print(f"  [{status}] {r['file']}: {score:.1%}")
        
        print(f"\nTotal: {passed_count}/{total_count} passed ({passed_count/total_count*100:.1f}%)")
        
        return 0 if passed_count == total_count else 1
    
    finally:
        client.disconnect()


def main():
    parser = argparse.ArgumentParser(
        description="SuperPitchMonitor Unified Test Client (Cross-Platform)"
    )
    parser.add_argument("--port", type=int, default=DEFAULT_TCP_PORT,
                       help=f"TCP port (default: {DEFAULT_TCP_PORT})")
    parser.add_argument("--test", type=str, default=None,
                       help="Run specific test (e.g., 'single_tone', 'chord')")
    parser.add_argument("--timeout", type=int, default=30,
                       help="Connection timeout in seconds (default: 30)")
    parser.add_argument("--wait-frames", type=int, default=DEFAULT_WAIT_FRAMES,
                       help=f"Frames to wait for analysis (default: {DEFAULT_WAIT_FRAMES})")
    parser.add_argument("--exe", type=str, default=None,
                       help="Path to SPM executable (auto-detect if not specified)")
    parser.add_argument("--keep-alive", action="store_true",
                       help="Keep SPM running after tests (for debugging)")
    parser.add_argument("--no-launch", action="store_true",
                       help="Don't launch SPM (assume it's already running)")
    
    args = parser.parse_args()
    
    spm_process = None
    
    try:
        if not args.no_launch:
            # Find SPM executable
            exe_path = args.exe
            if exe_path:
                exe_path = Path(exe_path)
            else:
                exe_path = find_spm_executable()
            
            if not exe_path or not exe_path.exists():
                print("Error: SPM executable not found")
                print("Use --exe to specify the path to SuperPitchMonitor executable")
                return 1
            
            print(f"SPM executable: {exe_path}")
            
            # Launch SPM in test mode (from project root so it can find Resources)
            cmd = [str(exe_path), "-TestMode", "-TestPort", str(args.port)]
            print(f"Launching SPM in test mode: {' '.join(cmd)}")
            print(f"Working directory: {PROJECT_ROOT}")
            
            if platform.system() == "Windows":
                spm_process = subprocess.Popen(cmd, creationflags=subprocess.CREATE_NEW_PROCESS_GROUP, cwd=PROJECT_ROOT)
            else:
                spm_process = subprocess.Popen(cmd, cwd=PROJECT_ROOT)
            
            # Wait for SPM to start
            print("Waiting for SPM to start...")
            time.sleep(3)
        
        # Run tests
        result = run_all_tests(args)
        
        if args.keep_alive and spm_process:
            print("\nKeeping SPM running (press Ctrl+C to stop)...")
            try:
                spm_process.wait()
            except KeyboardInterrupt:
                print("\nStopping SPM...")
        
        return result
    
    finally:
        if spm_process and not args.keep_alive:
            print("\nStopping SPM...")
            try:
                if platform.system() == "Windows":
                    spm_process.send_signal(signal.CTRL_BREAK_EVENT)
                else:
                    spm_process.terminate()
                spm_process.wait(timeout=5)
            except:
                spm_process.kill()


if __name__ == "__main__":
    sys.exit(main())

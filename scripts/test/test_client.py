#!/usr/bin/env python3
"""
SuperPitchMonitor Python Test Client with Ground Truth Validation

Usage:
    python test_client.py                    # Run all tests
    python test_client.py --test single_tone # Run specific test category
    python test_client.py --gui              # Start SPM in GUI mode
    python test_client.py --validate-only    # Validate ground truth data
    
Requirements:
    - Windows (uses named pipes)
    - pywin32: pip install pywin32
"""

import subprocess
import time
import json
import struct
import sys
import os
import argparse
from typing import List, Dict, Any, Tuple

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


class GroundTruth:
    """Ground truth data manager"""
    
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.data = None
        self.load()
    
    def load(self):
        """Load ground truth data from JSON"""
        with open(self.filepath, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        print(f"Loaded ground truth for {len(self.data['files'])} test files")
    
    def get_file_truth(self, filename: str) -> Dict[str, Any]:
        """Get ground truth for a specific file"""
        return self.data['files'].get(filename)
    
    def get_test_categories(self) -> Dict[str, Any]:
        """Get test categories"""
        return self.data.get('test_categories', {})
    
    def get_test_scenarios(self) -> Dict[str, Any]:
        """Get test scenarios"""
        return self.data.get('test_scenarios', {})
    
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
        
        # Check for false positives (detected but not expected)
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
            passed = len(results['matches']) >= 1  # Single tone must be detected
        
        return passed, results


class SPMTestClient:
    """SuperPitchMonitor test client"""
    
    def __init__(self, pipe_name=PIPE_NAME):
        self.pipe_name = pipe_name
        self.pipe = None
        self.ground_truth = GroundTruth(GROUND_TRUTH_FILE)
        
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
                if e.winerror == 2:  # ERROR_FILE_NOT_FOUND
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
    
    # Convenience methods
    def get_status(self):
        return self.send_command({"cmd": "getStatus"})
    
    def set_multi_res(self, enabled):
        return self.send_command({"cmd": "setMultiRes", "enabled": enabled})
    
    def load_file(self, filename):
        # Ensure file exists in test directory
        test_dir = os.path.join(SCRIPT_DIR, 'Resources', 'TestAudio')
        full_path = os.path.join(test_dir, filename)
        if os.path.exists(full_path):
            # Send just the filename - SPM will resolve path
            return self.send_command({"cmd": "loadFile", "filename": filename})
        else:
            return {"status": "error", "message": f"File not found: {full_path}"}
    
    def start(self):
        return self.send_command({"cmd": "start"})
    
    def stop(self):
        return self.send_command({"cmd": "stop"})
    
    def get_pitches(self):
        return self.send_command({"cmd": "getPitches"})
    
    def get_spectrum_peaks(self, freq_min=50, freq_max=2000):
        return self.send_command({
            "cmd": "getSpectrumPeaks",
            "freqMin": freq_min,
            "freqMax": freq_max
        })
    
    def wait_frames(self, count, timeout=5000):
        return self.send_command({"cmd": "wait", "frames": count, "timeout": timeout})
    
    def exit(self):
        return self.send_command({"cmd": "exit"})
    
    def test_file(self, filename: str, multi_res: bool = False, 
                  warmup_frames: int = 60, test_frames: int = 60) -> Tuple[bool, Dict]:
        """
        Test a single file against ground truth
        
        Returns: (passed, details)
        """
        print(f"\n{'='*60}")
        print(f"Testing: {filename}")
        print(f"Multi-resolution: {'ON' if multi_res else 'OFF'}")
        print(f"{'='*60}")
        
        truth = self.ground_truth.get_file_truth(filename)
        if not truth:
            print(f"ERROR: No ground truth for {filename}")
            return False, {"error": "No ground truth"}
        
        print(f"Description: {truth.get('description', 'N/A')}")
        print(f"Type: {truth.get('type', 'unknown')}")
        print(f"Expected fundamentals: {[f['note'] for f in truth.get('fundamentals', [])]}")
        
        # Configure and start
        self.set_multi_res(multi_res)
        
        if not self.load_file(filename).get('status') == 'ok':
            print(f"ERROR: Failed to load {filename}")
            return False, {"error": "Failed to load file"}
        
        self.start()
        
        # Warmup
        print(f"\nWarming up ({warmup_frames} frames)...")
        self.wait_frames(warmup_frames)
        
        # Collect results
        print(f"Collecting data ({test_frames} frames)...")
        all_pitches = []
        for _ in range(5):  # Sample 5 times
            result = self.get_pitches()
            pitches = result.get('pitches', [])
            all_pitches.extend(pitches)
            self.wait_frames(test_frames // 5)
        
        self.stop()
        
        # Merge duplicate detections (same frequency)
        merged = {}
        for p in all_pitches:
            freq = p['frequency']
            # Group by approximate frequency (within 5Hz)
            key = round(freq / 5) * 5
            if key not in merged:
                merged[key] = []
            merged[key].append(p)
        
        # Average duplicates
        unique_pitches = []
        for key, group in merged.items():
            avg_pitch = {
                'frequency': sum(p['frequency'] for p in group) / len(group),
                'midiNote': sum(p['midiNote'] for p in group) / len(group),
                'confidence': max(p['confidence'] for p in group),
                'harmonicCount': max(p['harmonicCount'] for p in group),
                'centsDeviation': sum(p['centsDeviation'] for p in group) / len(group)
            }
            unique_pitches.append(avg_pitch)
        
        # Validate against ground truth
        passed, details = self.ground_truth.validate_detection(
            filename, unique_pitches, 
            "multi_res" if multi_res else "standard"
        )
        
        # Print results
        print(f"\nDetected {len(unique_pitches)} unique pitches:")
        for p in sorted(unique_pitches, key=lambda x: x['frequency']):
            print(f"  {p['frequency']:7.1f} Hz | midi: {p['midiNote']:.1f} | conf: {p['confidence']:.2f} | H: {p['harmonicCount']}")
        
        print(f"\nValidation Results:")
        print(f"  Matches: {len(details['matches'])}/{details['total_expected']}")
        print(f"  Missed: {len(details['missed'])}")
        print(f"  False positives: {len(details['false_positives'])}")
        if 'score' in details:
            print(f"  F1 Score: {details['score']:.2f}")
        
        if details['missed']:
            print(f"  Missed fundamentals:")
            for m in details['missed']:
                print(f"    - {m['note']} ({m['freq']:.1f} Hz)")
        
        if details['false_positives']:
            print(f"  False positives:")
            for fp in details['false_positives']:
                print(f"    - {fp['freq']:.1f} Hz (conf: {fp['confidence']:.2f})")
        
        print(f"\nResult: {'PASS' if passed else 'FAIL'}")
        
        return passed, details


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


def run_category_test(client: SPMTestClient, category: str, ground_truth: GroundTruth):
    """Run tests for a specific category"""
    categories = ground_truth.get_test_categories()
    if category not in categories:
        print(f"Unknown category: {category}")
        print(f"Available: {list(categories.keys())}")
        return False
    
    cat_data = categories[category]
    files = cat_data.get('files', [])
    
    print(f"\n{'#'*70}")
    print(f"Category: {category}")
    print(f"Description: {cat_data.get('description', '')}")
    print(f"Files: {len(files)}")
    print(f"{'#'*70}")
    
    results = []
    
    # Test each file in both modes
    for filename in files:
        # Standard mode
        passed_std, details_std = client.test_file(filename, multi_res=False)
        results.append({
            'file': filename,
            'mode': 'standard',
            'passed': passed_std,
            'details': details_std
        })
        
        # Multi-res mode
        passed_mr, details_mr = client.test_file(filename, multi_res=True)
        results.append({
            'file': filename,
            'mode': 'multi_res',
            'passed': passed_mr,
            'details': details_mr
        })
    
    # Summary
    print(f"\n{'='*70}")
    print(f"CATEGORY SUMMARY: {category}")
    print(f"{'='*70}")
    
    for r in results:
        status = "PASS" if r['passed'] else "FAIL"
        score = r['details'].get('score', 0)
        print(f"  [{status}] {r['file']} ({r['mode']}) - F1: {score:.2f}")
    
    all_passed = all(r['passed'] for r in results)
    print(f"\nOverall: {'PASS' if all_passed else 'FAIL'}")
    
    return all_passed


def main():
    parser = argparse.ArgumentParser(description='SuperPitchMonitor Test Client')
    parser.add_argument('--gui', action='store_true', help='Start SPM in GUI mode')
    parser.add_argument('--test', type=str, help='Test category to run (single_tone, simple_chord, etc.)')
    parser.add_argument('--file', type=str, help='Test specific file')
    parser.add_argument('--validate-only', action='store_true', help='Validate ground truth data')
    parser.add_argument('--list-files', action='store_true', help='List all test files')
    args = parser.parse_args()
    
    # Load ground truth
    ground_truth = GroundTruth(GROUND_TRUTH_FILE)
    
    if args.list_files:
        print("Available test files:")
        for f in ground_truth.list_files():
            truth = ground_truth.get_file_truth(f)
            print(f"  - {f}: {truth.get('description', 'N/A')}")
        return
    
    if args.validate_only:
        print("Ground truth validation:")
        print(f"  Total files: {len(ground_truth.list_files())}")
        print(f"  Categories: {list(ground_truth.get_test_categories().keys())}")
        print(f"  Scenarios: {list(ground_truth.get_test_scenarios().keys())}")
        return
    
    if args.gui:
        exe = find_spm_exe()
        if exe:
            print(f"Starting SPM in GUI mode: {exe}")
            subprocess.run([exe])
        else:
            print("Error: SuperPitchMonitor.exe not found!")
        return
    
    # Run automated tests
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
        cwd=os.path.dirname(exe)
    )
    
    try:
        time.sleep(2)  # Wait for SPM to start
        
        client = SPMTestClient()
        if not client.connect(timeout=30):
            print("Failed to connect to SPM!")
            sys.exit(1)
        
        # Run tests
        if args.file:
            passed, _ = client.test_file(args.file, multi_res=False)
            passed_mr, _ = client.test_file(args.file, multi_res=True)
            all_passed = passed and passed_mr
        elif args.test:
            all_passed = run_category_test(client, args.test, ground_truth)
        else:
            # Run all categories
            categories = ['single_tone', 'single_note', 'simple_chord']
            results = []
            for cat in categories:
                results.append(run_category_test(client, cat, ground_truth))
            all_passed = all(results)
        
        client.exit()
        client.disconnect()
        
        sys.exit(0 if all_passed else 1)
        
    finally:
        proc.terminate()
        proc.wait()


if __name__ == '__main__':
    main()

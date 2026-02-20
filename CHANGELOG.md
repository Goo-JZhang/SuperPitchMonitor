# Changelog

All notable changes to SuperPitchMonitor project.

## [Unreleased]

### Added

#### Machine Learning Integration (ONNX Runtime)
- **MLPitchDetector**: ONNX Runtime-based neural network inference engine
  - Input: 4096 samples @ 44.1kHz (raw audio)
  - Output: 2048 frequency bins with (confidence, energy) pairs
  - GPU acceleration: CoreML (macOS/iOS), CUDA (Windows), NNAPI (Android)
  - Async inference with dedicated thread pool
  - Dual output: high-confidence detections + full spectrum for visualization

- **ML Mode UI Components**
  - SpectrumDisplay: Dual Y-axis (Confidence 0-1, Energy normalized)
  - PitchDisplay: Note name + cents deviation + ML energy display
  - Mode indicator: Semi-transparent label (MLMODE/FFTMODE) in bottom-left
  - Frequency cutoff: Values outside 20-5000Hz displayed as 0

- **AudioEngine ML Integration**
  - ML/FFT mode switching at runtime
  - GPU/CPU toggle support
  - Model path selection
  - Automatic model discovery from MLModel/ directory

- **Test Model** (`MLModel/export_test_model_with_peaks.py`)
  - Synthetic C Major chord peaks (C4, E4, G4)
  - Confidence range: [0.05, 0.99], Energy range: [0.3, 2.0]
  - File: `pitchnet_stub_v1.onnx` (~8.5MB)

- **Build System Updates**
  - `BUILD_ONNXRUNTIME_FROM_SOURCE` CMake option
  - Source build with CoreML support (20-30 min, full operator support)
  - Pre-built binary fallback for faster builds
  - `ThirdParty/` directory reorganization (juce/, onnxruntime/)

### Changed

#### Testing Framework - Complete Rewrite
- **BREAKING**: Replaced Windows-only named pipe testing with cross-platform TCP-based testing
- **New**: Unified test client works on Windows, macOS, and Linux
- **Removed**: `AutoTestManager` class (Windows named pipe implementation)
- **Removed**: Dependency on `pywin32` for testing

#### Command Line Parameters
- **Changed**: `-AutoTest` → `-TestMode` for headless testing mode
- **Added**: `-TestPort <port>` to specify TCP port for TestServer

#### File Locations
- **Changed**: Runtime logs moved from `build_logs/` to `Saved/Logs/`
- **Changed**: macOS app bundle now outputs to project root (like Windows/Linux)

### Added

- **TestServer**: TCP-based test server running inside application
  - Port 9999 by default (configurable)
  - JSON-based command protocol
  - Thread-safe implementation
  
- **test_client.py**: Cross-platform Python test client
  - Auto-detects SPM executable
  - Manages SPM process lifecycle
  - Validates results against ground truth
  - No external dependencies (pure Python stdlib)

- **Documentation**:
  - `scripts/test/README.md` - Test framework guide
  - `Docs/06_Development/Test_Framework_Architecture.md` - Architecture details

### Removed

- `Source/Test/AutoTestManager.h`
- `Source/Test/AutoTestManager.cpp`
- Windows-specific test scripts (replaced by unified `test_client.py`)

### Migration Guide

#### For Users

**Running tests:**
```bash
# Before (Windows only)
python scripts/test/test_detailed_analysis.py --file sine_440hz.wav

# After (All platforms)
python scripts/test/test_client.py --test sine_440
```

**Starting in test mode:**
```bash
# Before
./SuperPitchMonitor.exe -AutoTest

# After
./SuperPitchMonitor -TestMode -TestPort 9999
```

#### For Developers

**Finding logs:**
```bash
# Before
cat build_logs/app_*.log

# After
cat Saved/Logs/app_*.log
```

**Test server commands:**
- Still uses JSON protocol
- Same commands: `getStatus`, `setMultiRes`, `loadFile`, `startPlayback`, etc.
- Only transport changed: named pipes → TCP sockets

## [Previous Versions]

### Added
- Multi-resolution spectral analysis
- Cross-platform build system (CMake)
- JUCE framework integration
- Real-time pitch detection
- Spectrum visualization

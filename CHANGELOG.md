# Changelog

All notable changes to SuperPitchMonitor project.

## [Unreleased]

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

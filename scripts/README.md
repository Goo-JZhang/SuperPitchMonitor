# SuperPitchMonitor Scripts

This directory contains all utility scripts organized by category.

## Directory Structure

```
scripts/
├── audio/              # Audio generation scripts
├── build/              # Build setup scripts
├── test/               # Testing and analysis scripts
└── utils/              # Utility scripts
```

## Audio Scripts (`audio/`)

### generate_all_tests.py
Generates all test audio files (10-second duration for 0.1Hz FFT resolution).

**Usage:**
```bash
python scripts/audio/generate_all_tests.py
```

**Output:** `Resources/TestAudio/*.wav`

**Note:** Test audio files are not included in the Git repository. Run this script after cloning to generate them.

### generate_harmonic_chords.py
Generates harmonic-rich chords (piano, string, brass, organ timbres).

### generate_orchestral_test.py
Generates orchestral-style test audio.

### generate_test_signals.py
Generates basic test signals (sine waves, sweeps, noise).

## Build Scripts (`build/`)

### setup_windows.bat
One-click Windows build setup. Generates test audio and builds the project.

**Usage:**
```powershell
scripts\build\setup_windows.bat
```

### setup_macos.sh
One-click macOS build setup.

**Usage:**
```bash
chmod +x scripts/build/setup_macos.sh
scripts/build/setup_macos.sh
```

### setup_linux.sh
One-click Linux build setup.

**Usage:**
```bash
chmod +x scripts/build/setup_linux.sh
scripts/build/setup_linux.sh
```

## Test Scripts (`test/`)

### test_detailed_analysis.py
Comprehensive testing with FFT reference analysis.

**Features:**
- High-precision FFT (10s window, 0.1Hz resolution)
- Per-frame SPM data collection
- Statistical analysis (mean/std for frequency, confidence, amplitude)
- Both Multi-Resolution ON and OFF modes
- Detailed HTML reports

**Usage:**
```bash
# Test single file
python scripts/test/test_detailed_analysis.py --file chord_c_major_7_piano.wav

# Test all files
python scripts/test/test_detailed_analysis.py --all
```

**Output:** `Docs/test_reports/detailed_test_report.html`

### test_client.py
Basic test client for SPM validation.

### test_client_with_fft.py
Test client with FFT comparison.

### test_full_analysis.py
Full test suite with statistical analysis.

### test_polyphonic.py
Polyphonic detection specific tests.

## Utility Scripts (`utils/`)

### analyze_cmake.py
Analyzes CMake configuration.

### check_encoding.py
Checks file encoding.

### check_full.py
Full system check.

### fix_cmake.py
Fixes common CMake issues.

### view_cmake.py
Views CMake configuration.

## Legacy Scripts (root of scripts/)

These scripts are kept for compatibility but may be moved to subdirectories in the future:

- `auto_build.ps1` - Automated build script
- `build_android.bat` - Android build
- `build_android_fix.ps1` - Android build fixes
- `build_now.bat` - Quick build
- `build_windows.bat` - Windows build
- `check_cross_platform.ps1` - Cross-platform checks
- `check_system_health.ps1` - System health check
- `fix_android_json_error.ps1` - Android JSON fixes
- `init_git.bat` - Git initialization
- `open_vs2022.bat` - Open in VS2022
- `optimize_emulator.ps1` - Emulator optimization
- `quick_start.bat` - Quick start
- `setup_android_sdk.ps1` - Android SDK setup
- `verify_environment.ps1` - Environment verification

## Python Dependencies

Most scripts require:
```bash
pip install numpy scipy
```

Test client scripts additionally require:
```bash
pip install pywin32  # Windows only
```

## Notes

- All scripts should be run from the project root directory
- Audio generation scripts output to `Resources/TestAudio/`
- Test scripts output reports to `Docs/test_reports/`
- Build scripts create platform-specific directories (`build-windows/`, `build-macos/`, `build-linux/`)

# SuperPitchMonitor

A real-time polyphonic pitch detection and visualization application based on the JUCE framework. Supports multi-resolution spectral analysis for accurate pitch detection across the full frequency range.

## Features

- **Real-time Pitch Detection**: Monophonic and polyphonic pitch detection
- **Multi-resolution Analysis**: Adaptive FFT resolution for different frequency bands
- **Cross-platform**: Windows, macOS, Android, iOS
- **Visual Feedback**: Spectrum display, pitch waterfall, and pitch cards
- **Test Framework**: Python-based automated testing with FFT reference analysis

## Project Structure

```
SuperPitchMonitor/
├── Source/                 # Source code
│   ├── Audio/             # Audio processing (pitch detection, spectrum analysis)
│   ├── UI/                # User interface components
│   ├── Test/              # Automated testing framework
│   └── Utils/             # Utilities (logger, configuration)
├── Resources/             # Resources (images, test audio)
│   └── TestAudio/         # Test audio files
├── JUCE/                  # JUCE framework (optional, can use system JUCE)
├── Docs/                  # Documentation
├── scripts/               # Build and utility scripts
├── CMakeLists.txt         # Main CMake configuration
└── README.md             # This file
```

## Building

### Prerequisites

- CMake 3.15+
- C++17 compatible compiler
- JUCE framework (included as submodule or system-installed)

### Windows (Visual Studio)

```bash
mkdir build-windows
cd build-windows
cmake .. -G "Visual Studio 17 2022" -A x64
cmake --build . --config Release
```

### macOS (Xcode)

```bash
mkdir build-macos
cd build-macos
cmake .. -G Xcode -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release
```

Or using Makefile:

```bash
mkdir build-macos
cd build-macos
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build .
```

### Android

Use Android Studio or gradle from the `Builds/Android` directory.

## Testing

### Automated Testing

The project includes a Python-based test framework that:
- Generates 10-second test audio files (0.1Hz FFT resolution)
- Compares SPM real-time detection with high-precision FFT analysis
- Generates detailed HTML reports

```bash
cd Resources/TestAudio
python generate_all_tests.py  # Generate test audio files

cd ../..
python test_detailed_analysis.py --all  # Run full test suite
```

### Test Reports

Test reports are saved to `Docs/test_reports/`:
- `detailed_test_report.html` - Per-file detailed analysis
- `polyphonic_detection_debug.md` - Debugging notes

## Key Algorithms

- **Multi-resolution FFT**: Different FFT sizes for low/mid/high frequencies
- **YIN Pitch Detection**: Time-domain periodicity analysis
- **Harmonic Analysis**: Separates fundamentals from harmonics
- **Polyphonic Separation**: Identifies multiple simultaneous pitches

## Documentation

See the `Docs/` directory for:
- Technical analysis and architecture
- Algorithm descriptions
- Platform-specific build instructions
- Troubleshooting guides

## License

This project is licensed under the **GNU General Public License v3.0 (GPLv3)**.

This software uses the [JUCE](https://juce.com/) framework, which is also licensed under GPLv3 (with commercial licensing alternatives available).

See [LICENSE](LICENSE) file for details.

### Commercial Licensing

If you need to use this software in a closed-source commercial product, you must:
1. Purchase a JUCE commercial license from Raw Material Software
2. Contact the author for a commercial license to this software

## Development Notes

### Cross-platform Considerations

- Line endings: Repository uses LF (Unix-style). Windows developers should configure Git with `core.autocrlf=true`
- Build directories: Use platform-specific prefixes (`build-windows/`, `build-macos/`)
- Audio files: Test audio files are stored in Git LFS (if configured) or directly in the repo

### Git Workflow

```bash
# After making changes
git add .
git commit -m "Description of changes"
git push origin main

# On Mac, pull latest changes
git pull origin main
```

## Contact

[Your Contact Information]

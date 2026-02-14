# Build Scripts Reference

## Quick Start

### Windows

```powershell
# Release build (optimized, recommended for distribution)
scripts\build\build_release.bat

# Debug build (with debug symbols, for development)
scripts\build\build_debug.bat

# Interactive setup (choose build type)
scripts\build\setup_windows.bat [Debug|Release]
```

### macOS

```bash
# Release build
chmod +x scripts/build/build_release.sh
scripts/build/build_release.sh

# Debug build
chmod +x scripts/build/build_debug.sh
scripts/build/build_debug.sh
```

### Linux

```bash
# Release build
chmod +x scripts/build/build_release.sh
scripts/build/build_release.sh

# Debug build
chmod +x scripts/build/build_debug.sh
scripts/build/build_debug.sh
```

## Script Details

### build_release.bat / build_release.sh

- **Purpose**: Build optimized Release version
- **Output**: `SuperPitchMonitor.exe` (in project root)
- **Log**: `build-windows/logs/build_release_*.log`

### build_debug.bat / build_debug.sh

- **Purpose**: Build Debug version with symbols
- **Output**: `SuperPitchMonitor.exe` (in project root)
- **Log**: `build-windows/logs/build_debug_*.log`

### setup_windows.bat

- **Purpose**: Interactive build with setup
- **Parameters**: `Debug` or `Release` (default: Release)
- **Features**: 
  - Checks prerequisites
  - Generates test audio
  - Downloads JUCE if needed
  - Builds the project

## Build Output

All builds place the final executable in the **project root directory**:

```
SuperPitchMonitor/
├── SuperPitchMonitor.exe          ← Main executable (Debug or Release)
├── build-windows/                  ← Build intermediates
│   └── logs/                       ← Build logs
├── ThirdParty/
│   └── juce-src/                   ← JUCE framework
└── ...
```

## Switching Between Debug and Release

Since both builds output to the same filename, you can simply run the appropriate script:

```powershell
# Build Debug version
scripts\build\build_debug.bat

# Run Debug version
.\SuperPitchMonitor.exe

# Build Release version (overwrites executable)
scripts\build\build_release.bat

# Run Release version
.\SuperPitchMonitor.exe
```

## Troubleshooting

### Build fails

Check the build log:
```powershell
# View latest build log
type build-windows\logs\build_release_*.log
```

### Clean rebuild

```powershell
# Remove build directory (keeps ThirdParty cache)
rm -rf build-windows

# Rebuild
scripts\build\build_release.bat
```

### Full clean (including JUCE cache)

```powershell
# Remove everything
rm -rf build-windows ThirdParty/juce-src

# Rebuild (will re-download JUCE)
scripts\build\build_release.bat
```

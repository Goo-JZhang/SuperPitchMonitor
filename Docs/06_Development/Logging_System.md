# Logging System

## Overview

SuperPitchMonitor uses two separate logging systems:

1. **Runtime Logs** - Application execution logs
2. **Build Logs** - CMake and compilation logs

## Runtime Logs

### Location
```
Saved/Logs/
├── app_YYYY-MM-DD_HH-MM-SS.log          # Main log file
├── app_YYYY-MM-DD_HH-MM-SS_rotated.log  # Rotated logs (>10MB)
├── crash_YYYYMMDD_HHMMSS.txt            # Crash dump files
└── crash_assertion.txt                  # Latest assertion failure
```

### Purpose
- Application runtime information
- Debug messages
- Crash reports
- Performance metrics

### Characteristics
- ✅ Persistent (not deleted when cleaning build directories)
- ✅ Platform-independent location
- ✅ Automatically rotated when file size exceeds 10MB
- 📝 Not committed to Git

## Build Logs

### Location
```
build-windows/logs/     # Windows build logs
build-macos/logs/       # macOS build logs
build-linux/logs/       # Linux build logs
```

### Purpose
- CMake configuration output
- Compiler warnings and errors
- Build script execution logs

### Characteristics
- ❌ Deleted when cleaning build directories
- ✅ Useful for debugging build failures
- ✅ Can be analyzed by AI agents for troubleshooting
- 📝 Not committed to Git

## Usage

### Viewing Runtime Logs

```bash
# Windows
type Saved\Logs\app_*.log

# macOS/Linux
cat Saved/Logs/app_*.log
```

### Viewing Build Logs

```bash
# After build failure
cat build-windows/logs/build_*.log
```

### Cleaning Logs

```bash
# Clean runtime logs
rm Saved/Logs/*.log

# Clean build logs (or just clean entire build directory)
rm -rf build-*/
```

## Configuration

### Changing Log Directory

Edit `Source/Utils/Logger.cpp`:

```cpp
// Default: ProjectRoot/Saved/Logs
logDirectory_ = projectRoot.getChildFile("Saved/Logs");

// Custom location:
logDirectory_ = juce::File("/path/to/custom/logs");
```

### Build Script Logging

Build scripts automatically create logs in `build-*/logs/`:

```bash
# The setup scripts create timestamped log files:
# build-windows/logs/build_YYYYMMDD_HHMMSS.log
# build-macos/logs/build_YYYYMMDD_HHMMSS.log
```

## Migration from Old Structure

Previous versions used `build_logs/` directory. This has been replaced with:
- Runtime logs → `Saved/Logs/`
- Build logs → `build-*/logs/`

The old `build_logs/` directory can be safely deleted.

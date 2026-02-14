# SuperPitchMonitor Test Framework

Unified cross-platform testing framework for SuperPitchMonitor.

## Overview

This directory contains the unified test framework that replaces the previous Windows-only named pipe implementation with a cross-platform TCP-based solution.

## Architecture

```
┌─────────────────┐         TCP Socket         ┌─────────────────┐
│  Python Client  │  ◄──────────────────────►  │  TestServer     │
│  test_client.py │    Port 9999 (default)     │  (inside SPM)   │
└─────────────────┘                            └─────────────────┘
                                                        │
                              ┌─────────────────────────┘
                              ▼
                       ┌──────────────┐
                       │  AudioEngine │
                       └──────────────┘
```

## Test Server Commands

All commands use JSON format over TCP socket.

### Command Protocol

**Request format:**
```json
{"cmd": "commandName", "param1": value1, ...}
```

**Response format:**
```json
{"status": "ok|error", "data": {...}}
```

### Available Commands

| Command | Parameters | Description |
|---------|------------|-------------|
| `getStatus` | - | Get current status |
| `setMultiRes` | `enabled: bool` | Enable/disable multi-resolution analysis |
| `loadFile` | `filename: string` | Load test audio file |
| `startPlayback` | - | Start audio playback |
| `stopPlayback` | - | Stop audio playback |
| `getPitches` | - | Get detected pitches |
| `waitForFrames` | `count: int` | Wait for N processing frames |
| `getSpectrumPeaks` | `freqMin, freqMax: float` | Get spectrum peaks in range |

## Usage

### Run All Tests

```bash
python test_client.py
```

### Run Specific Test

```bash
# Test single category
python test_client.py --test single_tone

# Test specific file
python test_client.py --test sine_440
```

### Custom Options

```bash
# Use custom port
python test_client.py --port 9998

# Wait more frames for analysis
python test_client.py --wait-frames 100

# Longer connection timeout
python test_client.py --timeout 60

# Keep SPM running after tests (for debugging)
python test_client.py --keep-alive

# Don't launch SPM (assume it's already running)
python test_client.py --no-launch
```

### Manual Testing

If you want to run SPM manually and then test:

```bash
# Terminal 1: Start SPM in test mode
./SuperPitchMonitor -TestMode -TestPort 9999

# Terminal 2: Run tests (without launching SPM)
python test_client.py --no-launch
```

## Test Mode vs Normal Mode

### Normal Mode (GUI)

```bash
# Windows
./SuperPitchMonitor.exe

# macOS
./SuperPitchMonitor.app/Contents/MacOS/SuperPitchMonitor
# or
open SuperPitchMonitor.app

# Linux
./SuperPitchMonitor
```

In normal mode, TestServer also runs (on port 9999 by default) so you can still run tests against the GUI instance.

### Test Mode (Headless)

```bash
# All platforms
./SuperPitchMonitor -TestMode -TestPort 9999
```

Test mode runs without GUI, making it suitable for:
- Automated CI/CD pipelines
- Batch testing
- Remote testing

## File Structure

```
scripts/test/
├── test_client.py          # Main test client (cross-platform)
├── test_ground_truth.json  # Ground truth data (in Resources/TestAudio/)
└── README.md              # This file
```

## Migration from Old Framework

The old Windows-only test framework used named pipes (`pywin32`). The new framework uses TCP sockets and works on all platforms.

### Changes

| Aspect | Old (Windows) | New (Cross-Platform) |
|--------|---------------|----------------------|
| Communication | Named pipes (\\.\pipe\...) | TCP socket (127.0.0.1:9999) |
| Dependencies | pywin32 | None (pure Python stdlib) |
| Launch param | `-AutoTest` | `-TestMode` |
| Python script | Multiple Windows-specific scripts | Single `test_client.py` |

## Troubleshooting

### Connection Refused

Make sure SPM is running in test mode:
```bash
# Check if port is listening
netstat -an | grep 9999    # macOS/Linux
netstat -an | findstr 9999 # Windows
```

### Test Audio Not Found

Generate test audio first:
```bash
python scripts/audio/generate_all_tests.py
```

### Permission Denied (macOS/Linux)

Make sure the executable has proper permissions:
```bash
chmod +x SuperPitchMonitor
# or for macOS app bundle
chmod +x SuperPitchMonitor.app/Contents/MacOS/SuperPitchMonitor
```

## Implementation Details

### C++ Side (TestServer)

- **Location:** `Source/Test/TestServer.cpp/h`
- **Threading:** Runs in separate thread, thread-safe
- **Protocol:** Length-prefixed JSON messages (4-byte big-endian length prefix)

### Python Side (test_client.py)

- **Connection:** Standard `socket` module
- **Message format:** Same length-prefixed JSON
- **Ground truth:** Loaded from `Resources/TestAudio/test_ground_truth.json`

## Adding New Tests

1. Add test audio file to `Resources/TestAudio/`
2. Add ground truth entry to `test_ground_truth.json`
3. Run `python test_client.py --test your_test_name`

## Future Enhancements

- [ ] WebSocket support for browser-based testing
- [ ] Parallel test execution
- [ ] Performance benchmarking
- [ ] Continuous integration examples

# SuperPitchMonitor Cross-Platform Development Guide

## Supported Platforms

| Platform | Status | Notes |
|----------|--------|-------|
| Windows | ✅ Full Support | Primary development platform |
| Android | ✅ Supported | Mobile UI optimizations included |
| iOS | ✅ Supported | Requires macOS + Xcode for building |
| macOS | ✅ Supported | Native desktop support |

## Platform Abstraction Layer

All platform-specific code is centralized in `Source/Utils/PlatformUtils_*`:

```
Source/Utils/
├── PlatformUtils.h          # Common interface (all platforms)
├── PlatformUtils.cpp        # Shared implementations
├── PlatformUtils_Windows.cpp    # Windows-specific
├── PlatformUtils_Android.cpp    # Android-specific  
└── PlatformUtils_iOS.mm         # iOS/macOS-specific (Objective-C++)
```

### Key Abstractions

- **Permissions**: Audio input, storage, camera
- **File Paths**: App data, cache, documents directories
- **Window Management**: Fullscreen, display scale
- **Audio**: Buffer sizes, low latency support
- **Performance**: Thread count, power modes

## Building for Different Platforms

### Windows (Visual Studio)

```bash
mkdir build-windows && cd build-windows
cmake .. -G "Visual Studio 17 2022" -A x64
cmake --build . --config Release
```

### Android

Requirements:
- Android Studio
- Android NDK
- CMake 3.15+

```bash
mkdir build-android && cd build-android
cmake .. -G "Android Gradle" \
    -DCMAKE_SYSTEM_NAME=Android \
    -DCMAKE_ANDROID_ARCH_ABI=arm64-v8a \
    -DCMAKE_ANDROID_NDK=/path/to/ndk \
    -DCMAKE_ANDROID_STL=c++_shared
cmake --build .
```

### macOS

```bash
mkdir build-macos && cd build-macos
cmake .. -G "Xcode"
cmake --build . --config Release
```

### iOS

```bash
mkdir build-ios && cd build-ios
cmake .. -G "Xcode" \
    -DCMAKE_SYSTEM_NAME=iOS \
    -DCMAKE_OSX_ARCHITECTURES=arm64
cmake --build . --config Release
```

## Platform-Specific Considerations

### Audio Input Sources

| Platform | Device | System Audio | File Playback |
|----------|--------|--------------|---------------|
| Windows | ✅ Yes | ✅ WASAPI Loopback | ✅ Yes |
| Android | ✅ Yes | ❌ Not Supported | ✅ Yes |
| iOS | ✅ Yes | ❌ Not Supported* | ✅ Yes |
| macOS | ✅ Yes | ❌ Not Implemented | ✅ Yes |

*System audio capture requires special entitlements on iOS

### UI Adaptations

The UI automatically adapts:
- **Desktop**: Resizable window, 800x1200 default
- **Mobile**: Fullscreen, touch-optimized controls

Settings for mobile (in `PlatformUtils_*::configureMainWindow`):
- iOS/Android: Fullscreen, no window decorations
- Desktop: Centered, resizable

### File System

Test audio files are loaded from:

| Platform | Search Locations |
|----------|-----------------|
| Windows | Working dir, Exe dir, Parent dir |
| Android | App bundle assets, External storage |
| iOS/macOS | App bundle, Documents directory |

### Performance

Thread count is automatically limited:
- Desktop: `cores - 1`
- Mobile: `min(2, cores - 1)` (to save battery)

## Known Limitations

### Windows
- System audio loopback requires WASAPI
- Some antivirus may block audio capture

### Android  
- Requires RECORD_AUDIO permission at runtime
- Background audio capture restricted in newer versions
- File access limited by scoped storage (Android 10+)

### iOS
- Requires microphone permission in Info.plist
- Background audio requires audio background mode
- App Store requires purpose string for microphone usage

### macOS
- Microphone permission required (TCC)
- Notarized app required for distribution

## Development Workflow

### Adding Platform-Specific Code

1. Add interface to `PlatformUtils.h`
2. Implement in appropriate `PlatformUtils_*.cpp/.mm`
3. Use `#if JUCE_XXX` for conditional compilation
4. Provide fallback in `PlatformUtils.cpp` if possible

### Example: Platform Check

```cpp
#if JUCE_ANDROID
    // Android-specific code
#elif JUCE_IOS
    // iOS-specific code  
#elif JUCE_WINDOWS
    // Windows-specific code
#elif JUCE_MAC
    // macOS-specific code
#endif
```

## Testing Checklist

Before releasing on each platform:

- [ ] Audio input permission handling
- [ ] Test audio file loading
- [ ] UI layout on different screen sizes
- [ ] FPS/performance monitoring
- [ ] Crash logging works
- [ ] Settings persistence
- [ ] Background/foreground transitions (mobile)

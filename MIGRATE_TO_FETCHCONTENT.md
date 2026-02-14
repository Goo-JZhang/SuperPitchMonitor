# Migrating to FetchContent

This guide helps you migrate from a local JUCE directory to CMake FetchContent.

## What Changed

**Before:** JUCE was included as a git submodule or local directory
**After:** JUCE is downloaded automatically by CMake during first build

## Migration Steps

### Step 1: Remove Local JUCE Directory

```bash
# Windows Command Prompt
rmdir /s /q JUCE

# Windows PowerShell
Remove-Item -Recurse -Force JUCE

# macOS/Linux
rm -rf JUCE
```

### Step 2: Update Git Ignore

The `.gitignore` has been updated to exclude the JUCE directory.

### Step 3: Clean Build Directories

Remove old build directories to ensure clean state:

```bash
# Windows
rmdir /s /q build-windows

# macOS/Linux
rm -rf build-macos build-linux
```

### Step 4: Rebuild

Run the setup script for your platform:

```bash
# Windows
scripts\build\setup_windows.bat

# macOS
chmod +x scripts/build/setup_macos.sh
scripts/build/setup_macos.sh

# Linux
chmod +x scripts/build/setup_linux.sh
scripts/build/setup_linux.sh
```

Or manually:

```bash
mkdir build-[platform] && cd build-[platform]
cmake ..
# JUCE will be downloaded automatically
cmake --build .
```

## First Build Notes

- **Internet connection required** for first build
- JUCE will be downloaded to `ThirdParty/juce-src/`
- Download size: ~200MB (shallow clone of specific tag)
- Subsequent builds work offline

## Verification

After successful build, you should see:

```
ThirdParty/
└── juce-src/              ← JUCE downloaded here
    ├── CMakeLists.txt
    ├── modules/
    └── ...
```

Note: JUCE is cached in `ThirdParty/` (not inside build directory), so cleaning build directories won't delete it.

## Rollback (If Needed)

If you need to rollback to local JUCE:

1. Restore your JUCE directory:
   ```bash
   git submodule update --init --recursive  # If using submodule
   # OR manually place JUCE directory
   ```

2. Edit `CMakeLists.txt`:
   ```cmake
   # Comment out FetchContent section
   # Uncomment:
   add_subdirectory(JUCE)
   ```

3. Rebuild

## Benefits of FetchContent

1. **No submodule management** - Simpler git workflow
2. **Automatic dependency handling** - CMake manages everything
3. **Version locking** - Specific JUCE version guaranteed
4. **Cleaner repository** - No large external dependencies in git
5. **Cross-platform consistency** - Works the same on all platforms

## Troubleshooting

### Issue: "JUCE directory not found"
**Solution:** You've already deleted JUCE. Proceed with build, CMake will download it.

### Issue: CMake can't download JUCE (corporate proxy, etc.)
**Solutions:**
1. Configure git proxy:
   ```bash
   git config --global http.proxy http://proxy.company.com:8080
   ```
2. Manually download JUCE and place in `ThirdParty/juce-src/`
3. Use local JUCE path (see Dependency_Management.md)

### Issue: Old build files conflicting
**Solution:** Clean build completely:
```bash
# Clean only build directory (ThirdParty cache preserved)
rm -rf build-*
mkdir build-[platform]
cd build-[platform]
cmake ..
```

### Issue: Old build_logs directory exists
**Solution:** The old `build_logs/` directory is no longer used. You can safely delete it:
```bash
rm -rf build_logs/
```

Runtime logs are now in `Saved/Logs/` and build logs are in `build-*/logs/`.

### Issue: Want to force re-download JUCE
**Solution:** Delete the cached JUCE:
```bash
rm -rf ThirdParty/juce-src
cmake --build build-[platform]  # Will re-download
```

## Related Changes

### Log Directory Restructure

As part of this migration, the log directory structure has also been updated:

| Log Type | Old Location | New Location |
|----------|-------------|--------------|
| Runtime Logs | `build_logs/` | `Saved/Logs/` |
| Build Logs | `build_logs/` | `build-*/logs/` |

**Benefits:**
- Runtime logs persist across build directory cleanups
- Build logs are co-located with their respective builds
- Better separation of concerns

See `Docs/06_Development/Logging_System.md` for details.

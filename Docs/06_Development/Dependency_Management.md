# Dependency Management

## JUCE Framework

This project uses **CMake FetchContent** to automatically download and manage JUCE.

### How It Works

When you run CMake for the first time, it will:
1. Download JUCE from the official GitHub repository
2. Extract it to `build-*/_deps/juce-src/`
3. Build JUCE along with your project

No manual setup required!

### Locked JUCE Version

The JUCE version is locked in `CMakeLists.txt`:

```cmake
FetchContent_Declare(
    JUCE
    GIT_REPOSITORY https://github.com/juce-framework/JUCE.git
    GIT_TAG 7.0.12  # ← Locked version
    ...
)
```

### Updating JUCE

To update to a newer JUCE version:

1. Edit `CMakeLists.txt`:
   ```cmake
   GIT_TAG 7.0.13  # Change to desired version
   ```

2. Clean and rebuild:
   ```bash
   # Remove old JUCE download
   rm -rf build-*/_deps/juce-src
   
   # Rebuild (will download new version)
   cmake --build build-[platform]
   ```

### Viewing JUCE Version

After building, you can find the downloaded JUCE at:
```
ThirdParty/juce-src/              # All platforms (cached)
```

### Offline Builds

Once JUCE is cached in `ThirdParty/`, you can build offline:

```bash
# First build (requires internet, downloads to ThirdParty/)
cmake --build build-windows

# Clean build directory (JUCE cache preserved)
rm -rf build-windows

# Rebuild (offline OK, uses cached JUCE)
mkdir build-windows && cd build-windows
cmake ..
cmake --build .
```

To force a fresh download:
```bash
rm -rf ThirdParty/juce-src
cmake --build build-[platform]  # Will re-download
```

### Troubleshooting

**Issue:** CMake fails to download JUCE
- **Solution:** Check internet connection and proxy settings

**Issue:** JUCE download is slow
- **Solution:** Use `GIT_SHALLOW TRUE` (already enabled) to download only the specific tag

**Issue:** Want to use local JUCE copy
- **Solution:** Uncomment and modify in CMakeLists.txt:
  ```cmake
  # set(FetchContent_SOURCE_DIR_JUCE "C:/path/to/your/JUCE")
  ```

### Alternative: Using Local JUCE

If you prefer to use a local JUCE installation instead of FetchContent:

1. Clone JUCE to ThirdParty directory:
   ```bash
   git clone https://github.com/juce-framework/JUCE.git ThirdParty/juce-local
   ```

2. Modify CMakeLists.txt:
   ```cmake
   # Comment out FetchContent section
   # Then add:
   add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/ThirdParty/juce-local)
   ```

3. Build normally

**Note:** This approach is not recommended as it makes the project less portable.

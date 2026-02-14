# Third Party Dependencies

This directory is used by CMake FetchContent to cache external dependencies.

## How It Works

When you build the project for the first time, CMake will automatically download dependencies (like JUCE) into this directory. This cache persists even when you clean the build directories (`build-windows/`, `build-macos/`, etc.).

## Directory Structure

```
ThirdParty/
├── .gitkeep              # Keeps directory in git
├── README.md             # This file
└── juce-src/             # JUCE framework (downloaded automatically)
    ├── CMakeLists.txt
    ├── modules/
    └── ...
```

## Cached Dependencies

| Dependency | Version | Source |
|------------|---------|--------|
| JUCE       | 7.0.12  | https://github.com/juce-framework/JUCE |

## Management

### Clean Cache

If you need to force re-download dependencies:

```bash
# Delete specific dependency
rm -rf ThirdParty/juce-src

# Or delete entire cache
rm -rf ThirdParty/*
touch ThirdParty/.gitkeep  # Restore marker file
```

### Add New Dependencies

To add a new dependency, edit `CMakeLists.txt`:

```cmake
FetchContent_Declare(
    NewLibrary
    GIT_REPOSITORY https://github.com/user/repo.git
    GIT_TAG v1.0.0
    GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(NewLibrary)
```

The new library will be automatically downloaded to `ThirdParty/repo-src/`.

## Git

This directory is in `.gitignore` - downloaded libraries are **not** committed to the repository. Only `.gitkeep` and `README.md` are tracked.

## Offline Builds

Once dependencies are cached here, you can build offline. The cache is shared across all build configurations (Debug/Release, different platforms).

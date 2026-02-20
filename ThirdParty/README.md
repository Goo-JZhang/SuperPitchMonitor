# Third Party Dependencies

This directory manages external dependencies that are automatically downloaded during the build process.

## ONNX Runtime

ONNX Runtime is automatically downloaded and configured by CMake.

### Automatic Download

When you run CMake, it will automatically:
1. Download the platform-specific ONNX Runtime pre-built package
2. Extract it to `ThirdParty/onnxruntime-bin/`
3. Download additional headers for GPU support (if needed)
4. Configure include and library paths

### Platform-Specific Packages

| Platform | Package | GPU Support |
|----------|---------|-------------|
| macOS ARM64 | `onnxruntime-osx-arm64-*.tgz` | CoreML (optional) |
| macOS x86_64 | `onnxruntime-osx-x86_64-*.tgz` | CPU only |
| Windows x64 | `onnxruntime-win-x64-*.zip` | DirectML (optional) |
| Linux x64 | `onnxruntime-linux-x64-*.tgz` | CPU only |
| Android | `onnxruntime-android-*.aar` | NNAPI |

### Forcing GPU Version

To use the GPU-accelerated version, set the CMake option:

```bash
cmake .. -DUSE_GPU_ONNXRUNTIME=ON
```

### Manual Override

If you want to use a locally installed ONNX Runtime instead of the automatic download:

```bash
cmake .. -DONNXRUNTIME_ROOT=/path/to/your/onnxruntime
```

### Troubleshooting

**Download fails**: Check your internet connection and try again. CMake will retry on the next run.

**GPU not available**: The standard pre-built packages may not include GPU support. Check the logs for available execution providers. To get GPU support:
- Build ONNX Runtime from source with the appropriate EP flag
- Or download the specific GPU-enabled package from GitHub releases

## Build Output

Downloaded files are cached in:
- Source: `ThirdParty/onnxruntime-bin/<version>/`
- No need to commit these to git (already in .gitignore)

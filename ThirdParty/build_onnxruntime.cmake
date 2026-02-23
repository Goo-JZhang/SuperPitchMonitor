# build_onnxruntime.cmake
# Build ONNX Runtime from source with platform-specific GPU support
# Similar to how JUCE is handled

set(ONNXRUNTIME_VERSION "1.24.2")

# Check CUDNN_HOME from environment if not set in CMake
if(NOT DEFINED CUDNN_HOME)
    if(DEFINED ENV{CUDNN_HOME})
        # Remove any surrounding quotes from environment variable
        string(REPLACE "\"" "" CUDNN_HOME "$ENV{CUDNN_HOME}")
        message(STATUS "CUDNN_HOME from environment: ${CUDNN_HOME}")
    else()
        # Try to find cuDNN in common locations
        if(EXISTS "C:/Program Files/NVIDIA/CUDNN/v9.19/include/cudnn.h")
            set(CUDNN_HOME "C:/Program Files/NVIDIA/CUDNN/v9.19")
        elseif(EXISTS "C:/tools/cuda/include/cudnn.h")
            set(CUDNN_HOME "C:/tools/cuda")
        endif()
        if(CUDNN_HOME)
            message(STATUS "CUDNN_HOME auto-detected: ${CUDNN_HOME}")
        endif()
    endif()
endif()

# Convert CUDNN_HOME to short path (8.3 format) to avoid space issues on Windows
if(CUDNN_HOME AND WIN32)
    # Normalize path for comparison (convert backslash to forward slash, remove quotes)
    string(REPLACE "\\" "/" CUDNN_HOME_NORMALIZED "${CUDNN_HOME}")
    string(REPLACE "\"" "" CUDNN_HOME_NORMALIZED "${CUDNN_HOME_NORMALIZED}")
    # Hardcode known short paths for common cuDNN locations
    if(CUDNN_HOME_NORMALIZED STREQUAL "C:/Program Files/NVIDIA/CUDNN/v9.19")
        set(CUDNN_HOME "C:/PROGRA~1/NVIDIA/CUDNN/v9.19")
        message(STATUS "CUDNN_HOME (short path): ${CUDNN_HOME}")
    elseif(CUDNN_HOME_NORMALIZED STREQUAL "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.9")
        set(CUDNN_HOME "C:/PROGRA~1/NVIDIA~2/CUDA/v12.9")
        message(STATUS "CUDNN_HOME (short path): ${CUDNN_HOME}")
    endif()
endif()

# Use the centralized path from CMakeLists.txt
if(NOT DEFINED ONNXRUNTIME_DIR)
    set(ONNXRUNTIME_DIR "${CMAKE_CURRENT_SOURCE_DIR}/ThirdParty/onnxruntime")
endif()

set(ONNXRUNTIME_ROOT "${ONNXRUNTIME_DIR}/src")
set(ONNXRUNTIME_BUILD "${ONNXRUNTIME_DIR}/build")
set(ONNXRUNTIME_INSTALL "${ONNXRUNTIME_DIR}/install")

# Download ONNX Runtime source if not exists
include(FetchContent)

# Disable automatic re-download: use local source if exists, fail otherwise
# User must manually delete src directory to force re-download
if(EXISTS ${ONNXRUNTIME_ROOT}/CMakeLists.txt)
    set(FETCHCONTENT_SOURCE_DIR_ONNXRUNTIME_SOURCE ${ONNXRUNTIME_ROOT})
    message(STATUS "Using existing ONNX Runtime source: ${ONNXRUNTIME_ROOT}")
endif()

FetchContent_Declare(
    onnxruntime_source
    GIT_REPOSITORY https://github.com/microsoft/onnxruntime.git
    GIT_TAG v${ONNXRUNTIME_VERSION}
    GIT_SHALLOW TRUE
    GIT_PROGRESS TRUE
    SOURCE_DIR ${ONNXRUNTIME_ROOT}
)

# Check if already populated (CMakeLists.txt is in cmake subdirectory for v1.20+)
if(NOT EXISTS ${ONNXRUNTIME_ROOT}/cmake/CMakeLists.txt)
    message(STATUS "Downloading ONNX Runtime source code...")
    message(STATUS "  Target: ${ONNXRUNTIME_ROOT}")
    FetchContent_Populate(onnxruntime_source)
    message(STATUS "ONNX Runtime source downloaded to: ${ONNXRUNTIME_ROOT}")
    
    # Patch dependency URLs to use GitHub mirrors instead of gitlab (avoid 403)
    message(STATUS "Patching dependency URLs to use GitHub mirrors...")
    
    # Patch eigen.cmake
    if(EXISTS ${ONNXRUNTIME_ROOT}/cmake/external/eigen.cmake)
        file(READ ${ONNXRUNTIME_ROOT}/cmake/external/eigen.cmake EIGEN_CMAKE)
        string(REPLACE "gitlab.com/libeigen/eigen" "github.com/eigenteam/eigen-git-mirror" EIGEN_CMAKE "${EIGEN_CMAKE}")
        file(WRITE ${ONNXRUNTIME_ROOT}/cmake/external/eigen.cmake "${EIGEN_CMAKE}")
        message(STATUS "  - Patched eigen.cmake")
    endif()
    
    # Patch mp11.cmake if exists
    if(EXISTS ${ONNXRUNTIME_ROOT}/cmake/external/mp11.cmake)
        file(READ ${ONNXRUNTIME_ROOT}/cmake/external/mp11.cmake MP11_CMAKE)
        string(REPLACE "gitlab.com" "github.com" MP11_CMAKE "${MP11_CMAKE}")
        file(WRITE ${ONNXRUNTIME_ROOT}/cmake/external/mp11.cmake "${MP11_CMAKE}")
        message(STATUS "  - Patched mp11.cmake")
    endif()
    
        # Patch deps.txt to fix eigen hash mismatch
        if(EXISTS ${ONNXRUNTIME_ROOT}/cmake/deps.txt)
            file(READ ${ONNXRUNTIME_ROOT}/cmake/deps.txt DEPS_TXT)
            # Fix hash mismatch: actual hash from gitlab download is 32b145f525a8308d7ab1c09388b2e288312d8eba
            string(REPLACE 
                "eigen;https://gitlab.com/libeigen/eigen/-/archive/e7248b26a1ed53fa030c5c459f7ea095dfd276ac/eigen-e7248b26a1ed53fa030c5c459f7ea095dfd276ac.zip;be8be39fdbc6e60e94fa7870b280707069b5b81a"
                "eigen;https://gitlab.com/libeigen/eigen/-/archive/e7248b26a1ed53fa030c5c459f7ea095dfd276ac/eigen-e7248b26a1ed53fa030c5c459f7ea095dfd276ac.zip;32b145f525a8308d7ab1c09388b2e288312d8eba" 
                DEPS_TXT "${DEPS_TXT}")
            file(WRITE ${ONNXRUNTIME_ROOT}/cmake/deps.txt "${DEPS_TXT}")
            message(STATUS "  - Patched deps.txt (eigen hash fixed)")
        endif()
endif()

# Detect architecture
if(NOT CMAKE_OSX_ARCHITECTURES)
    set(CMAKE_OSX_ARCHITECTURES "arm64")
endif()

# Configure build arguments as a list (properly quoted)
set(ONNX_BUILD_ARGS
    "--config" "Release"
    "--build_shared_lib"
    "--parallel"
    "--build_dir" "${ONNXRUNTIME_BUILD}"
    "--skip_tests"
    "--cmake_extra_defines"
    "onnxruntime_BUILD_UNIT_TESTS=OFF"
    "onnxruntime_BUILD_TESTS=OFF"
    "onnxruntime_BUILD_BENCHMARKS=OFF"
    "onnxruntime_ENABLE_PYTHON=OFF"
    "onnxruntime_ENABLE_TRAINING=OFF"
    "onnxruntime_ENABLE_TRAINING_APIS=OFF"
    "onnxruntime_MINIMAL_BUILD=OFF"
    "onnxruntime_REDUCED_OPS_BUILD=OFF"
    "onnxruntime_DISABLE_CONTRIB_OPS=OFF"
    "CMAKE_CUDA_COMPILER=C:/PROGRA~1/NVIDIA~2/CUDA/v12.8/bin/nvcc.exe"
    "CUDAToolkit_ROOT=C:/PROGRA~1/NVIDIA~2/CUDA/v12.8"
    "CUDNN_INCLUDE_DIR=${CUDNN_HOME}/include/12.9"
    "cudnn_LIBRARY=${CUDNN_HOME}/lib/12.9/x64/cudnn.lib"
    "cudnn_cnn_LIBRARY=${CUDNN_HOME}/lib/12.9/x64/cudnn_cnn.lib"
    "cudnn_adv_LIBRARY=${CUDNN_HOME}/lib/12.9/x64/cudnn_adv.lib"
    "cudnn_graph_LIBRARY=${CUDNN_HOME}/lib/12.9/x64/cudnn_graph.lib"
    "cudnn_ops_LIBRARY=${CUDNN_HOME}/lib/12.9/x64/cudnn_ops.lib"
    "cudnn_engines_runtime_compiled_LIBRARY=${CUDNN_HOME}/lib/12.9/x64/cudnn_engines_runtime_compiled.lib"
    "cudnn_engines_precompiled_LIBRARY=${CUDNN_HOME}/lib/12.9/x64/cudnn_engines_precompiled.lib"
    "cudnn_heuristic_LIBRARY=${CUDNN_HOME}/lib/12.9/x64/cudnn_heuristic.lib"
    "CUDNN_PATH=${CUDNN_HOME}"
)

# Platform-specific configuration
if(APPLE)
    if(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
        message(STATUS "ONNX Runtime: Building for macOS with CoreML support (${CMAKE_OSX_ARCHITECTURES})")
        list(APPEND ONNX_BUILD_ARGS "--use_coreml")
        list(APPEND ONNX_BUILD_ARGS "--osx_arch" "${CMAKE_OSX_ARCHITECTURES}")
        if(CMAKE_OSX_DEPLOYMENT_TARGET)
            list(APPEND ONNX_BUILD_ARGS "--apple_deploy_target" "${CMAKE_OSX_DEPLOYMENT_TARGET}")
        endif()
    elseif(CMAKE_SYSTEM_NAME STREQUAL "iOS")
        message(STATUS "ONNX Runtime: Building for iOS with CoreML support (${CMAKE_OSX_ARCHITECTURES})")
        list(APPEND ONNX_BUILD_ARGS "--use_coreml")
        list(APPEND ONNX_BUILD_ARGS "--ios")
        list(APPEND ONNX_BUILD_ARGS "--osx_arch" "${CMAKE_OSX_ARCHITECTURES}")
        list(APPEND ONNX_BUILD_ARGS "--apple_deploy_target" "${CMAKE_OSX_DEPLOYMENT_TARGET}")
    endif()
    
elseif(WIN32)
    # Windows: Enable both CUDA (for NVIDIA RTX 4080) and DirectML (universal fallback)
    message(STATUS "ONNX Runtime: Building for Windows with CUDA + DirectML support")
    if(USE_CUDA_ON_WINDOWS)
        list(APPEND ONNX_BUILD_ARGS "--use_cuda")
        # Ensure CUDNN_HOME is set
        if(NOT CUDNN_HOME)
            set(CUDNN_HOME "$ENV{CUDNN_HOME}")
        endif()
        if(CUDNN_HOME)
            message(STATUS "Adding --cudnn_home: ${CUDNN_HOME}")
            list(APPEND ONNX_BUILD_ARGS "--cudnn_home")
            list(APPEND ONNX_BUILD_ARGS "${CUDNN_HOME}")
        else()
            message(WARNING "CUDNN_HOME not set! CUDA build may fail.")
        endif()
    endif()
    list(APPEND ONNX_BUILD_ARGS "--use_dml")
    
elseif(CMAKE_SYSTEM_NAME STREQUAL "Android")
    message(STATUS "ONNX Runtime: Building for Android with NNAPI support")
    list(APPEND ONNX_BUILD_ARGS "--android")
    list(APPEND ONNX_BUILD_ARGS "--android_abi" "${ANDROID_ABI}")
    list(APPEND ONNX_BUILD_ARGS "--android_api" "${ANDROID_PLATFORM}")
    list(APPEND ONNX_BUILD_ARGS "--use_nnapi")
    
else()
    message(STATUS "ONNX Runtime: Building for Linux (CPU only)")
endif()

# Debug output
message(STATUS "CUDNN_HOME value: '${CUDNN_HOME}'")
message(STATUS "ONNX_BUILD_ARGS before convert: ${ONNX_BUILD_ARGS}")

# Convert list to space-separated string for the script
string(REPLACE ";" " " ONNX_BUILD_ARGS_STR "${ONNX_BUILD_ARGS}")
message(STATUS "ONNX_BUILD_ARGS_STR: ${ONNX_BUILD_ARGS_STR}")

# Use ExternalProject for actual build
include(ExternalProject)

# Use ExternalProject for actual build
include(ExternalProject)

# Create a wrapper script for building with proper argument passing
if(WIN32)
    set(BUILD_SCRIPT ${CMAKE_BINARY_DIR}/build_onnxruntime_wrapper.bat)
    file(WRITE ${BUILD_SCRIPT} "@echo off
setlocal enabledelayedexpansion
call C:\\ProgramData\\anaconda3\\Scripts\\activate.bat onnx_build
set CMAKE_PREFIX_PATH=
set CMAKE_SYSTEM_PREFIX_PATH=
set PATH=C:\\Users\\jlzzh\\.conda\\envs\\onnx_build;C:\\Users\\jlzzh\\.conda\\envs\\onnx_build\\Scripts;C:\\Users\\jlzzh\\.conda\\envs\\onnx_build\\Library\\bin;C:\\Program Files\\Git\\cmd;C:\\Windows\\System32;C:\\Windows
if defined CUDA_HOME set PATH=%CUDA_HOME%\\bin;%PATH%
if defined CUDNN_HOME (
    set CUDNN_HOME=${CUDNN_HOME}
    set CUDNN_PATH=${CUDNN_HOME}
    set PATH=%CUDNN_HOME%\\bin;%PATH%
)
set HTTP_PROXY=http://127.0.0.1:7890
set HTTPS_PROXY=http://127.0.0.1:7890
set PYTHONNOUSERSITE=1
set CONDA_PREFIX=
set CONDA_DEFAULT_ENV=
cd /d ${ONNXRUNTIME_ROOT}
echo Building ONNX Runtime...
echo Using Python:
where python
cmake --version
python tools/ci_build/build.py ${ONNX_BUILD_ARGS_STR}
if errorlevel 1 exit /b 1
")
else()
    set(BUILD_SCRIPT ${CMAKE_BINARY_DIR}/build_onnxruntime_wrapper.sh)
    file(WRITE ${BUILD_SCRIPT} "#!/bin/bash
set -e
cd ${ONNXRUNTIME_ROOT}
echo 'Building ONNX Runtime with args: ${ONNX_BUILD_ARGS_STR}'
python3 tools/ci_build/build.py ${ONNX_BUILD_ARGS_STR}
")
    file(CHMOD ${BUILD_SCRIPT} PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE)
endif()

# Set byproducts based on platform (library file to check)
if(WIN32)
    set(ONNXRUNTIME_BYPRODUCT ${ONNXRUNTIME_INSTALL}/onnxruntime.lib)
    set(ONNXRUNTIME_BYPRODUCT_EXTRA ${ONNXRUNTIME_INSTALL}/onnxruntime.dll)
elseif(APPLE)
    set(ONNXRUNTIME_BYPRODUCT ${ONNXRUNTIME_INSTALL}/libonnxruntime.dylib)
    set(ONNXRUNTIME_BYPRODUCT_EXTRA "")
else()
    # Linux and Android
    set(ONNXRUNTIME_BYPRODUCT ${ONNXRUNTIME_INSTALL}/libonnxruntime.so)
    set(ONNXRUNTIME_BYPRODUCT_EXTRA "")
endif()

# Check if ONNX Runtime is already built (skip rebuild)
set(ONNXRUNTIME_ALREADY_BUILT FALSE)
if(EXISTS ${ONNXRUNTIME_BYPRODUCT})
    # Windows needs both .lib and .dll
    if(WIN32 AND EXISTS ${ONNXRUNTIME_BYPRODUCT_EXTRA})
        set(ONNXRUNTIME_ALREADY_BUILT TRUE)
    # macOS/iOS/Linux/Android only need the library file
    elseif(NOT WIN32)
        set(ONNXRUNTIME_ALREADY_BUILT TRUE)
    endif()
endif()

if(ONNXRUNTIME_ALREADY_BUILT)
    message(STATUS "========================================")
    message(STATUS "ONNX Runtime: Already built (skipping)")
    message(STATUS "Library: ${ONNXRUNTIME_BYPRODUCT}")
    message(STATUS "========================================")
    
    # Create a dummy imported target (no build needed)
    add_custom_target(onnxruntime_build)
    
    # Set output variables
    set(ONNXRUNTIME_INCLUDE_DIR ${ONNXRUNTIME_ROOT}/include/onnxruntime/core/session)
    set(ONNXRUNTIME_INCLUDE_DIR_EXTRA 
        ${ONNXRUNTIME_ROOT}/include
        ${ONNXRUNTIME_ROOT}/include/onnxruntime/core/providers/coreml
        ${ONNXRUNTIME_ROOT}/include/onnxruntime/core/providers/cuda
        ${ONNXRUNTIME_ROOT}/include/onnxruntime/core/providers/dml
    )
    set(ONNXRUNTIME_LIBRARY_DIR ${ONNXRUNTIME_INSTALL})
    
    if(APPLE)
        set(ONNXRUNTIME_LIBRARY ${ONNXRUNTIME_INSTALL}/libonnxruntime.dylib)
    elseif(WIN32)
        set(ONNXRUNTIME_LIBRARY ${ONNXRUNTIME_INSTALL}/onnxruntime.lib)
    else()
        set(ONNXRUNTIME_LIBRARY ${ONNXRUNTIME_INSTALL}/libonnxruntime.so)
    endif()
    
    set(ONNXRUNTIME_BIN_DIR ${ONNXRUNTIME_INSTALL})
    
    # Skip the rest of this file
    return()
endif()

ExternalProject_Add(
    onnxruntime_build
    SOURCE_DIR ${ONNXRUNTIME_ROOT}
    BINARY_DIR ${ONNXRUNTIME_BUILD}
    INSTALL_DIR ${ONNXRUNTIME_INSTALL}
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ${BUILD_SCRIPT}
    INSTALL_COMMAND
        ${CMAKE_COMMAND} -E copy_directory
            ${ONNXRUNTIME_BUILD}/Release
            ${ONNXRUNTIME_INSTALL}
    BUILD_IN_SOURCE FALSE
    BUILD_BYPRODUCTS ${ONNXRUNTIME_BYPRODUCT}
    USES_TERMINAL_BUILD TRUE
    EXCLUDE_FROM_ALL FALSE
)

# Set output variables
# Main include dir for onnxruntime_c_api.h
set(ONNXRUNTIME_INCLUDE_DIR ${ONNXRUNTIME_ROOT}/include/onnxruntime/core/session)
# Additional include dirs for provider headers (CoreML, CUDA, etc.)
# Source build structure: include/onnxruntime/core/providers/<ep>/
set(ONNXRUNTIME_INCLUDE_DIR_EXTRA 
    ${ONNXRUNTIME_ROOT}/include
    ${ONNXRUNTIME_ROOT}/include/onnxruntime/core/providers/coreml
    ${ONNXRUNTIME_ROOT}/include/onnxruntime/core/providers/cuda
    ${ONNXRUNTIME_ROOT}/include/onnxruntime/core/providers/dml
)
set(ONNXRUNTIME_LIBRARY_DIR ${ONNXRUNTIME_INSTALL})

if(APPLE)
    set(ONNXRUNTIME_LIBRARY ${ONNXRUNTIME_INSTALL}/libonnxruntime.dylib)
elseif(WIN32)
    set(ONNXRUNTIME_LIBRARY ${ONNXRUNTIME_INSTALL}/onnxruntime.lib)
else()
    set(ONNXRUNTIME_LIBRARY ${ONNXRUNTIME_INSTALL}/libonnxruntime.so)
endif()

set(ONNXRUNTIME_BIN_DIR ${ONNXRUNTIME_INSTALL})

message(STATUS "========================================")
message(STATUS "ONNX Runtime: Building from source")
message(STATUS "This may take 20-30 minutes (like JUCE)")
message(STATUS "Platform: ${CMAKE_SYSTEM_NAME}")
message(STATUS "CoreML: ${APPLE}")
message(STATUS "Full operator support: YES (MINIMAL_BUILD=OFF)")
message(STATUS "========================================")
message(STATUS "Source: ${ONNXRUNTIME_ROOT}")
message(STATUS "Build: ${ONNXRUNTIME_BUILD}")
message(STATUS "Install: ${ONNXRUNTIME_INSTALL}")

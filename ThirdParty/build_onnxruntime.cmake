# build_onnxruntime.cmake
# Build ONNX Runtime from source with platform-specific GPU support
# Similar to how JUCE is handled

set(ONNXRUNTIME_VERSION "1.16.3")

# Use the centralized path from CMakeLists.txt
if(NOT DEFINED ONNXRUNTIME_DIR)
    set(ONNXRUNTIME_DIR "${CMAKE_CURRENT_SOURCE_DIR}/ThirdParty/onnxruntime")
endif()

set(ONNXRUNTIME_ROOT "${ONNXRUNTIME_DIR}/src")
set(ONNXRUNTIME_BUILD "${ONNXRUNTIME_DIR}/build")
set(ONNXRUNTIME_INSTALL "${ONNXRUNTIME_DIR}/install")

# Download ONNX Runtime source if not exists
include(FetchContent)
FetchContent_Declare(
    onnxruntime_source
    GIT_REPOSITORY https://github.com/microsoft/onnxruntime.git
    GIT_TAG v${ONNXRUNTIME_VERSION}
    GIT_SHALLOW TRUE
    GIT_PROGRESS TRUE
    SOURCE_DIR ${ONNXRUNTIME_ROOT}
)

# Check if already populated
if(NOT EXISTS ${ONNXRUNTIME_ROOT}/CMakeLists.txt)
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
    
    # Patch deps.txt to replace gitlab eigen URL
    if(EXISTS ${ONNXRUNTIME_ROOT}/cmake/deps.txt)
        file(READ ${ONNXRUNTIME_ROOT}/cmake/deps.txt DEPS_TXT)
        # Replace gitlab eigen with working URL and updated hash
        string(REPLACE 
            "eigen;https://gitlab.com/libeigen/eigen/-/archive/e7248b26a1ed53fa030c5c459f7ea095dfd276ac/eigen-e7248b26a1ed53fa030c5c459f7ea095dfd276ac.zip;be8be39fdbc6e60e94fa7870b280707069b5b81a"
            "eigen;https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz;d222db69a9e87d9006608e029d1039039f360b52" 
            DEPS_TXT "${DEPS_TXT}")
        file(WRITE ${ONNXRUNTIME_ROOT}/cmake/deps.txt "${DEPS_TXT}")
        message(STATUS "  - Patched deps.txt (eigen)")
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
    if(USE_CUDA_ON_WINDOWS)
        message(STATUS "ONNX Runtime: Building for Windows with CUDA support")
        list(APPEND ONNX_BUILD_ARGS "--use_cuda")
        list(APPEND ONNX_BUILD_ARGS "--cuda_version" "11.8")
    else()
        message(STATUS "ONNX Runtime: Building for Windows with DirectML support")
        list(APPEND ONNX_BUILD_ARGS "--use_dml")
    endif()
    
elseif(CMAKE_SYSTEM_NAME STREQUAL "Android")
    message(STATUS "ONNX Runtime: Building for Android with NNAPI support")
    list(APPEND ONNX_BUILD_ARGS "--android")
    list(APPEND ONNX_BUILD_ARGS "--android_abi" "${ANDROID_ABI}")
    list(APPEND ONNX_BUILD_ARGS "--android_api" "${ANDROID_PLATFORM}")
    list(APPEND ONNX_BUILD_ARGS "--use_nnapi")
    
else()
    message(STATUS "ONNX Runtime: Building for Linux (CPU only)")
endif()

# Convert list to space-separated string for the script
string(REPLACE ";" " " ONNX_BUILD_ARGS_STR "${ONNX_BUILD_ARGS}")

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

# Set byproducts based on platform
if(WIN32)
    set(ONNXRUNTIME_BYPRODUCT ${ONNXRUNTIME_BUILD}/Release/onnxruntime.lib)
else()
    set(ONNXRUNTIME_BYPRODUCT ${ONNXRUNTIME_BUILD}/Release/libonnxruntime.dylib)
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
set(ONNXRUNTIME_LIBRARY_DIR ${ONNXRUNTIME_BUILD}/Release)

if(APPLE)
    set(ONNXRUNTIME_LIBRARY ${ONNXRUNTIME_BUILD}/Release/libonnxruntime.dylib)
elseif(WIN32)
    set(ONNXRUNTIME_LIBRARY ${ONNXRUNTIME_BUILD}/Release/onnxruntime.lib)
else()
    set(ONNXRUNTIME_LIBRARY ${ONNXRUNTIME_BUILD}/Release/libonnxruntime.so)
endif()

set(ONNXRUNTIME_BIN_DIR ${ONNXRUNTIME_BUILD}/Release)

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

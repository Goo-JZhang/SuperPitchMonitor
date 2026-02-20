# fetch_onnxruntime.cmake
# Automatically downloads ONNX Runtime with GPU support for current platform

set(ONNXRUNTIME_VERSION "1.16.3")

# Use centralized path
if(NOT DEFINED ONNXRUNTIME_DIR)
    set(ONNXRUNTIME_DIR "${CMAKE_CURRENT_SOURCE_DIR}/ThirdParty/onnxruntime")
endif()
set(ONNXRUNTIME_ROOT "${ONNXRUNTIME_DIR}/prebuilt")

# Check if onnxruntime is already present in ThirdParty/onnxruntime directory
if(EXISTS "${ONNXRUNTIME_DIR}/include/onnxruntime_c_api.h")
    message(STATUS "Using existing ONNX Runtime from ${ONNXRUNTIME_DIR}")
    set(ONNXRUNTIME_INCLUDE_DIR ${ONNXRUNTIME_DIR}/include)
    
    if(APPLE)
        set(ONNXRUNTIME_LIBRARY ${ONNXRUNTIME_DIR}/lib/libonnxruntime.dylib)
    elseif(WIN32)
        set(ONNXRUNTIME_LIBRARY ${ONNXRUNTIME_DIR}/lib/onnxruntime.lib)
    else()
        set(ONNXRUNTIME_LIBRARY ${ONNXRUNTIME_DIR}/lib/libonnxruntime.so)
    endif()
    
    set(ONNXRUNTIME_BIN_DIR ${ONNXRUNTIME_DIR}/bin)
    
    message(STATUS "ONNX Runtime include: ${ONNXRUNTIME_INCLUDE_DIR}")
    message(STATUS "ONNX Runtime library: ${ONNXRUNTIME_LIBRARY}")
    
    if(NOT EXISTS ${ONNXRUNTIME_LIBRARY})
        message(FATAL_ERROR "ONNX Runtime library not found at ${ONNXRUNTIME_LIBRARY}")
    endif()
    
    # For prebuilt, the provider headers are in the same directory
    set(ONNXRUNTIME_INCLUDE_DIR_EXTRA ${ONNXRUNTIME_INCLUDE_DIR})
    
    # CoreML header download if needed
    if(APPLE AND NOT EXISTS ${ONNXRUNTIME_INCLUDE_DIR}/coreml_provider_factory.h)
        message(STATUS "Downloading CoreML provider factory header...")
        file(DOWNLOAD 
            "https://raw.githubusercontent.com/microsoft/onnxruntime/v${ONNXRUNTIME_VERSION}/include/onnxruntime/core/providers/coreml/coreml_provider_factory.h"
            ${ONNXRUNTIME_INCLUDE_DIR}/coreml_provider_factory.h
            SHOW_PROGRESS
            STATUS DOWNLOAD_STATUS
        )
    endif()
    
    return()  # Skip download logic
endif()

# Determine platform-specific download URL
if(APPLE)
    if(CMAKE_SYSTEM_PROCESSOR STREQUAL "arm64")
        set(ONNXRUNTIME_URL "https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}/onnxruntime-osx-arm64-${ONNXRUNTIME_VERSION}.tgz")
        set(ONNXRUNTIME_GPU_URL "")  # CoreML requires custom build
    else()
        set(ONNXRUNTIME_URL "https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}/onnxruntime-osx-x86_64-${ONNXRUNTIME_VERSION}.tgz")
        set(ONNXRUNTIME_GPU_URL "")
    endif()
    set(ONNXRUNTIME_LIB "libonnxruntime.dylib")
    
elseif(WIN32)
    if(CMAKE_SIZEOF_VOID_P EQUAL 8)
        # 64-bit Windows - multiple GPU options
        set(ONNXRUNTIME_URL "https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}/onnxruntime-win-x64-${ONNXRUNTIME_VERSION}.zip")
        # CUDA first (best for NVIDIA RTX cards like 4080 Super)
        set(ONNXRUNTIME_CUDA_URL "https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}/onnxruntime-win-x64-gpu-${ONNXRUNTIME_VERSION}.zip")
        # DirectML as fallback (works on all GPUs)
        set(ONNXRUNTIME_GPU_URL "https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}/onnxruntime-win-x64-directml-${ONNXRUNTIME_VERSION}.zip")
    else()
        set(ONNXRUNTIME_URL "https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}/onnxruntime-win-x86-${ONNXRUNTIME_VERSION}.zip")
        set(ONNXRUNTIME_CUDA_URL "")
        set(ONNXRUNTIME_GPU_URL "")
    endif()
    set(ONNXRUNTIME_LIB "onnxruntime.lib")
    
elseif(CMAKE_SYSTEM_NAME STREQUAL "Android")
    # Android with NNAPI support
    set(ONNXRUNTIME_URL "https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}/onnxruntime-android-${ONNXRUNTIME_VERSION}.aar")
    set(ONNXRUNTIME_GPU_URL "")
    set(ONNXRUNTIME_LIB "libonnxruntime.so")
    
elseif(UNIX)
    if(CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64")
        set(ONNXRUNTIME_URL "https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}/onnxruntime-linux-aarch64-${ONNXRUNTIME_VERSION}.tgz")
    else()
        set(ONNXRUNTIME_URL "https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}/onnxruntime-linux-x64-${ONNXRUNTIME_VERSION}.tgz")
    endif()
    set(ONNXRUNTIME_GPU_URL "")  # CUDA requires specific setup
    set(ONNXRUNTIME_LIB "libonnxruntime.so")
endif()

# Use GPU version if available
if(USE_GPU_ONNXRUNTIME)
    if(WIN32 AND ONNXRUNTIME_CUDA_URL AND USE_CUDA_ON_WINDOWS)
        # Windows: Prefer CUDA for NVIDIA cards (RTX 4080S, etc.)
        set(ONNXRUNTIME_URL ${ONNXRUNTIME_CUDA_URL})
        message(STATUS "Using ONNX Runtime CUDA version (for NVIDIA RTX 4080S)")
    elseif(WIN32 AND ONNXRUNTIME_GPU_URL)
        # Windows: DirectML fallback (works on all GPUs)
        set(ONNXRUNTIME_URL ${ONNXRUNTIME_GPU_URL})
        message(STATUS "Using ONNX Runtime DirectML version (universal GPU)")
    elseif(ONNXRUNTIME_GPU_URL)
        set(ONNXRUNTIME_URL ${ONNXRUNTIME_GPU_URL})
        message(STATUS "Using ONNX Runtime GPU version")
    else()
        message(STATUS "Using ONNX Runtime standard version (CPU)")
    endif()
else()
    message(STATUS "Using ONNX Runtime standard version (CPU)")
endif()

# Download and extract
if(NOT EXISTS ${ONNXRUNTIME_ROOT}/${ONNXRUNTIME_VERSION})
    message(STATUS "Downloading ONNX Runtime ${ONNXRUNTIME_VERSION}...")
    message(STATUS "URL: ${ONNXRUNTIME_URL}")
    
    file(DOWNLOAD ${ONNXRUNTIME_URL} ${CMAKE_BINARY_DIR}/onnxruntime-download.tmp
        SHOW_PROGRESS
        STATUS DOWNLOAD_STATUS
    )
    
    list(GET DOWNLOAD_STATUS 0 STATUS_CODE)
    if(NOT STATUS_CODE EQUAL 0)
        list(GET DOWNLOAD_STATUS 1 ERROR_MESSAGE)
        message(FATAL_ERROR "Failed to download ONNX Runtime: ${ERROR_MESSAGE}")
    endif()
    
    # Extract
    file(MAKE_DIRECTORY ${ONNXRUNTIME_ROOT}/${ONNXRUNTIME_VERSION})
    
    if(ONNXRUNTIME_URL MATCHES "\\.zip$")
        execute_process(
            COMMAND ${CMAKE_COMMAND} -E tar xzf ${CMAKE_BINARY_DIR}/onnxruntime-download.tmp
            WORKING_DIRECTORY ${ONNXRUNTIME_ROOT}/${ONNXRUNTIME_VERSION}
        )
    else()
        execute_process(
            COMMAND ${CMAKE_COMMAND} -E tar xzf ${CMAKE_BINARY_DIR}/onnxruntime-download.tmp
            WORKING_DIRECTORY ${ONNXRUNTIME_ROOT}/${ONNXRUNTIME_VERSION}
        )
    endif()
    
    file(REMOVE ${CMAKE_BINARY_DIR}/onnxruntime-download.tmp)
    message(STATUS "ONNX Runtime extracted to ${ONNXRUNTIME_ROOT}/${ONNXRUNTIME_VERSION}")
endif()

# Find extracted directory
file(GLOB ONNXRUNTIME_DIRS ${ONNXRUNTIME_ROOT}/${ONNXRUNTIME_VERSION}/onnxruntime-*)
message(STATUS "Found ONNX Runtime directories: ${ONNXRUNTIME_DIRS}")

if(NOT ONNXRUNTIME_DIRS)
    # Maybe it was extracted flat (no subdirectory)
    if(EXISTS ${ONNXRUNTIME_ROOT}/${ONNXRUNTIME_VERSION}/include)
        set(ONNXRUNTIME_EXTRACTED_DIR ${ONNXRUNTIME_ROOT}/${ONNXRUNTIME_VERSION})
    else()
        message(FATAL_ERROR "Could not find ONNX Runtime extraction directory")
    endif()
else()
    list(GET ONNXRUNTIME_DIRS 0 ONNXRUNTIME_EXTRACTED_DIR)
endif()

message(STATUS "ONNX Runtime extracted dir: ${ONNXRUNTIME_EXTRACTED_DIR}")

# Set variables
set(ONNXRUNTIME_INCLUDE_DIR ${ONNXRUNTIME_EXTRACTED_DIR}/include)
set(ONNXRUNTIME_LIBRARY ${ONNXRUNTIME_EXTRACTED_DIR}/lib/${ONNXRUNTIME_LIB})
set(ONNXRUNTIME_BIN_DIR ${ONNXRUNTIME_EXTRACTED_DIR}/bin)
# For prebuilt, the provider headers are in the same directory
set(ONNXRUNTIME_INCLUDE_DIR_EXTRA ${ONNXRUNTIME_INCLUDE_DIR})

# Download EP headers if not included in prebuilt package
if(APPLE AND NOT EXISTS ${ONNXRUNTIME_INCLUDE_DIR}/coreml_provider_factory.h)
    message(STATUS "Downloading CoreML provider factory header...")
    file(DOWNLOAD 
        "https://raw.githubusercontent.com/microsoft/onnxruntime/v${ONNXRUNTIME_VERSION}/include/onnxruntime/core/providers/coreml/coreml_provider_factory.h"
        ${ONNXRUNTIME_INCLUDE_DIR}/coreml_provider_factory.h
        SHOW_PROGRESS
        STATUS DOWNLOAD_STATUS
    )
    list(GET DOWNLOAD_STATUS 0 STATUS_CODE)
    if(STATUS_CODE EQUAL 0)
        message(STATUS "CoreML header downloaded successfully")
    else()
        message(WARNING "Failed to download CoreML header, GPU support may be limited")
    endif()
endif()

if(WIN32 AND NOT EXISTS ${ONNXRUNTIME_INCLUDE_DIR}/cuda_provider_factory.h)
    message(STATUS "Downloading CUDA provider factory header...")
    file(DOWNLOAD 
        "https://raw.githubusercontent.com/microsoft/onnxruntime/v${ONNXRUNTIME_VERSION}/include/onnxruntime/core/providers/cuda/cuda_provider_factory.h"
        ${ONNXRUNTIME_INCLUDE_DIR}/cuda_provider_factory.h
        SHOW_PROGRESS
        STATUS DOWNLOAD_STATUS
    )
endif()

if(WIN32 AND NOT EXISTS ${ONNXRUNTIME_INCLUDE_DIR}/dml_provider_factory.h)
    message(STATUS "Downloading DirectML provider factory header...")
    file(DOWNLOAD 
        "https://raw.githubusercontent.com/microsoft/onnxruntime/v${ONNXRUNTIME_VERSION}/include/onnxruntime/core/providers/dml/dml_provider_factory.h"
        ${ONNXRUNTIME_INCLUDE_DIR}/dml_provider_factory.h
        SHOW_PROGRESS
        STATUS DOWNLOAD_STATUS
    )
endif()

message(STATUS "ONNX Runtime include: ${ONNXRUNTIME_INCLUDE_DIR}")
message(STATUS "ONNX Runtime library: ${ONNXRUNTIME_LIBRARY}")

# Verify library exists
if(NOT EXISTS ${ONNXRUNTIME_LIBRARY})
    message(FATAL_ERROR "ONNX Runtime library not found at ${ONNXRUNTIME_LIBRARY}")
endif()

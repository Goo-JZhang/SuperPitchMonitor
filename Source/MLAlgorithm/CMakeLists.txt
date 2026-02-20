# MLAlgorithm CMakeLists.txt
# Builds MLPitchDetector and test executable

cmake_minimum_required(VERSION 3.16)
project(MLAlgorithm VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

#==============================================================================
# Options
#==============================================================================
option(MLALGO_BUILD_TESTS "Build test executable" ON)
option(MLALGO_USE_JUCE "Use JUCE framework" OFF)

#==============================================================================
# Find ONNX Runtime
#==============================================================================
# User should set ONNXRUNTIME_ROOT to the ONNX Runtime installation directory
if(NOT DEFINED ONNXRUNTIME_ROOT)
    if(DEFINED ENV{ONNXRUNTIME_ROOT})
        set(ONNXRUNTIME_ROOT $ENV{ONNXRUNTIME_ROOT})
    else()
        # Try to find in common locations
        if(APPLE)
            set(ONNXRUNTIME_ROOT "/usr/local/onnxruntime")
        elseif(WIN32)
            set(ONNXRUNTIME_ROOT "C:/Program Files/onnxruntime")
        else()
            set(ONNXRUNTIME_ROOT "/usr/local/onnxruntime")
        endif()
    endif()
endif()

message(STATUS "Looking for ONNX Runtime at: ${ONNXRUNTIME_ROOT}")

find_path(ONNXRUNTIME_INCLUDE_DIR
    NAMES onnxruntime_cxx_api.h
    PATHS ${ONNXRUNTIME_ROOT}/include
    DOC "ONNX Runtime include directory"
)

find_library(ONNXRUNTIME_LIBRARY
    NAMES onnxruntime
    PATHS ${ONNXRUNTIME_ROOT}/lib
    DOC "ONNX Runtime library"
)

if(NOT ONNXRUNTIME_INCLUDE_DIR OR NOT ONNXRUNTIME_LIBRARY)
    message(FATAL_ERROR 
        "ONNX Runtime not found. Please set ONNXRUNTIME_ROOT to the installation directory.\n"
        "Download from: https://github.com/microsoft/onnxruntime/releases"
    )
endif()

message(STATUS "Found ONNX Runtime: ${ONNXRUNTIME_LIBRARY}")
message(STATUS "ONNX Runtime includes: ${ONNXRUNTIME_INCLUDE_DIR}")

#==============================================================================
# Library target
#==============================================================================
add_library(mlalgorithm STATIC
    MLPitchDetector.cpp
    MLPitchDetector.h
    RingBuffer.h
)

target_include_directories(mlalgorithm
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        ${ONNXRUNTIME_INCLUDE_DIR}
)

target_link_libraries(mlalgorithm
    PUBLIC
        ${ONNXRUNTIME_LIBRARY}
)

# Platform-specific libraries
if(APPLE)
    find_library(FOUNDATION_LIBRARY Foundation)
    target_link_libraries(mlalgorithm PUBLIC ${FOUNDATION_LIBRARY})
elseif(UNIX AND NOT APPLE)
    target_link_libraries(mlalgorithm PUBLIC pthread)
endif()

#==============================================================================
# Test executable (standalone, no JUCE)
#==============================================================================
if(MLALGO_BUILD_TESTS)
    add_executable(test_mlpitch
        test_mlpitch.cpp
        MLPitchDetector.cpp
        MLPitchDetector.h
        RingBuffer.h
    )
    
    # For standalone test, we need minimal JuceHeader replacement
    target_compile_definitions(test_mlpitch PRIVATE 
        JUCE_GLOBAL_MODULE_SETTINGS_INCLUDED=1
        JUCE_STANDALONE_APPLICATION=1
    )
    
    # Create a minimal JUCE compatibility header for standalone build
    file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/JuceHeader.h" "
#pragma once
#include <string>
#include <cstring>
#include <algorithm>
#include <cmath>
#include <memory>

// Minimal JUCE replacements for standalone test
namespace juce {
    using String = std::string;
    template<typename T> T jlimit(T min, T max, T val) { 
        return (val < min) ? min : ((val > max) ? max : val); 
    }
    inline void DBG(const std::string& msg) { 
        std::cout << msg << std::endl; 
    }
}
#define JUCE_MAC 1
#define JUCE_IOS 0
#define JUCE_WINDOWS 0
#define JUCE_ANDROID 0
")
    
    target_include_directories(test_mlpitch
        PRIVATE
            ${CMAKE_CURRENT_BINARY_DIR}
            ${ONNXRUNTIME_INCLUDE_DIR}
    )
    
    target_link_libraries(test_mlpitch
        PRIVATE
            ${ONNXRUNTIME_LIBRARY}
    )
    
    if(APPLE)
        target_link_libraries(test_mlpitch PRIVATE ${FOUNDATION_LIBRARY})
    elseif(UNIX)
        target_link_libraries(test_mlpitch PRIVATE pthread)
    endif()
    
    # Copy test model to build directory
    add_custom_command(TARGET test_mlpitch POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${CMAKE_SOURCE_DIR}/../../MLModel/pitchnet_stub_v1.onnx
            $<TARGET_FILE_DIR:test_mlpitch>/pitchnet_stub_v1.onnx
        COMMENT "Copying test model..."
    )
endif()

#==============================================================================
# Installation
#==============================================================================
install(TARGETS mlalgorithm
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)

install(FILES MLPitchDetector.h RingBuffer.h
    DESTINATION include/mlalgorithm
)

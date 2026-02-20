cmake_minimum_required(VERSION 3.15)

#==============================================================================
# Third Party Dependencies Setup
#==============================================================================

#==============================================================================
# Third Party Dependencies Setup
#==============================================================================

# JUCE uses FetchContent and will be stored in ThirdParty/juce/
set(FETCHCONTENT_BASE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/ThirdParty/juce" CACHE PATH "External dependencies cache directory")

# ONNX Runtime paths (source build)
set(ONNXRUNTIME_DIR "${CMAKE_CURRENT_SOURCE_DIR}/ThirdParty/onnxruntime" CACHE PATH "ONNX Runtime directory")

#==============================================================================
# SuperPitchMonitor CMake Configuration
#==============================================================================

project(SuperPitchMonitor VERSION 1.0.0 LANGUAGES CXX OBJCXX)

#==============================================================================
# Default Build Configuration (ONNX Runtime from source with CUDA + DirectML)
#==============================================================================
# Set defaults before any options are declared
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type" FORCE)
endif()

#==============================================================================
# Output Directory Configuration
#==============================================================================

# Set output directory to project root for easy access
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG "${CMAKE_CURRENT_SOURCE_DIR}")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_CURRENT_SOURCE_DIR}")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO "${CMAKE_CURRENT_SOURCE_DIR}")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_MINSIZEREL "${CMAKE_CURRENT_SOURCE_DIR}")

#==============================================================================
# JUCE Setup (via FetchContent)
#==============================================================================

include(FetchContent)

FetchContent_Declare(
    JUCE
    GIT_REPOSITORY https://github.com/juce-framework/JUCE.git
    GIT_TAG 7.0.12  # Use latest stable JUCE 7
    GIT_SHALLOW TRUE
    GIT_PROGRESS TRUE
)

FetchContent_MakeAvailable(JUCE)

#==============================================================================
# GUI Application
#==============================================================================

juce_add_gui_app(SuperPitchMonitor
    PRODUCT_NAME "SuperPitchMonitor"
    COMPANY_NAME "SuperPitchMonitor"
    BUNDLE_ID com.superpitchmonitor.app
    NEEDS_WEB_BROWSER FALSE
    NEEDS_INTERNET_CONNECTION FALSE
)

#==============================================================================
# Source Files
#==============================================================================

target_sources(SuperPitchMonitor PRIVATE
    # Main
    Source/Main.cpp
    Source/MainComponent.cpp
    Source/MainComponent.h
    
    # Audio
    Source/Audio/AudioEngine.cpp
    Source/Audio/AudioEngine.h
    Source/Audio/SpectrumData.h
    Source/Audio/SpectrumAnalyzer.cpp
    Source/Audio/SpectrumAnalyzer.h
    Source/Audio/PolyphonicDetector.cpp
    Source/Audio/PolyphonicDetector.h
    Source/Audio/MultiResolutionAnalyzer.cpp
    Source/Audio/MultiResolutionAnalyzer.h
    Source/Audio/YinPitchDetector.cpp
    Source/Audio/YinPitchDetector.h
    Source/Audio/QuickPitchDetector.cpp
    Source/Audio/QuickPitchDetector.h
    Source/Audio/PitchTracker.cpp
    Source/Audio/PitchTracker.h
    Source/Audio/AudioBufferFifo.h
    Source/Audio/AudioInputSource.h
    Source/Audio/DeviceAudioInput.cpp
    Source/Audio/DeviceAudioInput.h
    Source/Audio/SystemAudioInput.cpp
    Source/Audio/SystemAudioInput.h
    Source/Audio/FileAudioInput.cpp
    Source/Audio/FileAudioInput.h
    
    # UI
    Source/UI/SpectrumDisplay.cpp
    Source/UI/SpectrumDisplay.h
    Source/UI/PitchDisplay.cpp
    Source/UI/PitchDisplay.h
    Source/UI/TunerDisplay.cpp
    Source/UI/TunerDisplay.h
    Source/UI/PitchCard.cpp
    Source/UI/PitchCard.h
    Source/UI/SettingsPanel.cpp
    Source/UI/SettingsPanel.h
    Source/UI/PitchWaterfallDisplay.cpp
    Source/UI/PitchWaterfallDisplay.h
    
    # Utils
    Source/Utils/Logger.cpp
    Source/Utils/Logger.h
    
    # Algorithms
    Source/Algorithms/PeakDetector.cpp
    Source/Algorithms/PeakDetector.h
    Source/Algorithms/FFTUtils.h
    Source/Algorithms/HarmonicClustering.h
    Source/Algorithms/YINProcessor.h
    
    Source/Utils/Config.h
    Source/Utils/AudioUtils.h
    Source/Utils/MidiUtils.h
    Source/Utils/PerformanceMonitor.h
    Source/Utils/PlatformUtils.cpp
    Source/Utils/PlatformUtils.h
    
    # Debug
    Source/Debug/AudioSimulator.cpp
    Source/Debug/AudioSimulator.h
    Source/Debug/DebugControlPanel.cpp
    Source/Debug/DebugControlPanel.h
    
    # Test
    Source/Test/TestServer.cpp
    Source/Test/TestServer.h
    
    # ML Algorithm
    Source/MLAlgorithm/MLPitchDetector.cpp
    Source/MLAlgorithm/MLPitchDetector.h
    Source/MLAlgorithm/RingBuffer.h
)

#==============================================================================
# Platform-specific Source Files
#==============================================================================

if(WIN32)
    target_sources(SuperPitchMonitor PRIVATE
        Source/Utils/PlatformUtils_Windows.cpp
    )
elseif(CMAKE_SYSTEM_NAME STREQUAL "Android")
    target_sources(SuperPitchMonitor PRIVATE
        Source/Utils/PlatformUtils_Android.cpp
    )
elseif(CMAKE_SYSTEM_NAME STREQUAL "iOS")
    target_sources(SuperPitchMonitor PRIVATE
        Source/Utils/PlatformUtils_iOS.mm
    )
endif()

#==============================================================================
# JUCE Modules
#==============================================================================

target_link_libraries(SuperPitchMonitor PRIVATE
    juce::juce_core
    juce::juce_data_structures
    juce::juce_events
    juce::juce_graphics
    juce::juce_gui_basics
    juce::juce_gui_extra
    juce::juce_audio_basics
    juce::juce_audio_devices
    juce::juce_audio_formats
    juce::juce_audio_processors
    juce::juce_audio_utils
    juce::juce_dsp
)

#==============================================================================
# Include Directories
#==============================================================================

target_include_directories(SuperPitchMonitor PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/Source
)

#==============================================================================
# ONNX Runtime (Build from Source - Like JUCE)
#==============================================================================
option(BUILD_ONNXRUNTIME_FROM_SOURCE "Build ONNX Runtime from source (20-30 min, full GPU support)" ON)
option(USE_GPU_ONNXRUNTIME "Enable GPU acceleration" ON)

if(WIN32)
    option(USE_CUDA_ON_WINDOWS "Use CUDA for NVIDIA GPUs (vs DirectML universal)" ON)
endif()

if(BUILD_ONNXRUNTIME_FROM_SOURCE)
    message(STATUS "========================================")
    message(STATUS "ONNX Runtime: Building from source")
    message(STATUS "This may take 20-30 minutes (like JUCE)")
    message(STATUS "Platform: ${CMAKE_SYSTEM_NAME}")
    message(STATUS "========================================")
    include(${CMAKE_CURRENT_SOURCE_DIR}/ThirdParty/build_onnxruntime.cmake)
    
    add_dependencies(SuperPitchMonitor onnxruntime_build)
    target_include_directories(SuperPitchMonitor PRIVATE ${ONNXRUNTIME_INCLUDE_DIR})
    # Add extra provider include directories if defined
    if(DEFINED ONNXRUNTIME_INCLUDE_DIR_EXTRA)
        target_include_directories(SuperPitchMonitor PRIVATE ${ONNXRUNTIME_INCLUDE_DIR_EXTRA})
    endif()
    target_link_libraries(SuperPitchMonitor PRIVATE ${ONNXRUNTIME_LIBRARY})
    target_compile_definitions(SuperPitchMonitor PRIVATE ML_ENABLED=1)
    
    if(APPLE)
        add_custom_command(TARGET SuperPitchMonitor POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different ${ONNXRUNTIME_LIBRARY} $<TARGET_FILE_DIR:SuperPitchMonitor>
            COMMENT "Copying ONNX Runtime library..."
        )
    elseif(WIN32)
        add_custom_command(TARGET SuperPitchMonitor POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different ${ONNXRUNTIME_BIN_DIR}/onnxruntime.dll $<TARGET_FILE_DIR:SuperPitchMonitor>
            COMMENT "Copying ONNX Runtime DLL..."
        )
    endif()
else()
    message(STATUS "ONNX Runtime: Using pre-built binary (fast, limited operators)")
    include(${CMAKE_CURRENT_SOURCE_DIR}/ThirdParty/fetch_onnxruntime.cmake)
    
    if(EXISTS ${ONNXRUNTIME_LIBRARY})
        target_include_directories(SuperPitchMonitor PRIVATE ${ONNXRUNTIME_INCLUDE_DIR})
        target_link_libraries(SuperPitchMonitor PRIVATE ${ONNXRUNTIME_LIBRARY})
        target_compile_definitions(SuperPitchMonitor PRIVATE ML_ENABLED=1)
    else()
        message(WARNING "ONNX Runtime not found, ML disabled")
        target_compile_definitions(SuperPitchMonitor PRIVATE ML_ENABLED=0)
    endif()
endif()

# Copy ML models
add_custom_command(TARGET SuperPitchMonitor POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/MLModel $<TARGET_FILE_DIR:SuperPitchMonitor>/MLModel
    COMMENT "Copying ML models..."
)

#==============================================================================
# Compile Options
#==============================================================================

target_compile_features(SuperPitchMonitor PRIVATE cxx_std_17)

juce_generate_juce_header(SuperPitchMonitor)

# Windows specific
if(WIN32)
    target_compile_definitions(SuperPitchMonitor PRIVATE
        JUCE_WINDOWS=1
        _CRT_SECURE_NO_WARNINGS
        NOMINMAX
    )
    # Link DbgHelp for crash stack traces
    target_link_libraries(SuperPitchMonitor PRIVATE
        DbgHelp
    )
endif()

# Android specific
if(CMAKE_SYSTEM_NAME STREQUAL "Android")
    target_compile_definitions(SuperPitchMonitor PRIVATE
        JUCE_ANDROID=1
    )
endif()

# Apple specific (macOS & iOS)
if(APPLE)
    # Enable ARC for Objective-C++ files
    set_target_properties(SuperPitchMonitor PROPERTIES
        XCODE_ATTRIBUTE_CLANG_ENABLE_OBJC_ARC YES
    )
    
    # Platform-specific defines
    if(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
        target_compile_definitions(SuperPitchMonitor PRIVATE
            JUCE_MAC=1
        )
    elseif(CMAKE_SYSTEM_NAME STREQUAL "iOS")
        target_compile_definitions(SuperPitchMonitor PRIVATE
            JUCE_IOS=1
        )
    endif()
    
    # Framework dependencies for iOS/macOS
    if(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
        # macOS frameworks
        target_link_libraries(SuperPitchMonitor PRIVATE
            "-framework Foundation"
            "-framework AppKit"
            "-framework CoreGraphics"
            "-framework CoreAudio"
            "-framework CoreMidi"
            "-framework AudioToolbox"
            "-framework AudioUnit"
            "-framework Accelerate"
            "-framework Cocoa"
            "-framework IOKit"
        )
    elseif(CMAKE_SYSTEM_NAME STREQUAL "iOS")
        # iOS frameworks
        target_link_libraries(SuperPitchMonitor PRIVATE
            "-framework Foundation"
            "-framework UIKit"
            "-framework CoreGraphics"
        )
    endif()
endif()

# Debug/Release definitions
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_definitions(SuperPitchMonitor PRIVATE
        DEBUG=1
        _DEBUG=1
    )
else()
    target_compile_definitions(SuperPitchMonitor PRIVATE
        NDEBUG=1
    )
endif()

#==============================================================================
# Compiler Warnings (MSVC)
#==============================================================================

if(MSVC)
    target_compile_options(SuperPitchMonitor PRIVATE
        /W4
        /WX-
        /permissive-
        /Zc:__cplusplus
    )
endif()

#==============================================================================
# Binary Output
#==============================================================================

# All platforms: output to project root for easy access
set_target_properties(SuperPitchMonitor PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
)

# macOS: also set bundle output to project root
if(APPLE)
    set_target_properties(SuperPitchMonitor PROPERTIES
        MACOSX_BUNDLE_BUNDLE_NAME "SuperPitchMonitor"
        MACOSX_BUNDLE_GUI_IDENTIFIER "com.superpitchmonitor.app"
    )
endif()

#==============================================================================
# Installation
#==============================================================================

install(TARGETS SuperPitchMonitor
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    BUNDLE DESTINATION Applications
)

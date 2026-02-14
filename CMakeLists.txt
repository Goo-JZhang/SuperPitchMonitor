cmake_minimum_required(VERSION 3.15)

#==============================================================================
# Third Party Dependencies Setup
#==============================================================================

# Set external cache directory for FetchContent
# This prevents re-downloading dependencies when build directory is cleaned
set(FETCHCONTENT_BASE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/ThirdParty" CACHE PATH "External dependencies cache directory")

#==============================================================================
# SuperPitchMonitor CMake Configuration
#==============================================================================

project(SuperPitchMonitor VERSION 1.0.0 LANGUAGES CXX OBJCXX)

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
    Source/Utils/PlatformUtils_Windows.cpp
    Source/Utils/PlatformUtils_Android.cpp
    Source/Utils/PlatformUtils_iOS.mm
    
    # Debug
    Source/Debug/AudioSimulator.cpp
    Source/Debug/AudioSimulator.h
    Source/Debug/DebugControlPanel.cpp
    Source/Debug/DebugControlPanel.h
    
    # Test
    Source/Test/AutoTestManager.cpp
    Source/Test/AutoTestManager.h
)

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
    target_link_libraries(SuperPitchMonitor PRIVATE
        "-framework Foundation"
        "-framework UIKit"
        "-framework CoreGraphics"
    )
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

set_target_properties(SuperPitchMonitor PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/SuperPitchMonitor_artefacts"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/SuperPitchMonitor_artefacts"
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/SuperPitchMonitor_artefacts"
)

#==============================================================================
# Installation
#==============================================================================

install(TARGETS SuperPitchMonitor
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

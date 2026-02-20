/*
  =============================================================================-
    test_mlpitch.cpp
    Standalone test for MLPitchDetector
    
    Usage: ./test_mlpitch <path_to_onnx_model>
  =============================================================================-
*/

#include "MLPitchDetector.h"
#include "RingBuffer.h"
#include <iostream>
#include <cmath>
#include <chrono>
#include <vector>

using namespace spm;

// Generate a sine wave at given frequency
std::vector<float> generateSineWave(float frequency, float sampleRate, float duration)
{
    int numSamples = static_cast<int>(duration * sampleRate);
    std::vector<float> wave(numSamples);
    
    for (int i = 0; i < numSamples; ++i)
    {
        wave[i] = std::sin(2.0f * 3.14159265359f * frequency * i / sampleRate);
    }
    
    return wave;
}

// Generate a chord with multiple frequencies
std::vector<float> generateChord(const std::vector<float>& frequencies, float sampleRate, float duration)
{
    int numSamples = static_cast<int>(duration * sampleRate);
    std::vector<float> wave(numSamples, 0.0f);
    
    for (float freq : frequencies)
    {
        for (int i = 0; i < numSamples; ++i)
        {
            wave[i] += std::sin(2.0f * 3.14159265359f * freq * i / sampleRate) / frequencies.size();
        }
    }
    
    // Normalize
    float maxVal = 0.0f;
    for (float s : wave) maxVal = std::max(maxVal, std::abs(s));
    if (maxVal > 0)
    {
        for (float& s : wave) s /= maxVal;
    }
    
    return wave;
}

int main(int argc, char* argv[])
{
    std::cout << "==============================================\n";
    std::cout << "MLPitchDetector Quick Validation Test\n";
    std::cout << "==============================================\n\n";
    
    // Check arguments
    if (argc < 2)
    {
        std::cerr << "Usage: " << argv[0] << " <path_to_onnx_model>\n";
        std::cerr << "\nExample:\n";
        std::cerr << "  " << argv[0] << " ../MLModel/pitchnet_stub_v1.onnx\n";
        return 1;
    }
    
    std::string modelPath = argv[1];
    std::cout << "Model path: " << modelPath << "\n\n";
    
    // Initialize detector
    MLPitchDetector detector;
    MLPitchDetector::Config config;
    config.modelPath = modelPath;
    config.useGPU = true;
    config.threadPoolSize = 2;
    
    std::cout << "Initializing detector...\n";
    if (!detector.initialize(config))
    {
        std::cerr << "ERROR: Failed to initialize detector\n";
        return 1;
    }
    
    std::cout << "✓ Detector initialized\n";
    std::cout << "Model info:\n" << detector.getModelInfo().toStdString() << "\n\n";
    
    // Test 1: Single sine wave
    std::cout << "==============================================\n";
    std::cout << "Test 1: Single sine wave (440 Hz)\n";
    std::cout << "==============================================\n";
    
    auto sine440 = generateSineWave(440.0f, 44100.0f, 0.1f);  // 100ms
    std::vector<float> inputBuffer(4096, 0.0f);
    
    // Fill the last 4096 samples (or pad with zeros)
    size_t copySize = std::min(sine440.size(), inputBuffer.size());
    std::memcpy(inputBuffer.data() + (4096 - copySize), sine440.data(), copySize * sizeof(float));
    
    // Run inference
    std::vector<float> outputBuffer(4096);  // 2048 * 2
    
    std::cout << "Running inference...\n";
    auto start = std::chrono::high_resolution_clock::now();
    
    bool success = detector.inferenceSync(inputBuffer.data(), outputBuffer.data());
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    if (!success)
    {
        std::cerr << "ERROR: Inference failed\n";
        return 1;
    }
    
    std::cout << "✓ Inference completed in " << (duration.count() / 1000.0) << " ms\n\n";
    
    // Print top detections
    std::cout << "Top 5 detections (random weights, values are meaningless):\n";
    std::vector<std::pair<int, float>> confidences;
    for (int i = 0; i < 2048; ++i)
    {
        confidences.push_back({i, outputBuffer[i * 2]});  // confidence is at even indices
    }
    
    // Sort by confidence
    std::sort(confidences.begin(), confidences.end(),
        [](const auto& a, const auto& b) { return a.second > b.second; });
    
    for (int i = 0; i < 5 && i < confidences.size(); ++i)
    {
        int binIdx = confidences[i].first;
        float conf = confidences[i].second;
        float energy = outputBuffer[binIdx * 2 + 1];
        float freq = MLPitchDetector::binIndexToFrequency(binIdx);
        
        std::cout << "  " << (i+1) << ". Bin " << binIdx 
                  << " (" << freq << " Hz): conf=" << conf 
                  << ", energy=" << energy << "\n";
    }
    
    // Test 2: Multiple frequencies
    std::cout << "\n==============================================\n";
    std::cout << "Test 2: C Major chord (261.63, 329.63, 392.00 Hz)\n";
    std::cout << "==============================================\n";
    
    auto chord = generateChord({261.63f, 329.63f, 392.00f}, 44100.0f, 0.1f);
    std::fill(inputBuffer.begin(), inputBuffer.end(), 0.0f);
    copySize = std::min(chord.size(), inputBuffer.size());
    std::memcpy(inputBuffer.data() + (4096 - copySize), chord.data(), copySize * sizeof(float));
    
    success = detector.inferenceSync(inputBuffer.data(), outputBuffer.data());
    if (!success)
    {
        std::cerr << "ERROR: Inference failed\n";
        return 1;
    }
    
    std::cout << "✓ Inference completed in " << detector.getLastInferenceTimeMs() << " ms\n\n";
    
    std::cout << "Top 5 detections:\n";
    confidences.clear();
    for (int i = 0; i < 2048; ++i)
    {
        confidences.push_back({i, outputBuffer[i * 2]});
    }
    std::sort(confidences.begin(), confidences.end(),
        [](const auto& a, const auto& b) { return a.second > b.second; });
    
    for (int i = 0; i < 5 && i < confidences.size(); ++i)
    {
        int binIdx = confidences[i].first;
        float conf = confidences[i].second;
        float energy = outputBuffer[binIdx * 2 + 1];
        float freq = MLPitchDetector::binIndexToFrequency(binIdx);
        
        std::cout << "  " << (i+1) << ". Bin " << binIdx 
                  << " (" << freq << " Hz): conf=" << conf 
                  << ", energy=" << energy << "\n";
    }
    
    // Test 3: Ring buffer test
    std::cout << "\n==============================================\n";
    std::cout << "Test 3: RingBuffer functionality\n";
    std::cout << "==============================================\n";
    
    RingBuffer<float> ringBuffer(8192);  // Double buffer size
    
    // Simulate audio callback: write 512 samples at a time
    std::vector<float> callbackBuffer(512);
    for (int callback = 0; callback < 20; ++callback)
    {
        // Fill with some pattern
        for (int i = 0; i < 512; ++i)
        {
            callbackBuffer[i] = static_cast<float>(callback * 512 + i);
        }
        ringBuffer.write(callbackBuffer.data(), 512);
    }
    
    // Read latest 4096
    std::vector<float> latest4096(4096);
    ringBuffer.readLatest(4096, latest4096.data());
    
    std::cout << "✓ RingBuffer test passed\n";
    std::cout << "  Latest sample value: " << latest4096[4095] << " (expected ~10239)\n";
    std::cout << "  Oldest sample value: " << latest4096[0] << " (expected ~6144)\n";
    
    // Summary
    std::cout << "\n==============================================\n";
    std::cout << "All tests passed!\n";
    std::cout << "==============================================\n";
    std::cout << "\nNext steps:\n";
    std::cout << "1. Train actual model with PyTorch\n";
    std::cout << "2. Replace stub model with trained weights\n";
    std::cout << "3. Verify pitch detection accuracy\n";
    
    return 0;
}

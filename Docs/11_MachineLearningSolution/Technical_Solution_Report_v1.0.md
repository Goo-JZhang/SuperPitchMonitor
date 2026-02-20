# SuperPitchMonitor 机器学习化技术方案报告

**版本**: 1.1  
**日期**: 2026-02-19  
**目标**: 将传统FFT算法迁移至深度学习方案，满足实时多音高检测需求

**更新记录**:
- v1.1 (2026-02-19): 添加NumPy Shards存储格式，随机相位增强策略，Phase 3训练成果
- v1.0 (2026-02-16): 初始版本

---

## 一、需求规格定义

### 1.1 核心性能指标

| 指标 | 规格 | 说明 |
|------|------|------|
| **频率精度** | ≤ ±5 cents | 20Hz处约0.6Hz，5000Hz处约14.5Hz |
| **频率范围** | 20Hz - 5000Hz | 覆盖钢琴A0-C8，含低音贝斯/高音提琴 |
| **响应延迟** | ≤ 50ms | 从音发声到检测结果输出 |
| **多音能力** | ≥ 5个同时音 | 复杂和弦场景，独立相位泛音列 |
| **刷新率** | ≥ 30fps | UI显示帧率，推理需支持更高频率 |
| **输入规格** | 4096 samples @ 44.1kHz | 约93ms音频窗口 |

### 1.2 平台兼容性矩阵

| 平台 | 最低版本 | GPU后端 | 目标延迟 |
|------|---------|---------|----------|
| macOS | 12.0 (Monterey) | CoreML (ANE+GPU) | 5-8ms |
| iOS | 15.0 | CoreML Neural Engine | 5-8ms |
| Windows | 10 21H2 | DirectML / CUDA | 5-10ms |
| Android | 10 (API 29) | NNAPI GPU Delegate | 8-12ms |

### 1.3 输出规范

```cpp
struct PitchDetection {
    float frequency;      // 20-5000Hz，对数分布2048 bins对应
    float confidence;     // 0.0-1.0，Sigmoid输出
    float energy;         // 相对频谱能量，dB或线性
    uint32_t timestamp;   // 采样点时间戳
};

// 每帧输出：vector<PitchDetection>，按confidence排序
```

---

## 二、技术方案总览

### 2.1 方案对比分析

| 方案 | 模型大小 | 推理延迟(M1) | 5cent精度 | 多音分离 | 推荐度 |
|------|---------|-------------|-----------|----------|--------|
| **A. 纯时域TCN** | 200K | 3-5ms | ⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐ |
| **B. 时域端到端 (PitchNet)** | 2.5M | 6-10ms | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **C. Transformer** | 1.2M | 10-15ms | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| **D. 频谱图U-Net** | 800K | 8-12ms | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ |

### 2.2 推荐方案：B. 时域端到端网络 (PitchNet)

**核心设计**: 输入4096原始采样点 → 可学习前端 → 2048频谱bin (confidence+energy)

**选择理由**：
1. **端到端最优**：可学习前端替代固定CQT，网络自动发现最优时频表示
2. **输出精细**: 2048个对数等距bin，每个约4.67 cents间距，满足5 cents精度要求
3. **简化流程**: 无需预处理，时域直接到检测结果，减少误差累积
4. **泛化能力强**: 数据驱动的频谱表示，比手工设计的CQT更适应复杂音色

---

## 三、详细架构设计

### 3.1 系统架构图

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         SuperPitchMonitor ML Pipeline                    │
├─────────────────────────────────────────────────────────────────────────┤
│  Layer 1: Audio I/O (JUCE)                                               │
│  ├─ AudioDeviceManager @ 44.1kHz                                         │
│  ├─ RingBuffer: 2×4096 samples (双缓冲)                                  │
│  └─ 读取策略: 每次处理从最新采样点向前取4096点，无需等待hop填充        │
├─────────────────────────────────────────────────────────────────────────┤
│  Layer 2: 可选归一化 (GPU/CPU)                                           │
│  ├─ 输入: 4096点原始时域采样 @ 44.1kHz (约93ms)                          │
│  ├─ 归一化: RMS归一化到目标电平，或直接送入网络让其学习                  │
│  └─ 格式: [1, 4096] float32 → 直接送入神经网络                           │
│      └─ 耗时: ~0.1ms                                                     │
├─────────────────────────────────────────────────────────────────────────┤
│  Layer 3: Neural Network Inference (ONNX Runtime)                        │
│  ├─ Backend: CoreML/NNAPI/CUDA/DirectML (平台自适应)                     │
│  ├─ Model: PitchNet (详见3.2节)                                          │
│  └─ 耗时: ~5-8ms (M1/M2), ~8-12ms (移动端)                               │
├─────────────────────────────────────────────────────────────────────────┤
│  Layer 4: Post-Processing (CPU) - 可选/轻量级                            │
│  ├─ 阈值过滤: 根据confidence阈值筛选候选                                 │
│  ├─ 时序平滑: 1 Euro Filter (可选，用于UI显示稳定)                       │
│  └─ 耗时: ~0.5ms (若网络训练充分，此层可大幅简化或移除)                  │
├─────────────────────────────────────────────────────────────────────────┤
│  Layer 5: Output Interface                                               │
│  ├─ Thread-safe结果队列 → JUCE UI Thread                                 │
│  ├─ 回调接口: pitchDetected(vector<Pitch>)                               │
│  └─ 显示刷新: 30fps (33ms间隔，降采样自43fps内部流)                      │
└─────────────────────────────────────────────────────────────────────────┘
```

### 3.2 PitchNet 网络架构 (时域→频谱端到端)

**设计目标**: 直接从4096点时域采样学习映射到2048个对数等距频谱bin

```
Input: [batch=1, channels=1, samples=4096]  ← 原始时域波形
           │
           ▼
┌───────────────────────────────────────────┐
│ 可学习前端: Learnable Filterbank           │  替代固定CQT
│  Conv1d: 1→128, k=512, s=128, pad=256    │  分2048频带，overlap
│  BN + GELU                                │  输出: [128, 32] (channels, time)
│  转置 → [32, 128]                         │  
└───────────────────────────────────────────┘
           │
           ▼
┌───────────────────────────────────────────┐
│ 时频特征提取: 1D Conv Stack                │
│  Conv1d: 128→256, k=7, s=2               │  32→16
│  DepthwiseConv1d: 256, k=5, s=2          │  16→8
│  DepthwiseConv1d: 256, k=3, s=2          │  8→4
│  每步: BN + GELU + SE注意力               │  输出: [256, 4]
└───────────────────────────────────────────┘
           │
           ▼
┌───────────────────────────────────────────┐
│ 频率注意力聚合                             │
│  AdaptiveAvgPool1d: 4→1                  │  [256, 1]
│  Linear: 256→1024                        │  
│  GELU + Dropout(0.1)                      │
│  Linear: 1024→2048×2                     │  输出层
│  Split: [2048] confidence + [2048] energy │
└───────────────────────────────────────────┘
           │
           ▼
Output: Confidence [2048] → Sigmoid (存在基频的概率)
        Energy [2048] → Softplus (相对能量，非负)
```

**关键设计说明**:

1. **可学习前端 (Learnable Filterbank)**
   - 替代固定CQT，让网络学习最优时频表示
   - kernel=512 (约11.6ms), stride=128 (约2.9ms hop)
   - 2048个频带通过对数分布的初始化实现

2. **频率轴编码**
   - 2048个输出bin对应20-5000Hz对数等距分布
   - bin i 的中心频率: `f_i = 20 * (5000/20)^(i/2047)`
   - 每个bin宽度: ~4.67 cents (满足5 cents精度要求)

3. **时序压缩策略**
   - 通过stride卷积将4096点压缩到4个时间帧
   - 最后平均池化，输出与精确时间位置无关的检测结果
   - 适配50ms onset检测需求

**网络参数统计**:
- 总参数量: ~2.5M (约10MB FP32 / 2.5MB INT8)
- FLOPs: ~80M / inference
- 感受野: 覆盖全部4096输入点

**轻量化变体 (PitchNet-Lite)**:
若延迟需进一步优化，可缩减为:
- 前端: 1→64 channels
- 中间: 64→128 channels  
- 参数量: ~600K, 延迟: ~3-5ms

### 3.3 输出频谱定义 (对数等距2048 bins)

网络输出2048个频谱bin，覆盖20-5000Hz，对数等距分布：

| 参数 | 值 | 说明 |
|------|-----|------|
| 频率范围 | 20Hz - 5000Hz | 覆盖目标音高范围 |
| 总bins | 2048 | 精细频率分辨率 |
| Bin间距 | ~4.67 cents | `log2(5000/20)/2047 × 1200 ≈ 4.67 cents` |
| Bin中心频率 | `f_i = 20 × (5000/20)^(i/2047)` | 对数等距公式 |

**频率分辨率**:
- 对数等距分布: 2048 bins覆盖 log2(5000/20) ≈ 7.966 个八度
- **每bin固定宽度**: 7.966 / 2047 ≈ **0.00389 八度 = 4.67 cents**
- 线性Hz宽度随频率增长:
  - 20Hz附近: ~0.067 Hz
  - 440Hz附近: ~1.48 Hz
  - 5000Hz附近: ~16.8 Hz

*注: 4.67 cents < 5 cents要求，单bin即可满足精度，网络插值可进一步提升稳定性*

**可学习前端初始化**:
```python
# 初始化Conv1d滤波器为对数分布的带通滤波器
# 使网络从合理的频谱分解开始，而非随机
import numpy as np

freqs = 20 * (5000/20) ** (np.arange(128) / 127)  # 前端128通道
t = np.arange(512) / 44100
for i, f in enumerate(freqs):
    # Gabor滤波器初始化
    envelope = np.exp(-0.5 * ((t - 256/44100) / (128/44100))**2)
    carrier = np.cos(2 * np.pi * f * t)
    filter_kernel[i, 0, :] = envelope * carrier
```

---

## 四、训练策略

### 4.1 数据合成方案

由于真实多音高标注数据稀缺，采用**物理建模合成 + 域随机化**：

```python
# 合成流程 (PyTorch Dataset)
class PitchSynthesizer:
    def __init__(self):
        self.f0_range = np.logspace(np.log10(20), np.log10(5000), 2048)
        self.sample_rate = 44100
        
    def generate_note(self, f0, duration=0.1):
        """单个音符物理建模"""
        t = np.arange(int(duration * self.sample_rate)) / self.sample_rate
        
        # 1. 基频 + 谐波列 (含失谐)
        harmonics = []
        for n in range(1, 20):
            # 非谐性: 高频谐波略微偏高 (Inharmonicity)
            fn = f0 * n * (1 + 0.0001 * n**2)
            # 随机相位 (关键！确保网络学习泛音关系而非固定相位)
            phase = np.random.uniform(0, 2*np.pi)
            # 振幅衰减
            amp = np.random.uniform(0.5, 1.0) / n**np.random.uniform(0.8, 1.2)
            harmonics.append(amp * np.sin(2*np.pi*fn*t + phase))
        
        # 2. ADSR包络
        envelope = self.adsr_envelope(duration, 
                                     attack=np.random.uniform(0.01, 0.02),
                                     decay=np.random.uniform(0.02, 0.05))
        
        # 3. 混响 (不同房间IR)
        ir = self.random_room_ir()
        
        signal = sum(harmonics) * envelope
        signal = fftconvolve(signal, ir, mode='same')
        return signal
    
    def generate_chord(self, num_notes, min_semitones=3):
        """多音生成 (带相位关系约束)"""
        # 随机选择音高 (避免过于接近导致拍频干扰)
        f0s = np.random.choice(self.f0_range, num_notes, replace=False)
        f0s = sorted(f0s)
        
        # 确保音程符合音乐性 (最小3半音间隔)
        for i in range(1, len(f0s)):
            while 1200*np.log2(f0s[i]/f0s[i-1]) < min_semitones*100:
                f0s[i] *= 2**(3/12)  # 最小大三度
        
        # 独立合成每个音，随机起始时间偏移 (±10ms) 模拟真实演奏
        mixed = np.zeros(int(0.1 * self.sample_rate))
        labels = []
        for f0 in f0s:
            note = self.generate_note(f0)
            offset = np.random.randint(-int(0.01*self.sample_rate), 
                                        int(0.01*self.sample_rate))
            # 叠加...
            labels.append({'f0': f0, 'onset': max(0, offset/self.sample_rate)})
        
        return mixed, labels
```

### 4.2 标注策略与端到端学习

**核心原则**: 网络应直接学习从原始音频到音高检测的端到端映射，避免手工设计的后处理逻辑。

**标注方式**:
```python
# 对于输入信号（含泛音列），标注仅在基频位置
# 例: piano_like_c (261.63Hz基频 + 丰富泛音)
#     - 基频 bin (对应261.63Hz): confidence=1.0, energy=总能量
#     - 泛音 bins (523Hz, 785Hz...): confidence=0, energy=0
# 
# 网络通过训练自动学习：泛音模式 → 基频预测
# 因此后处理无需显式"谐波整合"，网络已内化此能力
```

**优势**:
- 泛音到基频的映射由数据驱动学习，比手工规则更鲁棒
- 支持复杂泛音结构（钢琴非谐性、打击乐非周期成分等）
- 推理阶段后处理大幅简化，仅需阈值过滤

### 4.3 数据增强策略

| 增强类型 | 参数范围 | 目的 |
|---------|---------|------|
| **噪声混合** | SNR: -10 to 40dB | 白噪、粉噪、环境噪 |
| **混响** | RT60: 0.1-3.0s | 模拟不同录制环境 |
| **时间拉伸** | factor: 0.95-1.05 | 模拟轻微速度变化 |
| **音高抖动** | ±10 cents | 模拟乐器调音偏差 |
| **时间偏移** | ±5ms | 强化 onset 鲁棒性 |
| **频谱掩蔽** | 随机bin置零 | 模拟频率选择性衰落 |

### 4.4 损失函数设计

```python
class PitchDetectionLoss(nn.Module):
    """
    音高检测组合损失函数
    
    设计原则:
    1. Confidence Loss: 所有bin参与学习，使用BCE
    2. Energy Loss: 使用target_conf作为软权重加权MSE
       (不使用pred_conf做mask，避免循环依赖)
    3. Sparsity Loss: 轻微抑制假阳性
    """
    def __init__(self, conf_weight=1.0, energy_weight=0.5, sparsity_weight=0.01):
        super().__init__()
        self.conf_weight = conf_weight
        self.energy_weight = energy_weight
        self.sparsity_weight = sparsity_weight
        
    def forward(self, pred_conf, pred_energy, target_conf, target_energy):
        """
        Args:
            pred_conf: [B, 2048] 预测置信度 (sigmoid后)
            pred_energy: [B, 2048] 预测能量
            target_conf: [B, 2048] 目标置信度 (软标签，高斯分布)
            target_energy: [B, 2048] 目标能量
        """
        # 1. 置信度损失: Soft-Label BCE
        # target_conf是预生成的高斯软标签，非one-hot
        loss_conf = F.binary_cross_entropy(pred_conf, target_conf)
        
        # 2. 能量损失: 使用target_conf作为软权重
        # 
        # 为什么不使用pred_conf做mask?
        # - pred_conf是网络输出，训练初期可能全预测0
        # - 用pred_conf做mask会导致energy loss被mask掉 → 梯度消失
        # - 形成循环依赖：预测错误 → mask错误 → 无法学习修正
        #
        # 为什么使用target_conf软权重而非硬mask?
        # - target_conf是ground truth，可靠
        # - 平滑学习：所有bin都有梯度，只是按重要性加权
        # - 高斯标签自然反映bin重要性（峰值权重高，边缘权重低）
        # - 避免threshold选择问题
        #
        weights = target_conf  # [B, 2048], 范围[0, 1]
        
        # 逐元素MSE，然后加权平均
        mse_per_bin = F.mse_loss(pred_energy, target_energy, reduction='none')
        loss_energy = (weights * mse_per_bin).sum() / (weights.sum() + 1e-8)
        
        # 3. 稀疏性约束: 抑制假阳性
        # 鼓励网络输出低confidence，除非有明确证据
        loss_sparsity = pred_conf.mean()
        
        # 总损失
        total_loss = (self.conf_weight * loss_conf + 
                     self.energy_weight * loss_energy +
                     self.sparsity_weight * loss_sparsity)
        
        return {
            'total': total_loss,
            'confidence': loss_conf,
            'energy': loss_energy,
            'sparsity': loss_sparsity
        }
```

**设计说明**:

| 损失项 | 计算范围 | 权重 | 设计理由 |
|--------|---------|------|---------|
| **Confidence Loss** | 所有2048 bins | 1.0 | 主任务，学习存在性判断 |
| **Energy Loss** | 软加权(用target_conf) | 0.5 | 只在有音区域学习能量，但用软权重避免硬截断 |
| **Sparsity Loss** | 所有bins的均值 | 0.01 | 轻微抑制假阳性，鼓励低confidence输出 |

**关键设计决策**:
- ✅ 使用 `target_conf` 软权重（ground truth，可靠）
- ❌ 不使用 `pred_conf` mask（避免循环依赖和梯度消失）
- ❌ 不使用硬threshold mask（避免截断软标签边缘信息）

### 4.5 训练配置

```yaml
# config.yaml
training:
  epochs: 200
  batch_size: 64
  optimizer: AdamW
  lr: 1e-3
  scheduler: CosineAnnealingWarmRestarts
  weight_decay: 1e-4
  
data:
  samples_per_epoch: 100000
  synthesis:
    note_duration: 0.1  # 100ms
    polyphony_dist: [0.3, 0.3, 0.2, 0.15, 0.05]  # 1-5音概率
    f0_distribution: log_uniform  # 20-5000Hz对数均匀
    
validation:
  datasets:
    - MIR-1K  # 真实歌声数据
    - MedleyDB  # 真实乐器数据
    - synthesized_chords  # 合成和弦测试集
```

---

## 五、推理引擎集成

### 5.1 ONNX Runtime 跨平台配置

```cpp
// MLPitchDetector.h
#pragma once
#include <onnxruntime_cxx_api.h>
#include <juce_audio_basics/juce_audio_basics.h>

class MLPitchDetector {
public:
    struct Config {
        int sampleRate = 44100;
        int inputSamples = 4096;
        int numFreqBins = 2048;
        float minFreq = 20.0f;
        float maxFreq = 5000.0f;
        juce::File modelPath;
    };
    
    struct Detection {
        float frequency;
        float confidence;
        float energy;
        uint64_t frameIndex;
    };
    
    bool initialize(const Config& config);
    void process(const float* audioData, int numSamples);
    std::vector<Detection> getLatestResults() const;
    
    // 回调接口供UI使用
    using ResultsCallback = std::function<void(const std::vector<Detection>&)>;
    void setCallback(ResultsCallback cb);
    
private:
    // ONNX Runtime
    std::unique_ptr<Ort::Session> session_;
    Ort::Env env_{ORT_LOGGING_LEVEL_WARNING, "PitchDetector"};
    Ort::MemoryInfo memoryInfo_{Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)};
    
    // 预分配Tensor (避免推理时malloc)
    std::vector<float> inputBuffer_;
    std::vector<float> outputConf_;
    std::vector<float> outputEnergy_;
    
    // 异步推理
    std::thread inferenceThread_;
    moodycamel::BlockingReaderWriterQueue<std::vector<float>> inputQueue_{16};
    moodycamel::ReaderWriterQueue<std::vector<Detection>> outputQueue_{16};
    
    // 滑动窗口 - 每次从最新位置向前读取4096点
    juce::AudioBuffer<float> ringBuffer_;
    std::atomic<int> writePos_{0};
    static constexpr int inputSize_ = 4096;
    
    void inferenceLoop();
    std::vector<Detection> postProcess(const float* conf, const float* energy);
};
```

### 5.2 平台特定优化

#### macOS/iOS (CoreML)

```cpp
// 配置CoreML执行提供程序
Ort::SessionOptions sessionOptions;
OrtCoreMLExecutionProviderOptions coremlOptions;
coremlOptions.enable_on_subgraph = true;
coremlOptions.use_cpu_only = false;  // 允许ANE
coremlOptions.model_format = ORT_COREML_MODEL_FORMAT_MLPROGRAM;  // 使用ML Program

sessionOptions.AppendExecutionProvider_CoreML(coremlOptions);
```

#### Windows (DirectML)

```cpp
// DirectML for AMD/Intel GPU
Ort::SessionOptions sessionOptions;
OrtSessionOptionsAppendExecutionProvider_DML(sessionOptions, 0);  // GPU 0

// 或TensorRT for NVIDIA
OrtTensorRTProviderOptions trtOptions;
trtOptions.device_id = 0;
trtOptions.trt_max_workspace_size = 1ULL << 30;  // 1GB
sessionOptions.AppendExecutionProvider_TensorRT(trtOptions);
```

#### Android (NNAPI)

```cpp
// NNAPI GPU Delegate
Ort::SessionOptions sessionOptions;
uint32_t nnapiFlags = NNAPI_FLAG_USE_NONE;
sessionOptions.AppendExecutionProvider_Nnapi(nnapiFlags);
```

### 5.3 与现有代码集成

```cpp
// AudioEngine.cpp 集成示例
void AudioEngine::prepareToPlay(int samplesPerBlock, double sampleRate) {
    // 原有FFT/相位声码器初始化
    // ...
    
    // 新增ML检测器初始化
    MLPitchDetector::Config mlConfig;
    mlConfig.sampleRate = static_cast<int>(sampleRate);
    mlConfig.modelPath = getModelPathForCurrentPlatform();
    
    mlDetector_ = std::make_unique<MLPitchDetector>();
    if (!mlDetector_->initialize(mlConfig)) {
        DBG("ML Detector initialization failed, falling back to FFT");
        useML_ = false;
    } else {
        mlDetector_->setCallback([this](const auto& results) {
            // 线程安全地更新UI数据
            juce::MessageManager::callAsync([this, results]() {
                latestPitchResults_ = results;
                sendChangeMessage();
            });
        });
        useML_ = true;
    }
}

void AudioEngine::processBlock(juce::AudioBuffer<float>& buffer) {
    if (useML_) {
        // 写入RingBuffer
        ringBuffer_.write(buffer.getReadPointer(0), buffer.getNumSamples());
        
        // 每次音频回调都触发推理，从最新位置向前取4096点
        // 无需等待hop填充，保证最低延迟
        std::vector<float> frame(4096);
        ringBuffer_.readLatest(4096, frame.data());
        mlDetector_->submitFrame(frame.data(), 4096);
    }
    
    // 保留原有处理用于可视化或其他功能
    // ...
}
```

---

## 六、性能基准与验证

### 6.1 预期性能指标 (PitchNet 2.5M FP32)

| 平台 | 模型推理 | 端到端延迟 | 内存占用 | CPU占用 |
|------|---------|-----------|---------|---------|
| M1 MacBook Air (ANE+GPU) | 6-10ms | ~15ms | ~60MB | <5% |
| M2 iPad Pro | 7-12ms | ~18ms | ~55MB | <8% |
| iPhone 13 | 8-15ms | ~20ms | ~50MB | <10% |
| Windows (RTX3060) | 3-5ms | ~10ms | ~100MB | <3% |
| Windows (iGPU) | 12-20ms | ~25ms | ~70MB | <15% |
| Android (Snapdragon 8Gen2) | 10-18ms | ~22ms | ~65MB | <12% |

**INT8量化后 (PitchNet 600KB)**:
- 推理延迟: 上述数值 × 0.6
- 内存占用: 减少 ~40%

### 6.2 精度验证方法

**合成测试集**:
- 单音: 20-5000Hz，每半音一个样本，含泛音失谐
- 双音: 所有音程组合(小三度、大三度、纯四、纯五等)
- 和弦: 三和弦、七和弦、九和弦
- 噪声场景: SNR -10dB to 40dB

**评估指标**:
```python
# 准确率计算
def evaluate(predictions, ground_truth, tolerance_cents=5):
    correct = 0
    for pred, gt in zip(predictions, ground_truth):
        # 允许5 cents误差
        pred_cents = 1200 * np.log2(pred / 440)
        gt_cents = 1200 * np.log2(gt / 440)
        if np.abs(pred_cents - gt_cents) < tolerance_cents:
            correct += 1
    return correct / len(ground_truth)

# 多音评估: Precision/Recall/F1
def evaluate_polyphonic(pred_multi, gt_multi):
    # 使用镜像方法计算帧级F1
    pass
```

**目标精度**:
- 单音 (clean): > 99%
- 单音 (-10dB SNR): > 95%
- 双音 (纯五度): > 90%
- 三和弦: > 85%
- 七和弦: > 80%

---

## 七、数据存储与增强策略

### 7.1 存储格式演进

#### 从HDF5到NumPy Shards

| 阶段 | 格式 | 问题 | 解决方案 |
|------|------|------|---------|
| 初期 | HDF5 | 单文件过大，随机读取慢 | 分片存储 |
| 中期 | NumPy Shards | 文件数量多，管理复杂 | 统一目录结构 |
| 优化 | Shards + 内存映射 | 大数据集内存不足 | mmap按需加载 |

**NumPy Shards优势**:
- **分片存储**: 每片~30MB，避免单文件过大损坏
- **内存映射**: `np.load(mmap_mode='r')`，按需加载
- **顺序读取优化**: 适合训练时的批量读取
- **跨平台**: 纯NumPy，无依赖

**文件结构**:
```
TrainingData/sanity_shards/
├── meta.pkl              # 元数据 (样本数、分片数)
├── shard_00000.npz       # 分片0 (waveforms, confs, energies)
├── shard_00001.npz
└── ...
```

### 7.2 随机相位增强 (关键改进)

#### 问题发现

**现象**: 模型对窗口位置极度敏感

| 窗口位置 | 预测频率 | 置信度 | 误差 |
|---------|---------|--------|-----|
| 文件开始(0) | 220.0 Hz | 0.866 | ✅ |
| +512 samples | 226.6 Hz | 0.312 | ❌ |
| 中间位置 | 217.1 Hz | 0.695 | ⚠️ |

**原因**: 220Hz在4096样本中只有20.43个周期（非整数），窗口滑动导致频谱泄漏程度不同。

#### 解决方案

**数据增强**: 每个样本使用随机初始相位

```python
# 生成时随机相位
phase = rng.uniform(0, 2 * np.pi)
waveform = np.sin(2 * np.pi * freq * t + phase) * 0.8
```

这让模型学习对相位不敏感，适应streaming场景。

#### 效果验证

| 频率 | 固定相位 | 随机相位 | 改进 |
|------|---------|---------|-----|
| 220Hz | +13.9 ± 25.3 cents | **+1.3 ± 3.3 cents** | **8x** |
| 440Hz | +4.8 ± 12.6 cents | **+2.0 ± 2.3 cents** | **5x** |
| 880Hz | 0.0 ± 5.2 cents | **-1.6 ± 2.3 cents** | 更稳定 |

**结论**: 随机相位增强是解决streaming场景窗口敏感性问题的关键。

### 7.3 训练性能对比

| 配置 | 数据格式 | 每epoch | 50 epochs | 瓶颈 |
|------|---------|---------|----------|-----|
| HDF5 + CPU | 密集存储 | ~90s | ~75min | IO读取 |
| HDF5 + MPS | 密集存储 | ~7s | ~6min | 无 |
| **Shards + MPS** | **分片+内存** | **~4s** | **~3.5min** | **无** ✅ |

---

## 八、风险评估与缓解

| 风险 | 可能性 | 影响 | 缓解措施 |
|------|--------|------|---------|
| **移动端推理延迟超标** | 中 | 高 | 1. 准备INT8量化方案<br>2. 降级到更轻量模型<br>3. 降低刷新率至20fps |
| **低音乐器检测失败** | 中 | 高 | 1. 增加低频训练数据<br>2. 延长CQT低频窗口<br>3. 专用低频子网络 |
| **多音遮挡 (较高音被抑制)** | 中 | 中 | 1. 数据增强: 动态范围压缩<br>2. 损失函数: 加权高频样本<br>3. 后处理: 谐波解卷积 |
| **跨平台模型精度不一致** | 低 | 高 | 1. 统一ONNX opset<br>2. 平台间交叉验证<br>3. 提供fallback纯CPU路径 |
| **模型文件过大** | 低 | 中 | 1. INT8量化 (4x压缩)<br>2. 分平台下载 (仅下载所需模型)<br>3. 模型压缩 (剪枝/蒸馏) |

---

## 八、开发路线图

### Phase 1: 原型验证 (4周)

**Week 1-2: 模型开发**
- [ ] 搭建PyTorch训练框架
- [ ] 实现CQT前端 + TinyCQT-Net
- [ ] 合成数据生成器
- [ ] 单音精度达到5 cents

**Week 3-4: 多音与鲁棒性**
- [ ] 多音合成数据训练
- [ ] 噪声增强与域随机化
- [ ] 验证集评估 (目标: 双音90%)

**交付物**: 训练好的PyTorch模型 + 验证报告

### Phase 2: 推理优化 (3周)

**Week 5-6: 模型转换**
- [ ] ONNX导出与验证
- [ ] INT8量化 (训练后/量化感知)
- [ ] CoreML/NNAPI转换测试

**Week 7: 跨平台基准测试**
- [ ] 延迟测试 (各平台)
- [ ] 精度一致性验证
- [ ] 内存占用分析

**交付物**: 优化后的ONNX模型 + 各平台基准数据

### Phase 3: C++集成 (4周)

**Week 8-9: 核心引擎**
- [ ] ONNX Runtime集成
- [ ] 异步推理框架
- [ ] CQT GPU预处理 (Metal/Vulkan)

**Week 10-11: 平台适配**
- [ ] macOS/iOS CoreML路径
- [ ] Windows DirectML路径
- [ ] Android NNAPI路径

**Week 12: 系统集成**
- [ ] JUCE AudioEngine集成
- [ ] 原有FFT模式共存/切换
- [ ] UI回调与显示

**交付物**: 完整C++实现 + 集成测试

### Phase 4: 优化与发布 (3周)

**Week 13-14: 性能调优**
- [ ] 端到端延迟优化
- [ ] 内存泄漏检查
- [ ] 长时间稳定性测试

**Week 15: 文档与发布**
- [ ] API文档
- [ ] 用户手册更新
- [ ] App Store/Play Store提交

**交付物**: 发布版本 + 完整文档

---

## 九、资源需求

### 人力资源

| 角色 | 时间投入 | 职责 |
|------|---------|------|
| ML Engineer | 8周 | 模型设计、训练、调优 |
| C++ Developer | 7周 | 推理引擎、跨平台集成 |
| Audio DSP Engineer | 3周 | CQT实现、轻量级后处理 |
| QA Engineer | 2周 | 测试、基准测试 |

### 计算资源

- **训练**: AWS g4dn.xlarge (T4 GPU) 或同等，约500小时
- **基准测试**: 
  - Mac: M1/M2/M3 设备各1台
  - iOS: iPhone 12/13/15 各1台
  - Android: 中端(骁龙7系)、旗舰(骁龙8系) 各1台
  - Windows: NVIDIA/AMD/Intel GPU 各1台

### 软件/服务

- ONNX Runtime (Apache 2.0, 免费)
- CoreML Tools (Apple, 免费)
- JUCE (GPL/商业授权)
- 训练数据: 合成生成 (无需购买)

---

## 十、结论与建议

### 方案优势

1. **精度保障**: CQT + 对数频率bin设计天然满足5 cents要求
2. **实时性**: 轻量网络 + GPU异步推理确保30fps+
3. **多音分离**: 注意力机制 + 数据驱动优于手工阈值
4. **可维护性**: 减少魔法值，通过数据迭代改进
5. **可扩展性**: 模型更新无需修改C++代码

### 关键成功因素

1. **数据质量**: 合成数据必须准确模拟真实乐器泛音结构
2. **量化策略**: INT8量化需仔细验证，避免精度损失
3. **异步架构**: 避免推理阻塞音频线程
4. **渐进交付**: 先单音后多音，降低技术风险

### 建议的下一步行动

1. **立即开始**: Phase 1模型原型开发
2. **并行准备**: 收集真实乐器样本用于验证
3. **技术预研**: 在目标设备上测试ONNX Runtime CoreML/NNAPI延迟
4. **风险对冲**: 保留现有FFT算法作为fallback模式

---

**报告编制**: AI Assistant  
**审核**: [待项目负责人审核]  
**版本历史**:  
- v1.0 (2026-02-16): 初始版本

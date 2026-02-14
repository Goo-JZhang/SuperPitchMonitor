# 测试音频文件设计

## 设计变更

从**代码生成测试信号**改为**外部脚本生成 WAV 文件**。

## 目录结构

```
Resources/
└── TestAudio/
    ├── sine_220hz.wav          # A3 正弦波
    ├── sine_440hz.wav          # A4 正弦波
    ├── sine_880hz.wav          # A5 正弦波
    ├── chord_c_major.wav       # C 大调和弦
    ├── chord_g_major.wav       # G 大调和弦
    ├── chord_a_minor.wav       # A 小调和弦
    ├── sweep_low_high.wav      # 100Hz → 2000Hz 扫频
    ├── sweep_high_low.wav      # 2000Hz → 100Hz 扫频
    ├── sweep_full_range.wav    # 50Hz → 8000Hz 全频段扫频
    ├── scale_c_major.wav       # C 大调音阶
    ├── white_noise.wav         # 白噪声
    ├── pink_noise.wav          # 粉红噪声
    ├── piano_like_c.wav        # 钢琴风格 C 和弦
    └── guitar_strum_e.wav      # 吉他 E 和弦扫弦
```

## 生成脚本

**文件**: `scripts/generate_test_audio.py`

### 使用方法

```bash
cd scripts
python generate_test_audio.py

# 指定输出目录
python generate_test_audio.py --output /path/to/output
```

### 依赖

```bash
pip install numpy
```

## 添加自定义测试音频

1. 将 WAV/MP3/FLAC/AIFF/OGG 文件复制到 `Resources/TestAudio/`
2. 重启程序或点击 "Refresh" 按钮
3. 文件会出现在下拉框中

## 代码变更

### FileAudioInput
- **移除**: `generateTestSignal()` 方法
- **移除**: `TestSignal` 枚举
- **保留**: `getAvailableTestFiles()` - 扫描目录获取文件列表
- **保留**: `loadTestFile()` - 加载指定文件

### InputSourceSelector
- 当选择 "File Playback" 时显示文件下拉框
- 下拉框显示 `Resources/TestAudio/` 目录中的所有音频文件名
- 默认自动加载第一个文件

### MainComponent
- 集成 InputSourceSelector 回调
- 当输入源改变时更新 AudioEngine

## 优点

1. **灵活性**: 可以轻松添加/替换测试音频
2. **真实性**: 可以使用真实录制的音频文件
3. **可维护性**: 测试音频生成逻辑独立于程序
4. **可扩展性**: 支持多种音频格式

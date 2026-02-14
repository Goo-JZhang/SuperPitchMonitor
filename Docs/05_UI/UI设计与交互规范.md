# UI 设计与交互规范

## 1. 设计理念

### 1.1 核心设计原则

1. **即时反馈**: 音高检测和频谱显示必须实时响应，延迟感知 < 50ms
2. **清晰易读**: 在演出环境下也能快速读取关键信息
3. **专业精准**: 满足音乐专业人士的需求，提供精确数据
4. **简约高效**: 减少视觉干扰，聚焦核心功能

### 1.2 视觉风格

- **深色主题**: 适合舞台和低光环境
- **高对比度**: 确保在各种光照条件下可读
- **色彩编码**: 使用颜色表示音高准确度

---

## 2. 界面布局

### 2.1 主界面结构

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  Status Bar (状态栏)                                                        │
│  [🔴 REC] [⚡高性能] [🔋85%] [📶-12dB]                          [⚙️设置] │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                                                                     │   │
│  │                    频谱显示器 (Spectrum Display)                     │   │
│  │                                                                     │   │
│  │     ┌─────────────────────────────────────────────────────────┐    │   │
│  │  dB │                                                         │    │   │
│  │  0  │▓▓▓▓▓▓▓░░░░░▓▓▓▓░░░░░░░░░░▓▓▓▓▓▓░░░░░░░░░░░░░░░░░░░░░│    │   │
│  │ -20 │▓▓▓▓▓░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░│    │   │
│  │ -40 │░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░│    │   │
│  │ -60 │░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░│    │   │
│  │     └─────────────────────────────────────────────────────────┘    │   │
│  │        100Hz        500Hz        1kHz        4kHz       8kHz       │   │
│  │                                                                     │   │
│  │        ▲ 检测到的峰值频率标记                                        │   │
│  │       ╱ ╲                                                           │   │
│  │      ╱   ╲  谐波指示线                                              │   │
│  │                                                                     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    音高显示器 (Pitch Display)                        │   │
│  │                                                                     │   │
│  │   ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐          │   │
│  │   │    A#    │  │    C#    │  │    F     │  │   ...    │          │   │
│  │   │   440.0  │  │   554.4  │  │   698.5  │  │          │          │   │
│  │   │   -2¢    │  │   +5¢    │  │   +1¢    │  │          │          │   │
│  │   │  ████░░  │  │  ██████  │  │  ███░░░  │  │          │          │   │
│  │   │  置信度  │  │  置信度  │  │  置信度  │  │          │          │   │
│  │   └──────────┘  └──────────┘  └──────────┘  └──────────┘          │   │
│  │                                                                     │   │
│  │   多音高卡片显示 (最多显示6个音)                                     │   │
│  │                                                                     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    调音指示器 (Tuner View)                           │   │
│  │                                                                     │   │
│  │                    │                                                │   │
│  │         ♭    ──────┼──────    ♯                                    │   │
│  │         -50   -25  │  +25   +50   音分                              │   │
│  │                    ▲                                                │   │
│  │                 当前偏差                                            │   │
│  │                                                                     │   │
│  │   [C4 ▲]  主检测音高                                                │   │
│  │                                                                     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 详细组件规范

#### 2.2.1 频谱显示器 (SpectrumDisplay)

```cpp
class SpectrumDisplay : public Component, private Timer
{
public:
    struct Config {
        Colour backgroundColour = Colours::black;
        Colour gridColour = Colours::darkgrey;
        Colour spectrumColour = Colour::fromRGB(0, 200, 255);
        Colour peakColour = Colours::yellow;
        
        int numBands = 128;           // 频带数量
        float minFreq = 20.0f;        // 最低显示频率
        float maxFreq = 8000.0f;      // 最高显示频率
        float minDb = -80.0f;         // 最小 dB 值
        float maxDb = 0.0f;           // 最大 dB 值
        
        bool showGrid = true;
        bool showPeaks = true;
        bool useLogScale = true;      // 频率轴对数刻度
    };
    
    void paint(Graphics& g) override
    {
        // 绘制背景
        g.fillAll(config_.backgroundColour);
        
        // 绘制网格
        if (config_.showGrid)
            drawGrid(g);
        
        // 绘制频谱
        drawSpectrum(g);
        
        // 绘制峰值标记
        if (config_.showPeaks)
            drawPeakMarkers(g);
    }
    
    void updateSpectrum(const std::vector<float>& frequencies,
                       const std::vector<float>& magnitudes)
    {
        currentFrequencies_ = frequencies;
        currentMagnitudes_ = magnitudes;
        
        // 检测峰值
        detectPeaks();
        
        repaint();
    }
    
private:
    Config config_;
    std::vector<float> currentFrequencies_;
    std::vector<float> currentMagnitudes_;
    std::vector<Peak> detectedPeaks_;
    
    void drawGrid(Graphics& g)
    {
        g.setColour(config_.gridColour.withAlpha(0.3f));
        
        // 垂直频率线 (对数刻度)
        const float freqMarkers[] = {100, 200, 500, 1000, 2000, 4000, 8000};
        for (float freq : freqMarkers)
        {
            float x = freqToX(freq);
            g.drawVerticalLine((int)x, 0.0f, (float)getHeight());
            
            // 频率标签
            g.setFont(12.0f);
            String label = freq >= 1000 ? String(freq/1000, 1) + "k" : String((int)freq);
            g.drawText(label, (int)x - 20, getHeight() - 20, 40, 20, Justification::centred);
        }
        
        // 水平 dB 线
        for (int db = (int)config_.minDb; db <= (int)config_.maxDb; db += 20)
        {
            float y = dbToY((float)db);
            g.drawHorizontalLine((int)y, 0.0f, (float)getWidth());
        }
    }
    
    void drawSpectrum(Graphics& g)
    {
        if (currentFrequencies_.empty()) return;
        
        Path spectrumPath;
        spectrumPath.startNewSubPath(0.0f, (float)getHeight());
        
        // 构建频谱路径
        for (size_t i = 0; i < currentFrequencies_.size(); ++i)
        {
            float x = freqToX(currentFrequencies_[i]);
            float y = magnitudeToY(currentMagnitudes_[i]);
            spectrumPath.lineTo(x, y);
        }
        
        spectrumPath.lineTo((float)getWidth(), (float)getHeight());
        spectrumPath.closeSubPath();
        
        // 渐变填充
        ColourGradient gradient(
            config_.spectrumColour.withAlpha(0.8f), 0.0f, 0.0f,
            config_.spectrumColour.withAlpha(0.1f), 0.0f, (float)getHeight(),
            false
        );
        
        g.setGradientFill(gradient);
        g.fillPath(spectrumPath);
        
        // 绘制频谱线
        g.setColour(config_.spectrumColour);
        g.strokePath(spectrumPath, PathStrokeType(2.0f));
    }
    
    void drawPeakMarkers(Graphics& g)
    {
        g.setColour(config_.peakColour);
        
        for (const auto& peak : detectedPeaks_)
        {
            float x = freqToX(peak.frequency);
            float y = magnitudeToY(peak.magnitude);
            
            // 绘制标记三角形
            Path marker;
            marker.addTriangle(x, y - 10, x - 5, y, x + 5, y);
            g.fillPath(marker);
            
            // 频率标签
            g.setFont(11.0f);
            String freqLabel = String(peak.frequency, 1) + " Hz";
            g.drawText(freqLabel, (int)x - 30, (int)y - 25, 60, 15, Justification::centred);
        }
    }
    
    float freqToX(float freq) const
    {
        if (config_.useLogScale)
        {
            float logMin = std::log10(config_.minFreq);
            float logMax = std::log10(config_.maxFreq);
            float logFreq = std::log10(jlimit(config_.minFreq, config_.maxFreq, freq));
            return (logFreq - logMin) / (logMax - logMin) * getWidth();
        }
        else
        {
            return (freq - config_.minFreq) / (config_.maxFreq - config_.minFreq) * getWidth();
        }
    }
    
    float magnitudeToY(float magnitude) const
    {
        float db = 20.0f * std::log10(magnitude + 1e-10f);
        db = jlimit(config_.minDb, config_.maxDb, db);
        return getHeight() * (1.0f - (db - config_.minDb) / (config_.maxDb - config_.minDb));
    }
};
```

#### 2.2.2 音高显示卡片 (PitchCard)

```cpp
class PitchCard : public Component
{
public:
    struct PitchData {
        String noteName;        // 音符名称 (如 "A#")
        int octave;             // 八度
        float frequency;        // 精确频率
        float cents;            // 音分偏差
        float confidence;       // 置信度 0-1
        int harmonicCount;      // 谐波数量
    };
    
    void setPitchData(const PitchData& data)
    {
        data_ = data;
        repaint();
    }
    
    void paint(Graphics& g) override
    {
        auto bounds = getLocalBounds().toFloat().reduced(4.0f);
        
        // 背景 (根据置信度着色)
        float hue = jlimit(0.0f, 0.33f, data_.confidence * 0.33f);  // 红 -> 绿
        Colour bgColour = Colour::fromHSV(hue, 0.8f, 0.2f, 1.0f);
        Colour borderColour = Colour::fromHSV(hue, 0.8f, 0.8f, 1.0f);
        
        g.setColour(bgColour);
        g.fillRoundedRectangle(bounds, 8.0f);
        
        g.setColour(borderColour);
        g.drawRoundedRectangle(bounds, 8.0f, 2.0f);
        
        // 音符名称
        g.setColour(Colours::white);
        g.setFont(28.0f);
        g.drawText(data_.noteName + String(data_.octave), 
                   bounds.removeFromTop(bounds.getHeight() * 0.4f),
                   Justification::centred);
        
        // 频率
        g.setFont(14.0f);
        g.setColour(Colours::lightgrey);
        g.drawText(String(data_.frequency, 1) + " Hz",
                   bounds.removeFromTop(20.0f),
                   Justification::centred);
        
        // 音分偏差 (带颜色)
        String centsText = (data_.cents >= 0 ? "+" : "") + String(data_.cents, 1) + "¢";
        Colour centsColour = std::abs(data_.cents) < 5.0f ? Colours::green :
                             std::abs(data_.cents) < 20.0f ? Colours::yellow : Colours::red;
        g.setColour(centsColour);
        g.drawText(centsText, bounds.removeFromTop(20.0f), Justification::centred);
        
        // 置信度条
        float barY = bounds.getCentreY();
        float barWidth = bounds.getWidth() * 0.8f;
        float barHeight = 6.0f;
        float barX = bounds.getCentreX() - barWidth / 2;
        
        // 背景条
        g.setColour(Colours::darkgrey);
        g.fillRoundedRectangle(barX, barY - barHeight/2, barWidth, barHeight, 3.0f);
        
        // 进度条
        g.setColour(Colour::fromHSV(data_.confidence * 0.33f, 0.8f, 1.0f, 1.0f));
        g.fillRoundedRectangle(barX, barY - barHeight/2, 
                               barWidth * data_.confidence, barHeight, 3.0f);
    }
    
private:
    PitchData data_;
};
```

#### 2.2.3 调音指示器 (TunerDisplay)

```cpp
class TunerDisplay : public Component
{
public:
    void setTargetPitch(float midiNote, float centsDeviation)
    {
        targetMidi_ = midiNote;
        centsDeviation_ = centsDeviation;
        repaint();
    }
    
    void paint(Graphics& g) override
    {
        auto bounds = getLocalBounds().toFloat();
        float centreX = bounds.getCentreX();
        float centreY = bounds.getCentreY();
        
        // 绘制刻度弧
        Path arc;
        arc.addCentredArc(centreX, centreY + 50, 120, 80, 0, 
                          -2.5f, 2.5f, true);
        g.setColour(Colours::darkgrey);
        g.strokePath(arc, PathStrokeType(4.0f));
        
        // 绘制刻度标记
        for (int i = -50; i <= 50; i += 10)
        {
            float angle = i / 50.0f * 2.5f;
            float x1 = centreX + std::sin(angle) * 100;
            float y1 = centreY + 50 - std::cos(angle) * 60;
            float x2 = centreX + std::sin(angle) * 120;
            float y2 = centreY + 50 - std::cos(angle) * 80;
            
            g.setColour(i == 0 ? Colours::green : Colours::grey);
            g.drawLine(x1, y1, x2, y2, i == 0 ? 3.0f : 1.0f);
            
            // 标签
            if (i % 25 == 0)
            {
                float lx = centreX + std::sin(angle) * 140;
                float ly = centreY + 50 - std::cos(angle) * 100;
                g.setFont(12.0f);
                g.setColour(Colours::lightgrey);
                g.drawText(String(i), (int)lx - 15, (int)ly - 10, 30, 20, 
                          Justification::centred);
            }
        }
        
        // 绘制指针
        float needleAngle = jlimit(-50.0f, 50.0f, centsDeviation_) / 50.0f * 2.5f;
        float nx = centreX + std::sin(needleAngle) * 110;
        float ny = centreY + 50 - std::cos(needleAngle) * 70;
        
        // 指针颜色根据偏差
        Colour needleColour = std::abs(centsDeviation_) < 5.0f ? Colours::green :
                              std::abs(centsDeviation_) < 20.0f ? Colours::yellow : Colours::red;
        
        Path needle;
        needle.addTriangle(centreX, centreY + 50 - 10, nx - 4, ny + 10, nx + 4, ny + 10);
        g.setColour(needleColour);
        g.fillPath(needle);
        
        // 当前音高显示
        g.setFont(32.0f);
        g.setColour(Colours::white);
        int note = (int)targetMidi_ % 12;
        int octave = (int)targetMidi_ / 12 - 1;
        const char* noteNames[] = {"C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"};
        g.drawText(String(noteNames[note]) + String(octave), 
                   (int)centreX - 50, (int)centreY - 80, 100, 40, 
                   Justification::centred);
        
        // 音准状态
        String status = std::abs(centsDeviation_) < 5.0f ? "IN TUNE" :
                        centsDeviation_ > 0 ? "SHARP" : "FLAT";
        g.setFont(14.0f);
        g.setColour(needleColour);
        g.drawText(status, (int)centreX - 50, (int)centreY - 40, 100, 20, 
                   Justification::centred);
    }
    
private:
    float targetMidi_ = 0.0f;
    float centsDeviation_ = 0.0f;
};
```

---

## 3. 设置界面

### 3.1 设置面板布局

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  Settings                          [✕]                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  Detection Settings                                                  │   │
│  │                                                                     │   │
│  │  检测范围:  [──────────●──────────────]  50Hz - 2000Hz              │   │
│  │                                                                     │   │
│  │  灵敏度:    [────●────────────────────]  中                          │   │
│  │                                                                     │   │
│  │  最大音数:  [4] ▲▼                                                  │   │
│  │                                                                     │   │
│  │  [✓] 启用多音检测                                                    │   │
│  │  [✓] 启用谐波增强                                                    │   │
│  │  [ ] 启用NMF (高性能模式)                                            │   │
│  │                                                                     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  Display Settings                                                    │   │
│  │                                                                     │   │
│  │  主题:     [深色 ▼]                                                 │   │
│  │  刷新率:   [60fps ▼]                                                │   │
│  │  频带数量: [128 ▼]                                                  │   │
│  │                                                                     │   │
│  │  [✓] 显示音分偏差                                                    │   │
│  │  [✓] 显示置信度                                                      │   │
│  │  [ ] 显示频谱网格                                                    │   │
│  │                                                                     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  Performance Settings                                                │   │
│  │                                                                     │   │
│  │  性能模式:  [平衡 ▼]                                                │   │
│  │                                                                     │   │
│  │  [✓] 自适应质量                                                      │   │
│  │  [✓] 低电量时降频                                                    │   │
│  │                                                                     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│                              [  恢复默认  ]  [  保存  ]                      │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 4. 交互设计

### 4.1 手势操作

| 手势 | 操作 | 功能 |
|-----|------|-----|
| 单指点击 | 频谱区域 | 查看该频率详情 |
| 双指捏合 | 频谱区域 | 缩放频率范围 |
| 水平滑动 | 频谱区域 | 调整参考频率 |
| 长按 | 音高卡片 | 显示详细信息 |
| 下拉 | 主界面 | 打开快速设置 |

### 4.2 状态指示

```cpp
enum class DetectionStatus {
    Idle,           // 灰色 - 未开始检测
    Listening,      // 绿色呼吸 - 监听中
    Detecting,      // 黄色 - 检测到信号分析中
    Detected,       // 绿色 - 成功检测到音高
    LowConfidence,  // 橙色 - 低置信度结果
    Error           // 红色 - 错误状态
};

class StatusIndicator : public Component
{
public:
    void setStatus(DetectionStatus status)
    {
        status_ = status;
        startAnimation();
    }
    
    void paint(Graphics& g) override
    {
        Colour colour;
        switch (status_)
        {
            case DetectionStatus::Idle: colour = Colours::grey; break;
            case DetectionStatus::Listening: colour = Colours::green.withAlpha(0.5f); break;
            case DetectionStatus::Detecting: colour = Colours::yellow; break;
            case DetectionStatus::Detected: colour = Colours::green; break;
            case DetectionStatus::LowConfidence: colour = Colours::orange; break;
            case DetectionStatus::Error: colour = Colours::red; break;
        }
        
        g.setColour(colour);
        
        // 呼吸效果
        if (status_ == DetectionStatus::Listening)
        {
            float pulse = (std::sin(animationPhase_) + 1.0f) * 0.5f;
            g.fillEllipse(getLocalBounds().toFloat().reduced(2.0f * pulse));
        }
        else
        {
            g.fillEllipse(getLocalBounds().toFloat());
        }
    }
    
private:
    DetectionStatus status_ = DetectionStatus::Idle;
    float animationPhase_ = 0.0f;
    
    void startAnimation()
    {
        // 启动定时器更新 animationPhase_
    }
};
```

---

## 5. 颜色系统

```cpp
namespace Colours {
    // 主色调
    const Colour primary = Colour::fromRGB(0, 200, 255);
    const Colour secondary = Colour::fromRGB(150, 100, 255);
    
    // 背景
    const Colour backgroundDark = Colour::fromRGB(20, 20, 25);
    const Colour backgroundCard = Colour::fromRGB(35, 35, 42);
    
    // 状态色
    const Colour success = Colour::fromRGB(50, 200, 100);
    const Colour warning = Colour::fromRGB(255, 180, 50);
    const Colour error = Colour::fromRGB(255, 80, 80);
    
    // 音准指示
    const Colour inTune = Colour::fromRGB(0, 255, 100);
    const Colour sharp = Colour::fromRGB(255, 200, 0);
    const Colour flat = Colour::fromRGB(255, 100, 100);
    
    // 频谱渐变
    const Colour spectrumLow = Colour::fromRGB(0, 255, 200);
    const Colour spectrumMid = Colour::fromRGB(255, 200, 0);
    const Colour spectrumHigh = Colour::fromRGB(255, 50, 100);
}
```

---

## 6. 响应式布局

### 6.1 屏幕适配

```cpp
class ResponsiveLayout
{
public:
    enum class ScreenSize {
        Compact,    // < 360dp 宽 (小屏手机)
        Medium,     // 360-600dp (标准手机)
        Expanded,   // > 600dp (平板/大屏)
        Large       // > 900dp (横屏平板)
    };
    
    static ScreenSize getScreenSize(int widthDp)
    {
        if (widthDp < 360) return ScreenSize::Compact;
        if (widthDp < 600) return ScreenSize::Medium;
        if (widthDp < 900) return ScreenSize::Expanded;
        return ScreenSize::Large;
    }
    
    static void configureLayout(Component& mainComponent, ScreenSize size)
    {
        switch (size)
        {
            case ScreenSize::Compact:
                // 紧凑布局: 频谱占50%，音高卡片垂直排列
                break;
            case ScreenSize::Medium:
                // 标准布局
                break;
            case ScreenSize::Expanded:
                // 平板布局: 频谱和音高并排
                break;
            case ScreenSize::Large:
                // 横屏布局: 扩展信息显示
                break;
        }
    }
};
```

---

## 7. 动画规范

| 动画 | 时长 | 缓动 | 说明 |
|-----|------|-----|------|
| 音高卡片出现 | 200ms | Ease-out | 缩放 + 淡入 |
| 频谱更新 | 16ms | Linear | 每帧平滑过渡 |
| 指针移动 | 100ms | Ease-in-out | 调音器指针 |
| 状态切换 | 300ms | Ease-out | 状态指示器颜色变化 |
| 面板滑动 | 250ms | Decelerate | 设置面板 |

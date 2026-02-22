# WaterfallPitches 自动追踪设计

## 1. 功能概述

### 1.1 目的
在音高瀑布流显示（PitchWaterfallDisplay）中，当用户无操作时，自动调整视图使频谱点/线保持在屏幕中间1/3区域，确保用户始终能看到最重要的音高信息。

### 1.2 核心概念
- **中间1/3区域（Center Zone）**: 屏幕垂直方向（频率轴Y轴）中间1/3区域
- **全局最佳频谱点**: 基于置信度和能量的综合评分确定
- **用户操作冷却期**: 10秒无操作后恢复自动追踪

**注意**: 横轴（X轴，时间）自动滚动，不参与区域划分。自动追踪只针对频率（Y轴）。

---

## 2. 界面区域定义

**只针对频率轴（Y轴）进行区域划分**，时间轴（X轴）自动滚动。

```
│  高频率
│
│ ┌──────────────────────────────────────┐
│ │           顶部1/3区域                 │  ◄── 高频率区域
│ │          (Top 1/3 Zone)              │
│ ├──────────────────────────────────────┤
│ │                                      │
│ │         中间1/3区域 (目标)            │  ◄── 目标区域：频谱点应对准这里
│ │        (Center Zone)                 │      将最佳频谱点移到视图正中心
│ │                                      │
│ ├──────────────────────────────────────┤
│ │           底部1/3区域                 │  ◄── 低频率区域
│ │        (Bottom 1/3 Zone)             │
│ └──────────────────────────────────────┘
│
│  低频率
│
└──────────────────────────────────────────► 时间 →

图1: 频率轴区域划分（仅Y轴）
```

### 2.1 区域判定算法（仅Y轴）
```cpp
// 只判断频率（Y轴）是否在中心区域（中间1/3）
bool isInCenterZone(float freq, float viewCenterFreq, float viewHeightSemitones) {
    float midi = freqToMidi(freq);
    float centerMidi = freqToMidi(viewCenterFreq);
    float offset = midi - centerMidi;
    
    // 中心区域 = 视图高度的1/3（上下各1/6）
    float minOffset = -(viewHeightSemitones / 6.0f);  // -1/6
    float maxOffset = (viewHeightSemitones / 6.0f);   // +1/6
    
    return (offset >= minOffset && offset <= maxOffset);
}

// 计算频率到中心区域边界的距离（仅Y轴）
// 返回0表示已在中心区域内
float distanceToCenterZone(float freq, float viewCenterFreq, float viewHeightSemitones) {
    float midi = freqToMidi(freq);
    float centerMidi = freqToMidi(viewCenterFreq);
    float offset = midi - centerMidi;
    
    float minOffset = -(viewHeightSemitones / 6.0f);
    float maxOffset = (viewHeightSemitones / 6.0f);
    
    if (offset < minOffset)
        return minOffset - offset;  // 低于中心区域
    else if (offset > maxOffset)
        return offset - maxOffset;  // 高于中心区域
    else
        return 0.0f;  // 在中心区域内
}
```

---

## 3. 自动追踪策略

### 3.1 决策流程图

**改进版流程**（只使用当前帧数据 + 智能暂停）：

```
                         ┌─────────────────────────┐
                         │   AutoTrack = ON ?      │
                         └────────────┬────────────┘
                                      │
                    ┌─────────────────┴──────────────────┐
                    ▼                                      ▼
              ┌─────────┐                            ┌─────────┐
              │   OFF   │                            │   ON    │
              │ 手动模式 │                            │ 检查用户 │
              │ 不追踪  │                            │ 操作状态 │
              └─────────┘                            └────┬────┘
                                                          │
                              ┌───────────────────────────┼───────────┐
                              │                           │           │
                              ▼                           ▼           ▼
                        ┌─────────┐                 ┌─────────┐  ┌─────────┐
                        │ 用户操作 │                 │ 用户空闲 │  │ 冷却中  │
                        │        │                 │        │  │        │
                        └────┬────┘                 └────┬────┘  └────┬────┘
                             │                           │           │
                             ▼                           ▼           ▼
                        ┌─────────┐                 ┌─────────┐  ┌─────────┐
                        │ 暂停追踪 │                 │ 当前帧  │  │ 暂停   │
                        │ 重置10s │                 │ 有检测? │  │ 等待   │
                        │ 计时器  │                 └───┬─────┘  └─────────┘
                        └─────────┘                     │
                                                  ┌─────┴─────┐
                                                  ▼           ▼
                                            ┌─────────┐  ┌─────────┐
                                            │   有    │  │   无    │
                                            │ 正常追踪 │  │ 检查   │
                                            │ Case1/2 │  │ 待定目标 │
                                            └────┬────┘  │ 或暂停  │
                                                 │       └────┬────┘
                          ┌──────────────────────┼────────────┘
                          │                      │
                          ▼                      ▼
                   ┌────────────┐       ┌────────────────┐
                   │ 全屏无点?  │       │ 连续3帧无检测?  │
                   │ Case 1     │       │                │
                   └─────┬──────┘       └───────┬────────┘
                         │                      │
              ┌──────────┴──────────┐          ▼
              ▼                     ▼    ┌────────────┐
        ┌──────────┐          ┌──────────┤   暂停追踪  │
        │ 有点在   │          │ 屏幕外有 │  等待新检测 │
        │ 屏幕外   │          │ 可信点?  │            │
        └────┬─────┘          └────┬─────┘            │
             │                     │                  │
             ▼                     ▼                  │
      ┌──────────────┐      ┌──────────────┐         │
      │ 立即跳转到   │      │ 有→跳转      │         │
      │ 最佳频谱点   │      │ 无→继续待定  │         │
      └──────────────┘      │   目标       │         │
                            └──────────────┘         │
                                                      │
                                                      ▼
                                               ┌────────────┐
                                               │ 继续接近   │
                                               │ 原定目标   │
                                               └────────────┘
```

**关键改进**:
1. **只使用当前帧数据**: 不依赖历史记录，避免追踪到旧音符
2. **智能暂停**: 连续3帧无检测才暂停追踪（避免瞬时噪音干扰）
3. **待定目标**: 追踪过程中短暂无检测时，继续接近原定目标
4. **ON/OFF切换**: 用户可完全关闭自动追踪，完全手动控制

### 3.2 三种处理情形

#### 情形1: 全屏无频谱点 (Global Jump)
**触发条件**: 当前屏幕范围内没有任何频谱点（**只使用当前帧数据，不包含历史**）

**处理逻辑**:
1. 扫描**当前帧**的所有频谱点
2. 选取**置信度最高**的点
3. 置信度相同时，比较**能量大小**
4. 直接将视图中心跳转到该点位置

**重要**: 不使用历史数据，避免追踪到已经停止演奏的音符

**智能暂停机制**:
- 如果当前帧没有检测到任何可信频谱点，检查连续无检测帧数
- 连续3帧无检测 → 暂停追踪，等待新的有效检测
- 1-2帧无检测 → 可能继续接近待定目标（如果正在追踪中）

**评分公式**:
```cpp
float score = point.confidence * 1000.0f + point.energy;
// 置信度权重更高 (0-1000范围)
// 能量作为 tie-breaker
```

**伪代码**:
```cpp
void handleGlobalJump() {
    PitchCandidate* best = nullptr;
    float bestScore = -1.0f;
    
    for (auto& pitch : allPitches) {
        float score = pitch.confidence * 1000.0f + pitch.amplitude;
        if (score > bestScore) {
            bestScore = score;
            best = &pitch;
        }
    }
    
    if (best) {
        targetCenterFreq = best->frequency;
        targetCenterTime = best->timestamp;
        jumpToTarget(); // 立即跳转
    }
}
```

---

#### 情形2: 中间区域无点但屏幕有点 (Smooth Approach)
**触发条件**: 中间1/3区域无频谱点，但屏幕其他区域有

**处理逻辑**:
1. 计算屏幕内所有频谱点到中心区域的**欧几里得距离**
2. 选取**距离最近**的点
3. 距离相同时，比较**置信度**
4. 置信度相同时，比较**能量**
5. 以**平滑动画**方式将目标点移入中心区域（到达边界即可停止）

**距离计算**（仅Y轴 - 频率）:
```cpp
float distanceToCenterZone(float freq, float viewCenterFreq, float viewHeightSemitones) {
    float midi = freqToMidi(freq);
    float centerMidi = freqToMidi(viewCenterFreq);
    float offset = midi - centerMidi;
    
    float minOffset = -(viewHeightSemitones / 6.0f);  // -1/6
    float maxOffset = (viewHeightSemitones / 6.0f);   // +1/6
    
    if (offset < minOffset)
        return minOffset - offset;  // 低于中心区域：返回到下边界的距离
    else if (offset > maxOffset)
        return offset - maxOffset;  // 高于中心区域：返回到上边界的距离
    else
        return 0.0f;  // 在中心区域内
}
```

**排序优先级**:
```cpp
bool compareCandidates(const PitchCandidate& a, const PitchCandidate& b) {
    float distA = distanceToCenter(a);
    float distB = distanceToCenter(b);
    
    if (std::abs(distA - distB) > 0.001f)
        return distA < distB;  // 距离优先
    
    if (std::abs(a.confidence - b.confidence) > 0.001f)
        return a.confidence > b.confidence;  // 置信度次之
    
    return a.amplitude > b.amplitude;  // 能量最后
}
```

**平滑移动算法**:
```cpp
void smoothApproach(const PitchCandidate& target) {
    // 目标：将目标点移动到中心区域边界内
    
    float targetFreq = target.frequency;
    float targetTime = target.timestamp;
    
    // 计算需要的视图偏移
    float viewCenterFreq = getViewCenterFreq();
    float viewCenterTime = getViewCenterTime();
    
    // 缓动函数（ease-in-out）
    auto easeInOut = [](float t) {
        return t < 0.5f ? 2.0f * t * t : 1.0f - std::pow(-2.0f * t + 2.0f, 2.0f) / 2.0f;
    };
    
    // 动画参数
    const float approachSpeed = 0.15f;  // 每帧移动比例
    const float minSpeed = 5.0f;        // 最小频率跨度/秒
    const float stopThreshold = 0.01f;  // 到达阈值
    
    // 检查是否已到达中心区域边界
    if (isInCenterZone(target)) {
        stopMovement();
        return;
    }
    
    // 计算新视图中心
    float newCenterFreq = lerp(viewCenterFreq, targetFreq, approachSpeed);
    float newCenterTime = lerp(viewCenterTime, targetTime, approachSpeed);
    
    setViewCenter(newCenterFreq, newCenterTime);
}
```

---

#### 情形3: 中间区域已有频谱点 (Hold Position)
**触发条件**: 中间1/3区域内已存在至少一个频谱点

**处理逻辑**: 不做任何操作，保持当前视图

---

## 4. 用户交互处理

### 4.0 初始状态和启动行为

**Timer 初始化**:
- 冷却期初始状态设为**已过期**（`lastInteractionTime_ = Time(0)`）
- 这样应用启动后自动追踪立即生效，无需等待

**Start 按钮点击**:
- 当用户点击 Start 开始音频分析时，重置冷却期为过期状态
- 确保开始分析后自动追踪立即生效，第一时间追踪到正确频谱
- 实现：`pitchWaterfall_->resetAutoTrackerCooldown()`

### 4.1 操作检测机制

**触发自动追踪暂停的操作**:

**桌面端 (Desktop)**:
- 鼠标拖拽（平移/滚动）
- 滚轮（缩放频率范围）
- 点击选择特定音高
- 双击：跳转到全局最佳频谱点

**移动端 (Mobile)**:
- 单指拖拽（触摸滑动平移）
- 双指捏合（缩放频率范围）
- 点击选择特定音高
- 双击：跳转到全局最佳频谱点

**双击行为** (特殊操作):

**PC端**: 鼠标双击
**移动端**: 触摸双击（快速双点，300ms内，20像素范围内）

```
双击/双点 ──► 执行 Case 1 (Global Jump)
    ├──► 有频谱点? ──► 跳转到置信度最高的点 (立即跳转)
    └──► 无频谱点? ──► 重置到 A4 (440Hz) 中心
```

**AutoTrack ON/OFF 切换**:
- **ON**: 启用自动追踪，用户操作时暂停，空闲时自动追踪
- **OFF**: 完全手动控制，不执行任何自动追踪逻辑
- 界面提供按钮供用户随时切换

**代码实现**:
```cpp
void performJumpToBestOrReset() {
    // 共享逻辑：PC双击 和 移动端双点 调用相同函数
    
    // 1. 收集最近1秒内的频谱点
    auto recentPitches = getRecentPitches(1.0);
    
    // 2. 查找置信度最高的点
    auto* best = findBestByConfidence(recentPitches);
    
    if (best) {
        // Case 1: 跳转到最佳点
        setViewCenterFreq(best->frequency);
    } else {
        // 无点：重置到 A4
        scrollOffset_ = 0;
    }
}

// PC端：鼠标双击
void mouseDoubleClick(const MouseEvent& event) {
    performJumpToBestOrReset();
}

// 移动端：触摸双点检测
void touchDown(const MouseEvent& event) {
    if (isDoubleTap(event)) {  // 300ms内，20像素内
        performJumpToBestOrReset();
        return;
    }
    // ... 单点处理
}
```

**为什么这样设计**:
- 双击 = "带我去看最重要的音高"
- 有信号时：直接跳转到当前最强音高 (快速定位)
- 无信号时：回到默认参考点 (A4标准音)
- PC和移动端行为完全一致

**交互设计原则**:
- PC端遵循 **"拖拽=平移，滚轮=缩放"** 的标准交互（类似地图应用）
- 移动端遵循 **"单指=平移，双指=缩放"** 的标准触摸交互

**输入设备检测**:
```cpp
void mouseDown(const MouseEvent& event) {
    // 自动检测触摸 vs 鼠标
    if (event.source.isTouch()) {
        // 移动端触摸处理
        handleTouchDown(event);
    } else {
        // 桌面端鼠标处理
        handleMouseDown(event);
    }
}
```

**代码实现**:
```cpp
class PitchWaterfallDisplay : public Component,
                              private Timer
{
public:
    void mouseDown(const MouseEvent& e) override {
        userInteracting_ = true;
        lastInteractionTime_ = Time::getCurrentTime();
        suspendAutoTracking();
    }
    
    void mouseDrag(const MouseEvent& e) override {
        // 手动拖拽时暂停自动追踪
        handleManualPan(e.getDistanceFromDragStart());
    }
    
    void mouseUp(const MouseEvent& e) override {
        userInteracting_ = false;
        lastInteractionTime_ = Time::getCurrentTime();
        // 不立即恢复，等待冷却期
    }
    
    void mouseWheelMove(const MouseEvent& e, const MouseWheelDetails& wheel) override {
        userInteracting_ = true;
        lastInteractionTime_ = Time::getCurrentTime();
        handleZoom(wheel.deltaY);
    }
    
private:
    std::atomic<bool> userInteracting_{false};
    Time lastInteractionTime_;
    static constexpr int COOLDOWN_SECONDS = 10;
    
    void timerCallback() override {
        if (!userInteracting_) {
            auto elapsed = Time::getCurrentTime() - lastInteractionTime_;
            if (elapsed.inSeconds() >= COOLDOWN_SECONDS) {
                resumeAutoTracking();
            }
        }
        
        if (autoTrackingEnabled_) {
            updateAutoTracking();
        }
    }
};
```

### 4.2 冷却期管理

```
用户操作 ──► 暂停自动追踪 ──► 启动10秒计时器 ──► 超时 ──► 恢复自动追踪
     │                                                    ▲
     └────────────────────────────────────────────────────┘
              （期间有新操作则重置计时器）
```

### 4.3 平台交互设计

#### 交互映射表
| 平台 | 平移(滚动) | 缩放 | 重置 |
|------|-----------|------|------|
| **桌面端 (PC)** | 鼠标拖拽 | 滚轮 | 双击 |
| **移动端** | 单指滑动 | 双指捏合 | 双击 |

**设计原则**: 每个平台使用最自然的交互方式，保持功能一致性。

#### 桌面端 (PC) 实现
```cpp
void mouseWheelMove(const MouseEvent& event, const MouseWheelDetails& wheel) {
    // PC: 滚轮 = 缩放 (Zoom)
    float zoomFactor = 1.0f - wheel.deltaY * 0.1f;
    
    float currentRange = maxFreq_ / minFreq_;
    float newRange = currentRange * zoomFactor;
    newRange = jlimit(2.0f, 100.0f, newRange);  // 2x - 100x 范围
    
    // 保持中心频率
    float centerFreq = sqrt(minFreq_ * maxFreq_);
    minFreq_ = centerFreq / sqrt(newRange);
    maxFreq_ = centerFreq * sqrt(newRange);
}

void mouseDrag(const MouseEvent& event) {
    // PC: 拖拽 = 平移 (Pan)
    float deltaY = event.getPosition().y - dragStartY_;
    scrollOffset_ += deltaY * sensitivity;
}
```

#### 移动端实现
```cpp
void touchDrag(const MouseEvent& event) {
    // 移动端: 单指滑动 = 平移
    float deltaY = event.getPosition().y - touchStartY_;
    scrollOffset_ += deltaY * sensitivity;
}

void handlePinchZoom(float scaleDelta) {
    // 移动端: 双指捏合 = 缩放
    float currentRange = maxFreq_ / minFreq_;
    float newRange = currentRange * (1.0f / scaleDelta);
    newRange = jlimit(2.0f, 100.0f, newRange);
    
    float centerFreq = sqrt(minFreq_ * maxFreq_);
    minFreq_ = centerFreq / sqrt(newRange);
    maxFreq_ = centerFreq * sqrt(newRange);
}
```

**自适应行为**:
- 自动检测输入类型（触摸 vs 鼠标）
- 触摸操作时禁用鼠标光标变化
- 捏合缩放时暂停自动追踪
- 支持双击重置（桌面和移动端）

---

## 5. 性能优化

### 5.1 更新频率
- 自动追踪计算：**30 FPS**（与渲染帧率同步）
- 距离排序优化：使用增量更新，避免每帧全量排序

### 5.2 避免抖动
- 引入**滞后区（Hysteresis）**: 目标点需要在中心区域外停留至少 200ms 才触发移动
- **速度限制**: 最大移动速度不超过屏幕高度/秒的 50%

### 5.3 内存优化
```cpp
// 使用对象池避免频繁分配
class PitchCandidatePool {
    std::array<PitchCandidate, 64> pool_;
    std::bitset<64> used_;
    
public:
    PitchCandidate* acquire() {
        for (size_t i = 0; i < 64; ++i) {
            if (!used_[i]) {
                used_[i] = true;
                return &pool_[i];
            }
        }
        return nullptr; // 池已满
    }
    
    void release(PitchCandidate* p) {
        // 标记为未使用
    }
};
```

---

## 6. 配置参数

```cpp
struct AutoTrackingConfig {
    // 区域定义
    float centerZoneRatio = 0.333f;      // 中心区域占屏幕比例
    
    // 优先级权重
    float confidenceWeight = 1000.0f;    // 置信度权重
    float energyWeight = 1.0f;           // 能量权重
    
    // 动画参数
    float approachSpeed = 0.15f;         // 接近速度（0-1）
    float minApproachSpeed = 5.0f;       // 最小频率跨度/秒
    float stopThreshold = 0.01f;         // 停止阈值
    
    // 交互参数
    int cooldownSeconds = 10;            // 用户操作冷却期（秒）
    int hysteresisMs = 200;              // 滞后区时间（毫秒）
    float maxPanSpeed = 0.5f;            // 最大移动速度（屏幕/秒）
    
    // 调试
    bool showCenterZone = false;         // 显示中心区域边框（调试用）
    bool showTrackingTarget = false;     // 显示追踪目标标记（调试用）
};
```

---

## 7. 测试验证

### 7.1 单元测试场景

| 场景 | 输入 | 预期行为 |
|------|------|---------|
| 全局跳转 | 全屏无点，远处有高置信度点 | 立即跳转到该点 |
| 平滑接近 | 中心无点，左侧有低频点 | 平滑右移直到点进入中心 |
| 保持不动 | 中心已有多个点 | 视图保持不动 |
| 用户中断 | 自动追踪中用户开始拖拽 | 立即暂停，10秒后恢复 |
| 置信度优先 | 近距离低置信 vs 远距离高置信 | 选择高置信点 |

### 7.2 手动测试步骤

1. **测试全局跳转**:
   ```
   1. 打开应用，等待几秒让视图稳定
   2. 快速滑动使所有点移出屏幕
   3. 观察是否自动跳转到最强音
   ```

2. **测试平滑接近**:
   ```
   1. 播放包含分散音符的音频
   2. 滑动视图使中心区域无点
   3. 观察最近点是否平滑移入中心
   ```

3. **测试用户中断**:
   ```
   1. 开始自动追踪
   2. 手指按住屏幕拖拽
   3. 确认自动追踪暂停
   4. 停止操作，等待10秒
   5. 确认自动追踪恢复
   ```

---

## 8. 待办事项

- [x] 实现 `AutoTracker` 类
- [x] 集成到 `PitchWaterfallDisplay`
- [x] 桌面端鼠标支持
- [x] 移动端触摸支持
- [x] 双指捏合缩放支持
- [x] 双击跳转逻辑 (PC: 鼠标双击, 移动端: 触摸双点)
- [x] **改进**: 只使用当前帧数据（不含历史）
- [x] **改进**: 智能暂停/恢复（连续3帧无检测才暂停）
- [x] **改进**: AutoTrack ON/OFF 按钮支持
- [x] **修复**: 滚轮缩放灵敏度
- [ ] 添加配置界面选项（AutoTrack按钮UI）
- [ ] 性能 profiling（目标：移动端 60 FPS）
- [ ] A/B 测试验证用户体验提升
- [ ] Android/iOS 真机测试

---

**文档版本**: 1.0  
**作者**: SuperPitchMonitor Team  
**日期**: 2026-02-22  
**状态**: 设计完成，待实现

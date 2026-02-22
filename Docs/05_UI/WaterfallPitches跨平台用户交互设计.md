# WaterfallPitches 跨平台用户交互设计

## 1. 功能概述

本文档定义 PitchWaterfallDisplay 组件的跨平台用户交互规范，支持桌面端（鼠标/键盘）和移动端（触摸）操作。

## 2. 交互映射表

| 功能 | 桌面端 (PC) | 移动端 (Touch) |
|------|------------|---------------|
| **平移 (Pan)** | 鼠标左键拖拽 | 单指滑动 |
| **缩放 (Zoom)** | 鼠标滚轮 | 双指捏合 (Pinch) |
| **快速定位** | 双击 | 双点 (Double Tap) |

## 3. 详细交互设计

### 3.1 平移操作 (Pan)

**功能**: 调整频率显示的中心位置（上下滚动查看不同频率范围）

**桌面端 - 鼠标拖拽**:
```cpp
void mouseDown(const MouseEvent& event) {
    if (event.mods.isLeftButtonDown()) {
        isDragging_ = true;
        dragStartY_ = event.getPosition().y;
        dragStartOffset_ = scrollOffset_;
        autoTracker_.onUserInteraction();  // 暂停自动追踪
    }
}

void mouseDrag(const MouseEvent& event) {
    if (isDragging_) {
        float deltaY = event.getPosition().y - dragStartY_;
        float semitoneDelta = (deltaY / plotHeight) * visibleSemitones;
        scrollOffset_ = dragStartOffset_ + semitoneDelta;
        scrollOffset_ = juce::jlimit(-36.0f, 36.0f, scrollOffset_);  // ±3八度限制
        repaint();
    }
}

void mouseUp(const MouseEvent& event) {
    if (isDragging_) {
        isDragging_ = false;
        autoTracker_.onUserInteractionEnd();  // 恢复自动追踪计时
    }
}
```

**移动端 - 单指滑动**:
- 与鼠标拖拽逻辑相同
- 自动检测触摸事件 (`event.source.isTouch()`)
- 单指按下并滑动时暂停自动追踪

### 3.2 缩放操作 (Zoom)

**功能**: 调整频率显示的范围（放大/缩小查看细节/概览）

**关键变量**: `visibleSemitones_` - 控制Y轴可见的半音数量（默认24 = 2个八度）
- 值越小 = 放大（看到更少的频率范围，更多细节）
- 值越大 = 缩小（看到更多的频率范围，更少细节）
- 范围限制: 12 - 60 半音（1个八度到5个八度）

**桌面端 - 鼠标滚轮**:
```cpp
void mouseWheelMove(const MouseEvent& event, const MouseWheelDetails& wheel) {
    autoTracker_.onUserInteraction();  // 暂停自动追踪
    
    if (std::abs(wheel.deltaY) > 0.01f) {
        // 滚轮向上 (deltaY > 0): 放大 (缩小可见范围)
        // 滚轮向下 (deltaY < 0): 缩小 (扩大可见范围)
        float zoomFactor = 1.0f - wheel.deltaY * 0.1f;
        visibleSemitones_ *= zoomFactor;
        visibleSemitones_ = juce::jlimit(minVisibleSemitones, maxVisibleSemitones, visibleSemitones_);
        
        repaint();
    }
    
    autoTracker_.onUserInteractionEnd();  // 恢复自动追踪计时
}
```

**注意**: 滚轮操作缩放频率范围（调整 `visibleSemitones_`），而非平移。平移使用拖拽操作。

**移动端 - 双指捏合 (Pinch)**:
```cpp
// 使用 JUCE 的 mouseMagnify 方法支持触控板/触摸屏捏合
void mouseMagnify(const MouseEvent& event, float scaleFactor) {
    autoTracker_.onUserInteraction();
    
    // scaleFactor > 1.0: 放大 (缩小可见范围)
    // scaleFactor < 1.0: 缩小 (扩大可见范围)
    // 注意：这里除以 scaleFactor，因为 scaleFactor > 1 表示放大（显示更少内容）
    visibleSemitones_ /= scaleFactor;
    visibleSemitones_ = juce::jlimit(minVisibleSemitones, maxVisibleSemitones, visibleSemitones_);
    
    repaint();
    autoTracker_.onUserInteractionEnd();
}
```

**替代方案 - 使用 GestureListener (Android/iOS)**:
```cpp
// 在 Android/iOS 上，可以使用原生手势检测
class PitchWaterfallDisplay : public Component,
                              private Timer,
                              public juce::GestureDetector::Listener  // 伪代码
{
    // 双指捏合回调
    void onPinchZoom(float scaleDelta) {
        autoTracker_.onUserInteraction();
        
        // scaleDelta > 1.0: 放大 (缩小可见范围)
        // scaleDelta < 1.0: 缩小 (扩大可见范围)
        visibleSemitones_ /= scaleDelta;
        visibleSemitones_ = juce::jlimit(minVisibleSemitones, maxVisibleSemitones, visibleSemitones_);
        
        repaint();
        autoTracker_.onUserInteractionEnd();
    }
};
```

### 3.3 快速定位 (Jump to Best/Reset)

**功能**: 双击/双点快速跳转到当前最佳频谱点，或重置到 A4

**桌面端 - 鼠标双击**:
```cpp
void mouseDown(const MouseEvent& event) {
    if (event.getNumberOfClicks() == 2) {
        performJumpToBestOrReset();
        return;
    }
    // ... 正常拖拽处理
}

void performJumpToBestOrReset() {
    // 找到置信度最高的频谱点
    const PitchCandidate* best = nullptr;
    float bestScore = -1.0f;
    
    for (const auto& pitch : currentPitches_) {
        if (pitch.confidence > 0.1f) {
            float score = pitch.confidence * 1000.0f + pitch.amplitude;
            if (score > bestScore) {
                bestScore = score;
                best = &pitch;
            }
        }
    }
    
    if (best) {
        // 跳转到最佳点
        float targetMidi = best->midiNote;
        scrollOffset_ = targetMidi - 69.0f;  // A4 = 69
    } else {
        // 无信号时重置到 A4
        scrollOffset_ = 0.0f;
    }
    
    repaint();
}
```

**移动端 - 双点 (Double Tap)**:
- 300ms 内两次点击，20 像素范围内
- 行为与桌面端双击完全一致

## 4. 自动追踪集成

所有用户交互都会暂停自动追踪：

```cpp
// 交互开始时调用
void onUserInteraction() {
    userInteracting_ = true;
    trackingActive_ = false;
    lastInteractionTime_ = juce::Time::getCurrentTime();
}

// 交互结束时调用
void onUserInteractionEnd() {
    userInteracting_ = false;
    lastInteractionTime_ = juce::Time::getCurrentTime();
    // 10秒后自动恢复追踪
}
```

## 5. 实现状态

| 功能 | 状态 | 说明 |
|------|------|------|
| 鼠标拖拽平移 | ✅ 已实现 | `mouseDrag()` - 调整中心频率 |
| 鼠标滚轮缩放 | ✅ 已实现 | `mouseWheelMove()` - 调整频率范围 |
| 鼠标双击跳转 | ✅ 已实现 | `mouseDown()` 检测双击 |
| 触控板捏合缩放 | ✅ 已实现 | `mouseMagnify()` - 支持触控板双指捏合 |
| 触摸单指平移 | ⚠️ 待测试 | JUCE 自动映射到 mouseDrag |
| 触摸双指捏合缩放 | ⚠️ 待测试 | `mouseMagnify()` 应该支持，需真机验证 |
| 触摸双点跳转 | ⚠️ 待测试 | JUCE 自动映射到双击 |

## 6. 待办事项

- [ ] 实现 `mouseMagnify()` 支持触控板捏合
- [ ] Android/iOS 原生手势检测（双指捏合）
- [ ] 移动端触摸测试验证
- [ ] 调整触摸灵敏度（与鼠标区分）
- [ ] 添加惯性滚动 (Inertial Scrolling) 支持

---

**文档版本**: 1.0  
**日期**: 2026-02-22  
**作者**: SuperPitchMonitor Team

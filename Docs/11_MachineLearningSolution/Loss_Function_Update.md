# 损失函数更新说明 - Tversky Loss 加入

**日期**: 2026-02-27  
**更新内容**: 在 ConfidenceLoss 中加入 Tversky Loss，重新平衡各损失权重

---

## 1. 更新概述

### 1.1 主要变更

| 变更项 | 原实现 | 新实现 |
|--------|--------|--------|
| **新增损失** | Focal + Sharpness | Focal + **Tversky** + Sharpness |
| **Tversky 权重** | 无 | **0.3~0.5** (推荐) |
| **Sharpness 权重** | 0.5 | **0.1~0.2** (降低) |
| **Focal 权重** | 隐含 1.0 | **1.0** (保持基准) |

### 1.2 为何加入 Tversky Loss

Tversky Loss 对 **峰位置偏离** 提供强惩罚：

```
Tversky = TP / (TP + α·FP + β·FN)
```

- **α = 0.3**: 假阳性惩罚（误检）
- **β = 0.7**: 假阴性惩罚（漏检）> α，强调不漏检

对于软标签（高斯分布目标），Tversky Loss 能敏感地检测预测 peak 与目标 peak 的重叠度，轻微的位置偏移会导致显著更高的损失。

---

## 2. 各损失取值区间与权重建议

### 2.1 设计理念

**所有配置都以 confidence 预测为核心**，因为：
- Confidence 决定"有哪些音"（分类问题，必须准确）
- Energy 决定"各音能量比例"（回归问题，容错较高）

### 2.2 问题诊断

如果看到以下现象，说明需要更强的 confidence 权重：
- Focal (f) 持续上升或不变
- Tversky (t) 接近 1.0（如 0.996）
- Energy (e) 下降很快但 conf 不降

**原因**：模型输出全 0，energy loss 容易优化，但 confidence 没有学到 peak。

### 2.3 损失取值范围（典型值）

| 损失项 | 健康范围 | 危险信号 | 说明 |
|--------|----------|----------|------|
| **Focal Loss** | 0.1 ~ 0.5 | <0.01 或 >1.0 | 太小=全0预测，太大=训练不稳定 |
| **Tversky Loss** | 0.1 ~ 0.5 | >0.9 | 接近1=预测与目标无交集 |
| **Sharpness Loss** | 0.1 ~ 0.5 | >1.0 | 过高=权重太大 |
| **Energy (KL)** | 0.1 ~ 1.0 | - | 相对次要 |
| **Sparsity** | 0.001 ~ 0.01 | - | 轻微正则化 |

### 2.4 配置说明

**`default` 配置已经是专注于 confidence 预测的配置**，其他配置在此基础上微调：

#### 默认配置 (`default`) - 推荐

最强的 confidence 预测配置：

```python
{
    'focal_weight': 10.0,       # 主导：confidence 分类
    'tversky_weight': 3.0,      # 强辅助：peak 位置精度
    'sharpness_weight': 0.1,    # 轻微：peak 形状
    'energy_weight': 0.3,       # 次要：能量分布
    'sparsity_weight': 0.1,     # 防止全 0 预测
    'focal_gamma': 0.5,         # 让更多样本参与
    'focal_alpha': 0.8,         # 高正样本权重（正样本很少）
}
```

#### 强调位置精度 (`position_focused`)

在 `default` 基础上，进一步提高位置精度要求：

```python
{
    'focal_weight': 8.0,        # 略降
    'tversky_weight': 5.0,      # 大幅提高位置权重
    'sharpness_weight': 0.1,
    'energy_weight': 0.3,
    'sparsity_weight': 0.1,
}
```

#### 强调 Peak 质量 (`peak_focused`)

在 `default` 基础上，强调 peak 锐利度：

```python
{
    'focal_weight': 8.0,        # 略降
    'tversky_weight': 2.5,      # 略降
    'sharpness_weight': 1.0,    # 大幅提高锐利度
    'energy_weight': 0.3,
    'sparsity_weight': 0.1,
}
```

#### 平衡配置 (`balanced`)

在 `default` 基础上，稍微提高 energy 权重：

```python
{
    'focal_weight': 8.0,
    'tversky_weight': 2.5,
    'sharpness_weight': 0.1,
    'energy_weight': 0.8,       # 提高 energy 权重
    'sparsity_weight': 0.08,
}
```

#### 简化版 (`simple`)

去掉 sharpness，保留核心 conf 损失：

```python
{
    'focal_weight': 10.0,
    'tversky_weight': 3.0,
    'sharpness_weight': 0.0,    # 关闭
    'energy_weight': 0.3,
    'sparsity_weight': 0.1,
}

#### 简化版 (`simple`)

```python
{
    'focal_weight': 1.0,
    'tversky_weight': 0.4,
    'sharpness_weight': 0.0,    # 关闭
    'energy_weight': 0.3,
    'sparsity_weight': 0.01,
}
```

---

## 3. 使用方式

### 3.1 训练脚本中的使用

```python
from loss import PitchDetectionLoss, get_loss_config

# 方法1: 使用默认配置（已经是专注于 confidence 的配置）
loss_config = get_loss_config('default')  # 或 'position_focused', 'peak_focused', etc.
criterion = PitchDetectionLoss(**loss_config).to(device)

# 方法2: 自定义配置
criterion = PitchDetectionLoss(
    focal_weight=10.0,       # confidence 主导
    tversky_weight=3.0,      # 位置精度辅助
    sharpness_weight=0.1,
    energy_weight=0.3,       # 次要
    sparsity_weight=0.1,     # 防止全0
    focal_gamma=0.5,
    focal_alpha=0.8,
    tversky_alpha=0.3,
    tversky_beta=0.7,
).to(device)
```

### 3.2 训练日志解读

新的训练日志格式：

```
Epoch 10/50 | 5.2s | Train: 0.8234 (f:0.234 t:0.123 s:0.045 e:0.312) | Val: 0.7654 (f:0.198 t:0.115 s:0.038 e:0.298) | LR: 0.001000
```

| 缩写 | 含义 |
|------|------|
| `f` | Focal Loss |
| `t` | Tversky Loss |
| `s` | Sharpness Loss |
| `e` | Energy Loss |

---

## 4. 技术细节

### 4.1 Tversky Loss 实现

```python
def _tversky_loss(self, pred, target, smooth=1e-7):
    """软标签版本的 Tversky Loss"""
    tp = (pred * target).sum(dim=-1)           # True Positive
    fp = (pred * (1 - target)).sum(dim=-1)     # False Positive  
    fn = ((1 - pred) * target).sum(dim=-1)     # False Negative
    
    tversky = (tp + smooth) / (tp + self.tversky_alpha * fp + self.tversky_beta * fn + smooth)
    return 1 - tversky  # 转换为损失
```

### 4.2 与 Focal Loss 的互补性

| 特性 | Focal Loss | Tversky Loss |
|------|------------|--------------|
| **关注点** | 难样本、类别不平衡 | 空间/位置重叠度 |
| **范围** | 无界 | [0, 1] 有界 |
| **梯度特性** | 动态调整 | 稳定 |
| **对位置偏移** | 不敏感 | **强惩罚** |

两者互补：
- Focal Loss 确保网络关注难分类样本
- Tversky Loss 确保预测 peak 位置精确

---

## 5. 调参建议

### 5.1 所有配置都以 Confidence 为核心

`default` 配置已经是最强的 confidence 预测配置。如果 conf 仍然不下降，请检查：
1. 学习率是否过高/过低
2. 数据是否有问题（如全 0 标签）
3. 模型结构是否有问题

### 5.2 正常调参顺序

1. **前 5 个 epoch**：使用 `default` 配置，观察 confidence 是否开始下降
   - f 应该逐渐下降（如 0.5 → 0.3 → 0.2）
   - t 应该逐渐下降（如 0.5 → 0.3 → 0.2）
   - 如果 f < 0.1 且 t > 0.9：检查数据或模型

2. **5-20 epoch**：根据需求切换到其他配置
   - 位置精度不够 → `position_focused`
   - peak 不够尖锐 → `peak_focused`
   - energy 也需要优化 → `balanced`

3. **20+ epoch**：精细调整或回到 `default` 继续训练

### 5.3 关键参数说明

| 参数 | 调高效果 | 调低效果 |
|------|----------|----------|
| `focal_weight` | 更强调分类准确性 | 可能输出全0 |
| `tversky_weight` | 更强调位置精度 | 位置可能偏移 |
| `focal_alpha` | 更关注正样本（有音的bin） | 类别更平衡 |
| `focal_gamma` | 只关注难样本 | 更多样本参与 |
| `energy_weight` | 能量分布更准 | confidence 可能更好 |

### 5.4 权重比例参考

健康的 loss 比例（加权后）：
```
total_loss ≈ 0.5 * focal + 0.3 * tversky + 0.15 * energy + 0.05 * others
```

 confidence 相关损失（focal + tversky）应该占总损失的 70% 以上。

---

## 6. 兼容性说明

### 6.1 向后兼容

- 旧版 `enhanced=True` 参数仍支持（内部映射到新配置）
- `PitchDetectionLoss` 的返回值增加 `focal`, `tversky`, `sharpness` 字段

### 6.2 文件修改列表

| 文件 | 修改内容 |
|------|----------|
| `loss.py` | 加入 Tversky Loss，重新设计接口 |
| `train_live.py` | 适配新接口，更新日志格式 |
| `train_test.py` | 适配新接口，更新日志格式 |
| `train_debugnoise.py` | 适配新接口，更新日志格式 |

---

## 7. 验证方法

### 7.1 健康指标

| 指标 | 健康范围 | 危险信号 | 处理方案 |
|------|----------|----------|----------|
| **Focal (f)** | 0.1~0.3 且下降 | <0.01 或持续上升 | 大幅提高 focal_weight |
| **Tversky (t)** | 0.1~0.3 且下降 | >0.9 不变 | 大幅提高 tversky_weight |
| **Energy (e)** | 0.1~0.3 | 主导总 loss | 降低 energy_weight |
| **总 loss** | 稳定下降 | 震荡或上升 | 降低整体学习率 |

### 7.2 示例对比

❌ 不健康（需要调整权重）：
```
Epoch 20/100 | Train: 0.4758 (f:0.062 t:0.996 s:0.677 e:0.158)
# 问题：t=0.996 太高，f=0.062 虽然下降但太慢
# 处理：切换到 conf_focused 配置
```

✅ 健康状态：
```
Epoch 50/100 | Train: 0.5234 (f:0.312 t:0.089 s:0.012 e:0.110) 
# Focal (0.312) 占主导 ✓
# Tversky (0.089) 适中，在下降 ✓
# Sharpness (0.012) 较低，正常 ✓
```

# Reward Hacking 检测与防御 + Reward 设计方法论

> Agentic RL 中最核心的问题之一

## 目录

- [1. Reward Hacking 是什么](#1-reward-hacking-是什么)
- [2. 典型 Hack 场景](#2-典型-hack-场景)
- [3. 如何发现 Reward Hacking](#3-如何发现-reward-hacking)
- [4. 如何解决 Reward Hacking](#4-如何解决-reward-hacking)
- [5. Reward 设计完整方法论](#5-reward-设计完整方法论)
- [6. 面试回答模板](#6-面试回答模板)

---

## 1. Reward Hacking 是什么

本质是 **Goodhart's Law**：当一个指标变成优化目标，它就不再是好指标。

模型不是在"解决任务"，而是在"最大化奖励函数"。当奖励函数与真实目标之间存在缝隙时，模型会找到捷径——奖励分数上升，但实际质量下降。

---

## 2. 典型 Hack 场景

### 场景 1: 长度 Hack

```
奖励: 越长越好（或隐式与长度相关）
hack: 模型输出重复废话凑长度
例:  "Paris. Paris is the capital. The capital is Paris. Indeed Paris..."
```

### 场景 2: 格式 Hack

```
奖励: 包含 <think> 和 <answer> 标签就加分
hack: 模型输出空标签 <think></think><answer>random</answer>
```

### 场景 3: 复制 Hack（Agent 特有）

```
奖励: 答案与 ground truth 的 F1
hack: 模型学会把搜索结果整段复制到 answer 里（F1 高但没有推理）
```

### 场景 4: 神经奖励模型 Hack

```
奖励: 学习的 reward model 打分
hack: 模型找到 RM 的盲区，生成 RM 给高分但人类觉得差的内容
例:  看起来"像推理"的胡说八道
```

---

## 3. 如何发现 Reward Hacking

### 3.1 训练时监控（Early Warning）

#### 监控指标 1: 奖励 vs 实际质量的背离

```python
def monitor_reward_quality_gap(rewards, manual_scores):
    """
    rewards: RL 训练中的奖励曲线
    manual_scores: 定期人工抽查评分

    当两者趋势背离时 → 报警
    """
    correlation = np.corrcoef(rewards[-100:], manual_scores[-100:])[0, 1]
    if correlation < 0.5:
        print("⚠️ WARNING: Reward-Quality gap detected!")
```

#### 监控指标 2: 输出多样性突然下降

```python
def monitor_diversity(outputs_per_epoch):
    """跟踪每 epoch 输出的唯一 n-gram 数，下降趋势 → 可能在 hack"""
    unique_trigrams = set()
    for output in outputs_per_epoch:
        tokens = output.split()
        for i in range(len(tokens) - 2):
            unique_trigrams.add(tuple(tokens[i:i + 3]))
    return len(unique_trigrams)
```

#### 监控指标 3: 输出长度分布异常

```python
def monitor_length(outputs):
    lengths = [len(o.split()) for o in outputs]
    mean_len = np.mean(lengths)
    std_len = np.std(lengths)
    # 长度持续增长或方差坍缩 → 报警
    return mean_len, std_len
```

#### 监控指标 4: 搜索行为异常（Agent 特有）

```python
def monitor_search_behavior(trajectories):
    """跟踪搜索频率和搜索内容"""
    stats = {
        'avg_searches': np.mean([t.num_searches for t in trajectories]),
        'empty_searches': sum(1 for t in trajectories
                              if any(q.strip() == '' for q in t.queries)),
        'duplicate_searches': sum(1 for t in trajectories
                                  if len(set(t.queries)) < len(t.queries)),
    }
    # 空搜索或重复搜索增加 → hack
    return stats
```

### 3.2 事后分析（Post-hoc Detection）

最有效的方法其实很朴素：**手动看 top-k 高奖励样本**。

```
Checklist:
□ 随机抽 20 个最高奖励的输出，逐条检查
□ 答案是推理出来的，还是从搜索结果复制的？
□ 推理链是否合理，还是"看起来像推理"的废话？
□ 搜索查询是否有意义？
□ 对比高奖励 vs 低奖励样本，差异是否合理？
```

---

## 4. 如何解决 Reward Hacking

### 解法 1: 用规则化奖励（最根本的防线）

```python
# ❌ 神经奖励模型 — 可被 hack
reward = reward_model(output)  # RM 有盲区

# ✅ 规则化奖励 — 不可被 hack（前提是规则正确）
reward = exact_match(extract_answer(output), ground_truth)
```

DeepSeek-R1 的核心发现：**对有可验证答案的任务，规则化奖励严格优于神经 RM。**

局限：不是所有任务都有明确的正确答案。对于开放式任务，仍需神经 RM + 其他防线。

### 解法 2: 多维度奖励组合（让 hack 变难）

核心思路：hack 一个指标容易，同时 hack 多个指标难。

```python
def anti_hack_reward(output, ground_truth):
    scores = {}

    # 维度 1: 答案正确性（必须）
    answer = extract_answer(output)
    scores['accuracy'] = exact_match(answer, ground_truth)

    # 维度 2: 格式合规
    scores['format'] = check_format(output)

    # 维度 3: 推理质量（用合理范围，不直接奖励长度）
    think_content = extract_think(output)
    scores['reasoning_length'] = min(len(think_content.split()) / 50, 1.0)
    if len(think_content.split()) > 500:  # 过长惩罚
        scores['reasoning_length'] *= 0.5

    # 维度 4: 搜索效率
    num_searches = output.count('<search>')
    if scores['accuracy'] > 0 and num_searches == 0:
        scores['efficiency'] = 1.0    # 不搜也能对，最高效
    elif scores['accuracy'] > 0:
        scores['efficiency'] = 1.0 / (1 + num_searches * 0.1)
    else:
        scores['efficiency'] = 0.0

    # 加权组合
    final = (0.5 * scores['accuracy'] +
             0.2 * scores['format'] +
             0.1 * scores['reasoning_length'] +
             0.2 * scores['efficiency'])
    return final, scores  # 返回各维度分数用于监控
```

### 解法 3: KL 约束（防止策略漂移太远）

```python
# GRPO 的 KL 惩罚
L = L_clip - β * D_KL(π_θ || π_ref)

# β 的选择：
# β 太小 → 策略自由漂移 → 容易找到 hack 路径
# β 太大 → 策略几乎不变 → 学不到东西
# 实践中 β = 0.001 ~ 0.05

# 特例：ToolRL 发现对工具调用任务，完全去掉 KL 反而更好
# 原因：强结果奖励本身就是约束，不需要额外 KL
```

### 解法 4: 奖励裁剪和归一化

```python
def clip_and_normalize_reward(raw_reward, clip_range=(-3, 3)):
    """防止极端奖励值导致的训练不稳定和 hack"""
    clipped = max(clip_range[0], min(clip_range[1], raw_reward))
    return clipped

# ToolRL 的动态缩放
def dynamic_scale_reward(reward, running_mean, running_std):
    """用滑动窗口统计归一化"""
    return (reward - running_mean) / (running_std + 1e-8)
```

### 解法 5: Retrieved Token Masking（Agent 特有防线）

```python
def compute_policy_gradient(output_tokens, advantages, token_mask):
    """
    Search-R1 的关键技术：环境返回的 token 不参与梯度计算

    token_mask[t] = 1: 模型生成的 token（参与梯度）
    token_mask[t] = 0: 环境返回的 token（不参与梯度）
    """
    loss = 0
    for t, (token, adv, mask) in enumerate(
        zip(output_tokens, advantages, token_mask)
    ):
        if mask == 1:  # 只对模型自己的输出计算梯度
            loss -= log_prob(token) * adv
    return loss

# 没有这个 masking 会怎样？
# → 模型会学到"记住搜索结果的分布"而不是"学会搜索"
# → 本质上也是一种 reward hacking
```

### 解法汇总表

| 解法 | 防御目标 | 复杂度 | 来源 |
|------|---------|--------|------|
| 规则化奖励 | 根本性防御 | 低 | DeepSeek-R1 |
| 多维度组合 | 增加 hack 难度 | 中 | ToolRL |
| KL 约束 | 防策略漂移 | 低 | GRPO/PPO 标配 |
| 奖励裁剪归一化 | 防极端奖励 | 低 | ToolRL |
| Retrieved Token Masking | 防复制 hack | 中 | Search-R1 |
| 人工抽查 | 兜底检测 | 高（人力） | 通用 |

---

## 5. Reward 设计完整方法论

### 5.1 核心原则

```
规则化 > 神经模型        ← 防 hack
多维度 > 单维度          ← 增加鲁棒性
分阶段 > 一步到位        ← 解决冷启动
能用 EM 就不用 F1        ← 越简单越不容易出错
能用规则就不用模型        ← 同上
```

### 5.2 奖励类型全谱系

```
粗 ←─────────────────────────────────────→ 细

结果奖励      轨迹奖励      步骤奖励      Token奖励
 (0/1)        (整条)        (每步)        (每词)

简单、稀疏     ↕             ↕          复杂、密集
不容易hack     ↕             ↕          容易hack
信用分配难     ↕             ↕          信用分配易

推荐起点: 结果奖励 + 格式奖励（两级粒度混合）
```

### 5.3 逐级递进设计

```python
# Level 0: 最简单 — 答案对了就行
R = exact_match(pred, gold)                     # 0 或 1

# Level 1: 加上格式 — 确保工具调用可解析
R = 0.7 * EM(pred, gold) + 0.3 * format_valid(output)

# Level 2: 加上效率 — 奖励高效的工具使用
R = 0.6 * EM + 0.2 * format + 0.2 * efficiency

# Level 3: 过程奖励 — 每步工具调用的质量
R = 0.5 * outcome + 0.2 * format + 0.3 * process
```

**Search-R1 用 Level 0 就达到了 26% 的提升。不要过度设计。**

### 5.4 分阶段引入复杂性（R1-Searcher 的核心洞察）

```
Phase 1 (前 N 步): 只训格式
  R = format_reward
  目的: 让模型先学会"怎么调用工具"

Phase 2 (之后): 格式 + 准确性
  R = format_reward + accuracy_reward
  目的: 在会用工具的基础上学会"用好工具"
```

**为什么分阶段有效？**
- 一开始就用准确性奖励，但模型还不会正确调用工具
- 工具调用格式错 → 环境返回错误 → 奖励始终为 0 → 学不到任何东西
- 先学格式 = 解决"冷启动"问题

### 5.5 针对常见问题打补丁

```python
# 问题 1: 模型总是搜索（即使不需要）
# 补丁: β-GRPO 置信度感知
if model_confidence > threshold and did_search:
    R *= 0.5  # 惩罚不必要的搜索

# 问题 2: 模型复制搜索结果而不推理
# 补丁: 检查答案与搜索结果的重叠度
overlap = compute_overlap(answer, search_results)
if overlap > 0.9:
    R *= 0.3  # 惩罚纯复制

# 问题 3: 推理链太长或太短
# 补丁: 软长度约束（不是奖励长度！是惩罚极端长度）
think_len = len(extract_think(output).split())
if think_len < 10 or think_len > 500:
    R *= 0.7  # 温和惩罚极端长度

# 问题 4: 格式时好时坏
# 补丁: 格式违规重罚（R1-Searcher 用 -2）
if not format_valid(output):
    R = -2.0  # 硬惩罚，优先级高于一切
```

### 5.6 完整 Decision Tree

```
                     你的任务有明确正确答案吗？
                    /                          \
                  是                            否
                  |                              |
           用规则化奖励                    用神经 RM + 多重防线
           (EM/F1/代码测试)               (KL约束+裁剪+人工抽查)
                  |
        模型需要调用工具吗？
        /                    \
      是                      否
      |                        |
  分阶段设计                 直接结果奖励
  Phase1: 格式               (DeepSeek-R1 Style)
  Phase2: 格式+准确
      |
  模型学会搜索后还有问题吗？
  /          |            \
过度搜索    复制hack     推理链质量差
  |          |              |
β-GRPO    overlap惩罚    过程奖励
```

### 5.7 各经典项目的奖励设计对比

| 项目 | 奖励类型 | 具体实现 | 特点 |
|------|---------|---------|------|
| DeepSeek-R1 | 结果+格式+语言一致性 | 数学验答案/代码跑测试 + 标签检查 + 目标语言占比 | 规则化，零 hack |
| Search-R1 | 纯结果 | `EM(pred, gold)` | 极简，26% 提升 |
| R1-Searcher | 两阶段(格式→格式+F1) | Stage1: +0.5 格式 / Stage2: +F1 答案 / 违规: -2 | 解决冷启动 |
| ToolRL | 细粒度工具调用 | 工具名Jaccard + 参数名Jaccard + 参数值EM, 归一化到[-3,3] | 多维度防hack |
| Agent-R1 | 过程+结果 | PRIME框架归一化 | 密集信号 |
| Tool-Star | Self-Critic | 模型自我评估工具使用质量 | 自动化 |
| β-GRPO | 置信度感知 | `R=1 if (prob≥β AND correct)` | 防过度搜索 |

---

## 6. 面试回答模板

### Q: 怎么发现 reward hacking？

> 三个层面：
> 1. **训练时监控**：跟踪奖励曲线与人工抽查质量的相关性，背离即 hack；同时监控输出多样性、长度分布、搜索频率等辅助指标
> 2. **事后分析**：定期人工检查 top-k 高奖励样本——这是最可靠的方法
> 3. **指标交叉验证**：如果 EM 涨了但 F1 没涨，或奖励涨了但 BLEU 降了，说明优化的不是真正的目标

### Q: 怎么设计不容易被 hack 的奖励？

> 核心原则：规则化 > 神经模型，多维度 > 单维度，分阶段 > 一步到位。
> 具体做法：
> 1. 有可验证答案的任务用 EM/F1 规则化奖励
> 2. 组合格式+准确+效率多个维度增加 hack 难度
> 3. 先训格式再训准确（R1-Searcher 的分阶段策略）
> 4. Agent 场景做 Retrieved Token Masking 防止复制 hack
> 5. 配合 KL 约束和奖励裁剪作为安全网

### Q: 如果发现了 hacking 怎么处理？

> 1. **定位 hack 维度**：是长度、格式、复制还是 RM 盲区？
> 2. **加入针对性惩罚**：如复制 hack → overlap 惩罚；长度 hack → 极端长度惩罚
> 3. **考虑换奖励类型**：神经 RM → 规则化；单维度 → 多维度
> 4. **调整 KL 系数**：增大 β 限制策略漂移
> 5. **回退检查点**：如果 hack 严重，回退到 hack 前的 checkpoint，修改奖励后重训

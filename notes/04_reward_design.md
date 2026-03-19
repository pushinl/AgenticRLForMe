# 奖励设计模式

## 1. 总览

奖励设计是 Agentic RL 最关键的环节。核心原则：
- **规则化奖励 >> 神经奖励模型**（DeepSeek-R1 的核心发现）
- **分阶段设计**: 先学格式，再学准确性
- **可验证性是关键**: 选择有明确正确答案的任务

## 2. 奖励类型详解

### 2.1 结果奖励 (Outcome Reward)

最常用，最简单，最不容易被 hack。

```python
def outcome_reward(prediction, ground_truth):
    """二值奖励：答案对了 = 1，错了 = 0"""
    if exact_match(prediction, ground_truth):
        return 1.0
    return 0.0
```

**使用者**: Search-R1, DeepSeek-R1 (数学题验证答案)

**变体 - F1 奖励**（R1-Searcher Stage 2）:
```python
def f1_reward(prediction, ground_truth):
    """Token 级 F1 overlap，适用于开放式问答"""
    pred_tokens = set(prediction.split())
    gold_tokens = set(ground_truth.split())
    if not pred_tokens or not gold_tokens:
        return 0.0
    precision = len(pred_tokens & gold_tokens) / len(pred_tokens)
    recall = len(pred_tokens & gold_tokens) / len(gold_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)
```

### 2.2 格式奖励 (Format Reward)

确保模型输出可被环境解析的结构化格式。

```python
def format_reward(output):
    """检查必需的标签是否存在且格式正确"""
    has_think = "<think>" in output and "</think>" in output
    has_answer = "<answer>" in output and "</answer>" in output

    # 搜索标签可选，但如果有必须成对出现
    search_opens = output.count("<search>")
    search_closes = output.count("</search>")
    search_valid = search_opens == search_closes

    if has_think and has_answer and search_valid:
        return 1.0
    return 0.0  # 或负值惩罚, 如 R1-Searcher 用 -2
```

**R1-Searcher 的格式奖励设计**:
- Stage 1: +0.5 正确检索格式
- 格式违规: -2.0（重罚）

### 2.3 过程奖励 (Process Reward)

对每一步工具调用给予奖励，而非只看最终结果。

```python
def process_reward(trajectory):
    """Agent-R1 风格：过程奖励 + 结果奖励"""
    step_rewards = []
    for step in trajectory.tool_calls:
        # 每步的有效性评分
        relevance = evaluate_tool_relevance(step)
        step_rewards.append(relevance)

    # 最终结果奖励
    outcome = outcome_reward(trajectory.final_answer, trajectory.ground_truth)

    # PRIME 框架归一化平衡
    process_score = normalize(mean(step_rewards))
    return alpha * process_score + (1 - alpha) * outcome
```

### 2.4 Self-Critic 奖励 (Tool-Star)

```python
def self_critic_reward(model, query, output, tool_results):
    """模型自我评估工具使用质量"""
    critic_prompt = f"""
    Query: {query}
    Model output: {output}
    Tool results: {tool_results}
    Rate the quality of tool usage (0-1):
    """
    score = model.generate(critic_prompt)
    return float(score)
```

### 2.5 细粒度工具调用奖励 (ToolRL)

```python
def toolrl_reward(pred_tool_call, gold_tool_call):
    """ToolRL: 多维度工具调用评分"""
    score = 0.0

    # 工具名匹配 (Jaccard 相似度)
    pred_names = set(pred_tool_call.tool_names)
    gold_names = set(gold_tool_call.tool_names)
    name_score = len(pred_names & gold_names) / len(pred_names | gold_names)
    score += name_score

    # 参数名匹配
    pred_params = set(pred_tool_call.param_names)
    gold_params = set(gold_tool_call.param_names)
    param_name_score = len(pred_params & gold_params) / len(pred_params | gold_params)
    score += param_name_score

    # 参数值精确匹配
    value_matches = sum(1 for k in pred_params & gold_params
                       if pred_tool_call.params[k] == gold_tool_call.params[k])
    value_score = value_matches / max(len(gold_params), 1)
    score += value_score

    # 归一化到 [-3, 3]
    return (score - 1.5) * 2
```

## 3. 奖励组合策略

### 策略 1: 简单叠加（Search-R1）
```
R = outcome_reward(answer)   # 就这么简单
```
Search-R1 发现纯结果奖励就够了，26% 提升。

### 策略 2: 分阶段递进（R1-Searcher）
```
Stage 1: R = format_reward        # 先学格式
Stage 2: R = format_reward + f1   # 再学准确
```

### 策略 3: 多维度组合（Agent-R1）
```
R = α × normalize(process_reward) + (1-α) × outcome_reward
```

### 策略 4: 效率感知（β-GRPO）
```
R = 1  if (min_token_prob ≥ β) AND correct   # 高置信且正确
R = 0  otherwise                               # 惩罚不确定时不搜索
```

## 4. DeepSeek-R1 的奖励设计（完整版）

```
Stage 2 (推理 RL):
  - 数学: 验证最终答案是否正确
  - 代码: 编译 + 运行测试用例
  - 格式: 输出必须包含 <think> 和 <answer> 标签
  - 语言一致性: CoT 中目标语言词汇的比例

Stage 4 (全场景 RL):
  - 推理任务: 同上 (规则化)
  - 通用任务: 学习的奖励模型 (helpful + harmless)
```

## 5. 设计原则与反面教训

### ✅ DO

| 原则 | 原因 |
|------|------|
| 优先规则化奖励 | 避免 reward hacking |
| 分阶段设计 | 先格式再准确 |
| 格式违规重罚 | 确保工具调用可解析 |
| 归一化奖励 | 避免梯度爆炸 |

### ❌ DON'T

| 反模式 | 原因 |
|--------|------|
| 不要用长度奖励 | ToolRL 发现会损害小模型性能 |
| 不要用神经奖励模型做推理 | DeepSeek-R1 发现会被 hack |
| 不要一开始就用准确性奖励 | 模型还不会格式时学不到有效信号 |
| 不要奖励过于稀疏 | 多轮交互中信号太弱 |

## 6. 面试高频问题

**Q: 为什么不用神经奖励模型？**
> 对于有可验证答案的任务（数学、代码、事实 QA），规则化奖励更精确、不可被 hack。
> 神经奖励模型会找到捷径（生成看起来合理但实际错误的推理链）。

**Q: 怎么处理奖励稀疏问题？**
> 1. 分阶段设计（先格式后准确）
> 2. 过程奖励提供中间信号
> 3. F1 奖励代替精确匹配（连续值更平滑）
> 4. β-GRPO 用置信度提供额外信号

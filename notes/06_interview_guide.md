# Agentic RL 面试准备指南

## Part 1: 概念理解题

### Q1: 什么是 Agentic RL？和标准 RLHF 有什么区别？

**答**: Agentic RL 将 RL 应用于训练能够使用工具的 LLM Agent，核心区别在于：
1. **多轮交互** vs 单轮生成
2. **动作空间** 包含工具调用（搜索、代码执行等）
3. **奖励** 基于规则可验证，而非神经奖励模型
4. **训练基建** 需要并行环境交互

标准 RLHF 对齐人类偏好，Agentic RL 训练任务完成能力。

---

### Q2: 解释 GRPO 算法。为什么它比 PPO 更适合 Agentic RL？

**答**: GRPO (Group Relative Policy Optimization) 的核心创新是用**组内相对排名替代 Critic 模型**：

1. 对每个 prompt 采样 G 个输出
2. 计算各输出奖励 {r₁...r_G}
3. 优势 = (r_i - mean) / std（**无需值函数**）
4. 用 PPO 风格的 clipped objective 优化

**为什么更适合**:
- 节省 50% 显存（无 Critic）
- 可验证奖励天然适合组内比较
- 更简单、更少超参数
- DeepSeek-R1 验证了其在推理任务上的有效性

**局限**: 不适合 dense per-token reward；需要足够大的 G 保证方差估计。

---

### Q3: 如何为搜索 Agent 设计奖励函数？

**答**: 分三层设计：

```
Level 1 (基础): 结果奖励
  R = EM(prediction, ground_truth)  # 答案对了 = 1

Level 2 (格式): + 格式合规奖励
  R += format_valid(output)  # 正确使用 <search>/<answer> 标签

Level 3 (效率): + 搜索效率感知
  R 仅在模型置信度低时奖励搜索 (β-GRPO)
```

**关键原则**:
- 规则化 > 神经奖励模型（防 hack）
- 分阶段：先格式后准确
- 格式违规重罚（R1-Searcher: -2）

---

### Q4: 解释 Retrieved Token Masking。为什么重要？

**答**: Search-R1 的关键技术。

在 Agent 与环境交互时，输出包含两类 token：
- **模型生成**: think, search query, answer（梯度应作用于这些）
- **环境返回**: 搜索结果（梯度不应作用于这些）

```python
# 对每个 token
I(y_t) = 1  # 模型生成的 token → 参与策略梯度
I(y_t) = 0  # 环境返回的 token → masked from loss
```

**为什么重要**:
1. 防止模型学会"记忆"环境输出
2. 确保梯度只优化模型自身的决策
3. 类似于 seq2seq 中只对 decoder 输出计算 loss

---

### Q5: Agentic RL 中的 mode collapse 怎么处理？

**答**: Mode collapse 是模型输出变得单一的现象。

**诊断**:
- 监控输出熵和多样性指标
- 检查 KL 散度是否过低（策略几乎不变）或过高（完全偏离）
- RAGEN 建议用**互信息**（不是熵）检测 template collapse

**解决方案**:
1. **增大 KL 系数 β** — 拉回参考模型附近
2. **提高采样温度** — rollout 时用 T > 1.0
3. **确保奖励有方差** — 避免 echo trap
4. **多样化训练数据** — 初始状态多样
5. **轨迹过滤** — 跳过方差太低的组

---

## Part 2: 技术深潜题

### Q6: DeepSeek-R1 的四阶段训练管线是什么？每阶段的目的？

**答**:

| 阶段 | 内容 | 目的 |
|------|------|------|
| 1. 冷启动 SFT | 数千条 long-CoT 示例微调 | 让模型学会推理格式 |
| 2. 推理 RL | GRPO + 规则奖励 | 提升推理能力 |
| 3. 拒绝采样 SFT | 600K 推理 + 200K 非推理 | "重置"模型，吸收高质量数据 |
| 4. 全场景 RL | 规则 + 神经奖励模型 | 全面提升（推理 + 通用） |

**R1-Zero**（跳过 Stage 1）证明纯 RL 可以涌现推理，但有可读性问题。

---

### Q7: 如何处理 RL rollout 中的变长工具调用？

**答**: 这是 Agentic RL 独有的系统挑战。

```
核心问题: 不同样本的工具调用次数不同
  样本 A: 2 次搜索
  样本 B: 5 次搜索
  → batch 效率低下

解决方案:
  1. Step-independent rollout (verl-agent)
     - 每次工具调用是独立的推理请求
     - 支持异步处理

  2. 最大交互预算
     - 设置硬上限 B（如最多 5 次搜索）
     - 超过则强制输出答案

  3. Padding + Masking
     - 短序列补齐到最长
     - 环境 token 用 mask 排除

  4. 异步 rollout 服务器
     - 解耦生成和训练
     - 用 session ID 管理并发交互
```

---

### Q8: 对比 on-policy 和 off-policy 方法在 Agent 训练中的表现

**答**:

**On-policy (GRPO/PPO)**: 用当前策略生成 rollout → 计算奖励 → 更新策略
**Off-policy (DPO/RFT)**: 用旧策略或固定数据集训练

**DeepSeekMath 关键发现**: Online >> Offline

```
Online RFT  >>>  Offline RFT  (相同数据量)
On-policy RL >>>  DPO          (显著优势)
```

**原因**: on-policy 数据更好地反映当前策略的分布，提供更有效的梯度信号。

**对于 Agent**: on-policy 更加重要，因为：
- Agent 动作空间更复杂（工具调用 + 文本）
- 环境反馈依赖于具体的动作选择
- 旧数据的分布偏移更严重

---

### Q9: KL 散度在 Agentic RL 中的作用？可以去掉吗？

**答**:

**作用**: 防止策略偏离预训练模型太远，维持文本生成的连贯性，防止 mode collapse。

**PPO vs GRPO 的 KL 实现差异**:
- PPO: 逐 token KL 加到奖励中 → 通过 GAE 传播
- GRPO: KL 直接加到损失函数 → 更简洁

**可以去掉吗？**
- ToolRL 发现：完全移除 KL 在工具任务上**收敛更快且效果相当**
- 但这是任务依赖的：工具任务有强结果奖励约束
- 对于开放式任务（如通用对话），KL 仍然重要

---

### Q10: 什么是 Echo Trap 和 Template Collapse？如何检测和解决？

**答**: 两个来自 RAGEN 论文的关键发现。

**Echo Trap**:
- 奖励方差 → 0，所有采样输出得分相似
- GRPO 分母 std → 0，优势估计爆炸
- 表现：梯度尖峰、训练不稳定
- **检测**: 监控每组 G 个样本的奖励标准差
- **解决**: SNR 自适应过滤（奖励方差太低时跳过该组）

**Template Collapse**:
- 对单个输入有多样输出 → 表面正常
- 对不同输入输出相同策略 → 实质退化
- **检测**: 熵指标看不出！需要计算输入-输出互信息
- **解决**: 多样初始状态、中等交互粒度、更频繁采样

---

## Part 3: 系统设计题

### Q11: 如何设计可扩展的 Agentic RL 训练系统？

**答**: 参考 AgentRL (THUDM) 和 veRL 架构。

```
┌─────────────────────────────────────────┐
│            Orchestrator (单控制器)         │
└─────────┬──────────┬──────────┬─────────┘
          │          │          │
    ┌─────▼────┐ ┌───▼─────┐ ┌──▼────────┐
    │ Rollout  │ │ Actor   │ │ Reference │
    │ Workers  │ │ Workers │ │ Workers   │
    │(vLLM推理)│ │(梯度计算)│ │(冻结基线) │
    │          │ │(FSDP)   │ │(KL计算)   │
    └─────┬────┘ └───▲─────┘ └───────────┘
          │          │
    ┌─────▼──────────┘
    │ Environment Pool
    │ (Go-based, 10K+ sessions)
    │ - 搜索引擎
    │ - 代码沙箱
    │ - API 模拟器
    └────────────────────
```

**关键设计决策**:
1. **推理-训练分离**: vLLM 做 rollout，PyTorch 做梯度更新
2. **异步环境**: 不阻塞训练循环
3. **权重同步**: 训练更新后同步到推理服务器
4. **并行度**: Ray-based 多 worker 并行

---

### Q12: vLLM 在 RL 训练中的角色？

**答**:

RL 训练的瓶颈是 **rollout 生成**。GRPO 需要每 prompt 采样 G 个输出。

**vLLM 的作用**:
1. **连续 batching**: 不同长度请求高效并行
2. **PagedAttention**: 节省 KV cache 显存
3. **CUDA graphs**: 减少 kernel launch 开销
4. **Tensor parallelism**: 大模型多卡推理

**集成模式**:
```
训练循环:
  1. 收集 prompt batch
  2. → 发送到 vLLM 推理服务器
  3. ← 返回 G 个 completion / prompt
  4. 计算奖励、优势
  5. 策略更新
  6. 同步权重到 vLLM
  7. 重复
```

---

## Part 4: 速答题

**Q: 为什么 GRPO 而不是 PPO？**
> 无 Critic = 50% 显存节省。组内 baseline 对可验证奖励任务效果好。DeepSeek-R1 后的标准选择。

**Q: 为什么规则化奖励？**
> 神经奖励模型可被 hack。有可验证结果的任务（数学、代码、搜索），规则化严格更优。

**Q: Agentic RL 最大的挑战？**
> 跨多轮的信用分配、变长 rollout 的 batch 效率、训练中的环境交互延迟。

**Q: 怎么调试 reward hacking？**
> 手动检查高奖励样本。如果看起来差，奖励有问题。切换到规则化奖励、加 KL、裁剪奖励。

**Q: 小模型该做 RL 还是蒸馏？**
> DeepSeek 发现蒸馏 > RL。小模型 RL 能力有限，不如从大模型蒸馏。但如果没有大模型蒸馏源，分阶段 RL 仍可行。

**Q: veRL 和 TRL 的区别？**
> veRL 专为多轮 Agent RL 设计，支持环境交互、异步 rollout、vLLM 集成。TRL 主要面向单轮 RLHF。

---

## Part 5: 必读论文清单（按重要性排序）

| 优先级 | 论文 | 要点 |
|--------|------|------|
| ⭐⭐⭐ | DeepSeek-R1 | 基石。4 阶段管线、GRPO、规则奖励、涌现推理 |
| ⭐⭐⭐ | DeepSeekMath | 提出 GRPO。统一 RL 框架对比 |
| ⭐⭐⭐ | RAGEN | 失败诊断。Echo Trap、Template Collapse |
| ⭐⭐ | Search-R1 | 干净的开源实现。Retrieved token masking |
| ⭐⭐ | R1-Searcher | 两阶段设计。Reinforce++ |
| ⭐⭐ | ToolRL | 细粒度奖励。冷启动 > SFT init 发现 |
| ⭐ | Agent-R1 | 模块化多工具。过程 + 结果奖励 |
| ⭐ | AgentRL (THUDM) | 系统设计参考 |

## Part 6: 面试展示 Demo 建议

如果面试需要展示项目经验：

```
方案 1 (最佳): 复现 Search-R1
  - 代码清晰、社区活跃
  - 可以在 CPU 上跑通数据管线和奖励计算
  - 讨论 retrieved token masking 的实现

方案 2: 复现 R1-Searcher 的奖励设计
  - 两阶段奖励是好的讨论话题
  - 可以在 CPU 上实现奖励函数并测试

方案 3: 实现简化版 GRPO
  - 手写 GRPO 训练循环
  - 在小规模任务（如数学）上验证
```

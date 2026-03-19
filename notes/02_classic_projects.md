# 经典项目详解

## 1. DeepSeek-R1 — 基石

- **论文**: https://arxiv.org/abs/2501.12948
- **GitHub**: https://github.com/deepseek-ai/DeepSeek-R1
- **发表于 Nature (2025)**

### 核心思想
纯 RL（无监督推理示范）可以在 LLM 中涌现推理能力 — 自我反思、验证、动态策略适应。

### 训练管线（4 阶段）

```
Stage 1: 冷启动 SFT
  └─ 在数千条 long-CoT 示例上微调 DeepSeek-V3-Base

Stage 2: 推理导向 RL
  └─ GRPO + 准确性奖励 + 格式奖励 + 语言一致性奖励

Stage 3: 拒绝采样 SFT
  └─ ~600K 推理样本 + ~200K 非推理样本，重新 SFT

Stage 4: 全场景 RL
  └─ 推理任务: 规则奖励; 通用任务: 神经奖励模型
```

### R1-Zero（纯 RL 变体）
- 直接在 DeepSeek-V3-Base 上做 RL，**没有任何 SFT**
- **涌现行为**：自发出现 Chain-of-Thought、自我验证、反思（"aha moment"）
- **问题**：可读性差、语言混合

### 关键发现
- 小模型做 RL **不如** 从大模型蒸馏
- 蒸馏用 800K 精选样本 SFT 小模型（Qwen2.5 1.5B-32B, Llama 8B-70B）

---

## 2. Search-R1 — 搜索 Agent 的 RL 训练

- **GitHub**: https://github.com/PeterGriffinJin/Search-R1 (Apache-2.0)
- **核心框架**: veRL

### 核心思想
将 DeepSeek-R1 的 RL 方法扩展到**交替推理与实时搜索调用**。开源版 DeepResearch。

### 技术细节
- **算法**: 支持 PPO、GRPO、REINFORCE
- **基座模型**: Llama-3.2-3B-base, Qwen2.5-7B-base
- **奖励**: 规则化 + ground-truth 答案匹配
- **数据**: NQ (Natural Questions) + Wikipedia 语料库

### 交互格式
```
<think>推理过程</think>
<search>搜索查询</search>
<information>搜索结果（环境返回，masked from loss）</information>
<think>基于搜索结果继续推理</think>
<answer>最终答案</answer>
```

### 关键技术：Retrieved Token Masking
- 环境返回的 token 设置 `I(y_t) = 0`（不参与梯度计算）
- 模型生成的 token 设置 `I(y_t) = 1`
- 确保策略梯度只作用于模型自身的输出

---

## 3. R1-Searcher — 两阶段 RL 搜索

- **GitHub**: https://github.com/RUCAIBox/R1-Searcher
- **机构**: 中国人民大学 AI Box

### 核心思想
两阶段 outcome-supervised RL，**无需 SFT 冷启动**即可学会搜索。

### 两阶段训练

```
Stage 1: 学习 "何时搜索"
  ├─ 只有格式奖励 (+0.5 正确检索格式)
  ├─ 强制要求检索
  └─ 学会 <begin_of_search>/<end_of_search> 格式

Stage 2: 学习 "搜索什么"
  ├─ 格式奖励 + F1-based 答案奖励
  ├─ 移除强制检索要求
  └─ 模型自主决定是否搜索
```

### 算法
- **Reinforce++**（REINFORCE 变体，带方差减少）
- **格式违规惩罚**: -2 分

### 结果
- 7B 模型在 HotpotQA、2WikiMultiHopQA 等多跳 QA 上接近或超过 GPT-4o-mini
- 从本地检索训练到在线搜索的**零样本泛化**能力

---

## 4. ReSearch / ReCall — 任意工具调用的 RL

- **GitHub**: https://github.com/Agent-RL/ReSearch → https://github.com/Agent-RL/ReCall

### 核心思想
训练 LLM "通过 RL 学会使用和组合任意工具"，**无需监督工具使用轨迹数据**。

### 技术细节
- **算法**: PPO, 基于 veRL (v0.3.0 + vLLM 0.8.4)
- **基座**: Qwen2.5-7B-Instruct
- **训练数据**: SynTool 合成数据 + MuSiQue 训练集
- **评估**: FlashRAG 多跳 QA, Bamboogle, BFCL

### 突破
通过纯 RL 实现**零样本工具组合**（zero-shot tool composition）。

---

## 5. RAGEN — Agentic RL 的诊断框架（必读！）

- **GitHub**: https://github.com/RAGEN-AI/RAGEN
- **论文**: https://arxiv.org/abs/2504.20073

### 核心思想
识别 Agentic RL **为什么会失败**并提供解决方案。这可能是理解 pitfalls 最重要的论文。

### StarPO 框架
State-Thinking-Actions-Reward Policy Optimization — 通用轨迹级 Agent RL 框架。

### 关键发现：失败模式

#### Echo Trap（回声陷阱）
- 奖励方差坍缩到接近零
- 所有采样输出得分相似
- 梯度尖峰导致训练不稳定
- **解决**: 轨迹过滤、SNR 自适应过滤

#### Template Collapse（模板坍缩）
- 对单个输入看起来多样
- 但对不同输入使用相同策略
- **标准熵指标无法检测**
- 需要互信息指标

### StarPO-S（稳定变体）
轨迹过滤 + Critic 集成 + 梯度稳定化

---

## 6. Agent-R1 — 模块化多工具 Agent

- **GitHub**: https://github.com/tonyzhoup/Agent-R1

### 核心思想
端到端 RL 框架，支持多工具协调。开发者定义工具和奖励函数，无需复杂工作流工程。

### 技术细节
- **算法**: PPO, GRPO, REINFORCE++
- **架构**: `BaseTool`（单个工具）+ `BaseToolEnv`（状态转移）
- **奖励设计**: 双重奖励
  - **过程奖励**: 每次工具调用的有效性评分
  - **结果奖励**: 最终任务准确性
  - 通过 PRIME 框架归一化平衡

### 应用
- PaperScout（学术搜索）
- TableMind（表格推理，WSDM 2026）

---

## 7. AgentRL (THUDM) — 大规模 Agentic RL 基建

- **GitHub**: https://github.com/THUDM/AgentRL (MIT)
- **论文**: https://arxiv.org/abs/2510.04206
- **机构**: 清华大学 THUDM

### 核心思想
可扩展的多轮多任务 RL 框架，支持大规模并行环境交互。

### 架构：三池 Ray 系统

```
┌─────────────┐   ┌─────────────┐   ┌─────────────┐
│  Rollout     │   │   Actor      │   │  Reference   │
│  Workers     │   │   Workers    │   │  Workers     │
│  (探索任务,  │   │  (计算梯度,  │   │  (维护冻结   │
│   流式轨迹)  │   │   更新策略)  │   │   基线, KL)  │
└─────────────┘   └─────────────┘   └─────────────┘
```

- **环境**: Go-based 控制器，支持 10,000 并发 sessions
- **特性**: 跨策略采样、任务优势归一化、异步任务生成

---

## 8. Tool-Star — 自驱动工具使用 RL

- **GitHub**: https://github.com/dongguanting/Tool-Star

### 核心思想
LLM 在逐步推理中自主学习调用 6 种外部工具。

### 两阶段训练
1. **冷启动 SFT**: 用 Llama Factory
2. **Self-Critic RL**: GRPO 训练 + 可选 Self-Critic DPO 精炼

### 奖励
答案准确性 + 工具执行结果（Self-Critic 设计）

### 结果
在 10+ 基准测试上表现强劲（AIME24, MATH500, WebWalker, HotpotQA）。

---

## 项目对比总结

| 项目 | 算法 | 基座模型 | 奖励设计 | 特色 |
|------|------|----------|----------|------|
| DeepSeek-R1 | GRPO | V3-Base | 规则化 | 纯 RL 涌现推理 |
| Search-R1 | GRPO/PPO | Qwen2.5-7B | 结果奖励 | Retrieved token masking |
| R1-Searcher | Reinforce++ | Qwen-2.5-7B | 两阶段奖励 | 无 SFT 冷启动 |
| ReSearch | PPO | Qwen2.5-7B-Inst | 结果奖励 | 零样本工具组合 |
| RAGEN | GRPO/PPO | 多种 | 多种 | 失败诊断框架 |
| Agent-R1 | GRPO/PPO | Qwen | 过程+结果 | 模块化多工具 |
| AgentRL | GRPO | 多种 | 多种 | 大规模基建 |
| Tool-Star | GRPO | Qwen2.5 | Self-Critic | 6 种工具类型 |

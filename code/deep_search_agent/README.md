# DeepSearch Agent with Intent-Aware Process Reward Model

> 用 RL 训练 LLM 搜索 Agent，核心创新：Intent-Aware Process Reward Model (IA-PRM)

## 项目定位

这是一个可在 4×A100 40G 上真实跑通的 toy demo，核心讲述**从静态意图分类 → 动态意图感知的序列决策**的技术升级故事。

## 核心创新：IA-PRM

传统 Process Reward Model (PRM) 只评估"这一步对不对"，IA-PRM 额外评估"这一步是否与用户原始意图对齐"：

```
                    ┌─────────────────────┐
                    │   Qwen2.5-1.5B      │
                    │   (Base Encoder)     │
                    └──────────┬──────────┘
                               │
                    ┌──────────┴──────────┐
                    │    Hidden States     │
                    └──────────┬──────────┘
                         ┌─────┴─────┐
                         │           │
                 ┌───────▼──┐  ┌────▼────────┐
                 │ Progress │  │   Intent     │
                 │  Head    │  │ Alignment    │
                 │ (0~1)    │  │ Head (0~1)   │
                 └───────┬──┘  └────┬────────┘
                         │          │
                         ▼          ▼
              step_reward = α × progress + β × intent_alignment
```

**面试叙事**: "我做二分类意图识别时发现静态判断不够，所以在这个项目里我把意图对齐做成了每一步的动态信号——这就是 IA-PRM 的 intent_alignment head。"

## 架构

```
deep_search_agent/
├── configs/default.yaml          # 超参配置
├── env/
│   ├── wiki_search_env.py        # Wikipedia 搜索环境 (gym-like)
│   └── dataset.py                # HotpotQA 数据加载 + F1/EM 指标
├── models/
│   ├── agent.py                  # 搜索 Agent (Qwen2.5-3B-Instruct + LoRA)
│   └── intent_prm.py            # IA-PRM (Qwen2.5-1.5B + 双 head) ⭐
├── training/
│   ├── sft_warmstart.py          # Phase 1: SFT 暖启动
│   ├── prm_trainer.py            # Phase 2: PRM 训练
│   └── grpo_trainer.py           # Phase 3: GRPO RL 训练
├── evaluation/
│   └── evaluate.py               # 评估 + 对比实验
└── scripts/
    ├── run_sft.sh                # ~1h, 单卡
    ├── run_prm.sh                # ~30min, 单卡
    ├── run_grpo.sh               # ~3-5h, 4×A100
    └── run_eval.sh
```

## 搜索环境

Agent 在 HotpotQA 多跳问题上与 Wikipedia 交互：

- **State**: `(question, search_history, retrieved_passages, step_count)`
- **Action Space** (结构化文本输出):
  - `<search>query</search>` — 搜索 Wikipedia
  - `<refine>new_query</refine>` — 重构搜索 query
  - `<answer>final_answer</answer>` — 给出最终答案
- **终止条件**: 输出 answer 或达到最大步数 (5步)
- **Wikipedia API**: `wikipedia-api` 库 + 本地磁盘缓存

## 三阶段训练

### Phase 1: SFT 暖启动
- 规则 agent 在 HotpotQA 上生成 ~2000 条成功搜索轨迹
- Qwen2.5-3B LoRA SFT (rank=64)
- 资源: 单卡, ~1小时

### Phase 2: PRM 训练
- 收集 SFT 模型的 rollout 轨迹 (~5000 条)
- 启发式标注每步的 (progress, intent_alignment) 分数
- 训练 Qwen2.5-1.5B + 双头 PRM
- 资源: 单卡, ~30分钟

### Phase 3: GRPO 训练
- Group Relative Policy Optimization
- Reward = outcome_reward (F1) + Σ(step_rewards from IA-PRM)
- 4×A100 40G, batch_size=4/GPU, gradient_accumulation=4
- ~500 episodes, 3-5小时

## 评估指标

| Metric | Description |
|--------|-------------|
| Answer F1 | Token-level F1 vs ground truth |
| Answer EM | Exact match |
| Avg Steps | 平均搜索步数 (越少越高效) |
| Intent Drift Rate | 搜索过程中偏离意图的步骤比例 (IA-PRM 度量) |
| Success Rate | F1 > 0.5 的比例 |

对比实验: Base → SFT-only → SFT+GRPO(outcome) → SFT+GRPO(outcome+IA-PRM)

## Quick Start

```bash
# 安装依赖
pip install -r requirements.txt

# 验证环境
python -m env.wiki_search_env --test
python -m models.agent --test

# 训练流程
bash scripts/run_sft.sh           # Phase 1
bash scripts/run_prm.sh           # Phase 2
bash scripts/run_grpo.sh          # Phase 3

# 评估
bash scripts/run_eval.sh
```

### 小数据快速验证

```bash
# 仅用 50 条数据验证 SFT 流程
MAX_SAMPLES=50 bash scripts/run_sft.sh

# 仅跑 10 个 episode 验证 GRPO
NUM_EPISODES=10 bash scripts/run_grpo.sh

# 评估 100 条
MAX_SAMPLES=100 bash scripts/run_eval.sh
```

## 关键技术点 (面试用)

### 1. 为什么用 GRPO 而不是 PPO？
- GRPO 不需要 value function (少一个模型)，用 group 内相对排名做 advantage
- 更适合 LLM fine-tuning：内存开销更小，训练更稳定
- DeepSeek-R1 验证了 GRPO 在推理任务上的有效性

### 2. IA-PRM 的创新点？
- **双头设计**: progress head 评估搜索进度，intent alignment head 评估意图对齐
- **Cross-attention**: intent head 用 question 表示做 cross-attention，显式建模意图与动作的关联
- **动态信号**: 不同于静态意图分类，IA-PRM 在每个搜索步骤上动态评估意图对齐

### 3. Process Reward vs Outcome Reward？
- Outcome reward 稀疏 (只有最后一步有)，credit assignment 困难
- Process reward 密集，提供逐步反馈，减少 reward hacking
- IA-PRM 的 intent alignment 可以防止 agent 的"意图漂移" (搜索着搜着就跑题了)

### 4. 启发式标注 vs 人工标注？
- 用 BM25/embedding similarity 作为 proxy label 构造初始训练数据
- 可以少量人工标注做校准
- 面试可以说"在资源有限时用启发式快速迭代，后续可接入人工标注提升质量"

## 硬件需求

| Phase | GPU | 时间 | 内存 |
|-------|-----|------|------|
| SFT | 1× A100 40G | ~1h | ~20GB |
| PRM | 1× A100 40G | ~30min | ~10GB |
| GRPO | 4× A100 40G | ~3-5h | ~35GB/卡 |
| Eval | 1× A100 40G | ~1h | ~20GB |

## 依赖

- PyTorch ≥ 2.1
- Transformers ≥ 4.40
- PEFT ≥ 0.10 (LoRA)
- TRL ≥ 0.8 (GRPO)
- Accelerate ≥ 0.28 (多卡)
- wikipedia-api (搜索环境)
- datasets (HotpotQA)

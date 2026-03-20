# Harness Engineering 在 Agentic RL 中的应用

> 从 AgentHarness 项目出发，研究"训练夹具"如何系统性解决 Agentic RL 的工程断层

---

## 目录

- [一、什么是 Harness Engineering](#一什么是-harness-engineering)
- [二、Agentic RL 的三大工程断层](#二agentic-rl-的三大工程断层)
- [三、AgentHarness 的四层架构解析](#三agentharness-的四层架构解析)
- [四、对标分析：现有项目如何处理 Harness 问题](#四对标分析现有项目如何处理-harness-问题)
- [五、核心技术深挖：Reward Engine](#五核心技术深挖reward-engine)
- [六、核心技术深挖：Credit Assignment](#六核心技术深挖credit-assignment)
- [七、核心技术深挖：Reward Debugger](#七核心技术深挖reward-debugger)
- [八、Harness 如何解决 Search-R1 中的真实问题](#八harness-如何解决-search-r1-中的真实问题)
- [九、面试中如何讲 Harness Engineering](#九面试中如何讲-harness-engineering)
- [十、Harness Engineering 的未来方向](#十harness-engineering-的未来方向)

---

## 一、什么是 Harness Engineering

### 1.1 从传统 RL 说起

传统 RL 有一个干净的抽象：

```
Agent ←→ Environment
  └── action → env.step() → observation, reward, done
```

OpenAI Gymnasium 把这个抽象标准化了。任何任务只要实现 `reset()` / `step()` / `reward`，就能被任何 RL 算法训练。**Gymnasium 就是传统 RL 的 harness。**

### 1.2 Agentic RL 的问题

Agentic RL（用 RL 训练 LLM Agent）比传统 RL **复杂一个数量级**：

```
传统 RL:     Agent (小网络)  ←→  Environment (Gym)
Agentic RL:  Agent (7B LLM)  ←→  Environment (搜索/代码/浏览器)
                                      ↕
                               外部工具 (APIs)
                                      ↕
                               奖励计算 (规则/模型/多维度)
                                      ↕
                               训练框架 (veRL/OpenRLHF/TRL)
                                      ↕
                               轨迹管理 (收集/过滤/课程)
```

**问题**：没有 Gymnasium 这样的标准层把这些粘合在一起。每个项目都从零搭建自己的粘合代码。

### 1.3 Harness Engineering 的定义

**Harness（夹具）**：在制造业中，夹具用于固定工件，使加工过程标准化、可重复。

**Harness Engineering 在 Agentic RL 中**：设计和实现连接 Environment、Reward、Training 三者的标准化中间层，使任何任务能快速变成可训练的 Agent RL 环境。

```
类比：
  Gymnasium       → 传统 RL 的 harness
  pytest          → 测试的 harness
  AgentHarness    → Agentic RL 的 harness
```

---

## 二、Agentic RL 的三大工程断层

### 断层 1: Environment → Reward

```
痛点: 每个环境的 reward 都要从零写，无法复用

Search-R1 的 reward:     手写 EM/F1 + format check
R1-Searcher 的 reward:   手写 7 种 format check + F1 + 惩罚
Agent-R1 的 reward:      手写 process reward + outcome reward
Tool-Star 的 reward:     手写 self-critic + accuracy

全都是 ad-hoc 的一次性代码，没有任何复用。
```

### 断层 2: Reward → Training

```
痛点: reward signal 格式不统一，换训练框架要重写

veRL 期望:        token_level_rewards tensor, shape (bs, seq_len)
OpenRLHF 期望:    HTTP API 返回 {"rewards": [float]}
TRL 期望:         reward_fn(samples, prompts, outputs) -> list[float]

同一个 reward 逻辑，要为三个框架写三个版本。
```

### 断层 3: Environment → Training

```
痛点: trajectory 收集、格式、replay 无标准

Search-R1:    DataProto + meta_info（veRL 私有格式）
R1-Searcher:  OpenRLHF experience maker（另一种私有格式）
Agent-R1:     自定义 trajectory 结构

切换框架 = 重写整个数据管线。
```

**Harness Engineering 的目标就是消除这三大断层。**

---

## 三、AgentHarness 的四层架构解析

### 3.1 架构总览

```
                          AgentHarness
┌──────────────────────────────────────────────────────┐
│                                                      │
│  ┌─────────────┐  ┌──────────────┐  ┌─────────────┐ │
│  │ Environment  │  │   Reward     │  │  Trajectory │ │
│  │ Protocol     │  │   Engine     │  │  Store      │ │
│  └──────┬──────┘  └──────┬───────┘  └──────┬──────┘ │
│         └────────────────┼─────────────────┘        │
│              ┌───────────▼──────────┐               │
│              │   Harness Runtime    │               │
│              └───────────┬──────────┘               │
│  ┌───────────────────────▼───────────────────────┐  │
│  │           Training Backend Adapters           │  │
│  │  ┌──────┐  ┌────────┐  ┌─────┐  ┌─────────┐  │  │
│  │  │ veRL │  │OpenRLHF│  │ TRL │  │ Custom  │  │  │
│  │  └──────┘  └────────┘  └─────┘  └─────────┘  │  │
│  └───────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────┘
```

### 3.2 Layer 1: Environment Protocol

AgentHarness 定义了 `AgentEnv` 协议——5 个方法，任何任务实现即可接入：

```python
class AgentEnv(ABC):
    def reset(self, task: dict) -> Observation:        # 初始化
    def step(self, action: Action) -> (Observation, bool):  # 执行
    def get_ground_truth(self) -> Any:                  # 正确答案
    def get_available_tools(self) -> list[ToolSpec]:    # 工具列表
    def get_state_snapshot(self) -> dict:               # 状态快照
```

**对比 Search-R1 的做法**：Search-R1 把环境逻辑分散在 `generation.py` 的 `execute_predictions()` 和 `batch_search()` 中，没有统一协议。如果要换一个环境（比如从搜索换成代码执行），需要重写整个 generation 循环。

**AgentHarness 的优势**：换环境只需换一个 `AgentEnv` 实例，训练循环不变。

### 3.3 Layer 2: Reward Engine（最核心的创新）

**声明式组合 vs 命令式硬编码**

```python
# ❌ Search-R1 / R1-Searcher 的做法：一大坨 if-else
def get_reward(queries):
    scores = []
    for i in range(len(queries)):
        f1_score_now = f1_score(preds[t], answers[t])
        scores.append(float(f1_score_now))
        if "<answer>" not in solutions[i]:
            scores[i] = 0.0
        if count_1 != count_2:
            format_punishment = True
        if have_chinese:
            format_punishment = True
        if format_punishment:
            scores[i] = scores[i] - 2
    return scores

# ✅ AgentHarness 的做法：声明式组合
reward = RewardComposer([
    exact_match(key="answer", weight=0.6),
    format_follows(pattern="<think>...</think><answer>...</answer>", weight=0.2),
    trajectory_efficiency(max_turns=5, weight=0.1),
    tool_call_valid(weight=0.1),
])
```

**关键区别**：
- 可复用：`exact_match` 在任何 QA 任务上都能用
- 可调试：`compute_breakdown()` 看每个维度的得分
- 可组合：像乐高积木一样拼接
- 可替换：换一个组件不影响其他

### 3.4 Layer 3: Trajectory Store

统一的轨迹数据结构：

```python
class Trajectory:
    task: dict              # 任务定义
    turns: list[Turn]       # 交互轮次序列
    total_reward: float     # 总奖励
    success: bool           # 是否成功
    metadata: dict          # 额外元数据

class Turn:
    action: Action          # Agent 的动作
    observation: Observation # 环境的反馈
    reward: float | None    # 单轮奖励（credit assignment 后填入）
```

**关键方法**：
- `to_messages()` → 转换为 chat 格式（适配任何训练框架）
- `to_dict()` / `from_dict()` → 序列化/反序列化
- `get_final_answer()` → 提取最终答案

**对比 Search-R1**：Search-R1 用 `DataProto`（veRL 私有格式）+ `meta_info`（字典）+ 各种 tensor（`input_ids`, `info_mask`, `responses_with_info_mask`），完全绑定 veRL。换到 OpenRLHF 需要重写所有数据处理。

### 3.5 Layer 4: Training Backend Adapters

```python
# 一行代码切换训练框架
harness = AgentHarness(
    env=SearchEnv(),
    reward=reward,
    backend=veRLBackend(algorithm="grpo", model="Qwen/Qwen2.5-7B"),
)

# 想换框架？改一行
harness = AgentHarness(
    env=SearchEnv(),
    reward=reward,  # 同一个 reward，不用改
    backend=OpenRLHFBackend(algorithm="reinforce++", model="Qwen/Qwen2.5-7B"),
)
```

---

## 四、对标分析：现有项目如何处理 Harness 问题

| 维度 | Search-R1 | R1-Searcher | Agent-R1 | AgentHarness |
|------|-----------|-------------|----------|--------------|
| **环境协议** | 无，硬编码在 generation.py | 无，散在 experience_maker 中 | BaseTool + BaseToolEnv | AgentEnv (5 methods) |
| **Reward** | 一个 reward_fn.py 文件 | 一个 reward_server.py + 大量 if-else | PRIME 框架但不可拆分 | RewardComposer (声明式) |
| **Reward 调试** | 无 | 手动日志 | 无 | RewardDebugger ✅ |
| **Credit Assignment** | outcome only | outcome + format | process + outcome (PRIME) | CreditAssigner (3 种策略) |
| **轨迹格式** | DataProto (veRL) | OpenRLHF experience | 自定义 | Trajectory (标准化) |
| **训练框架** | 绑定 veRL | 绑定 OpenRLHF | 绑定 veRL | 可插拔 |
| **换环境成本** | 重写 generation.py | 重写 experience_maker | 实现 BaseTool | 实现 AgentEnv (5 methods) |
| **换 reward 成本** | 重写 reward_fn | 重写 reward_server | 重写 reward 逻辑 | 替换 Reward 组件 |

---

## 五、核心技术深挖：Reward Engine

### 5.1 Reward 基类设计

```python
class Reward(ABC):
    def __init__(self, weight: float = 1.0, name: str = None):
        self.weight = weight
        self._name = name

    @abstractmethod
    def compute(self, trajectory: Trajectory, **kwargs) -> float:
        """返回 [0, 1] 范围的奖励分数"""
        ...
```

**设计原则**：
- 统一范围 [0, 1]：不同维度可加权求和
- 接受 Trajectory 而非原始文本：可以做多轮分析
- kwargs 传递额外上下文（ground_truth, env_state）

### 5.2 RewardComposer：声明式组合

```python
class RewardComposer(Reward):
    def compute(self, trajectory, **kwargs):
        score = 0.0
        for r in self.rewards:
            component_score = r.compute(trajectory, **kwargs)
            score += r.weight / total_weight * component_score
        return max(0.0, min(1.0, score))

    def compute_breakdown(self, trajectory, **kwargs):
        """关键方法：返回每个组件的独立得分"""
        return {r.name: r.compute(trajectory, **kwargs) for r in self.rewards}
```

`compute_breakdown` 是 debug 的基础——你可以看到哪个维度拖了后腿。

### 5.3 与 RL 训练的接口

```
AgentHarness Reward Engine
        ↓
  compute(trajectory) → float (composite score)
  compute_breakdown(trajectory) → dict (per-component)
        ↓
  CreditAssigner.assign(trajectory) → list[float] (per-turn rewards)
        ↓
  Backend Adapter 转换为框架期望的格式:
    veRL:      → token_level_rewards tensor
    OpenRLHF:  → HTTP JSON {"rewards": [...]}
    TRL:       → list[float]
```

---

## 六、核心技术深挖：Credit Assignment

### 6.1 为什么 Credit Assignment 是关键

```
多轮交互中，最终 reward=1（答对了）。但中间哪一步贡献了？

Turn 1: <think> 分析问题 </think>              ← 贡献多少？
Turn 2: <search> 搜索 query </search>           ← 这次搜索有用吗？
Turn 3: <think> 分析搜索结果 </think>            ← 推理质量如何？
Turn 4: <search> 更精确的 query </search>        ← 第二次搜索必要吗？
Turn 5: <answer> 最终答案 </answer>              ← reward=1

如果所有 turn 都得 reward=1（outcome only），那有效搜索和无效搜索得到相同的梯度信号。
```

### 6.2 AgentHarness 的三种策略

```python
class CreditAssigner:
    strategies = {
        "outcome_only": 所有 turn 得相同奖励,          # 简单但粗糙
        "turn_level":   每个 turn 独立评分,             # 精确但需要 turn reward fn
        "hybrid":       0.7 * outcome + 0.3 * turn,   # 推荐
    }
```

**outcome_only**（Search-R1 用的）：
```
Turn 1: 1.0 ← 所有 turn 都得 1.0
Turn 2: 1.0
Turn 3: 1.0
Turn 4: 1.0
Turn 5: 1.0
```

**hybrid**（AgentHarness 推荐）：
```
Turn 1: 0.7 * 1.0 + 0.3 * 0.5 = 0.85  ← 推理质量中等
Turn 2: 0.7 * 1.0 + 0.3 * 0.8 = 0.94  ← 搜索很相关
Turn 3: 0.7 * 1.0 + 0.3 * 0.7 = 0.91  ← 分析合理
Turn 4: 0.7 * 1.0 + 0.3 * 0.2 = 0.76  ← 这次搜索不太必要
Turn 5: 0.7 * 1.0 + 0.3 * 1.0 = 1.00  ← 答对了
```

**对应 GRPO 的接口**：credit assignment 的输出 list[float] 可以直接作为 GRPO 的 reward 输入，不需要额外转换。

### 6.3 与学术方法的对应

| AgentHarness 策略 | 对应学术方法 | 使用项目 |
|-------------------|-------------|---------|
| outcome_only | Outcome-supervised RL | Search-R1, DeepSeek-R1 |
| turn_level | Process Reward Model (PRM) | Agent-R1 (PRIME) |
| hybrid | StarPO (RAGEN) | RAGEN v2 |

---

## 七、核心技术深挖：Reward Debugger

### 7.1 为什么需要 Reward Debugger

R1-Searcher 的 reward_server.py 有 **7 种不同的格式检查**，每一种都是在发现 hack 行为后才加上的。如果有 debugger，可以提前发现问题。

### 7.2 Debugger 的三个功能

**功能 1: 统计分析**

```python
report = debugger.analyze(trajectories, ground_truth="Paris")
print(report.summary())

# 输出:
# Component            Mean    Std    Min    Max  Risk
# --------------------------------------------------------
# exact_match          0.420  0.494  0.000  1.000 Low
# format_follows       0.950  0.080  0.500  1.000 High(!)
# tool_call_valid      0.870  0.150  0.000  1.000 Medium
```

**功能 2: Hacking 检测**

```python
alerts = debugger.detect_hacking(trajectories)

# 检测 3 种风险:
# 1. 饱和 (saturation):  mean > 0.9 且 85%+ 样本得分 > 0.9
#    → format_follows 太容易拿分，模型可能在 hack
# 2. 低方差 (low variance): std < 0.05 且 mean > 0.5
#    → 奖励没有区分度，梯度信号弱
# 3. 触底 (floor):  mean < 0.1
#    → 奖励太严格，模型学不到任何东西
```

**功能 3: A/B 对比**

```python
comparison = debugger.compare(reward_v1, reward_v2, trajectories)
print(comparison.summary())

# 输出:
# A (reward_v1): mean=0.42, std=0.30
# B (reward_v2): mean=0.55, std=0.25
# Correlation: 0.78
# Ranking Agreement: 85.2%
```

### 7.3 Debugger 如何防止 Reward Hacking

```
传统流程（无 Debugger）:
  设计 reward → 训练 → 发现 hacking → 修改 reward → 重训
  周期: 数天到数周

AgentHarness 流程（有 Debugger）:
  设计 reward → Debugger 预分析 → 发现潜在风险 → 提前修复 → 训练
  周期: 数小时

具体例子:
  你设计了 format_follows 作为 reward 组件。
  Debugger 发现 95% 的样本在这个维度得分 > 0.9。
  报警: "near-saturated, model can trivially maximize this"
  建议: "increase difficulty or reduce weight"
  你还没开始训练就知道了这个问题。
```

---

## 八、Harness 如何解决 Search-R1 中的真实问题

### 问题 1: Retrieved Token Masking

```
Search-R1 的问题: info_mask 和 attention_mask 混淆，v0.1 有 bug

AgentHarness 的解法:
  Trajectory 的 Turn 结构天然区分了 action（模型生成）和 observation（环境返回）。
  Backend Adapter 在转换时自动生成正确的 mask:
    action tokens → I(y_t) = 1
    observation tokens → I(y_t) = 0
  不需要手动维护 info_mask——结构本身就是 mask。
```

### 问题 2: 格式检查遗漏

```
R1-Searcher 的问题: 7 种格式检查是逐步加上的，容易遗漏

AgentHarness 的解法:
  format_follows 是一个标准的 Reward 组件:
    - 预置常见检查（标签配对、标签嵌套、禁止角色泄露等）
    - 可组合自定义检查
    - Debugger 会检测 format reward 是否饱和

  reward = RewardComposer([
      format_follows(required_tags=["think", "answer"],
                     optional_tags=["search"],
                     forbidden_strings=["Assistant", "Human"],
                     max_answer_length=20,
                     weight=0.2),
      ...
  ])
```

### 问题 3: 变长 Rollout 的 Batch 对齐

```
Search-R1 的问题: 手动维护 active_mask、GPU padding、序列截断

AgentHarness 的解法:
  Environment Protocol 的 step() 返回 (observation, done)。
  Harness Runtime 自动管理 active/done 状态:
    - 自动跳过已完成的 episode
    - 自动 padding 到 GPU 对齐
    - 自动截断超长序列
  开发者只需实现 AgentEnv，不需要关心 batch 管理。
```

### 问题 4: 训练框架切换

```
Search-R1 绑定 veRL，R1-Searcher 绑定 OpenRLHF。

AgentHarness 的解法:
  Training Backend Adapter 抽象:
    - veRL Adapter: 将 Trajectory 转为 DataProto + info_mask
    - OpenRLHF Adapter: 将 Reward 包装为 HTTP reward server
    - TRL Adapter: 将 Trajectory 转为 HuggingFace Dataset
  切换框架 = 改一行 backend= 参数。
```

---

## 九、面试中如何讲 Harness Engineering

### 讲法 1: 从痛点出发

> "我调研了 Search-R1、R1-Searcher、Agent-R1 等项目，发现一个共同的工程问题：每个项目都从零搭建环境-奖励-训练的粘合代码，代码不可复用、不可调试、不可切换。
>
> 我把这个问题抽象为 **Harness Engineering**——类比传统 RL 的 Gymnasium、测试领域的 pytest，Agentic RL 需要一个标准化的夹具层。
>
> 核心设计：
> 1. **Environment Protocol**: 5 个方法定义任何任务
> 2. **Reward Engine**: 声明式组合，不再写 ad-hoc 的 if-else
> 3. **Reward Debugger**: 训练前就能检测 reward hacking 风险
> 4. **Training Adapter**: 一行切换 veRL / OpenRLHF / TRL"

### 讲法 2: 从具体 Bug 出发

> "Search-R1 v0.1 有一个关键 Bug：Retrieved Token Masking 实现错误，导致训练极度不稳定。这个 Bug 的根因是环境返回的 token 和模型生成的 token 混在一起，需要手动维护一个 info_mask。
>
> 我的思考是：如果有一个标准化的 Trajectory 结构，天然区分 action（模型生成）和 observation（环境返回），这类 Bug 就不会发生。这就是 Harness Engineering 的核心价值——通过好的抽象消灭一整类 Bug。"

### 讲法 3: 从 Reward Debugger 出发

> "R1-Searcher 的 reward server 有 7 种格式检查，每一种都是在发现 hack 行为后才加上的。这是一个 reactive 的方式——等出了问题才修。
>
> 我设计了 Reward Debugger，可以在训练前就分析 reward 的分布：哪个组件太容易被满足（饱和风险）、哪个组件区分度不够（低方差）、哪个组件太严格（触底）。这让 reward 设计变成了一个 proactive 的过程。"

### 高频追问

**Q: 为什么不直接用 Gymnasium？**

> Gymnasium 是为传统 RL 设计的：单个 scalar action、固定维度 observation、单步 reward。Agentic RL 需要：自然语言 action、变长 observation（搜索结果）、多维度组合 reward、多轮 credit assignment。AgentHarness 是 Agentic RL 的 Gymnasium。

**Q: 现有项目已经能训练了，为什么还需要 Harness？**

> 能训练 ≠ 工程上好。Search-R1 能训练，但 reward 全是 ad-hoc 代码、绑定 veRL、没有调试工具、换环境要重写。这就像"能跑的代码"vs"可维护的代码"的区别。Harness 的价值是让 Agentic RL 从"一次性实验代码"进化到"可复用的工程基建"。

**Q: Reward Debugger 具体检测什么？**

> 三种风险：
> 1. **饱和**: 某个 reward 组件 95% 的样本都得高分 → 模型无需努力就能拿分 → 这个组件变成了 free reward
> 2. **低方差**: 所有样本得分几乎相同 → 梯度信号接近零 → 等于没有这个组件
> 3. **触底**: 几乎所有样本得分接近零 → 太严格 → 模型得不到正反馈
> 另外还做组件间相关性分析——如果两个组件高度相关，说明其中一个是冗余的。

---

## 十、Harness Engineering 的未来方向

### 10.1 自动化 Reward 搜索

```
当前: 人工设计 reward 组合和权重
未来: 用 meta-learning / AutoML 自动搜索最优 reward 组合

reward_search = AutoRewardSearch(
    components=[exact_match, f1, format, efficiency, ...],
    objective="maximize_val_accuracy",
    budget=100,  # 100 次 reward 配置实验
)
best_reward = reward_search.run(env, val_data)
```

### 10.2 动态 Reward 调度

```
当前: 固定的 reward 权重
未来: 根据训练进度动态调整

scheduler = RewardScheduler(
    phases=[
        Phase(epoch=0,  weights={"format": 0.8, "accuracy": 0.2}),   # 先学格式
        Phase(epoch=10, weights={"format": 0.3, "accuracy": 0.7}),   # 再学准确
        Phase(epoch=20, weights={"format": 0.1, "accuracy": 0.5,
                                 "efficiency": 0.4}),                 # 最后学效率
    ]
)
```

### 10.3 跨环境迁移

```
当前: 每个环境独立训练
未来: 在一个环境学到的 reward 工程迁移到另一个

# 搜索 Agent 的 reward 组合迁移到代码 Agent
search_reward = load_reward("search_agent_best")
code_reward = search_reward.adapt(
    replace={"exact_match": "code_passes_tests"},
    keep=["format_follows", "trajectory_efficiency"],
)
```

### 10.4 Reward 版本管理

```
当前: reward 改了就覆盖了，无法回溯
未来: reward 像代码一样版本管理

store = RewardStore("./rewards")
store.save(reward_v1, tag="v1-baseline")
store.save(reward_v2, tag="v2-add-efficiency")

# 对比任意两个版本
debugger.compare(store.load("v1"), store.load("v2"), trajectories)
```

---

## 附录: AgentHarness 在 Agentic RL 生态中的位置

```
┌─────────────────────────────────────────────────────┐
│                 Agentic RL 全景图                     │
│                                                     │
│  模型层:     Qwen2.5 / Llama / DeepSeek             │
│       ↕                                             │
│  训练层:     veRL / OpenRLHF / TRL                   │
│       ↕                                             │
│  ★ Harness 层: AgentHarness ★                       │
│       ↕          ↕           ↕                      │
│  环境层:    搜索     代码      浏览器                  │
│  评测层:    NQ/HotpotQA  SWE-bench  WebArena        │
│                                                     │
│  AgentHarness = 训练层和环境层之间的标准化粘合层       │
└─────────────────────────────────────────────────────┘
```

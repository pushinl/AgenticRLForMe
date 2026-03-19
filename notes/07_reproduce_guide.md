# CPU 环境复现指南

> ⚠️ 本机为 CPU 机器，无 GPU。以下内容聚焦于可在 CPU 上完成的部分。

## 1. 可在 CPU 上做的事情

| ✅ 可做 | ❌ 需要 GPU |
|---------|------------|
| 阅读和理解代码架构 | 实际训练（RL rollout + 梯度更新） |
| 运行数据预处理管线 | vLLM 推理 |
| 实现和测试奖励函数 | 模型推理（大模型） |
| 搭建搜索环境（Pyserini） | PPO/GRPO 完整训练循环 |
| 手写简化版 GRPO 算法 | 多卡并行训练 |
| 单元测试各组件 | - |
| 理解 veRL 框架架构 | - |

## 2. 项目克隆与代码阅读

### 2.1 克隆核心项目

```bash
cd /data/workspace/agentic-rl-research/code

# Search-R1 (最推荐阅读)
git clone https://github.com/PeterGriffinJin/Search-R1.git --depth 1

# R1-Searcher
git clone https://github.com/RUCAIBox/R1-Searcher.git --depth 1

# RAGEN (失败诊断框架)
git clone https://github.com/RAGEN-AI/RAGEN.git --depth 1

# veRL (训练框架)
git clone https://github.com/volcengine/verl.git --depth 1
```

### 2.2 代码阅读路线（Search-R1 为例）

```
Search-R1/
├── search_r1/
│   ├── search/          # ← 第1步: 读搜索环境实现
│   │   └── search_engine.py
│   ├── reward/          # ← 第2步: 读奖励函数
│   │   └── reward_fn.py
│   ├── data/            # ← 第3步: 读数据处理
│   │   └── dataset.py
│   └── train/           # ← 第4步: 读训练循环
│       └── train_grpo.py
├── configs/             # ← 理解超参数配置
└── scripts/             # ← 理解启动脚本
```

**重点关注**:
1. `reward_fn.py` — 奖励是怎么计算的
2. Retrieved token masking 的实现位置
3. 与 veRL 的集成接口
4. 搜索环境的接口设计

## 3. CPU 上可运行的实践

### 3.1 实现 GRPO 算法（简化版）

```python
"""
简化版 GRPO 实现 — 可在 CPU 上运行
用于理解算法核心逻辑，不涉及 LLM
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np

class SimplePolicyNet(nn.Module):
    """简单的策略网络（替代 LLM）"""
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return F.softmax(self.net(x), dim=-1)


def grpo_step(policy, ref_policy, optimizer, prompts, reward_fn,
              G=8, epsilon=0.2, beta=0.01):
    """
    GRPO 单步训练

    Args:
        policy: 当前策略网络
        ref_policy: 参考策略（冻结）
        optimizer: 优化器
        prompts: 输入 batch [B, input_dim]
        reward_fn: 奖励函数 (prompt, action) -> scalar
        G: 组大小
        epsilon: PPO clip 比例
        beta: KL 系数
    """
    total_loss = 0.0
    batch_rewards = []

    for prompt in prompts:
        # Step 1: 采样 G 个动作
        with torch.no_grad():
            old_probs = policy(prompt.unsqueeze(0))  # [1, action_dim]
            dist = Categorical(old_probs.squeeze())
            actions = torch.stack([dist.sample() for _ in range(G)])  # [G]
            old_log_probs = dist.log_prob(actions)  # [G]

        # Step 2: 计算奖励
        rewards = torch.tensor([reward_fn(prompt, a) for a in actions])
        batch_rewards.append(rewards.mean().item())

        # Step 3: 组相对优势
        mean_r = rewards.mean()
        std_r = rewards.std() + 1e-8
        advantages = (rewards - mean_r) / std_r  # [G]

        # Step 4: 计算当前策略的 log prob
        new_probs = policy(prompt.unsqueeze(0))
        new_dist = Categorical(new_probs.squeeze())
        new_log_probs = new_dist.log_prob(actions)

        # Step 5: 重要性采样比
        ratio = torch.exp(new_log_probs - old_log_probs.detach())

        # Step 6: Clipped objective
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - epsilon, 1 + epsilon) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        # Step 7: KL 惩罚
        ref_probs = ref_policy(prompt.unsqueeze(0))
        kl = (ref_probs.squeeze() * (ref_probs.squeeze().log() -
              new_probs.squeeze().log())).sum()
        kl_loss = beta * kl

        total_loss += policy_loss + kl_loss

    # 反向传播
    optimizer.zero_grad()
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
    optimizer.step()

    return np.mean(batch_rewards)


# === 使用示例 ===
if __name__ == "__main__":
    # 简单任务: 选择能让奖励最大的动作
    input_dim = 10
    hidden_dim = 32
    action_dim = 5

    policy = SimplePolicyNet(input_dim, hidden_dim, action_dim)
    ref_policy = SimplePolicyNet(input_dim, hidden_dim, action_dim)
    ref_policy.load_state_dict(policy.state_dict())
    for p in ref_policy.parameters():
        p.requires_grad = False

    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)

    # 奖励函数: action == 3 时奖励为 1，否则为 0
    def reward_fn(prompt, action):
        return 1.0 if action.item() == 3 else 0.0

    # 训练循环
    for epoch in range(100):
        prompts = torch.randn(16, input_dim)  # batch of 16
        avg_reward = grpo_step(policy, ref_policy, optimizer, prompts,
                               reward_fn, G=8)
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}, Avg Reward: {avg_reward:.3f}")
```

### 3.2 实现奖励函数并测试

```python
"""
测试不同的奖励函数设计
"""
import re
from collections import Counter

def exact_match_reward(prediction: str, ground_truth: str) -> float:
    """精确匹配奖励"""
    pred = prediction.strip().lower()
    gold = ground_truth.strip().lower()
    return 1.0 if pred == gold else 0.0

def f1_reward(prediction: str, ground_truth: str) -> float:
    """F1 token overlap 奖励"""
    pred_tokens = Counter(prediction.lower().split())
    gold_tokens = Counter(ground_truth.lower().split())

    common = sum((pred_tokens & gold_tokens).values())
    if common == 0:
        return 0.0

    precision = common / sum(pred_tokens.values())
    recall = common / sum(gold_tokens.values())
    return 2 * precision * recall / (precision + recall)

def format_reward(output: str) -> float:
    """格式合规奖励"""
    score = 0.0

    # 检查 think 标签
    if re.search(r'<think>.*?</think>', output, re.DOTALL):
        score += 0.25

    # 检查 answer 标签
    if re.search(r'<answer>.*?</answer>', output, re.DOTALL):
        score += 0.25

    # 检查 search 标签（如果有，必须成对）
    search_opens = len(re.findall(r'<search>', output))
    search_closes = len(re.findall(r'</search>', output))
    if search_opens == search_closes:
        score += 0.25
    else:
        score -= 2.0  # R1-Searcher 风格重罚

    # 检查顺序合理性
    answer_pos = output.find('<answer>')
    last_think = output.rfind('</think>')
    if answer_pos > last_think and answer_pos != -1:
        score += 0.25

    return score

def combined_reward(output: str, ground_truth: str) -> float:
    """组合奖励"""
    # 提取答案
    answer_match = re.search(r'<answer>(.*?)</answer>', output, re.DOTALL)
    if not answer_match:
        return -1.0  # 无答案重罚

    prediction = answer_match.group(1).strip()

    # 格式分 (0-1) + 准确性分 (0-1)
    fmt = max(format_reward(output), 0)  # 不让格式分为负
    acc = f1_reward(prediction, ground_truth)

    return 0.3 * fmt + 0.7 * acc

# === 测试 ===
if __name__ == "__main__":
    # 测试用例 1: 完美输出
    output1 = """<think>The user asks about the capital of France.
    Let me search for this.</think>
    <search>capital of France</search>
    <think>The search results confirm it's Paris.</think>
    <answer>Paris</answer>"""

    print("Test 1 (perfect):")
    print(f"  Format: {format_reward(output1):.2f}")
    print(f"  EM: {exact_match_reward('Paris', 'Paris'):.2f}")
    print(f"  F1: {f1_reward('Paris', 'Paris'):.2f}")
    print(f"  Combined: {combined_reward(output1, 'Paris'):.2f}")

    # 测试用例 2: 格式错误
    output2 = "The answer is Paris"
    print("\nTest 2 (no format):")
    print(f"  Format: {format_reward(output2):.2f}")
    print(f"  Combined: {combined_reward(output2, 'Paris'):.2f}")

    # 测试用例 3: 部分正确
    output3 = """<think>Let me think</think>
    <answer>Paris is the capital city of France</answer>"""
    print("\nTest 3 (partial match):")
    print(f"  F1: {f1_reward('Paris is the capital city of France', 'Paris'):.2f}")
    print(f"  Combined: {combined_reward(output3, 'Paris'):.2f}")
```

### 3.3 搭建本地搜索环境

```bash
# 安装 Pyserini (Search-R1 使用的搜索引擎)
pip install pyserini

# 下载小规模测试索引（Wikipedia 子集）
# 注意: 完整索引很大，CPU 机器建议用小规模测试
python -c "
from pyserini.search.lucene import LuceneSearcher
# 使用预构建的小索引进行测试
searcher = LuceneSearcher.from_prebuilt_index('wikipedia-dpr-100w')
hits = searcher.search('what is reinforcement learning', k=3)
for hit in hits:
    print(f'Score: {hit.score:.4f}')
    print(f'Content: {hit.raw[:200]}')
    print('---')
"
```

### 3.4 理解 veRL 框架结构

```
veRL 代码阅读路线:

verl/
├── verl/
│   ├── trainer/
│   │   ├── ppo/           # PPO 训练器 ← 重点读
│   │   │   ├── ppo_trainer.py
│   │   │   └── reward_fn.py
│   │   └── grpo/          # GRPO 训练器 ← 重点读
│   ├── workers/
│   │   ├── rollout/       # Rollout 工作者（vLLM 集成）
│   │   └── actor/         # Actor 工作者（梯度计算）
│   ├── envs/              # 环境接口 ← Agent 相关
│   └── utils/
│       └── reward_score/  # 奖励计算工具
└── examples/
    └── search_r1/         # Search-R1 的 veRL 集成示例
```

## 4. 推荐学习路线（CPU 机器）

```
Week 1: 理论基础
  ├─ 读 DeepSeek-R1 论文 (核心: Section 3)
  ├─ 读 DeepSeekMath 论文 (核心: GRPO 推导)
  └─ 手写 GRPO 简化实现 (上面的代码)

Week 2: 代码阅读
  ├─ Clone Search-R1, 通读代码
  ├─ 重点: reward_fn, token masking, 搜索接口
  └─ Clone veRL, 理解 trainer 架构

Week 3: 奖励设计实践
  ├─ 实现上面的奖励函数
  ├─ 搭建本地 Pyserini 搜索环境
  └─ 读 R1-Searcher 的两阶段设计

Week 4: 深入理解 & 面试准备
  ├─ 读 RAGEN 论文 (失败模式)
  ├─ 整理面试笔记
  └─ 准备面试 demo (奖励函数 + GRPO 实现)
```

## 5. 如果后续有 GPU

当获得 GPU 后，可以：

```bash
# 1. 用 Search-R1 的脚本启动完整训练
cd Search-R1
bash scripts/train_grpo.sh

# 2. 配置要求（最低）
# - 单卡: A100 80GB 或 H100 (7B 模型)
# - 推荐: 4× A100 (FSDP 并行)
# - vLLM 推理服务器: 额外 1-2 卡

# 3. 小规模验证
# - 先用 Qwen2.5-1.5B 跑通管线
# - 再扩展到 7B
```

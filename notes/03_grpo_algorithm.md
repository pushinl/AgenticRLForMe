# GRPO 算法深度解析

## 1. 算法起源

GRPO (Group Relative Policy Optimization) 由 DeepSeekMath (2024.02) 提出，后在 DeepSeek-R1 中发扬光大，成为 Agentic RL 领域的主流算法。

## 2. 动机：为什么不用 PPO？

PPO 需要一个与策略模型同等大小的 **Critic（价值模型）**，这意味着：
- **显存翻倍**：训练 7B 策略 + 7B Critic = 14B 参数的显存开销
- **Critic 训练不稳定**：值函数拟合本身就是一个难题
- **超参数更多**：Critic 学习率、训练步数等额外超参

GRPO 的核心创新：**用组内相对排名替代 Critic，完全消除价值模型**。

## 3. 算法流程

### Step 1: 采样
对每个问题 q，从**当前（旧）策略** π_θ_old 采样 G 个输出：

```
{o₁, o₂, ..., o_G} ~ π_θ_old(· | q)
```

典型 G = 4~16。

### Step 2: 计算奖励
用规则化奖励函数给每个输出打分：

```
{r₁, r₂, ..., r_G} = R(o₁, q), R(o₂, q), ..., R(o_G, q)
```

### Step 3: 计算组相对优势（核心创新）
```
Â_i = (r_i - mean(r₁..r_G)) / std(r₁..r_G)
```

- 不需要学习的价值函数
- 组均值自动作为 baseline
- 标准差归一化控制方差

**对比 PPO 的 GAE (Generalized Advantage Estimation)**:
```
PPO:  Â_t = Σ(k=0..∞) (γλ)^k × δ_{t+k}
      δ_t = r_t + γ V(s_{t+1}) - V(s_t)    # 需要 V(s)!

GRPO: Â_i = (r_i - μ_group) / σ_group       # 不需要 V(s)!
```

### Step 4: 策略优化
使用与 PPO 相同的 clipped 重要性采样比：

```
ratio_i,t = π_θ(o_{i,t} | q, o_{i,<t}) / π_θ_old(o_{i,t} | q, o_{i,<t})

L_clip = min(ratio × Â, clip(ratio, 1-ε, 1+ε) × Â)
```

### Step 5: KL 惩罚
GRPO 将 KL 散度**直接加到损失函数中**（而非加到奖励中）：

```
L = L_clip - β × D_KL(π_θ || π_ref)
```

使用无偏估计器：
```
D_KL ≈ (π_ref(o_t | ...) / π_θ(o_t | ...)) - log(π_ref(o_t | ...) / π_θ(o_t | ...)) - 1
```

## 4. 完整伪代码

```python
def grpo_training_step(questions, policy, ref_policy, reward_fn,
                        G=8, epsilon=0.2, beta=0.01):
    """GRPO 训练一步"""
    all_loss = 0

    for q in questions:
        # Step 1: 采样 G 个输出
        outputs = [policy.generate(q) for _ in range(G)]

        # Step 2: 计算奖励
        rewards = [reward_fn(q, o) for o in outputs]

        # Step 3: 组相对优势
        mean_r = mean(rewards)
        std_r = std(rewards)
        advantages = [(r - mean_r) / (std_r + 1e-8) for r in rewards]

        # Step 4: Clipped objective
        for i, (output, adv) in enumerate(zip(outputs, advantages)):
            for t, token in enumerate(output):
                ratio = policy.prob(token | q, output[:t]) / \
                        old_policy.prob(token | q, output[:t])
                clip_ratio = clip(ratio, 1 - epsilon, 1 + epsilon)
                loss_token = -min(ratio * adv, clip_ratio * adv)

                # Step 5: KL 惩罚
                ref_ratio = ref_policy.prob(token | ...) / policy.prob(token | ...)
                kl = ref_ratio - log(ref_ratio) - 1
                loss_token += beta * kl

                all_loss += loss_token

    all_loss.backward()
    optimizer.step()
```

## 5. GRPO vs PPO vs REINFORCE 对比

| 特性 | REINFORCE | PPO | GRPO |
|------|-----------|-----|------|
| **Critic** | 无 | 需要（同等大小） | **无** |
| **优势估计** | 原始奖励（高方差） | GAE + 学习值函数 | 组归一化 |
| **显存开销** | 低 | **非常高**（2× 参数） | 低 |
| **稳定性** | 低 | 高 | 中等 |
| **收敛速度** | 慢 | 中等 | 快 |
| **KL 实现** | N/A | 每 token KL 加到奖励 | KL 直接加到损失 |
| **最适场景** | 简单任务 | 需要最大稳定性 | 可验证奖励任务 |

## 6. GRPO 的变体

### Reinforce++ (R1-Searcher 使用)
- REINFORCE 的方差减少变体
- 类似 GRPO 但实现细节不同

### β-GRPO（效率感知）
```
R = 1  if (min_token_prob ≥ β) AND (answer correct)
R = 0  otherwise
```
惩罚不必要的搜索（~20-28% 的情况）和遗漏必要搜索（最高 63% 错误率）。

### GiGPO (Group-in-Group, verl-agent)
- 轨迹级优化，适用于长期任务（50+ 步）
- 在 ALFWorld、WebShop 上优于 PPO/GRPO
- 解决 Echo Trap 问题

## 7. 关键超参数

| 超参数 | 典型值 | 说明 |
|--------|--------|------|
| G (组大小) | 4-16 | 越大越稳定但越慢 |
| ε (clip 比例) | 0.1-0.2 | 控制策略更新幅度 |
| β (KL 系数) | 0.001-0.05 | 控制与参考策略的偏离 |
| 学习率 | 1e-6 ~ 5e-6 | 低 = 更稳定 |
| 采样温度 | 0.7-1.0 | rollout 采样用 |
| PPO epochs | 1-4 | 每批数据的优化次数 |

## 8. GRPO 的局限

1. **需要多次采样**: 每个 prompt 采样 G 个输出，推理开销大 → 需要 vLLM 加速
2. **方差估计不稳定**: 组太小时 std 估计不准
3. **不适合 dense reward**: 如果有逐 token 的奖励信号，PPO 的 GAE 更好
4. **instruct 模型上不稳定**: RAGEN 发现在微调过的模型上 GRPO 可能奖励坍缩

## 9. 面试必备一句话

> "GRPO 用组内采样的均值和标准差替代了 PPO 中的 Critic 模型作为 baseline，
> 在保持 clipped objective 的同时节省了 50% 的显存，特别适合具有可验证奖励的
> 推理和 Agent 任务。"

"""
简化版 GRPO (Group Relative Policy Optimization) 实现
可在 CPU 上运行，用于理解算法核心逻辑

用一个简单的 bandit 问题演示 GRPO 的工作原理
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
import copy


class SimplePolicyNet(nn.Module):
    """简单的策略网络（替代 LLM）"""
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
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

    Returns:
        avg_reward: 平均奖励
        stats: 训练统计信息
    """
    total_loss = 0.0
    batch_rewards = []
    batch_kl = []
    batch_advantages = []

    for prompt in prompts:
        # ========== Step 1: 采样 G 个动作 ==========
        with torch.no_grad():
            old_probs = policy(prompt.unsqueeze(0)).squeeze()  # [action_dim]
            dist = Categorical(old_probs)
            actions = torch.stack([dist.sample() for _ in range(G)])  # [G]
            old_log_probs = dist.log_prob(actions)  # [G]

        # ========== Step 2: 计算奖励 ==========
        rewards = torch.tensor([reward_fn(prompt, a) for a in actions],
                               dtype=torch.float32)
        batch_rewards.append(rewards.mean().item())

        # ========== Step 3: 组相对优势（GRPO 核心） ==========
        # 这就是 GRPO 的精髓：用组内统计替代 Critic
        mean_r = rewards.mean()
        std_r = rewards.std()

        if std_r < 1e-8:
            # Echo Trap 检测！方差太小时跳过
            continue

        advantages = (rewards - mean_r) / (std_r + 1e-8)  # [G]
        batch_advantages.append(advantages.abs().mean().item())

        # ========== Step 4: 计算新策略的概率 ==========
        new_probs = policy(prompt.unsqueeze(0)).squeeze()
        new_dist = Categorical(new_probs)
        new_log_probs = new_dist.log_prob(actions)

        # ========== Step 5: Clipped 重要性采样 ==========
        ratio = torch.exp(new_log_probs - old_log_probs.detach())
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - epsilon, 1 + epsilon) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        # ========== Step 6: KL 散度惩罚 ==========
        with torch.no_grad():
            ref_probs = ref_policy(prompt.unsqueeze(0)).squeeze()
        # 无偏 KL 估计器 (GRPO 论文的方式)
        kl = (ref_probs / (new_probs + 1e-8) -
              torch.log(ref_probs / (new_probs + 1e-8)) - 1).sum()
        batch_kl.append(kl.item())

        total_loss += policy_loss + beta * kl

    if total_loss == 0:
        return np.mean(batch_rewards) if batch_rewards else 0, {}

    # ========== 反向传播 ==========
    optimizer.zero_grad()
    total_loss.backward()
    # 梯度裁剪（防止梯度爆炸）
    grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
    optimizer.step()

    stats = {
        'avg_reward': np.mean(batch_rewards),
        'avg_kl': np.mean(batch_kl) if batch_kl else 0,
        'avg_advantage': np.mean(batch_advantages) if batch_advantages else 0,
        'grad_norm': grad_norm.item(),
        'loss': total_loss.item(),
    }
    return stats['avg_reward'], stats


def demo_grpo():
    """演示 GRPO 在简单任务上的训练"""
    print("=" * 60)
    print("GRPO (Group Relative Policy Optimization) Demo")
    print("=" * 60)

    # 任务设置
    input_dim = 10
    hidden_dim = 64
    action_dim = 5
    target_action = 3  # 目标动作

    # 初始化策略
    policy = SimplePolicyNet(input_dim, hidden_dim, action_dim)
    ref_policy = copy.deepcopy(policy)
    for p in ref_policy.parameters():
        p.requires_grad = False

    optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)

    # 奖励函数: 选择 target_action 得 1 分，否则 0 分
    def reward_fn(prompt, action):
        return 1.0 if action.item() == target_action else 0.0

    # 训练
    print(f"\n任务: 学会选择 action={target_action} (共 {action_dim} 个动作)")
    print(f"超参数: G=8, ε=0.2, β=0.01, lr=3e-4")
    print("-" * 60)

    for epoch in range(200):
        prompts = torch.randn(32, input_dim)  # batch of 32
        avg_reward, stats = grpo_step(
            policy, ref_policy, optimizer, prompts,
            reward_fn, G=8, epsilon=0.2, beta=0.01
        )

        if (epoch + 1) % 20 == 0:
            # 评估当前策略
            with torch.no_grad():
                test_prompts = torch.randn(100, input_dim)
                probs = policy(test_prompts)
                target_prob = probs[:, target_action].mean().item()

            print(f"Epoch {epoch+1:3d} | "
                  f"Reward: {avg_reward:.3f} | "
                  f"P(a={target_action}): {target_prob:.3f} | "
                  f"KL: {stats.get('avg_kl', 0):.4f} | "
                  f"Grad: {stats.get('grad_norm', 0):.4f}")

    print("-" * 60)
    print("训练完成！")

    # 最终评估
    with torch.no_grad():
        test_prompts = torch.randn(1000, input_dim)
        probs = policy(test_prompts)
        print(f"\n最终动作概率分布 (平均):")
        for i in range(action_dim):
            marker = " ← target" if i == target_action else ""
            print(f"  action {i}: {probs[:, i].mean():.4f}{marker}")


def compare_grpo_vs_reinforce():
    """对比 GRPO 和 REINFORCE 的训练效果"""
    print("\n" + "=" * 60)
    print("GRPO vs REINFORCE 对比实验")
    print("=" * 60)

    input_dim = 10
    hidden_dim = 64
    action_dim = 5
    target_action = 3

    def reward_fn(prompt, action):
        return 1.0 if action.item() == target_action else 0.0

    results = {'GRPO': [], 'REINFORCE': []}

    for method in ['GRPO', 'REINFORCE']:
        policy = SimplePolicyNet(input_dim, hidden_dim, action_dim)
        ref_policy = copy.deepcopy(policy)
        for p in ref_policy.parameters():
            p.requires_grad = False
        optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)

        for epoch in range(200):
            prompts = torch.randn(32, input_dim)

            if method == 'GRPO':
                avg_reward, _ = grpo_step(
                    policy, ref_policy, optimizer, prompts,
                    reward_fn, G=8, epsilon=0.2, beta=0.01
                )
            else:
                # 简单 REINFORCE (无 baseline)
                total_loss = 0
                all_rewards = []
                for prompt in prompts:
                    probs = policy(prompt.unsqueeze(0)).squeeze()
                    dist = Categorical(probs)
                    action = dist.sample()
                    reward = reward_fn(prompt, action)
                    all_rewards.append(reward)
                    total_loss -= dist.log_prob(action) * reward
                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
                optimizer.step()
                avg_reward = np.mean(all_rewards)

            results[method].append(avg_reward)

    # 打印对比结果
    print(f"\n{'Epoch':>6} | {'GRPO':>8} | {'REINFORCE':>10}")
    print("-" * 30)
    for i in range(0, 200, 20):
        grpo_r = np.mean(results['GRPO'][max(0,i-5):i+5])
        reinf_r = np.mean(results['REINFORCE'][max(0,i-5):i+5])
        print(f"{i+1:6d} | {grpo_r:8.3f} | {reinf_r:10.3f}")

    print(f"\n最终 50 轮平均:")
    print(f"  GRPO:      {np.mean(results['GRPO'][-50:]):.3f}")
    print(f"  REINFORCE: {np.mean(results['REINFORCE'][-50:]):.3f}")


if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)

    demo_grpo()
    compare_grpo_vs_reinforce()

# 必读论文清单

## 核心论文

| 优先级 | 论文 | 链接 | 要点 |
|--------|------|------|------|
| ⭐⭐⭐ | DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via RL | [arxiv 2501.12948](https://arxiv.org/abs/2501.12948) | 基石。4阶段管线、GRPO、规则奖励、涌现推理 |
| ⭐⭐⭐ | DeepSeekMath: Pushing the Limits of Mathematical Reasoning | [arxiv 2402.03300](https://arxiv.org/abs/2402.03300) | 提出 GRPO 算法。统一 RL 框架对比 |
| ⭐⭐⭐ | RAGEN: Training Agents by Reinforcing Reasoning | [arxiv 2504.20073](https://arxiv.org/abs/2504.20073) | 失败诊断。Echo Trap、Template Collapse |
| ⭐⭐ | Search-R1: Training LLMs to Reason and Leverage Search Engines | [arxiv 2503.09516](https://arxiv.org/abs/2503.09516) | 开源搜索Agent。Retrieved token masking |
| ⭐⭐ | R1-Searcher: Incentivizing Search Capability in LLMs via RL | [github](https://github.com/RUCAIBox/R1-Searcher) | 两阶段RL设计。Reinforce++ |
| ⭐⭐ | ToolRL: Reward is All Tool Learning Needs | [arxiv 2504.13958](https://arxiv.org/abs/2504.13958) | 细粒度工具奖励。冷启动发现 |
| ⭐ | Agent-R1 | [github](https://github.com/tonyzhoup/Agent-R1) | 模块化多工具。过程+结果奖励 |
| ⭐ | AgentRL: Training Language Model Agents with RL | [arxiv 2510.04206](https://arxiv.org/abs/2510.04206) | 系统设计。大规模基建 |
| ⭐ | Tool-Star: Self-Driven Tool-Using LLM | [github](https://github.com/dongguanting/Tool-Star) | Self-Critic RL。6种工具 |

## 相关 GitHub 仓库

| 项目 | URL | Stars 趋势 |
|------|-----|-----------|
| Search-R1 | https://github.com/PeterGriffinJin/Search-R1 | 🔥 |
| R1-Searcher | https://github.com/RUCAIBox/R1-Searcher | 🔥 |
| RAGEN | https://github.com/RAGEN-AI/RAGEN | 🔥 |
| Agent-R1 | https://github.com/tonyzhoup/Agent-R1 | ⭐ |
| AgentRL | https://github.com/THUDM/AgentRL | ⭐ |
| Tool-Star | https://github.com/dongguanting/Tool-Star | ⭐ |
| ReSearch/ReCall | https://github.com/Agent-RL/ReSearch | ⭐ |
| veRL | https://github.com/volcengine/verl | 🔥🔥 |
| DeepSeek-R1 | https://github.com/deepseek-ai/DeepSeek-R1 | 🔥🔥🔥 |

## 推荐阅读顺序

```
1. DeepSeekMath (理解 GRPO)
   ↓
2. DeepSeek-R1 (理解完整管线)
   ↓
3. Search-R1 论文 + 代码 (理解搜索 Agent RL)
   ↓
4. R1-Searcher (对比两阶段设计)
   ↓
5. RAGEN (理解失败模式)
   ↓
6. ToolRL (理解奖励设计细节)
```

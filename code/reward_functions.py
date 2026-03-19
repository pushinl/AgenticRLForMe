"""
Agentic RL 奖励函数实现与测试

包含 Search-R1, R1-Searcher, ToolRL 等项目的奖励设计
可在 CPU 上运行测试
"""
import re
from collections import Counter
from typing import Optional


# ============================================================
# 1. 结果奖励 (Outcome Reward)
# ============================================================

def exact_match_reward(prediction: str, ground_truth: str) -> float:
    """
    精确匹配奖励 (Search-R1 风格)
    最简单但最有效的奖励
    """
    pred = prediction.strip().lower()
    gold = ground_truth.strip().lower()
    return 1.0 if pred == gold else 0.0


def f1_reward(prediction: str, ground_truth: str) -> float:
    """
    F1 token overlap 奖励 (R1-Searcher Stage 2)
    比 EM 更平滑，适合开放式问答
    """
    pred_tokens = Counter(prediction.lower().split())
    gold_tokens = Counter(ground_truth.lower().split())

    common = sum((pred_tokens & gold_tokens).values())
    if common == 0:
        return 0.0

    precision = common / max(sum(pred_tokens.values()), 1)
    recall = common / max(sum(gold_tokens.values()), 1)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


# ============================================================
# 2. 格式奖励 (Format Reward)
# ============================================================

def format_reward_basic(output: str) -> float:
    """
    基础格式奖励: 检查必需标签是否存在
    """
    score = 0.0

    if re.search(r'<think>.*?</think>', output, re.DOTALL):
        score += 0.5
    if re.search(r'<answer>.*?</answer>', output, re.DOTALL):
        score += 0.5

    return score


def format_reward_strict(output: str) -> float:
    """
    严格格式奖励 (R1-Searcher 风格)
    格式违规会被重罚
    """
    # 检查 search 标签配对
    search_opens = len(re.findall(r'<search>', output))
    search_closes = len(re.findall(r'</search>', output))

    if search_opens != search_closes:
        return -2.0  # 重罚！

    # 检查基本格式
    has_think = bool(re.search(r'<think>.*?</think>', output, re.DOTALL))
    has_answer = bool(re.search(r'<answer>.*?</answer>', output, re.DOTALL))

    if has_think and has_answer:
        return 0.5
    elif has_answer:
        return 0.25
    else:
        return -1.0  # 没有答案标签也要罚


def format_reward_r1searcher(output: str, stage: int = 1) -> float:
    """
    R1-Searcher 的分阶段格式奖励

    Stage 1: 只看格式是否正确 (+0.5)
    Stage 2: 格式 + 答案质量
    """
    # 检查检索格式
    has_search = bool(re.search(
        r'<begin_of_search>.*?<end_of_search>', output, re.DOTALL
    ))
    has_answer = bool(re.search(r'<answer>.*?</answer>', output, re.DOTALL))

    if stage == 1:
        # Stage 1: 只关心格式
        if has_search and has_answer:
            return 0.5
        elif has_answer:
            return 0.1
        else:
            return -2.0
    else:
        # Stage 2: 格式是基础分
        return 0.5 if has_answer else -2.0


# ============================================================
# 3. 工具调用奖励 (ToolRL 风格)
# ============================================================

def tool_call_reward(
    pred_tool_names: list[str],
    pred_param_names: list[str],
    pred_param_values: dict,
    gold_tool_names: list[str],
    gold_param_names: list[str],
    gold_param_values: dict,
) -> float:
    """
    ToolRL 细粒度工具调用奖励

    三个维度:
    - 工具名匹配 (Jaccard)
    - 参数名匹配 (Jaccard)
    - 参数值匹配 (Exact)
    """
    # 工具名 Jaccard 相似度
    pred_set = set(pred_tool_names)
    gold_set = set(gold_tool_names)
    if pred_set or gold_set:
        name_score = len(pred_set & gold_set) / len(pred_set | gold_set)
    else:
        name_score = 1.0

    # 参数名 Jaccard
    pred_params = set(pred_param_names)
    gold_params = set(gold_param_names)
    if pred_params or gold_params:
        param_name_score = len(pred_params & gold_params) / len(pred_params | gold_params)
    else:
        param_name_score = 1.0

    # 参数值精确匹配
    common_params = pred_params & gold_params
    if common_params:
        value_matches = sum(
            1 for k in common_params
            if pred_param_values.get(k) == gold_param_values.get(k)
        )
        param_value_score = value_matches / len(common_params)
    else:
        param_value_score = 0.0

    # 归一化到 [-3, 3] (ToolRL 论文设计)
    raw_score = name_score + param_name_score + param_value_score  # [0, 3]
    normalized = (raw_score - 1.5) * 2  # [-3, 3]
    return normalized


# ============================================================
# 4. 效率感知奖励 (β-GRPO 风格)
# ============================================================

def beta_grpo_reward(
    answer_correct: bool,
    min_token_prob: float,
    beta_threshold: float = 0.7,
    did_search: bool = True,
) -> float:
    """
    β-GRPO: 置信度感知的搜索奖励

    只在模型"知道自己不确定"时奖励搜索
    """
    if answer_correct:
        if did_search and min_token_prob >= beta_threshold:
            # 搜了但其实很确定 → 不必要的搜索
            return 0.5  # 部分奖励（答对了但浪费了搜索）
        else:
            return 1.0  # 完全正确的行为
    else:
        if not did_search and min_token_prob < beta_threshold:
            # 不确定但没搜 → 该搜没搜
            return -0.5  # 惩罚
        else:
            return 0.0  # 搜了但还是错了，不额外惩罚


# ============================================================
# 5. 组合奖励 (实际使用的完整版)
# ============================================================

def search_agent_reward(
    output: str,
    ground_truth: str,
    stage: int = 2,
    format_weight: float = 0.3,
    accuracy_weight: float = 0.7,
) -> float:
    """
    搜索 Agent 完整奖励函数

    组合格式奖励和准确性奖励
    """
    # 格式分
    fmt_score = format_reward_strict(output)
    if fmt_score <= -1.0:
        # 格式严重错误，直接返回低分
        return fmt_score

    # 提取答案
    answer_match = re.search(r'<answer>(.*?)</answer>', output, re.DOTALL)
    if not answer_match:
        return -1.0

    prediction = answer_match.group(1).strip()

    # 准确性分 (EM + F1 的混合)
    em = exact_match_reward(prediction, ground_truth)
    f1 = f1_reward(prediction, ground_truth)
    acc_score = max(em, f1)  # 取较高分

    return format_weight * max(fmt_score, 0) + accuracy_weight * acc_score


# ============================================================
# 测试
# ============================================================

def run_tests():
    print("=" * 60)
    print("Agentic RL 奖励函数测试")
    print("=" * 60)

    # --- 测试 1: 结果奖励 ---
    print("\n--- 1. 结果奖励 ---")
    print(f"EM('Paris', 'Paris') = {exact_match_reward('Paris', 'Paris')}")
    print(f"EM('paris', 'Paris') = {exact_match_reward('paris', 'Paris')}")
    print(f"EM('London', 'Paris') = {exact_match_reward('London', 'Paris')}")
    print(f"F1('capital of France is Paris', 'Paris') = "
          f"{f1_reward('capital of France is Paris', 'Paris'):.3f}")
    print(f"F1('Paris is beautiful', 'Paris') = "
          f"{f1_reward('Paris is beautiful', 'Paris'):.3f}")

    # --- 测试 2: 格式奖励 ---
    print("\n--- 2. 格式奖励 ---")

    good_output = """<think>Let me search for the answer.</think>
<search>capital of France</search>
<think>Based on the search, it's Paris.</think>
<answer>Paris</answer>"""

    bad_format = "The answer is Paris"

    unclosed_search = """<think>Searching</think>
<search>query
<answer>Paris</answer>"""

    print(f"Good format: {format_reward_strict(good_output):.2f}")
    print(f"No format:   {format_reward_strict(bad_format):.2f}")
    print(f"Unclosed:    {format_reward_strict(unclosed_search):.2f}")

    # --- 测试 3: R1-Searcher 分阶段 ---
    print("\n--- 3. R1-Searcher 分阶段奖励 ---")

    r1s_output = """<think>Need to search</think>
<begin_of_search>capital of France<end_of_search>
<answer>Paris</answer>"""

    print(f"Stage 1: {format_reward_r1searcher(r1s_output, stage=1):.2f}")
    print(f"Stage 2: {format_reward_r1searcher(r1s_output, stage=2):.2f}")

    # --- 测试 4: 工具调用奖励 ---
    print("\n--- 4. ToolRL 工具调用奖励 ---")

    # 完美匹配
    score1 = tool_call_reward(
        pred_tool_names=['web_search'],
        pred_param_names=['query', 'num_results'],
        pred_param_values={'query': 'capital of France', 'num_results': 5},
        gold_tool_names=['web_search'],
        gold_param_names=['query', 'num_results'],
        gold_param_values={'query': 'capital of France', 'num_results': 5},
    )
    print(f"Perfect match: {score1:.2f}")

    # 部分匹配
    score2 = tool_call_reward(
        pred_tool_names=['web_search'],
        pred_param_names=['query'],
        pred_param_values={'query': 'France capital'},
        gold_tool_names=['web_search'],
        gold_param_names=['query', 'num_results'],
        gold_param_values={'query': 'capital of France', 'num_results': 5},
    )
    print(f"Partial match: {score2:.2f}")

    # 完全不匹配
    score3 = tool_call_reward(
        pred_tool_names=['calculator'],
        pred_param_names=['expression'],
        pred_param_values={'expression': '2+2'},
        gold_tool_names=['web_search'],
        gold_param_names=['query'],
        gold_param_values={'query': 'capital of France'},
    )
    print(f"No match: {score3:.2f}")

    # --- 测试 5: β-GRPO ---
    print("\n--- 5. β-GRPO 效率感知奖励 ---")
    print(f"正确+必要搜索:     {beta_grpo_reward(True, 0.3, did_search=True):.2f}")
    print(f"正确+不必要搜索:   {beta_grpo_reward(True, 0.9, did_search=True):.2f}")
    print(f"错误+未搜索(该搜): {beta_grpo_reward(False, 0.3, did_search=False):.2f}")
    print(f"错误+已搜索:       {beta_grpo_reward(False, 0.5, did_search=True):.2f}")

    # --- 测试 6: 组合奖励 ---
    print("\n--- 6. 组合奖励 (完整搜索 Agent) ---")
    print(f"完美输出:  {search_agent_reward(good_output, 'Paris'):.3f}")
    print(f"无格式:    {search_agent_reward(bad_format, 'Paris'):.3f}")
    print(f"格式好但答案错: {search_agent_reward(good_output.replace('Paris</answer>', 'London</answer>'), 'Paris'):.3f}")

    print("\n" + "=" * 60)
    print("所有测试完成！")


if __name__ == "__main__":
    run_tests()

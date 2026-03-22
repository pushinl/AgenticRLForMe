"""
Evaluation script for the DeepSearch Agent.

Metrics:
- Answer F1 / EM (final answer quality)
- Avg Steps (search efficiency)
- Intent Drift Rate (fraction of steps where intent alignment drops below threshold)
- Comparison across: SFT-only vs SFT+GRPO(outcome) vs SFT+GRPO(outcome+IA-PRM)
"""

import os
import json
import argparse
from typing import Dict, List, Optional
from collections import defaultdict

import yaml
import torch
import numpy as np

from env.dataset import HotpotQADataset, compute_f1, compute_em
from env.wiki_search_env import WikiSearchEnv
from models.agent import SearchAgent
from models.intent_prm import IntentAwarePRM, PRMInputFormatter
from transformers import AutoTokenizer


@torch.no_grad()
def evaluate_agent(
    agent: SearchAgent,
    dataset: HotpotQADataset,
    env: WikiSearchEnv,
    prm: Optional[IntentAwarePRM] = None,
    prm_formatter: Optional[PRMInputFormatter] = None,
    max_samples: int = 500,
    intent_drift_threshold: float = 0.3,
    device: str = "cuda",
) -> Dict:
    """
    Evaluate an agent on HotpotQA.

    Returns dict with:
    - avg_f1, avg_em: answer quality
    - avg_steps: search efficiency
    - intent_drift_rate: fraction of steps with low intent alignment
    - success_rate: fraction with F1 > 0.5
    - per_type_metrics: breakdown by question type (bridge/comparison)
    """
    metrics = defaultdict(list)
    per_type = defaultdict(lambda: defaultdict(list))
    detailed_results = []

    num_samples = min(max_samples, len(dataset))
    print(f"Evaluating on {num_samples} examples...")

    for idx in range(num_samples):
        example = dataset[idx]
        state = env.reset(example)
        history = []
        step_rewards = []
        intent_scores = []

        for step_idx in range(env.max_steps):
            if state.done:
                break

            # Generate action
            try:
                response, action_type, action_content = agent.generate_action(
                    example.question,
                    history if history else None,
                    temperature=0.1,  # near-greedy for eval
                    do_sample=True,
                )
            except Exception as e:
                print(f"  Error at example {idx}, step {step_idx}: {e}")
                break

            # Execute
            state, reward, done, info = env.step(response)

            # Compute PRM scores if available
            prm_progress = 0.0
            prm_intent = 0.0
            if prm is not None and prm_formatter is not None and action_type != "invalid":
                tokenized = prm_formatter.tokenize(
                    example.question, history, action_type, action_content,
                )
                input_ids = tokenized["input_ids"].to(device)
                attention_mask = tokenized["attention_mask"].to(device)
                question_mask = tokenized["question_mask"].to(device)

                prm_out = prm(input_ids, attention_mask, question_mask)
                prm_progress = prm_out["progress_score"].item()
                prm_intent = prm_out["intent_alignment_score"].item()
                step_rewards.append(prm_out["combined_reward"].item())
                intent_scores.append(prm_intent)

            history.append({
                "step_num": step_idx + 1,
                "action_type": action_type,
                "action_content": action_content,
                "result": info.get("search_results", ""),
                "prm_progress": prm_progress,
                "prm_intent": prm_intent,
            })

        # Compute answer metrics
        predicted = state.final_answer or ""
        f1 = compute_f1(predicted, example.answer)
        em = compute_em(predicted, example.answer)
        num_steps = state.step_count

        # Intent drift: fraction of steps below threshold
        if intent_scores:
            drift_count = sum(1 for s in intent_scores if s < intent_drift_threshold)
            intent_drift = drift_count / len(intent_scores)
        else:
            intent_drift = 0.0

        metrics["f1"].append(f1)
        metrics["em"].append(em)
        metrics["steps"].append(num_steps)
        metrics["intent_drift"].append(intent_drift)
        metrics["success"].append(float(f1 > 0.5))
        if step_rewards:
            metrics["avg_step_reward"].append(np.mean(step_rewards))

        # Per-type metrics
        qtype = example.question_type
        per_type[qtype]["f1"].append(f1)
        per_type[qtype]["em"].append(em)
        per_type[qtype]["steps"].append(num_steps)

        # Detailed results
        detailed_results.append({
            "id": example.id,
            "question": example.question,
            "gold_answer": example.answer,
            "predicted_answer": predicted,
            "f1": f1,
            "em": em,
            "num_steps": num_steps,
            "intent_drift": intent_drift,
            "type": qtype,
            "trajectory": history,
        })

        if (idx + 1) % 50 == 0:
            running_f1 = np.mean(metrics["f1"])
            running_em = np.mean(metrics["em"])
            print(f"  [{idx+1}/{num_samples}] F1={running_f1:.3f}, EM={running_em:.3f}")

    # Aggregate
    results = {
        "num_samples": num_samples,
        "avg_f1": float(np.mean(metrics["f1"])),
        "avg_em": float(np.mean(metrics["em"])),
        "avg_steps": float(np.mean(metrics["steps"])),
        "intent_drift_rate": float(np.mean(metrics["intent_drift"])),
        "success_rate": float(np.mean(metrics["success"])),
        "std_f1": float(np.std(metrics["f1"])),
        "std_em": float(np.std(metrics["em"])),
    }

    if metrics["avg_step_reward"]:
        results["avg_step_reward"] = float(np.mean(metrics["avg_step_reward"]))

    # Per-type breakdown
    results["per_type"] = {}
    for qtype, type_metrics in per_type.items():
        results["per_type"][qtype] = {
            "avg_f1": float(np.mean(type_metrics["f1"])),
            "avg_em": float(np.mean(type_metrics["em"])),
            "avg_steps": float(np.mean(type_metrics["steps"])),
            "count": len(type_metrics["f1"]),
        }

    return results, detailed_results


def run_comparison(config: Dict, args):
    """
    Run comparative evaluation across model variants:
    1. SFT-only
    2. SFT + GRPO (outcome only)
    3. SFT + GRPO (outcome + IA-PRM)
    """
    print("=" * 60)
    print("DeepSearch Agent Evaluation")
    print("=" * 60)

    # Load dataset
    dataset = HotpotQADataset(
        split=config["dataset"]["split_val"],
        max_samples=config["evaluation"]["max_samples"],
    )

    env = WikiSearchEnv(
        max_steps=config["env"]["max_steps"],
        max_search_results=config["env"]["max_search_results"],
        passage_max_tokens=config["env"]["passage_max_tokens"],
        cache_dir=config["env"]["cache_dir"],
    )

    # Load PRM if available
    prm = None
    prm_formatter = None
    prm_path = args.prm_path or config["prm_training"]["output_dir"]
    if os.path.exists(prm_path):
        print(f"Loading IA-PRM from {prm_path}")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        prm = IntentAwarePRM.from_pretrained(prm_path).to(device)
        prm.eval()

        prm_tokenizer = AutoTokenizer.from_pretrained(
            config["prm_model"]["name"], trust_remote_code=True,
        )
        if prm_tokenizer.pad_token is None:
            prm_tokenizer.pad_token = prm_tokenizer.eos_token
        prm_formatter = PRMInputFormatter(
            prm_tokenizer, max_length=config["prm_training"]["max_seq_length"],
        )

    # Define model variants to evaluate
    variants = []

    # 1. Base model (no fine-tuning)
    if args.eval_base:
        variants.append(("Base (no fine-tuning)", None))

    # 2. SFT-only
    sft_path = args.sft_path or config["sft"]["output_dir"]
    if os.path.exists(sft_path):
        variants.append(("SFT-only", sft_path))

    # 3. GRPO model
    grpo_path = args.grpo_path or os.path.join(config["grpo"]["output_dir"], "final")
    if os.path.exists(grpo_path):
        variants.append(("SFT + GRPO (outcome + IA-PRM)", grpo_path))

    # Custom model path
    if args.model_path and args.model_path not in [sft_path, grpo_path]:
        variants.append(("Custom model", args.model_path))

    if not variants:
        print("No model variants found. Evaluating base model only.")
        variants.append(("Base model", None))

    # Run evaluations
    all_results = {}
    output_dir = config["evaluation"]["output_dir"]
    os.makedirs(output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    for variant_name, model_path in variants:
        print(f"\n{'='*60}")
        print(f"Evaluating: {variant_name}")
        print(f"{'='*60}")

        agent = SearchAgent(
            model_name=config["agent_model"]["name"],
            lora_path=model_path,
            device="auto",
            temperature=0.1,
            do_sample=True,
        )

        results, detailed = evaluate_agent(
            agent=agent,
            dataset=dataset,
            env=env,
            prm=prm,
            prm_formatter=prm_formatter,
            max_samples=config["evaluation"]["max_samples"],
            device=device,
        )

        all_results[variant_name] = results

        # Save detailed results
        detail_path = os.path.join(
            output_dir,
            f"detailed_{variant_name.replace(' ', '_').lower()}.json",
        )
        with open(detail_path, "w") as f:
            json.dump(detailed, f, indent=2)

        # Print summary
        print(f"\n--- {variant_name} ---")
        print(f"  Answer F1:          {results['avg_f1']:.3f} ± {results['std_f1']:.3f}")
        print(f"  Answer EM:          {results['avg_em']:.3f} ± {results['std_em']:.3f}")
        print(f"  Avg Steps:          {results['avg_steps']:.2f}")
        print(f"  Intent Drift Rate:  {results['intent_drift_rate']:.3f}")
        print(f"  Success Rate (F1>0.5): {results['success_rate']:.3f}")

        if "avg_step_reward" in results:
            print(f"  Avg Step Reward:    {results['avg_step_reward']:.3f}")

        if results.get("per_type"):
            for qtype, tmetrics in results["per_type"].items():
                print(f"  [{qtype}] F1={tmetrics['avg_f1']:.3f}, "
                      f"EM={tmetrics['avg_em']:.3f}, "
                      f"Steps={tmetrics['avg_steps']:.1f} "
                      f"(n={tmetrics['count']})")

        # Free GPU memory
        del agent
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Print comparison table
    print("\n" + "=" * 80)
    print("COMPARISON TABLE")
    print("=" * 80)
    header = f"{'Model':<35} {'F1':>6} {'EM':>6} {'Steps':>6} {'Drift':>6} {'Success':>8}"
    print(header)
    print("-" * 80)
    for name, res in all_results.items():
        row = (
            f"{name:<35} "
            f"{res['avg_f1']:>6.3f} "
            f"{res['avg_em']:>6.3f} "
            f"{res['avg_steps']:>6.2f} "
            f"{res['intent_drift_rate']:>6.3f} "
            f"{res['success_rate']:>8.3f}"
        )
        print(row)
    print("=" * 80)

    # Save comparison
    comparison_path = os.path.join(output_dir, "comparison.json")
    with open(comparison_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {output_dir}/")

    return all_results


def main():
    parser = argparse.ArgumentParser(description="Evaluate DeepSearch Agent")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--model-path", type=str, default=None, help="Path to model/adapter")
    parser.add_argument("--sft-path", type=str, default=None, help="Path to SFT model")
    parser.add_argument("--grpo-path", type=str, default=None, help="Path to GRPO model")
    parser.add_argument("--prm-path", type=str, default=None, help="Path to PRM model")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--eval-base", action="store_true", help="Also evaluate base model")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    if args.max_samples:
        config["evaluation"]["max_samples"] = args.max_samples

    run_comparison(config, args)


if __name__ == "__main__":
    main()

"""
Phase 1: SFT Warmstart for the DeepSearch Agent.

Generates demonstration trajectories (via rule-based agent or teacher model),
then fine-tunes Qwen2.5-3B-Instruct with LoRA on successful trajectories.
"""

import os
import json
import random
import argparse
from typing import List, Dict, Optional

import yaml
import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)
from peft import LoraConfig, TaskType, get_peft_model
from trl import SFTTrainer

from env.dataset import HotpotQADataset, HotpotQAExample, compute_f1
from env.wiki_search_env import WikiSearchEnv
from models.agent import SYSTEM_PROMPT


def generate_rule_based_trajectory(
    example: HotpotQAExample,
    env: WikiSearchEnv,
) -> Optional[List[Dict]]:
    """
    Generate a demonstration trajectory using a rule-based strategy.

    Strategy:
    1. Extract key entities from the question
    2. Search for each entity
    3. Answer based on retrieved info

    Returns trajectory if successful (F1 > 0.5), else None.
    """
    state = env.reset(example)
    question = example.question

    # Simple entity extraction: use noun phrases from the question
    # For HotpotQA, supporting fact titles are good search queries
    search_queries = []

    # Use supporting fact titles as oracle queries (for demo generation)
    sf_titles = list(set(title for title, _ in example.supporting_facts))
    search_queries.extend(sf_titles[:3])

    # If we have fewer than 2 queries, add question-based ones
    if len(search_queries) < 2:
        # Simple heuristic: split question into clauses
        words = question.split()
        if len(words) > 6:
            mid = len(words) // 2
            search_queries.append(" ".join(words[:mid]))
            search_queries.append(" ".join(words[mid:]))
        else:
            search_queries.append(question)

    trajectory = []

    # Execute searches
    for i, query in enumerate(search_queries[:3]):
        action = f"I need to find information about {query}.\n\n<search>{query}</search>"
        state, reward, done, info = env.step(action)

        trajectory.append({
            "step_num": i + 1,
            "action_type": "search",
            "action_content": query,
            "result": info.get("search_results", ""),
        })

        if done:
            break

    # Generate answer
    if not state.done:
        # Use the gold answer for demonstration
        answer = example.answer
        action = (
            f"Based on the information I found, the answer is clear.\n\n"
            f"<answer>{answer}</answer>"
        )
        state, reward, done, info = env.step(action)

        trajectory.append({
            "step_num": len(trajectory) + 1,
            "action_type": "answer",
            "action_content": answer,
            "result": None,
        })

    # Verify quality
    if state.final_answer:
        f1 = compute_f1(state.final_answer, example.answer)
        if f1 >= 0.5:
            return trajectory

    return None


def trajectory_to_sft_format(
    question: str,
    trajectory: List[Dict],
) -> List[Dict[str, str]]:
    """
    Convert a trajectory into multi-turn chat format for SFT.

    Returns a list of messages: [system, user, assistant, user, assistant, ...]
    """
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    history_so_far = []

    for step in trajectory:
        # Build user message with current state
        user_content = f"Question: {question}\n"

        if history_so_far:
            user_content += "\nSearch History:\n"
            for h in history_so_far:
                if h["action_type"] in ("search", "refine"):
                    result = h.get("result", "No results")
                    result_short = result[:300] + "..." if len(result) > 300 else result
                    user_content += (
                        f'Step {h["step_num"]}: {h["action_type"]}ed '
                        f'"{h["action_content"]}" → found: "{result_short}"\n'
                    )

        user_content += (
            "\nAvailable actions: <search>query</search>, "
            "<refine>query</refine>, <answer>answer</answer>\n"
            "Think step by step, then output your next action."
        )

        messages.append({"role": "user", "content": user_content})

        # Build assistant response
        action_type = step["action_type"]
        action_content = step["action_content"]

        if action_type == "search":
            assistant_msg = (
                f"I need to search for information about {action_content} "
                f"to help answer this question.\n\n<search>{action_content}</search>"
            )
        elif action_type == "refine":
            assistant_msg = (
                f"The previous search wasn't sufficient. Let me try a more "
                f"specific query.\n\n<refine>{action_content}</refine>"
            )
        elif action_type == "answer":
            assistant_msg = (
                f"Based on the information gathered from my searches, "
                f"I can now answer the question.\n\n<answer>{action_content}</answer>"
            )
        else:
            continue

        messages.append({"role": "assistant", "content": assistant_msg})
        history_so_far.append(step)

    return messages


def prepare_sft_dataset(
    config: Dict,
    output_path: str = "./data/sft_trajectories.json",
) -> str:
    """
    Generate demonstration trajectories and save as SFT training data.

    Returns path to the saved dataset.
    """
    print("=" * 60)
    print("Phase 1a: Generating demonstration trajectories")
    print("=" * 60)

    # Load HotpotQA
    dataset = HotpotQADataset(
        split=config["dataset"]["split_train"],
        max_samples=config["sft"]["num_demo_trajectories"] * 2,  # oversample
    )

    env = WikiSearchEnv(
        max_steps=config["env"]["max_steps"],
        max_search_results=config["env"]["max_search_results"],
        passage_max_tokens=config["env"]["passage_max_tokens"],
        cache_dir=config["env"]["cache_dir"],
    )

    trajectories = []
    success_count = 0
    total_tried = 0

    for example in dataset.examples:
        total_tried += 1
        traj = generate_rule_based_trajectory(example, env)

        if traj is not None:
            messages = trajectory_to_sft_format(example.question, traj)
            trajectories.append({
                "id": example.id,
                "question": example.question,
                "answer": example.answer,
                "messages": messages,
                "num_steps": len(traj),
            })
            success_count += 1

            if success_count % 100 == 0:
                print(f"  Generated {success_count} trajectories ({total_tried} tried)")

        if success_count >= config["sft"]["num_demo_trajectories"]:
            break

    print(f"\nGenerated {success_count}/{total_tried} successful trajectories")

    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(trajectories, f, indent=2)

    print(f"Saved to {output_path}")
    return output_path


def train_sft(config: Dict, data_path: str):
    """
    Fine-tune Qwen2.5-3B-Instruct with LoRA on demonstration trajectories.
    """
    print("=" * 60)
    print("Phase 1b: SFT Training")
    print("=" * 60)

    model_name = config["agent_model"]["name"]
    sft_config = config["sft"]

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        padding_side="right",  # right padding for training
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model
    print(f"Loading model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    # Apply LoRA
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=config["agent_model"]["lora_rank"],
        lora_alpha=config["agent_model"]["lora_alpha"],
        lora_dropout=config["agent_model"]["lora_dropout"],
        target_modules=config["agent_model"]["lora_target_modules"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Load and format training data
    print(f"Loading training data from {data_path}")
    with open(data_path, "r") as f:
        raw_data = json.load(f)

    # Convert to chat-formatted strings
    def format_example(item):
        """Convert messages to a single training string using chat template."""
        text = tokenizer.apply_chat_template(
            item["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )
        return {"text": text}

    dataset = Dataset.from_list(raw_data)
    dataset = dataset.map(format_example)

    print(f"Training on {len(dataset)} examples")
    if len(dataset) > 0:
        print(f"Sample text (first 500 chars): {dataset[0]['text'][:500]}")

    # Training arguments
    output_dir = sft_config["output_dir"]
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=sft_config["num_epochs"],
        per_device_train_batch_size=sft_config["batch_size"],
        gradient_accumulation_steps=sft_config["gradient_accumulation_steps"],
        learning_rate=sft_config["learning_rate"],
        warmup_ratio=sft_config["warmup_ratio"],
        logging_steps=10,
        save_strategy="epoch",
        bf16=True,
        remove_unused_columns=False,
        dataloader_pin_memory=False,
        report_to="wandb" if config["logging"]["use_wandb"] else "none",
    )

    # SFT Trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=dataset,
        max_seq_length=sft_config["max_seq_length"],
    )

    print("Starting SFT training...")
    trainer.train()

    # Save
    print(f"Saving model to {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("SFT training complete!")


def main():
    parser = argparse.ArgumentParser(description="Phase 1: SFT Warmstart")
    parser.add_argument(
        "--config", type=str, default="configs/default.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--generate-only", action="store_true",
        help="Only generate trajectories, don't train",
    )
    parser.add_argument(
        "--train-only", action="store_true",
        help="Only train, skip trajectory generation",
    )
    parser.add_argument(
        "--data-path", type=str, default="./data/sft_trajectories.json",
        help="Path to trajectory data",
    )
    parser.add_argument(
        "--max-samples", type=int, default=None,
        help="Override max number of demo trajectories",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    if args.max_samples:
        config["sft"]["num_demo_trajectories"] = args.max_samples

    if not args.train_only:
        data_path = prepare_sft_dataset(config, args.data_path)
    else:
        data_path = args.data_path

    if not args.generate_only:
        train_sft(config, data_path)


if __name__ == "__main__":
    main()

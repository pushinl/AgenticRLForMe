"""
Phase 2: PRM Training.

Trains the Intent-Aware Process Reward Model on rollout trajectories
from the SFT model with heuristic + optional human labels.
"""

import os
import json
import argparse
from typing import Dict, List, Tuple

import yaml
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from env.dataset import HotpotQADataset, HotpotQAExample
from env.wiki_search_env import WikiSearchEnv
from models.agent import SearchAgent
from models.intent_prm import (
    IntentAwarePRM,
    PRMInputFormatter,
    compute_heuristic_labels,
)
from transformers import AutoTokenizer


class PRMDataset(Dataset):
    """Dataset for PRM training: each example is a single step with labels."""

    def __init__(
        self,
        data: List[Dict],
        formatter: PRMInputFormatter,
    ):
        self.data = data
        self.formatter = formatter

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        item = self.data[idx]

        tokenized = self.formatter.tokenize(
            question=item["question"],
            history=item["history"],
            current_action_type=item["action_type"],
            current_action_content=item["action_content"],
        )

        return {
            "input_ids": tokenized["input_ids"].squeeze(0),
            "attention_mask": tokenized["attention_mask"].squeeze(0),
            "question_mask": tokenized["question_mask"].squeeze(0),
            "progress_label": torch.tensor(item["progress_label"], dtype=torch.float32),
            "intent_label": torch.tensor(item["intent_label"], dtype=torch.float32),
        }


def collect_rollout_trajectories(
    config: Dict,
    agent: SearchAgent,
    num_trajectories: int = 5000,
    output_path: str = "./data/prm_rollouts.json",
) -> str:
    """
    Collect rollout trajectories from the SFT agent for PRM training.
    Labels each step with heuristic progress + intent alignment scores.
    """
    print("=" * 60)
    print("Phase 2a: Collecting rollout trajectories for PRM training")
    print("=" * 60)

    dataset = HotpotQADataset(
        split=config["dataset"]["split_train"],
        max_samples=num_trajectories * 2,
    )

    env = WikiSearchEnv(
        max_steps=config["env"]["max_steps"],
        max_search_results=config["env"]["max_search_results"],
        passage_max_tokens=config["env"]["passage_max_tokens"],
        cache_dir=config["env"]["cache_dir"],
    )

    all_step_data = []
    trajectories_collected = 0

    for example in dataset.examples:
        if trajectories_collected >= num_trajectories:
            break

        state = env.reset(example)
        gold_passages = dataset.get_gold_passages(example)
        history = []

        for step_idx in range(config["env"]["max_steps"]):
            if state.done:
                break

            # Get agent action
            try:
                response, action_type, action_content = agent.generate_action(
                    example.question,
                    history if history else None,
                    temperature=0.8,  # slightly more exploration
                )
            except Exception as e:
                print(f"  Error generating action: {e}")
                break

            # Execute in environment
            state, reward, done, info = env.step(response)

            # Compute heuristic labels
            progress, intent_align = compute_heuristic_labels(
                question=example.question,
                action_type=action_type,
                action_content=action_content,
                result=info.get("search_results", None),
                gold_answer=example.answer,
                gold_passages=gold_passages,
                step_num=step_idx + 1,
                total_steps=config["env"]["max_steps"],
            )

            # Save step data
            step_data = {
                "question": example.question,
                "answer": example.answer,
                "history": list(history),  # copy
                "action_type": action_type,
                "action_content": action_content,
                "result": info.get("search_results", ""),
                "progress_label": progress,
                "intent_label": intent_align,
                "step_num": step_idx + 1,
                "example_id": example.id,
            }
            all_step_data.append(step_data)

            # Update history
            history.append({
                "step_num": step_idx + 1,
                "action_type": action_type,
                "action_content": action_content,
                "result": info.get("search_results", ""),
            })

        trajectories_collected += 1
        if trajectories_collected % 100 == 0:
            print(f"  Collected {trajectories_collected} trajectories, {len(all_step_data)} steps")

    print(f"\nCollected {len(all_step_data)} steps from {trajectories_collected} trajectories")

    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_step_data, f, indent=2)

    print(f"Saved rollout data to {output_path}")
    return output_path


def train_prm(config: Dict, data_path: str):
    """Train the Intent-Aware PRM."""
    print("=" * 60)
    print("Phase 2b: Training Intent-Aware PRM")
    print("=" * 60)

    prm_config = config["prm_model"]
    train_config = config["prm_training"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        prm_config["name"],
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Initialize PRM
    print(f"Initializing IA-PRM with base: {prm_config['name']}")
    prm = IntentAwarePRM(
        model_name=prm_config["name"],
        hidden_size=prm_config["hidden_size"],
        alpha=prm_config["progress_weight"],
        beta=prm_config["intent_weight"],
    ).to(device)
    prm.print_trainable_parameters()

    # Load data
    print(f"Loading training data from {data_path}")
    with open(data_path, "r") as f:
        raw_data = json.load(f)

    print(f"Total steps: {len(raw_data)}")

    # Create dataset
    formatter = PRMInputFormatter(tokenizer, max_length=train_config["max_seq_length"])
    dataset = PRMDataset(raw_data, formatter)

    # Split train/val
    val_size = min(500, len(dataset) // 10)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_config["batch_size"],
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_config["batch_size"],
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # Optimizer
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, prm.parameters()),
        lr=train_config["learning_rate"],
        weight_decay=0.01,
    )

    total_steps = len(train_loader) * train_config["num_epochs"]
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps)

    # Loss functions
    progress_criterion = nn.MSELoss()
    intent_criterion = nn.MSELoss()

    progress_loss_weight = train_config["progress_loss_weight"]
    intent_loss_weight = train_config["intent_loss_weight"]

    # Training loop
    best_val_loss = float("inf")
    output_dir = train_config["output_dir"]

    for epoch in range(train_config["num_epochs"]):
        # Training
        prm.train()
        total_loss = 0
        total_progress_loss = 0
        total_intent_loss = 0
        num_batches = 0

        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            question_mask = batch["question_mask"].to(device)
            progress_labels = batch["progress_label"].to(device)
            intent_labels = batch["intent_label"].to(device)

            # Forward
            outputs = prm(input_ids, attention_mask, question_mask)

            # Compute losses
            p_loss = progress_criterion(
                outputs["progress_score"].squeeze(-1), progress_labels
            )
            i_loss = intent_criterion(
                outputs["intent_alignment_score"].squeeze(-1), intent_labels
            )
            loss = progress_loss_weight * p_loss + intent_loss_weight * i_loss

            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(prm.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            total_progress_loss += p_loss.item()
            total_intent_loss += i_loss.item()
            num_batches += 1

            if (batch_idx + 1) % 50 == 0:
                avg_loss = total_loss / num_batches
                print(
                    f"  Epoch {epoch+1}, Batch {batch_idx+1}/{len(train_loader)}: "
                    f"loss={avg_loss:.4f} (progress={total_progress_loss/num_batches:.4f}, "
                    f"intent={total_intent_loss/num_batches:.4f})"
                )

        avg_train_loss = total_loss / max(num_batches, 1)

        # Validation
        prm.eval()
        val_loss = 0
        val_progress_loss = 0
        val_intent_loss = 0
        val_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                question_mask = batch["question_mask"].to(device)
                progress_labels = batch["progress_label"].to(device)
                intent_labels = batch["intent_label"].to(device)

                outputs = prm(input_ids, attention_mask, question_mask)

                p_loss = progress_criterion(
                    outputs["progress_score"].squeeze(-1), progress_labels
                )
                i_loss = intent_criterion(
                    outputs["intent_alignment_score"].squeeze(-1), intent_labels
                )
                loss = progress_loss_weight * p_loss + intent_loss_weight * i_loss

                val_loss += loss.item()
                val_progress_loss += p_loss.item()
                val_intent_loss += i_loss.item()
                val_batches += 1

        avg_val_loss = val_loss / max(val_batches, 1)
        print(
            f"Epoch {epoch+1}/{train_config['num_epochs']}: "
            f"train_loss={avg_train_loss:.4f}, val_loss={avg_val_loss:.4f} "
            f"(val_progress={val_progress_loss/max(val_batches,1):.4f}, "
            f"val_intent={val_intent_loss/max(val_batches,1):.4f})"
        )

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            prm.save_pretrained(output_dir)
            print(f"  → Saved best model (val_loss={best_val_loss:.4f})")

    print(f"\nPRM training complete! Best val_loss: {best_val_loss:.4f}")
    print(f"Model saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Phase 2: PRM Training")
    parser.add_argument(
        "--config", type=str, default="configs/default.yaml",
    )
    parser.add_argument(
        "--collect-only", action="store_true",
        help="Only collect rollouts, don't train PRM",
    )
    parser.add_argument(
        "--train-only", action="store_true",
        help="Only train PRM, skip rollout collection",
    )
    parser.add_argument(
        "--data-path", type=str, default="./data/prm_rollouts.json",
    )
    parser.add_argument(
        "--sft-model-path", type=str, default=None,
        help="Path to SFT model (LoRA adapter). Uses base model if not provided.",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    if not args.train_only:
        # Load agent for rollout collection
        agent = SearchAgent(
            model_name=config["agent_model"]["name"],
            lora_path=args.sft_model_path,
            device="auto",
            temperature=0.8,
        )

        data_path = collect_rollout_trajectories(
            config, agent,
            num_trajectories=config["prm_training"]["num_rollout_trajectories"],
            output_path=args.data_path,
        )
    else:
        data_path = args.data_path

    if not args.collect_only:
        train_prm(config, data_path)


if __name__ == "__main__":
    main()

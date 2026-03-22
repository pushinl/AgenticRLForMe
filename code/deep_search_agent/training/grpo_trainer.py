"""
Phase 3: GRPO (Group Relative Policy Optimization) Training.

Core RL training loop that combines:
- Outcome reward: Final answer F1 score
- Process reward: IA-PRM step-level rewards (progress + intent alignment)

Uses the trl library's GRPO-style optimization with custom reward integration.
"""

import os
import json
import argparse
import math
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

import yaml
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, LoraConfig, TaskType, get_peft_model
from accelerate import Accelerator

from env.dataset import HotpotQADataset, HotpotQAExample, compute_f1
from env.wiki_search_env import WikiSearchEnv
from models.agent import SearchAgent, SYSTEM_PROMPT
from models.intent_prm import IntentAwarePRM, PRMInputFormatter


@dataclass
class Episode:
    """A complete search episode."""
    question: str
    gold_answer: str
    trajectory: List[Dict]  # list of step dicts
    final_answer: Optional[str]
    outcome_reward: float
    step_rewards: List[float]
    total_reward: float
    # For GRPO: store the generated tokens and log probs
    token_ids: List[List[int]]  # per-step token ids
    log_probs: List[torch.Tensor]  # per-step log probs


class GRPOTrainer:
    """
    GRPO Trainer for the DeepSearch Agent.

    Algorithm:
    1. For each question, generate G rollout episodes
    2. Compute rewards: outcome (F1) + process (IA-PRM)
    3. Normalize rewards within the group (GRPO's relative advantage)
    4. Update policy with clipped surrogate objective + KL penalty

    Reward = outcome_weight * outcome_reward + step_weight * mean(step_rewards)
    """

    def __init__(
        self,
        config: Dict,
        sft_model_path: Optional[str] = None,
        prm_path: Optional[str] = None,
    ):
        self.config = config
        self.grpo_config = config["grpo"]
        self.accelerator = Accelerator(
            gradient_accumulation_steps=self.grpo_config["gradient_accumulation_steps"],
        )

        # Load agent model
        model_name = config["agent_model"]["name"]
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            padding_side="left",
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.accelerator.print(f"Loading agent model: {model_name}")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )

        # Load SFT adapter if provided
        if sft_model_path:
            self.accelerator.print(f"Loading SFT adapter: {sft_model_path}")
            self.model = PeftModel.from_pretrained(self.model, sft_model_path)
            self.model = self.model.merge_and_unload()

        # Apply fresh LoRA for GRPO training
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=config["agent_model"]["lora_rank"],
            lora_alpha=config["agent_model"]["lora_alpha"],
            lora_dropout=config["agent_model"]["lora_dropout"],
            target_modules=config["agent_model"]["lora_target_modules"],
        )
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()

        # Reference model (frozen copy for KL penalty)
        self.accelerator.print("Creating reference model...")
        self.ref_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        if sft_model_path:
            self.ref_model = PeftModel.from_pretrained(self.ref_model, sft_model_path)
            self.ref_model = self.ref_model.merge_and_unload()
        self.ref_model.eval()
        for p in self.ref_model.parameters():
            p.requires_grad = False

        # Load IA-PRM
        if prm_path and os.path.exists(prm_path):
            self.accelerator.print(f"Loading IA-PRM from: {prm_path}")
            self.prm = IntentAwarePRM.from_pretrained(prm_path)
            self.prm.eval()
            for p in self.prm.parameters():
                p.requires_grad = False
        else:
            self.accelerator.print("No PRM loaded — using outcome-only rewards")
            self.prm = None

        # PRM formatter
        if self.prm is not None:
            prm_tokenizer = AutoTokenizer.from_pretrained(
                config["prm_model"]["name"],
                trust_remote_code=True,
            )
            if prm_tokenizer.pad_token is None:
                prm_tokenizer.pad_token = prm_tokenizer.eos_token
            self.prm_formatter = PRMInputFormatter(
                prm_tokenizer,
                max_length=config["prm_training"]["max_seq_length"],
            )
        else:
            self.prm_formatter = None

        # Environment
        self.env = WikiSearchEnv(
            max_steps=config["env"]["max_steps"],
            max_search_results=config["env"]["max_search_results"],
            passage_max_tokens=config["env"]["passage_max_tokens"],
            cache_dir=config["env"]["cache_dir"],
        )

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.grpo_config["learning_rate"],
            weight_decay=0.01,
        )

        # Prepare with accelerator
        self.model, self.optimizer = self.accelerator.prepare(
            self.model, self.optimizer
        )
        self.ref_model = self.ref_model.to(self.accelerator.device)
        if self.prm is not None:
            self.prm = self.prm.to(self.accelerator.device)

        # Hyperparams
        self.kl_coef = self.grpo_config["kl_coef"]
        self.num_generations = self.grpo_config["num_generations"]
        self.outcome_weight = self.grpo_config["outcome_reward_weight"]
        self.step_weight = self.grpo_config["step_reward_weight"]

    @torch.no_grad()
    def generate_episode(
        self,
        example: HotpotQAExample,
        temperature: float = 0.7,
    ) -> Episode:
        """
        Generate a complete search episode by interacting with the environment.
        Records token-level log probs for policy gradient computation.
        """
        state = self.env.reset(example)
        trajectory = []
        all_token_ids = []
        all_log_probs = []
        step_rewards = []

        history = []

        for step_idx in range(self.config["env"]["max_steps"]):
            if state.done:
                break

            # Build prompt
            messages = self._build_messages(example.question, history)
            prompt_text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
            inputs = self.tokenizer(
                prompt_text, return_tensors="pt",
            ).to(self.accelerator.device)

            # Generate with log probs
            model_to_use = self.accelerator.unwrap_model(self.model)
            outputs = model_to_use.generate(
                **inputs,
                max_new_tokens=self.grpo_config["max_new_tokens"],
                temperature=temperature,
                do_sample=True,
                top_p=0.9,
                return_dict_in_generate=True,
                output_scores=True,
                pad_token_id=self.tokenizer.pad_token_id,
            )

            # Extract generated tokens and compute log probs
            generated_ids = outputs.sequences[0][inputs["input_ids"].shape[1]:]
            scores = outputs.scores  # tuple of [vocab_size] tensors

            token_log_probs = []
            for t, score in enumerate(scores):
                if t >= len(generated_ids):
                    break
                log_prob = F.log_softmax(score[0], dim=-1)
                token_log_probs.append(log_prob[generated_ids[t]].item())

            response = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
            action_type, action_content = self._parse_action(response)

            # Execute in environment
            state, reward, done, info = self.env.step(response)

            # Compute PRM step reward
            prm_reward = 0.0
            if self.prm is not None and action_type != "invalid":
                prm_reward = self._compute_prm_reward(
                    example.question, history, action_type, action_content,
                )

            step_rewards.append(prm_reward)

            # Record
            step_data = {
                "step_num": step_idx + 1,
                "action_type": action_type,
                "action_content": action_content,
                "result": info.get("search_results", ""),
                "prm_reward": prm_reward,
            }
            trajectory.append(step_data)
            history.append(step_data)
            all_token_ids.append(generated_ids.tolist())
            all_log_probs.append(torch.tensor(token_log_probs))

        # Compute outcome reward
        outcome_reward = 0.0
        if state.final_answer:
            outcome_reward = compute_f1(state.final_answer, example.answer)

        # Total reward = outcome + mean(step_rewards)
        mean_step_reward = sum(step_rewards) / max(len(step_rewards), 1)
        total_reward = (
            self.outcome_weight * outcome_reward
            + self.step_weight * mean_step_reward
        )

        return Episode(
            question=example.question,
            gold_answer=example.answer,
            trajectory=trajectory,
            final_answer=state.final_answer,
            outcome_reward=outcome_reward,
            step_rewards=step_rewards,
            total_reward=total_reward,
            token_ids=all_token_ids,
            log_probs=all_log_probs,
        )

    @torch.no_grad()
    def _compute_prm_reward(
        self,
        question: str,
        history: List[Dict],
        action_type: str,
        action_content: str,
    ) -> float:
        """Compute PRM reward for a single step."""
        tokenized = self.prm_formatter.tokenize(
            question, history, action_type, action_content,
        )

        input_ids = tokenized["input_ids"].to(self.accelerator.device)
        attention_mask = tokenized["attention_mask"].to(self.accelerator.device)
        question_mask = tokenized["question_mask"].to(self.accelerator.device)

        reward = self.prm.compute_step_reward(input_ids, attention_mask, question_mask)
        return reward.item()

    def compute_grpo_loss(
        self,
        episodes: List[Episode],
    ) -> torch.Tensor:
        """
        Compute GRPO loss for a group of episodes.

        GRPO normalizes rewards within a group (per-question):
        advantage_i = (reward_i - mean(rewards)) / std(rewards)

        Loss = -Σ advantage_i * Σ_t log π(a_t|s_t) + kl_coef * KL(π || π_ref)
        """
        # Group relative advantage
        rewards = torch.tensor([ep.total_reward for ep in episodes])
        if len(rewards) > 1 and rewards.std() > 1e-8:
            advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        else:
            advantages = rewards - rewards.mean()

        total_loss = torch.tensor(0.0, device=self.accelerator.device, requires_grad=True)

        for ep_idx, episode in enumerate(episodes):
            advantage = advantages[ep_idx].item()
            if abs(advantage) < 1e-8:
                continue

            # For each step in the episode, recompute log probs and KL
            for step_idx, (step, token_ids, old_log_probs) in enumerate(
                zip(episode.trajectory, episode.token_ids, episode.log_probs)
            ):
                if not token_ids:
                    continue

                # Rebuild the prompt for this step
                history = episode.trajectory[:step_idx]
                messages = self._build_messages(episode.question, history)
                prompt_text = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True,
                )

                prompt_inputs = self.tokenizer(
                    prompt_text, return_tensors="pt",
                ).to(self.accelerator.device)

                # Concatenate prompt + generated tokens
                gen_tensor = torch.tensor(
                    token_ids, device=self.accelerator.device,
                ).unsqueeze(0)
                full_ids = torch.cat([prompt_inputs["input_ids"], gen_tensor], dim=1)
                full_mask = torch.ones_like(full_ids)

                # Forward pass through current policy
                outputs = self.model(
                    input_ids=full_ids,
                    attention_mask=full_mask,
                )
                logits = outputs.logits

                # Get log probs for generated tokens
                prompt_len = prompt_inputs["input_ids"].shape[1]
                gen_logits = logits[0, prompt_len - 1:-1]  # shift by 1
                gen_log_probs = F.log_softmax(gen_logits, dim=-1)

                gen_ids = torch.tensor(token_ids, device=self.accelerator.device)
                num_tokens = min(len(gen_ids), gen_log_probs.shape[0])
                policy_log_probs = gen_log_probs[
                    torch.arange(num_tokens), gen_ids[:num_tokens]
                ]

                # Reference model log probs for KL
                with torch.no_grad():
                    ref_outputs = self.ref_model(
                        input_ids=full_ids.to(self.ref_model.device),
                        attention_mask=full_mask.to(self.ref_model.device),
                    )
                    ref_logits = ref_outputs.logits
                    ref_gen_logits = ref_logits[0, prompt_len - 1:-1]
                    ref_log_probs = F.log_softmax(ref_gen_logits, dim=-1)
                    ref_policy_log_probs = ref_log_probs[
                        torch.arange(num_tokens), gen_ids[:num_tokens]
                    ].to(self.accelerator.device)

                # KL divergence per token
                kl_div = (policy_log_probs - ref_policy_log_probs).mean()

                # Policy gradient loss for this step
                step_log_prob_sum = policy_log_probs.sum()
                step_loss = -advantage * step_log_prob_sum + self.kl_coef * kl_div

                total_loss = total_loss + step_loss

        return total_loss / max(len(episodes), 1)

    def train(self):
        """Main training loop."""
        self.accelerator.print("=" * 60)
        self.accelerator.print("Phase 3: GRPO Training")
        self.accelerator.print("=" * 60)

        # Load dataset
        dataset = HotpotQADataset(
            split=self.config["dataset"]["split_train"],
            max_samples=self.grpo_config["num_episodes"] * 2,
        )

        output_dir = self.grpo_config["output_dir"]
        os.makedirs(output_dir, exist_ok=True)

        # Training metrics
        metrics_log = []
        best_avg_reward = -float("inf")

        num_episodes = min(self.grpo_config["num_episodes"], len(dataset))
        self.accelerator.print(f"Training for {num_episodes} episodes")
        self.accelerator.print(f"Generations per question: {self.num_generations}")

        for episode_idx in range(num_episodes):
            example = dataset[episode_idx]

            # Generate G episodes for the same question (GRPO group)
            self.model.eval()
            episodes = []
            for g in range(self.num_generations):
                ep = self.generate_episode(
                    example,
                    temperature=self.grpo_config["temperature"],
                )
                episodes.append(ep)

            # Compute GRPO loss
            self.model.train()
            with self.accelerator.accumulate(self.model):
                loss = self.compute_grpo_loss(episodes)
                self.accelerator.backward(loss)

                # Gradient clipping
                if self.accelerator.sync_gradients:
                    self.accelerator.clip_grad_norm_(
                        self.model.parameters(), max_norm=1.0,
                    )
                self.optimizer.step()
                self.optimizer.zero_grad()

            # Log metrics
            avg_reward = sum(ep.total_reward for ep in episodes) / len(episodes)
            avg_outcome = sum(ep.outcome_reward for ep in episodes) / len(episodes)
            avg_steps = sum(
                len(ep.trajectory) for ep in episodes
            ) / len(episodes)

            metrics = {
                "episode": episode_idx + 1,
                "loss": loss.item(),
                "avg_reward": avg_reward,
                "avg_outcome_f1": avg_outcome,
                "avg_steps": avg_steps,
                "question": example.question[:80],
            }
            metrics_log.append(metrics)

            if (episode_idx + 1) % self.config["logging"]["log_every_n_steps"] == 0:
                recent = metrics_log[-self.config["logging"]["log_every_n_steps"]:]
                avg_r = sum(m["avg_reward"] for m in recent) / len(recent)
                avg_f1 = sum(m["avg_outcome_f1"] for m in recent) / len(recent)
                avg_s = sum(m["avg_steps"] for m in recent) / len(recent)

                self.accelerator.print(
                    f"Episode {episode_idx+1}/{num_episodes}: "
                    f"loss={loss.item():.4f}, avg_reward={avg_r:.3f}, "
                    f"avg_f1={avg_f1:.3f}, avg_steps={avg_s:.1f}"
                )

            # Save checkpoint
            if (episode_idx + 1) % self.grpo_config["save_steps"] == 0:
                if avg_reward > best_avg_reward:
                    best_avg_reward = avg_reward
                    save_path = os.path.join(output_dir, "best")
                    self.accelerator.unwrap_model(self.model).save_pretrained(save_path)
                    self.tokenizer.save_pretrained(save_path)
                    self.accelerator.print(f"  → Saved best model (reward={best_avg_reward:.3f})")

                # Save latest
                save_path = os.path.join(output_dir, f"checkpoint-{episode_idx+1}")
                self.accelerator.unwrap_model(self.model).save_pretrained(save_path)
                self.tokenizer.save_pretrained(save_path)

        # Save final model
        final_path = os.path.join(output_dir, "final")
        self.accelerator.unwrap_model(self.model).save_pretrained(final_path)
        self.tokenizer.save_pretrained(final_path)

        # Save metrics
        metrics_path = os.path.join(output_dir, "training_metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics_log, f, indent=2)

        self.accelerator.print(f"\nGRPO training complete!")
        self.accelerator.print(f"Best avg reward: {best_avg_reward:.3f}")
        self.accelerator.print(f"Model saved to {final_path}")
        self.accelerator.print(f"Metrics saved to {metrics_path}")

    def _build_messages(
        self,
        question: str,
        history: Optional[List[Dict]] = None,
    ) -> List[Dict[str, str]]:
        """Build chat messages for model input."""
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]

        user_content = f"Question: {question}\n"
        if history:
            user_content += "\nSearch History:\n"
            for h in history:
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
        return messages

    def _parse_action(self, text: str) -> Tuple[str, str]:
        """Parse action from model output."""
        import re
        patterns = {
            "search": re.compile(r"<search>(.*?)</search>", re.DOTALL),
            "refine": re.compile(r"<refine>(.*?)</refine>", re.DOTALL),
            "answer": re.compile(r"<answer>(.*?)</answer>", re.DOTALL),
        }
        for action_type, pattern in patterns.items():
            match = pattern.search(text)
            if match:
                return action_type, match.group(1).strip()
        return "invalid", text


def main():
    parser = argparse.ArgumentParser(description="Phase 3: GRPO Training")
    parser.add_argument(
        "--config", type=str, default="configs/default.yaml",
    )
    parser.add_argument(
        "--sft-model-path", type=str, default=None,
        help="Path to SFT model checkpoint",
    )
    parser.add_argument(
        "--prm-path", type=str, default=None,
        help="Path to trained IA-PRM model",
    )
    parser.add_argument(
        "--num-episodes", type=int, default=None,
        help="Override number of training episodes",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    if args.num_episodes:
        config["grpo"]["num_episodes"] = args.num_episodes

    # Use default checkpoint paths if not specified
    sft_path = args.sft_model_path or config["sft"]["output_dir"]
    prm_path = args.prm_path or config["prm_training"]["output_dir"]

    trainer = GRPOTrainer(
        config=config,
        sft_model_path=sft_path if os.path.exists(sft_path) else None,
        prm_path=prm_path if os.path.exists(prm_path) else None,
    )

    trainer.train()


if __name__ == "__main__":
    main()

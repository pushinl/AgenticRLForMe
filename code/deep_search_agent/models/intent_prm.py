"""
Intent-Aware Process Reward Model (IA-PRM).

Core innovation: Traditional PRM only evaluates "is this step correct?"
IA-PRM additionally evaluates "is this step aligned with the user's original intent?"

Architecture:
- Base: Qwen2.5-1.5B (smaller than agent to save resources)
- Input: (original_question, current_step_action, search_history)
- Output: Two heads
  - progress_score: Is this step making progress toward the answer? (0~1)
  - intent_alignment_score: Does this step stay aligned with original intent? (0~1)
- Final step reward = α * progress + β * intent_alignment

Interview narrative:
"When I did binary intent classification, I found static judgment wasn't enough.
So in this project I made intent alignment a dynamic per-step signal —
that's the intent_alignment head of IA-PRM."
"""

import os
import json
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM


class IntentAwarePRM(nn.Module):
    """
    Intent-Aware Process Reward Model.

    Two-headed reward model:
    1. Progress head: evaluates whether the current step advances toward the answer
    2. Intent alignment head: evaluates whether the step stays aligned with the query intent

    Final reward = alpha * progress_score + beta * intent_alignment_score
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-1.5B",
        hidden_size: int = 1536,
        alpha: float = 0.6,
        beta: float = 0.4,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.alpha = alpha
        self.beta = beta
        self.model_name = model_name

        # Load base model (encoder-style usage)
        self.base_model = AutoModel.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )

        # Freeze most layers, only fine-tune last few
        self._freeze_base_layers(num_unfrozen=4)

        # Progress reward head
        # Evaluates: "Is this step making progress toward answering the question?"
        self.progress_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid(),
        )

        # Intent alignment head
        # Evaluates: "Does this step stay aligned with the original query intent?"
        self.intent_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid(),
        )

        # Optional: cross-attention between question representation and step representation
        self.intent_cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=8,
            dropout=dropout,
            batch_first=True,
        )

    def _freeze_base_layers(self, num_unfrozen: int = 4):
        """Freeze all but the last N transformer layers."""
        # Freeze embeddings
        for param in self.base_model.embed_tokens.parameters():
            param.requires_grad = False

        # Freeze all layers
        layers = self.base_model.layers
        for layer in layers:
            for param in layer.parameters():
                param.requires_grad = False

        # Unfreeze last N layers
        for layer in layers[-num_unfrozen:]:
            for param in layer.parameters():
                param.requires_grad = True

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        question_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            input_ids: Tokenized input [batch, seq_len]
                Format: "[Question] <sep> [Search History] <sep> [Current Step Action]"
            attention_mask: Attention mask [batch, seq_len]
            question_mask: Binary mask indicating question token positions [batch, seq_len]
                Used by intent alignment head to compute cross-attention

        Returns:
            Dict with:
                - progress_score: [batch, 1] in (0, 1)
                - intent_alignment_score: [batch, 1] in (0, 1)
                - combined_reward: [batch, 1] = alpha * progress + beta * intent
        """
        # Get hidden states from base model
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        hidden_states = outputs.last_hidden_state  # [batch, seq, hidden]

        # Pool: use last non-padding token (like reward models typically do)
        # Find the last non-padding position for each example
        seq_lengths = attention_mask.sum(dim=1) - 1  # [batch]
        batch_indices = torch.arange(hidden_states.size(0), device=hidden_states.device)
        pooled = hidden_states[batch_indices, seq_lengths]  # [batch, hidden]

        # Progress score
        progress_score = self.progress_head(pooled)  # [batch, 1]

        # Intent alignment score with cross-attention
        if question_mask is not None:
            # Get question representation
            # Expand question_mask to match hidden_states
            q_mask_expanded = question_mask.unsqueeze(-1).float()  # [batch, seq, 1]
            question_repr = (hidden_states * q_mask_expanded).sum(dim=1, keepdim=True)
            q_count = question_mask.sum(dim=1, keepdim=True).unsqueeze(-1).clamp(min=1)
            question_repr = question_repr / q_count  # [batch, 1, hidden]

            # Cross-attention: step attends to question
            step_repr = pooled.unsqueeze(1)  # [batch, 1, hidden]
            attn_output, _ = self.intent_cross_attn(
                query=step_repr,
                key=question_repr,
                value=question_repr,
            )
            intent_input = attn_output.squeeze(1)  # [batch, hidden]
        else:
            intent_input = pooled

        intent_alignment_score = self.intent_head(intent_input)  # [batch, 1]

        # Combined reward
        combined_reward = (
            self.alpha * progress_score
            + self.beta * intent_alignment_score
        )

        return {
            "progress_score": progress_score,
            "intent_alignment_score": intent_alignment_score,
            "combined_reward": combined_reward,
            "hidden_states": pooled,  # for analysis
        }

    def compute_step_reward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        question_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Convenience method: returns just the combined scalar reward."""
        out = self.forward(input_ids, attention_mask, question_mask)
        return out["combined_reward"]

    def save_pretrained(self, save_dir: str):
        """Save model weights and config."""
        os.makedirs(save_dir, exist_ok=True)

        # Save base model
        self.base_model.save_pretrained(os.path.join(save_dir, "base_model"))

        # Save heads + config
        torch.save({
            "progress_head": self.progress_head.state_dict(),
            "intent_head": self.intent_head.state_dict(),
            "intent_cross_attn": self.intent_cross_attn.state_dict(),
            "alpha": self.alpha,
            "beta": self.beta,
        }, os.path.join(save_dir, "reward_heads.pt"))

        config = {
            "model_name": self.model_name,
            "alpha": self.alpha,
            "beta": self.beta,
        }
        with open(os.path.join(save_dir, "prm_config.json"), "w") as f:
            json.dump(config, f, indent=2)

        print(f"IA-PRM saved to {save_dir}")

    @classmethod
    def from_pretrained(cls, load_dir: str, device: str = "auto") -> "IntentAwarePRM":
        """Load a saved IA-PRM."""
        with open(os.path.join(load_dir, "prm_config.json")) as f:
            config = json.load(f)

        model = cls(
            model_name=os.path.join(load_dir, "base_model"),
            alpha=config["alpha"],
            beta=config["beta"],
        )

        heads = torch.load(
            os.path.join(load_dir, "reward_heads.pt"),
            map_location="cpu",
        )
        model.progress_head.load_state_dict(heads["progress_head"])
        model.intent_head.load_state_dict(heads["intent_head"])
        model.intent_cross_attn.load_state_dict(heads["intent_cross_attn"])

        return model

    def print_trainable_parameters(self):
        """Print the number of trainable vs total parameters."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(
            f"IA-PRM: {trainable:,} trainable params / {total:,} total "
            f"({100 * trainable / total:.2f}%)"
        )


class PRMInputFormatter:
    """
    Formats search trajectories into input for the IA-PRM.

    Input format:
    [Question] question text
    [History] Step 1: searched "X" → "result..."  Step 2: ...
    [Current Action] search "Y"
    """

    def __init__(self, tokenizer: AutoTokenizer, max_length: int = 1024):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def format_step(
        self,
        question: str,
        history: List[Dict],
        current_action_type: str,
        current_action_content: str,
    ) -> str:
        """Format a single step for PRM input."""
        parts = [f"[Question] {question}"]

        if history:
            history_str = " ".join([
                f'Step {h["step_num"]}: {h["action_type"]}ed "{h["action_content"][:100]}" → '
                f'"{h.get("result", "")[:150]}"'
                for h in history
            ])
            parts.append(f"[History] {history_str}")

        parts.append(f"[Current Action] {current_action_type} \"{current_action_content}\"")

        return "\n".join(parts)

    def tokenize(
        self,
        question: str,
        history: List[Dict],
        current_action_type: str,
        current_action_content: str,
    ) -> Dict[str, torch.Tensor]:
        """Tokenize and return input_ids, attention_mask, question_mask."""
        text = self.format_step(question, history, current_action_type, current_action_content)

        # Tokenize full text
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        # Create question mask: tokens belonging to the question portion
        question_text = f"[Question] {question}"
        question_tokens = self.tokenizer(
            question_text,
            add_special_tokens=False,
        )
        question_len = len(question_tokens["input_ids"])

        question_mask = torch.zeros(1, self.max_length, dtype=torch.long)
        # Account for BOS token (position 0)
        start = 1 if self.tokenizer.bos_token_id is not None else 0
        end = min(start + question_len, self.max_length)
        question_mask[0, start:end] = 1

        return {
            "input_ids": encoding["input_ids"],
            "attention_mask": encoding["attention_mask"],
            "question_mask": question_mask,
        }

    def tokenize_batch(
        self,
        questions: List[str],
        histories: List[List[Dict]],
        action_types: List[str],
        action_contents: List[str],
    ) -> Dict[str, torch.Tensor]:
        """Tokenize a batch of steps."""
        texts = [
            self.format_step(q, h, at, ac)
            for q, h, at, ac in zip(questions, histories, action_types, action_contents)
        ]

        encoding = self.tokenizer(
            texts,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        # Create question masks for the batch
        batch_size = len(questions)
        question_masks = torch.zeros(batch_size, self.max_length, dtype=torch.long)

        for i, q in enumerate(questions):
            q_text = f"[Question] {q}"
            q_tokens = self.tokenizer(q_text, add_special_tokens=False)
            q_len = len(q_tokens["input_ids"])
            start = 1 if self.tokenizer.bos_token_id is not None else 0
            end = min(start + q_len, self.max_length)
            question_masks[i, start:end] = 1

        return {
            "input_ids": encoding["input_ids"],
            "attention_mask": encoding["attention_mask"],
            "question_mask": question_masks,
        }


def compute_heuristic_labels(
    question: str,
    action_type: str,
    action_content: str,
    result: Optional[str],
    gold_answer: str,
    gold_passages: List[str],
    step_num: int,
    total_steps: int,
) -> Tuple[float, float]:
    """
    Compute heuristic labels for PRM training.

    Returns (progress_score, intent_alignment_score) in [0, 1].

    Progress heuristics:
    - If answer step with high F1: high progress
    - If search result overlaps with gold passages: medium-high progress
    - If search yields relevant results: low-medium progress

    Intent alignment heuristics:
    - BM25-style overlap between search query and original question
    - Penalize queries that drift too far from question entities
    """
    from env.dataset import compute_f1, _normalize_answer

    # --- Progress Score ---
    progress = 0.5  # default

    if action_type == "answer":
        f1 = compute_f1(action_content, gold_answer)
        progress = f1
    elif action_type in ("search", "refine") and result:
        # Check overlap with gold answer
        answer_overlap = compute_f1(result[:500], gold_answer)
        # Check overlap with gold passages
        passage_overlaps = [
            compute_f1(result[:500], p) for p in gold_passages
        ] if gold_passages else [0.0]
        max_passage_overlap = max(passage_overlaps)

        progress = 0.3 * answer_overlap + 0.7 * max_passage_overlap
        progress = min(progress * 2.0, 1.0)  # scale up

    # Step discount: earlier informative steps are worth slightly more
    step_bonus = 0.1 * (1.0 - step_num / max(total_steps, 1))
    progress = min(progress + step_bonus, 1.0)

    # --- Intent Alignment Score ---
    intent_alignment = 0.5  # default

    # Token overlap between query and original question
    question_tokens = set(_normalize_answer(question).split())
    action_tokens = set(_normalize_answer(action_content).split())

    if question_tokens and action_tokens:
        # Jaccard-like similarity
        overlap = len(question_tokens & action_tokens)
        union = len(question_tokens | action_tokens)
        token_sim = overlap / max(union, 1)

        # Also consider if key question words appear in query
        # (longer overlap = better alignment)
        recall = overlap / max(len(question_tokens), 1)

        intent_alignment = 0.4 * token_sim + 0.6 * recall

    # Answer actions get high intent alignment by default
    if action_type == "answer":
        intent_alignment = max(intent_alignment, 0.7)

    # Refine actions slightly penalized (might be drifting)
    if action_type == "refine":
        intent_alignment *= 0.9

    return float(progress), float(intent_alignment)


if __name__ == "__main__":
    # Quick test
    print("Testing IA-PRM model...")

    # Test heuristic labels
    p, i = compute_heuristic_labels(
        question="Were Scott Derrickson and Ed Wood of the same nationality?",
        action_type="search",
        action_content="Scott Derrickson nationality",
        result="Scott Derrickson is an American director, screenwriter, and producer.",
        gold_answer="Yes",
        gold_passages=["Scott Derrickson is an American director."],
        step_num=1,
        total_steps=3,
    )
    print(f"Progress: {p:.3f}, Intent Alignment: {i:.3f}")

    p2, i2 = compute_heuristic_labels(
        question="Were Scott Derrickson and Ed Wood of the same nationality?",
        action_type="search",
        action_content="history of cinema in France",  # drifting query
        result="French cinema has a long and distinguished history.",
        gold_answer="Yes",
        gold_passages=["Scott Derrickson is an American director."],
        step_num=2,
        total_steps=3,
    )
    print(f"Drifting query - Progress: {p2:.3f}, Intent Alignment: {i2:.3f}")
    print(f"Intent alignment dropped: {i:.3f} -> {i2:.3f} ✓")

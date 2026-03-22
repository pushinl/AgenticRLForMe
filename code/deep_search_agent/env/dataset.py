"""
HotpotQA dataset loading and preprocessing.
Provides multi-hop questions for the search agent to answer.
"""

import json
import random
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from datasets import load_dataset


@dataclass
class HotpotQAExample:
    """A single HotpotQA example."""
    question: str
    answer: str
    supporting_facts: List[Tuple[str, int]]  # (title, sentence_idx)
    context: Dict[str, List[str]]  # title -> sentences
    question_type: str  # "bridge" or "comparison"
    level: str  # "easy", "medium", "hard"
    id: str


class HotpotQADataset:
    """Loads and manages HotpotQA data for the search agent environment."""

    def __init__(
        self,
        split: str = "train",
        max_samples: Optional[int] = None,
        difficulty_filter: Optional[str] = None,
        seed: int = 42,
    ):
        self.split = split
        self.max_samples = max_samples
        self.difficulty_filter = difficulty_filter
        self.seed = seed
        self.examples: List[HotpotQAExample] = []
        self._load()

    def _load(self):
        """Load HotpotQA from HuggingFace datasets."""
        # HotpotQA distractor setting
        dataset = load_dataset("hotpot_qa", "distractor", split=self.split)

        for item in dataset:
            example = HotpotQAExample(
                question=item["question"],
                answer=item["answer"],
                supporting_facts=list(zip(
                    item["supporting_facts"]["title"],
                    item["supporting_facts"]["sent_id"],
                )),
                context={
                    title: sents
                    for title, sents in zip(
                        item["context"]["title"],
                        item["context"]["sentences"],
                    )
                },
                question_type=item["type"],
                level=item["level"],
                id=item["id"],
            )

            if self.difficulty_filter and example.level != self.difficulty_filter:
                continue

            self.examples.append(example)

            if self.max_samples and len(self.examples) >= self.max_samples:
                break

        # Shuffle with fixed seed for reproducibility
        random.seed(self.seed)
        random.shuffle(self.examples)
        print(f"Loaded {len(self.examples)} HotpotQA examples (split={self.split})")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx) -> HotpotQAExample:
        return self.examples[idx]

    def get_batch(self, batch_size: int, offset: int = 0) -> List[HotpotQAExample]:
        """Get a batch of examples."""
        return self.examples[offset:offset + batch_size]

    def get_gold_passages(self, example: HotpotQAExample) -> List[str]:
        """Extract gold supporting passages for an example."""
        passages = []
        for title, sent_idx in example.supporting_facts:
            if title in example.context:
                sents = example.context[title]
                if sent_idx < len(sents):
                    passages.append(f"[{title}] {sents[sent_idx]}")
        return passages


def compute_f1(prediction: str, ground_truth: str) -> float:
    """Compute token-level F1 between prediction and ground truth."""
    pred_tokens = _normalize_answer(prediction).split()
    gt_tokens = _normalize_answer(ground_truth).split()

    if not pred_tokens or not gt_tokens:
        return float(pred_tokens == gt_tokens)

    common = set(pred_tokens) & set(gt_tokens)
    if not common:
        return 0.0

    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(gt_tokens)
    return 2 * precision * recall / (precision + recall)


def compute_em(prediction: str, ground_truth: str) -> float:
    """Compute exact match between prediction and ground truth."""
    return float(_normalize_answer(prediction) == _normalize_answer(ground_truth))


def _normalize_answer(s: str) -> str:
    """Normalize answer string for evaluation."""
    import re
    import string

    s = s.lower()
    # Remove articles
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    # Remove punctuation
    s = ''.join(ch for ch in s if ch not in string.punctuation)
    # Remove extra whitespace
    s = ' '.join(s.split())
    return s


if __name__ == "__main__":
    # Quick test
    print("Loading HotpotQA dataset...")
    ds = HotpotQADataset(split="train", max_samples=10)

    for i, ex in enumerate(ds.examples[:3]):
        print(f"\n--- Example {i+1} ---")
        print(f"Question: {ex.question}")
        print(f"Answer: {ex.answer}")
        print(f"Type: {ex.question_type}, Level: {ex.level}")
        print(f"Supporting facts: {ex.supporting_facts[:3]}")
        gold = ds.get_gold_passages(ex)
        print(f"Gold passages: {gold[:2]}")

    # Test F1/EM
    print(f"\nF1('the cat sat', 'a cat sat down'): {compute_f1('the cat sat', 'a cat sat down'):.3f}")
    print(f"EM('yes', 'yes'): {compute_em('yes', 'yes'):.1f}")
    print(f"EM('Yes', 'yes'): {compute_em('Yes', 'yes'):.1f}")

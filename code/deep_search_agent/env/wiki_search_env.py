"""
Wikipedia Search Environment for the DeepSearch Agent.

Provides a gym-like interface where the agent interacts with Wikipedia
to answer multi-hop questions from HotpotQA.
"""

import os
import re
import json
import hashlib
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field

import wikipediaapi

from env.dataset import HotpotQAExample, compute_f1, compute_em


@dataclass
class SearchResult:
    """A single search result from Wikipedia."""
    title: str
    passage: str  # truncated passage
    full_text: str  # full page text (cached)


@dataclass
class StepRecord:
    """Record of a single agent step."""
    step_num: int
    action_type: str  # "search", "refine", "answer"
    action_content: str  # the query or answer text
    result: Optional[str] = None  # search result summary
    reward: float = 0.0


@dataclass
class EnvState:
    """Current state of the search environment."""
    question: str
    search_history: List[StepRecord] = field(default_factory=list)
    retrieved_passages: List[str] = field(default_factory=list)
    step_count: int = 0
    done: bool = False
    final_answer: Optional[str] = None
    total_reward: float = 0.0


class WikiSearchEnv:
    """
    Wikipedia search environment for multi-hop QA.

    The agent can:
    - <search>query</search>: Search Wikipedia for a query
    - <refine>new_query</refine>: Refine search query based on history
    - <answer>final_answer</answer>: Submit a final answer

    Episode ends when agent submits an answer or max steps reached.
    """

    # Regex patterns for parsing agent actions
    ACTION_PATTERNS = {
        "search": re.compile(r"<search>(.*?)</search>", re.DOTALL),
        "refine": re.compile(r"<refine>(.*?)</refine>", re.DOTALL),
        "answer": re.compile(r"<answer>(.*?)</answer>", re.DOTALL),
    }

    def __init__(
        self,
        max_steps: int = 5,
        max_search_results: int = 3,
        passage_max_tokens: int = 200,
        cache_dir: str = "./cache/wiki",
        use_cache: bool = True,
    ):
        self.max_steps = max_steps
        self.max_search_results = max_search_results
        self.passage_max_tokens = passage_max_tokens
        self.cache_dir = cache_dir
        self.use_cache = use_cache

        # Initialize Wikipedia API
        self.wiki = wikipediaapi.Wikipedia(
            user_agent="DeepSearchAgent/1.0 (research project)",
            language="en",
        )

        # Setup cache
        if use_cache:
            os.makedirs(cache_dir, exist_ok=True)

        self.state: Optional[EnvState] = None
        self.current_example: Optional[HotpotQAExample] = None

    def reset(self, example: HotpotQAExample) -> EnvState:
        """Reset the environment with a new question."""
        self.current_example = example
        self.state = EnvState(question=example.question)
        return self.state

    def step(self, action_text: str) -> Tuple[EnvState, float, bool, Dict[str, Any]]:
        """
        Execute one agent action.

        Args:
            action_text: Raw text output from the agent containing an action tag.

        Returns:
            (state, reward, done, info)
        """
        if self.state is None:
            raise RuntimeError("Must call reset() before step()")

        if self.state.done:
            raise RuntimeError("Episode already done. Call reset().")

        # Parse action
        action_type, action_content = self._parse_action(action_text)
        self.state.step_count += 1

        info = {"action_type": action_type, "action_content": action_content}
        reward = 0.0

        if action_type == "answer":
            # Episode ends with answer
            self.state.final_answer = action_content
            self.state.done = True

            # Compute outcome reward
            if self.current_example:
                f1 = compute_f1(action_content, self.current_example.answer)
                em = compute_em(action_content, self.current_example.answer)
                reward = f1  # Use F1 as primary reward
                info["f1"] = f1
                info["em"] = em
                info["gold_answer"] = self.current_example.answer

            step_record = StepRecord(
                step_num=self.state.step_count,
                action_type="answer",
                action_content=action_content,
                reward=reward,
            )

        elif action_type in ("search", "refine"):
            # Execute search
            results = self._search_wikipedia(action_content)
            result_summary = self._format_results(results)

            self.state.retrieved_passages.extend(
                [r.passage for r in results]
            )

            step_record = StepRecord(
                step_num=self.state.step_count,
                action_type=action_type,
                action_content=action_content,
                result=result_summary,
            )
            info["search_results"] = result_summary

        else:
            # Invalid action — small penalty
            reward = -0.1
            step_record = StepRecord(
                step_num=self.state.step_count,
                action_type="invalid",
                action_content=action_text[:200],
                reward=reward,
            )
            info["error"] = "Could not parse valid action from output"

        self.state.search_history.append(step_record)
        self.state.total_reward += reward

        # Check if max steps reached
        if self.state.step_count >= self.max_steps and not self.state.done:
            self.state.done = True
            info["terminated"] = "max_steps_reached"

        return self.state, reward, self.state.done, info

    def _parse_action(self, text: str) -> Tuple[str, str]:
        """Parse action type and content from agent output."""
        for action_type, pattern in self.ACTION_PATTERNS.items():
            match = pattern.search(text)
            if match:
                return action_type, match.group(1).strip()
        return "invalid", text

    def _search_wikipedia(self, query: str) -> List[SearchResult]:
        """Search Wikipedia and return results."""
        # Check cache first
        cache_key = hashlib.md5(query.encode()).hexdigest()
        cache_path = os.path.join(self.cache_dir, f"{cache_key}.json")

        if self.use_cache and os.path.exists(cache_path):
            with open(cache_path, "r") as f:
                cached = json.load(f)
            return [SearchResult(**r) for r in cached]

        results = []
        try:
            # Try direct page lookup first
            page = self.wiki.page(query)
            if page.exists():
                passage = self._truncate_text(page.summary)
                results.append(SearchResult(
                    title=page.title,
                    passage=passage,
                    full_text=page.text[:2000],
                ))

            # Also search for related pages via links
            if page.exists() and len(results) < self.max_search_results:
                for link_title in list(page.links.keys())[:5]:
                    if len(results) >= self.max_search_results:
                        break
                    link_page = self.wiki.page(link_title)
                    if link_page.exists() and link_page.summary:
                        passage = self._truncate_text(link_page.summary)
                        results.append(SearchResult(
                            title=link_page.title,
                            passage=passage,
                            full_text=link_page.text[:2000],
                        ))

        except Exception as e:
            # Fallback: return empty results
            results.append(SearchResult(
                title="Search Error",
                passage=f"Could not search for '{query}': {str(e)}",
                full_text="",
            ))

        # Cache results
        if self.use_cache and results:
            with open(cache_path, "w") as f:
                json.dump([
                    {"title": r.title, "passage": r.passage, "full_text": r.full_text}
                    for r in results
                ], f)

        return results

    def _truncate_text(self, text: str) -> str:
        """Truncate text to max_passage_tokens words."""
        words = text.split()
        if len(words) > self.passage_max_tokens:
            return " ".join(words[:self.passage_max_tokens]) + "..."
        return text

    def _format_results(self, results: List[SearchResult]) -> str:
        """Format search results into a readable string."""
        if not results:
            return "No results found."

        parts = []
        for i, r in enumerate(results, 1):
            parts.append(f"[{i}] {r.title}: {r.passage}")
        return "\n".join(parts)

    def get_prompt(self) -> str:
        """
        Build the prompt to send to the agent model.
        Includes question, search history, and available actions.
        """
        if self.state is None:
            raise RuntimeError("Must call reset() first.")

        prompt_parts = [f"Question: {self.state.question}\n"]

        if self.state.search_history:
            prompt_parts.append("Search History:")
            for step in self.state.search_history:
                if step.action_type in ("search", "refine"):
                    prompt_parts.append(
                        f"Step {step.step_num}: {step.action_type}ed \"{step.action_content}\" → "
                        f"found: \"{step.result[:300]}...\""
                        if step.result and len(step.result) > 300
                        else f"Step {step.step_num}: {step.action_type}ed \"{step.action_content}\" → "
                             f"found: \"{step.result}\""
                    )
                elif step.action_type == "answer":
                    prompt_parts.append(
                        f"Step {step.step_num}: answered \"{step.action_content}\""
                    )
            prompt_parts.append("")

        prompt_parts.append(
            "Available actions: <search>query</search>, <refine>query</refine>, <answer>answer</answer>\n"
            "Think step by step, then output your next action."
        )

        return "\n".join(prompt_parts)

    def get_trajectory(self) -> List[Dict]:
        """Get the full trajectory as a list of dicts (for training data)."""
        if self.state is None:
            return []

        trajectory = []
        for step in self.state.search_history:
            trajectory.append({
                "step_num": step.step_num,
                "action_type": step.action_type,
                "action_content": step.action_content,
                "result": step.result,
                "reward": step.reward,
            })
        return trajectory


def run_test():
    """Test the search environment with a mock example."""
    from env.dataset import HotpotQAExample

    # Create a mock example
    example = HotpotQAExample(
        question="Were Scott Derrickson and Ed Wood of the same nationality?",
        answer="Yes",
        supporting_facts=[("Scott Derrickson", 0), ("Ed Wood", 0)],
        context={
            "Scott Derrickson": ["Scott Derrickson is an American director."],
            "Ed Wood": ["Edward Davis Wood Jr. was an American filmmaker."],
        },
        question_type="comparison",
        level="medium",
        id="test_001",
    )

    env = WikiSearchEnv(max_steps=5, use_cache=True)
    state = env.reset(example)

    print(f"Question: {state.question}")
    print(f"\n--- Agent Prompt ---")
    print(env.get_prompt())

    # Simulate agent actions
    actions = [
        "Let me search for information about Scott Derrickson. <search>Scott Derrickson nationality</search>",
        "Now let me find out about Ed Wood. <search>Ed Wood filmmaker nationality</search>",
        "Both are American, so the answer is yes. <answer>Yes</answer>",
    ]

    for i, action in enumerate(actions):
        print(f"\n--- Step {i+1} ---")
        print(f"Agent output: {action}")
        state, reward, done, info = env.step(action)
        print(f"Action: {info['action_type']} -> {info['action_content']}")
        if "search_results" in info:
            print(f"Results: {info['search_results'][:200]}...")
        if "f1" in info:
            print(f"F1: {info['f1']:.3f}, EM: {info['em']:.1f}")
        print(f"Done: {done}, Reward: {reward:.3f}")

        if done:
            break

        print(f"\n--- Updated Prompt ---")
        print(env.get_prompt()[:500])

    print(f"\n--- Trajectory ---")
    for step in env.get_trajectory():
        print(f"  Step {step['step_num']}: {step['action_type']}({step['action_content'][:50]})")


if __name__ == "__main__":
    run_test()

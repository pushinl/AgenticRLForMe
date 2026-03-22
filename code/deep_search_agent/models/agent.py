"""
DeepSearch Agent based on Qwen2.5-3B-Instruct.

Handles:
- Loading the model with optional LoRA adapters
- Generating structured actions (<search>, <refine>, <answer>)
- Building prompts from environment state
"""

import re
from typing import Optional, Dict, List, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from peft import PeftModel, LoraConfig, get_peft_model, TaskType


SYSTEM_PROMPT = """You are a research agent. Given a question, search Wikipedia step by step to find the answer.

Rules:
1. Use <search>query</search> to search Wikipedia for information
2. Use <refine>new_query</refine> to search again with a better query if results are insufficient
3. Use <answer>your_answer</answer> to give your final answer when you have enough information
4. Keep your answers concise and factual
5. Try to answer within 5 search steps

Think step by step before each action."""


class SearchAgent:
    """
    Search Agent based on Qwen2.5-3B-Instruct.
    Generates structured search actions given a question and search history.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-3B-Instruct",
        lora_path: Optional[str] = None,
        device: str = "auto",
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        do_sample: bool = True,
    ):
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.do_sample = do_sample

        print(f"Loading agent model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            padding_side="left",
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map=device,
            trust_remote_code=True,
        )

        # Load LoRA adapters if provided
        if lora_path:
            print(f"Loading LoRA adapters from: {lora_path}")
            self.model = PeftModel.from_pretrained(self.model, lora_path)

        self.model.eval()
        print("Agent model loaded successfully.")

    def apply_lora(self, lora_config: Optional[Dict] = None) -> None:
        """Apply LoRA adapters for training."""
        if lora_config is None:
            lora_config = {
                "r": 64,
                "lora_alpha": 128,
                "lora_dropout": 0.05,
                "target_modules": [
                    "q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj",
                ],
            }

        config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_config["r"],
            lora_alpha=lora_config["lora_alpha"],
            lora_dropout=lora_config["lora_dropout"],
            target_modules=lora_config["target_modules"],
        )
        self.model = get_peft_model(self.model, config)
        self.model.print_trainable_parameters()

    def build_messages(
        self,
        question: str,
        search_history: Optional[List[Dict]] = None,
    ) -> List[Dict[str, str]]:
        """Build chat messages for the model."""
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]

        user_content = f"Question: {question}\n"

        if search_history:
            user_content += "\nSearch History:\n"
            for step in search_history:
                step_num = step["step_num"]
                action_type = step["action_type"]
                action_content = step["action_content"]
                result = step.get("result", "")

                if action_type in ("search", "refine"):
                    result_truncated = result[:300] + "..." if len(result) > 300 else result
                    user_content += (
                        f'Step {step_num}: {action_type}ed "{action_content}" → '
                        f'found: "{result_truncated}"\n'
                    )
                elif action_type == "answer":
                    user_content += f'Step {step_num}: answered "{action_content}"\n'

        user_content += (
            "\nAvailable actions: <search>query</search>, "
            "<refine>query</refine>, <answer>answer</answer>\n"
            "Think step by step, then output your next action."
        )

        messages.append({"role": "user", "content": user_content})
        return messages

    @torch.no_grad()
    def generate_action(
        self,
        question: str,
        search_history: Optional[List[Dict]] = None,
        temperature: Optional[float] = None,
        do_sample: Optional[bool] = None,
    ) -> Tuple[str, str, str]:
        """
        Generate the next action given question and search history.

        Returns:
            (full_response, action_type, action_content)
        """
        messages = self.build_messages(question, search_history)

        # Apply chat template
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            temperature=temperature or self.temperature,
            do_sample=do_sample if do_sample is not None else self.do_sample,
            top_p=0.9,
            pad_token_id=self.tokenizer.pad_token_id,
        )

        # Decode only the generated part
        generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
        response = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        # Parse action
        action_type, action_content = self._parse_action(response)

        return response, action_type, action_content

    @torch.no_grad()
    def generate_batch(
        self,
        questions: List[str],
        search_histories: Optional[List[List[Dict]]] = None,
        num_generations: int = 1,
        temperature: Optional[float] = None,
    ) -> List[List[Tuple[str, str, str]]]:
        """
        Generate actions for a batch of questions, possibly multiple per question.

        Returns:
            List of lists of (full_response, action_type, action_content)
        """
        if search_histories is None:
            search_histories = [None] * len(questions)

        all_results = []
        for q, hist in zip(questions, search_histories):
            generations = []
            for _ in range(num_generations):
                result = self.generate_action(
                    q, hist,
                    temperature=temperature or self.temperature,
                )
                generations.append(result)
            all_results.append(generations)

        return all_results

    def _parse_action(self, text: str) -> Tuple[str, str]:
        """Parse action type and content from model output."""
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

    def get_tokenizer(self):
        return self.tokenizer

    def get_model(self):
        return self.model

    def format_trajectory_for_sft(
        self,
        question: str,
        trajectory: List[Dict],
    ) -> List[Dict[str, str]]:
        """
        Format a full trajectory into chat messages for SFT training.

        Each (prompt, response) pair in the trajectory becomes a turn.
        """
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]

        # Build the conversation turn by turn
        history_so_far = []
        for step in trajectory:
            # User message: current state
            user_msg = self.build_messages(question, history_so_far if history_so_far else None)
            messages.append(user_msg[-1])  # take only the user message

            # Assistant response
            action_type = step["action_type"]
            action_content = step["action_content"]

            if action_type == "search":
                assistant_msg = f"I need to search for more information.\n\n<search>{action_content}</search>"
            elif action_type == "refine":
                assistant_msg = f"Let me refine my search query.\n\n<refine>{action_content}</refine>"
            elif action_type == "answer":
                assistant_msg = f"Based on the information gathered, I can now answer.\n\n<answer>{action_content}</answer>"
            else:
                continue

            messages.append({"role": "assistant", "content": assistant_msg})
            history_so_far.append(step)

        return messages


def run_test():
    """Test the agent model with a sample question."""
    print("=" * 60)
    print("Testing SearchAgent")
    print("=" * 60)

    # Test with a smaller model or just test the prompt building
    agent = SearchAgent(
        model_name="Qwen/Qwen2.5-3B-Instruct",
        device="auto",
        temperature=0.7,
    )

    question = "Were Scott Derrickson and Ed Wood of the same nationality?"

    # Test prompt building
    messages = agent.build_messages(question)
    print("\n--- Messages (no history) ---")
    for m in messages:
        print(f"[{m['role']}]: {m['content'][:200]}...")

    # Test with history
    history = [
        {
            "step_num": 1,
            "action_type": "search",
            "action_content": "Scott Derrickson nationality",
            "result": "[1] Scott Derrickson: Scott Derrickson is an American director...",
        }
    ]
    messages = agent.build_messages(question, history)
    print("\n--- Messages (with history) ---")
    for m in messages:
        print(f"[{m['role']}]: {m['content'][:300]}...")

    # Test generation
    print("\n--- Generating Action ---")
    response, action_type, action_content = agent.generate_action(question)
    print(f"Response: {response[:300]}")
    print(f"Action: {action_type} -> {action_content}")

    print("\n--- Generating with History ---")
    response, action_type, action_content = agent.generate_action(question, history)
    print(f"Response: {response[:300]}")
    print(f"Action: {action_type} -> {action_content}")


if __name__ == "__main__":
    run_test()

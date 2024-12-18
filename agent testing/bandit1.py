"""
simple_prompt_bandit_openai.py

Minimal epsilon-greedy bandit that:
- Loads prompt variants from prompt_variants.json
- Connects to OpenAI using client = OpenAI(api_key=api_key)
- Tests variants by calling the model
- Uses your reward function
"""

import os
import json
import random
from dataclasses import dataclass
from typing import List, Dict, Any
from openai import OpenAI


# ---------------------------------------------------------
# OpenAI connection
# ---------------------------------------------------------

api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    raise EnvironmentError("OPENAI_API_KEY is not set")

client = OpenAI(api_key=api_key)


# ---------------------------------------------------------
# Prompt Variant Structure
# ---------------------------------------------------------

@dataclass
class PromptVariant:
    name: str
    schema_name: str
    n_shots: int
    temperature: float = 0.0
    max_tokens: int = 1500


def load_prompt_variants(path: str) -> List[PromptVariant]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return [
        PromptVariant(
            name=v["name"],
            schema_name=v["schema_name"],
            n_shots=v["n_shots"],
            temperature=v.get("temperature", 0.0),
            max_tokens=v.get("max_tokens", 1500),
        )
        for v in data
    ]


# ---------------------------------------------------------
# Simple epsilon-greedy bandit
# ---------------------------------------------------------

class EpsilonGreedyBandit:
    def __init__(self, n_arms: int, epsilon: float = 0.25):
        self.n_arms = n_arms
        self.epsilon = epsilon

        self.q_values = [0.0] * n_arms
        self.counts = [0] * n_arms

    def select_arm(self) -> int:
        if random.random() < self.epsilon:
            return random.randrange(self.n_arms)  # explore
        best_value = max(self.q_values)
        candidates = [i for i, v in enumerate(self.q_values) if v == best_value]
        return random.choice(candidates)

    def update(self, arm: int, reward: float):
        self.counts[arm] += 1
        n = self.counts[arm]

        old = self.q_values[arm]
        new = old + (reward - old) / n
        self.q_values[arm] = new


# ---------------------------------------------------------
# Build prompt (super simple placeholder)
# ---------------------------------------------------------

def build_prompt(example_text: str, variant: PromptVariant) -> str:
    """
    Replace with your real schema logic (plan vs partial, shots, etc.)
    """
    header = f"You are using schema '{variant.schema_name}' with {variant.n_shots} shots.\n"
    return header + "\nTask:\n" + example_text


# ---------------------------------------------------------
# Run a single trial for an arm (variant)
# ---------------------------------------------------------

def run_trial_with_variant(example_text: str, variant: PromptVariant) -> float:
    prompt = build_prompt(example_text, variant)

    try:
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=variant.temperature,
            max_tokens=variant.max_tokens,
        )

        output = response.choices[0].message.content

        # TODO: plug in your real SRED reward
        reward = simple_reward(output)

        return reward

    except Exception as e:
        print("API error:", e)
        return 0.0  # punish failures


# ---------------------------------------------------------
# Placeholder reward function
# ---------------------------------------------------------

def simple_reward(output_text: str) -> float:
    """
    Replace with your real SRED evaluation.
    This is only to keep the file runnable.
    """
    if "step" in output_text.lower():
        return 1.0
    return 0.3


# ---------------------------------------------------------
# Main bandit loop
# ---------------------------------------------------------

def main():
    variants = load_prompt_variants("prompt_variants.json")

    bandit = EpsilonGreedyBandit(len(variants), epsilon=0.25)

    example_text = "Organize my tasks into steps with constraints."

    N_TRIALS = 40

    for t in range(N_TRIALS):
        arm = bandit.select_arm()
        variant = variants[arm]

        reward = run_trial_with_variant(example_text, variant)
        bandit.update(arm, reward)

        print(
            f"[{t:03d}] Arm={arm} ({variant.name}) "
            f"Reward={reward:.3f} Q={bandit.q_values[arm]:.3f}"
        )

    print("\nFinal Q-values:")
    for i, v in enumerate(variants):
        print(f"  Arm {i} ({v.name}): Q={bandit.q_values[i]:.3f}, N={bandit.counts[i]}")


if __name__ == "__main__":
    main()

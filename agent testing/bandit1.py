"""
simple_prompt_bandit_templates.py

Minimal epsilon-greedy bandit over prompt templates.
"""

import os
import json
import random
from dataclasses import dataclass
from typing import List
from openai import OpenAI


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

DATA_PATH    = "agent_dataset.json"
PROMPTS_PATH = "prompt_variants.json"
CSV_LOG      = "bandit_fewshot_agent_log_router_hybrid_steps_patch_semantic.csv"


# ---------------------------------------------------------
# OpenAI client
# ---------------------------------------------------------

api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    raise EnvironmentError("OPENAI_API_KEY is not set")

client = OpenAI(api_key=api_key)


# ---------------------------------------------------------
# Prompt variant structure
# ---------------------------------------------------------

@dataclass
class PromptVariant:
    id: str
    template: str
    category: str
    intent: str
    temperature: float = 0.0
    max_tokens: int = 1500


def load_prompt_variants() -> List[PromptVariant]:
    """
    Loads prompt templates from PROMPTS_PATH.
    """
    with open(PROMPTS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    variants: List[PromptVariant] = []
    for v in data:
        variants.append(
            PromptVariant(
                id=v["id"],
                template=v["template"],
                category=v.get("category", ""),
                intent=v.get("intent", ""),
                temperature=v.get("temperature", 0.0),
                max_tokens=v.get("max_tokens", 1500),
            )
        )
    return variants


# ---------------------------------------------------------
# Epsilon-greedy bandit
# ---------------------------------------------------------

class EpsilonGreedyBandit:
    def __init__(self, n_arms: int, epsilon: float = 0.25):
        self.n_arms = n_arms
        self.epsilon = epsilon
        self.q_values = [0.0] * n_arms
        self.counts = [0] * n_arms

    def select_arm(self) -> int:
        if random.random() < self.epsilon:
            return random.randrange(self.n_arms)
        best = max(self.q_values)
        idx = [i for i, v in enumerate(self.q_values) if v == best]
        return random.choice(idx)

    def update(self, arm: int, reward: float):
        self.counts[arm] += 1
        n = self.counts[arm]
        old = self.q_values[arm]
        new = old + (reward - old) / n
        self.q_values[arm] = new


# ---------------------------------------------------------
# Prompt assembly
# ---------------------------------------------------------

def build_prompt(example_text: str, variant: PromptVariant) -> str:
    """
    Combines template text with the task input.
    """
    return f"{variant.template}\n\nTask:\n{example_text}\nReturn JSON only."


# ---------------------------------------------------------
# Trial execution
# ---------------------------------------------------------

def run_trial_with_variant(example_text: str, variant: PromptVariant) -> float:
    """
    Calls the model and returns a reward signal.
    """
    prompt = build_prompt(example_text, variant)

    try:
        resp = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=variant.temperature,
            max_tokens=variant.max_tokens,
        )
        output = resp.choices[0].message.content or ""
        reward = simple_reward(output)
        return reward

    except Exception as e:
        print(f"API error for {variant.id}: {e}")
        return 0.0


# ---------------------------------------------------------
# Reward function
# ---------------------------------------------------------

def simple_reward(text: str) -> float:
    """
    Produces a scalar reward based on JSON-like structure detection.
    """
    t = text.lower()
    has_braces = "{" in t and "}" in t
    has_steps = "step" in t or "steps" in t

    if has_braces and has_steps:
        return 1.0
    if has_braces:
        return 0.7
    if has_steps:
        return 0.5
    return 0.2


# ---------------------------------------------------------
# Main loop
# ---------------------------------------------------------

def main():
    variants = load_prompt_variants()
    bandit = EpsilonGreedyBandit(len(variants), epsilon=0.25)

    example_text = "Plan activities for the day including study, exercise, and cooking."

    N = 30

    for t in range(N):
        arm = bandit.select_arm()
        variant = variants[arm]

        reward = run_trial_with_variant(example_text, variant)
        bandit.update(arm, reward)

        print(
            f"[{t:03d}] Arm={arm} ({variant.id}) "
            f"Reward={reward:.3f}, Q={bandit.q_values[arm]:.3f}, n={bandit.counts[arm]}"
        )

    print("\nFinal Q-values:")
    for i, v in enumerate(variants):
        print(f"  Arm {i} ({v.id}): Q={bandit.q_values[i]:.3f}, n={bandit.counts[i]}")


if __name__ == "__main__":
    main()

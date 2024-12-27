"""
functional_prompt_bandit_ucb1.py

Minimal UCB1 bandit over prompt templates (functional style).
"""

import os
import json
import math
import random
from typing import List, Dict, Any
from openai import OpenAI


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

DATA_PATH    = "agent_dataset.json"
PROMPTS_PATH = "prompt_variants.json"
CSV_LOG      = "bandit2.csv"


# ---------------------------------------------------------
# OpenAI client
# ---------------------------------------------------------

api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    raise EnvironmentError("OPENAI_API_KEY is not set")

client = OpenAI(api_key=api_key)


# ---------------------------------------------------------
# Prompt variants
# ---------------------------------------------------------

def load_prompt_variants() -> List[Dict[str, Any]]:
    """
    Loads prompt templates from PROMPTS_PATH.
    Each entry should contain: id, template, category, intent.
    """
    with open(PROMPTS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    variants: List[Dict[str, Any]] = []
    for v in data:
        variants.append(
            {
                "id": v["id"],
                "template": v["template"],
                "category": v.get("category", ""),
                "intent": v.get("intent", ""),
                "temperature": v.get("temperature", 0.0),
                "max_tokens": v.get("max_tokens", 1500),
            }
        )
    return variants


# ---------------------------------------------------------
# UCB1 bandit state and operations (functional)
# ---------------------------------------------------------

def init_bandit_state(n_arms: int) -> Dict[str, Any]:
    """
    Initializes bandit state for UCB1.
    """
    return {
        "q_values": [0.0] * n_arms,
        "counts": [0] * n_arms,
        "total_pulls": 0,
    }


def select_arm_ucb1(state: Dict[str, Any], c: float = 1.0) -> int:
    """
    Selects an arm using UCB1.
    Arms with zero pulls are selected first.
    """
    q_values = state["q_values"]
    counts = state["counts"]
    total = state["total_pulls"]

    # Untried arms
    untried = [i for i, n in enumerate(counts) if n == 0]
    if untried:
        return random.choice(untried)

    # UCB1 score for each arm
    scores = []
    for i, (q, n) in enumerate(zip(q_values, counts)):
        bonus = c * math.sqrt(2.0 * math.log(total) / n)
        scores.append(q + bonus)

    max_score = max(scores)
    best_indices = [i for i, s in enumerate(scores) if s == max_score]
    return random.choice(best_indices)


def update_bandit_state(
    state: Dict[str, Any],
    arm: int,
    reward: float,
) -> Dict[str, Any]:
    """
    Returns updated bandit state after observing reward for selected arm.
    """
    q_values = state["q_values"]
    counts = state["counts"]
    total = state["total_pulls"] + 1

    counts[arm] += 1
    n = counts[arm]
    old_q = q_values[arm]
    new_q = old_q + (reward - old_q) / n
    q_values[arm] = new_q

    return {
        "q_values": q_values,
        "counts": counts,
        "total_pulls": total,
    }


# ---------------------------------------------------------
# Prompt assembly
# ---------------------------------------------------------

def build_prompt(example_text: str, variant: Dict[str, Any]) -> str:
    """
    Combines template text with the task input.
    """
    template = variant["template"]
    return f"{template}\n\nTask:\n{example_text}\nReturn JSON only."


# ---------------------------------------------------------
# Trial execution
# ---------------------------------------------------------

def run_trial_with_variant(example_text: str, variant: Dict[str, Any]) -> float:
    """
    Calls the model and returns a reward signal.
    """
    prompt = build_prompt(example_text, variant)

    try:
        resp = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=variant["temperature"],
            max_tokens=variant["max_tokens"],
        )
        output = resp.choices[0].message.content or ""
        reward = simple_reward(output)
        return reward

    except Exception as e:
        print(f"API error for {variant['id']}: {e}")
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
    n_arms = len(variants)
    state = init_bandit_state(n_arms)

    example_text = "Plan activities for the day including study, exercise, and cooking."
    n_trials = 30

    for t in range(n_trials):
        arm = select_arm_ucb1(state, c=1.0)
        variant = variants[arm]

        reward = run_trial_with_variant(example_text, variant)
        state = update_bandit_state(state, arm, reward)

        print(
            f"[{t:03d}] Arm={arm} ({variant['id']}) "
            f"Reward={reward:.3f}, Q={state['q_values'][arm]:.3f}, n={state['counts'][arm]}"
        )

    print("\nFinal Q-values:")
    for i, v in enumerate(variants):
        print(
            f"  Arm {i} ({v['id']} | {v['category']}): "
            f"Q={state['q_values'][i]:.3f}, n={state['counts'][i]}"
        )


if __name__ == "__main__":
    main()

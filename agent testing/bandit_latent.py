"""
functional_prompt_bandit_ucb1_csv_timestamps.py

Pure functional UCB1 bandit over prompt templates with CSV logging,
timestamps, and latency measurement.
"""

import os
import json
import math
import random
import csv
import time
from datetime import datetime
from typing import List, Dict, Any
from openai import OpenAI


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

DATA_PATH    = "agent_dataset.json"
PROMPTS_PATH = "prompt_variants.json"
CSV_LOG      = "bandit_log_ucb1_timestamps.csv"


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
    with open(PROMPTS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    variants = []
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
# Bandit state
# ---------------------------------------------------------

def init_bandit_state(n_arms: int) -> Dict[str, Any]:
    return {
        "q_values": [0.0] * n_arms,
        "counts": [0] * n_arms,
        "total_pulls": 0,
    }


def select_arm_ucb1(state: Dict[str, Any], c: float = 1.0) -> int:
    q = state["q_values"]
    n = state["counts"]
    total = state["total_pulls"]

    untried = [i for i, ct in enumerate(n) if ct == 0]
    if untried:
        return random.choice(untried)

    scores = []
    for i in range(len(q)):
        bonus = c * math.sqrt(2.0 * math.log(total) / n[i])
        scores.append(q[i] + bonus)

    max_score = max(scores)
    best = [i for i, s in enumerate(scores) if s == max_score]
    return random.choice(best)


def update_bandit_state(state: Dict[str, Any], arm: int, reward: float) -> Dict[str, Any]:
    q = state["q_values"]
    n = state["counts"]

    state["total_pulls"] += 1
    n[arm] += 1

    k = n[arm]
    old = q[arm]
    new = old + (reward - old) / k
    q[arm] = new

    return state


# ---------------------------------------------------------
# Prompt assembly
# ---------------------------------------------------------

def build_prompt(example_text: str, variant: Dict[str, Any]) -> str:
    return f"{variant['template']}\n\nTask:\n{example_text}\nReturn JSON only."


# ---------------------------------------------------------
# Trial execution
# ---------------------------------------------------------

def run_trial_with_variant(example_text: str, variant: Dict[str, Any]) -> Dict[str, Any]:
    prompt = build_prompt(example_text, variant)
    start = time.time()

    try:
        resp = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=variant["temperature"],
            max_tokens=variant["max_tokens"],
        )
        latency = time.time() - start

        output = resp.choices[0].message.content or ""
        reward = simple_reward(output)

        return {
            "reward": reward,
            "output": output,
            "error": "",
            "latency_sec": latency,
        }

    except Exception as e:
        latency = time.time() - start
        return {
            "reward": 0.0,
            "output": "",
            "error": str(e),
            "latency_sec": latency,
        }


# ---------------------------------------------------------
# Reward function
# ---------------------------------------------------------

def simple_reward(text: str) -> float:
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
# CSV logging
# ---------------------------------------------------------

def init_csv(path: str):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "trial",
                "timestamp",
                "variant_id",
                "category",
                "reward",
                "q_value",
                "count",
                "latency_sec",
                "output",
                "error",
            ]
        )


def append_csv(path: str, row: List[Any]):
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(row)


# ---------------------------------------------------------
# Main loop
# ---------------------------------------------------------

def main():
    variants = load_prompt_variants()
    n_arms = len(variants)
    state = init_bandit_state(n_arms)

    example_text = "Plan activities for the day including study, exercise, and cooking."
    n_trials = 30

    init_csv(CSV_LOG)

    for t in range(n_trials):
        arm = select_arm_ucb1(state)
        variant = variants[arm]

        timestamp = datetime.utcnow().isoformat()
        result = run_trial_with_variant(example_text, variant)
        reward = result["reward"]

        state = update_bandit_state(state, arm, reward)

        q_val = state["q_values"][arm]
        count = state["counts"][arm]
        latency = result["latency_sec"]

        append_csv(
            CSV_LOG,
            [
                t,
                timestamp,
                variant["id"],
                variant["category"],
                reward,
                q_val,
                count,
                latency,
                result["output"],
                result["error"],
            ],
        )

        print(
            f"[{t:03d}] Arm={arm} ({variant['id']}) "
            f"Reward={reward:.3f}, Q={q_val:.3f}, n={count}, latency={latency:.3f}s"
        )

    print("\nFinal Q-values:")
    for i, v in enumerate(variants):
        print(
            f"  Arm {i} ({v['id']} | {v['category']}): "
            f"Q={state['q_values'][i]:.3f}, n={state['counts'][i]}"
        )


if __name__ == "__main__":
    main()
#
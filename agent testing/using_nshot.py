"""


UCB1 bandit over prompt templates with 0/1/2/3-shot variants,

"""

import os
import json
import math
import random
import csv
import time
from datetime import datetime
from typing import List, Dict, Any, Optional
from openai import OpenAI


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

DATA_PATH    = "agent_dataset.json"
PROMPTS_PATH = "prompt_variants.json"
CSV_LOG      = "bandit_log_ucb1_dataset_shots.csv"
##

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

def load_base_prompt_variants() -> List[Dict[str, Any]]:
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


def expand_variants_with_shots(
    base_variants: List[Dict[str, Any]],
    shot_counts: List[int] = None,
) -> List[Dict[str, Any]]:
    if shot_counts is None:
        shot_counts = [0, 1, 2, 3]

    expanded: List[Dict[str, Any]] = []
    for v in base_variants:
        for k in shot_counts:
            new_v = dict(v)
            new_v["shots"] = k
            new_v["id"] = f"{v['id']}_shot{k}"
            expanded.append(new_v)
    return expanded


# ---------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------

def load_agent_dataset(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("agent_dataset.json must be a JSON array")
    return data


def get_example_text(example: Dict[str, Any]) -> str:
    for key in ("input", "user_input", "prompt", "text"):
        if key in example and isinstance(example[key], str):
            return example[key]
    return json.dumps(example, ensure_ascii=False)


def get_example_id(example: Dict[str, Any], default_id: str) -> str:
    for key in ("id", "example_id", "uid"):
        if key in example:
            return str(example[key])
    return default_id


def get_example_output(example: Dict[str, Any]) -> Optional[str]:
    keys = [
        "output_json",
        "target_json",
        "label_json",
        "answer_json",
        "output",
        "target",
        "label",
        "answer",
    ]
    for key in keys:
        if key in example and isinstance(example[key], str):
            return example[key]
    return None


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
# Few-shot construction
# ---------------------------------------------------------

def build_few_shot_block(
    examples: List[Dict[str, Any]],
    current_index: int,
    k: int,
) -> str:
    if k <= 0:
        return ""

    indices = [i for i in range(len(examples)) if i != current_index]
    random.shuffle(indices)

    blocks = []
    for idx in indices:
        ex = examples[idx]
        inp = get_example_text(ex)
        out = get_example_output(ex)
        if out is None:
            continue

        block = f"Example:\nUser: {inp}\nIdeal JSON:\n{out}"
        blocks.append(block)

        if len(blocks) >= k:
            break

    if not blocks:
        return ""

    return "\n\n".join(blocks)


# ---------------------------------------------------------
# Prompt assembly
# ---------------------------------------------------------

def build_prompt(
    examples: List[Dict[str, Any]],
    example_index: int,
    variant: Dict[str, Any],
) -> str:
    current = examples[example_index]
    example_text = get_example_text(current)

    shots = variant.get("shots", 0)
    few_shot_block = build_few_shot_block(examples, example_index, shots)

    parts = [variant["template"]]
    if few_shot_block:
        parts.append(few_shot_block)
    parts.append(f"Task:\n{example_text}\nReturn JSON only.")

    return "\n\n".join(parts)


# ---------------------------------------------------------
# Trial execution
# ---------------------------------------------------------

def run_trial_with_variant(
    examples: List[Dict[str, Any]],
    example_index: int,
    variant: Dict[str, Any],
) -> Dict[str, Any]:
    prompt = build_prompt(examples, example_index, variant)
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
                "example_index",
                "example_id",
                "variant_id",
                "category",
                "shots",
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
    base_variants = load_base_prompt_variants()
    variants = expand_variants_with_shots(base_variants, shot_counts=[0, 1, 2, 3])
    examples = load_agent_dataset(DATA_PATH)

    n_arms = len(variants)
    n_examples = len(examples)
    state = init_bandit_state(n_arms)

    n_trials = 60  # adjust as needed

    init_csv(CSV_LOG)

    for t in range(n_trials):
        example_index = t % n_examples
        example = examples[example_index]
        example_id = get_example_id(example, default_id=str(example_index))

        arm = select_arm_ucb1(state)
        variant = variants[arm]

        timestamp = datetime.utcnow().isoformat()
        result = run_trial_with_variant(examples, example_index, variant)
        reward = result["reward"]

        state = update_bandit_state(state, arm, reward)

        q_val = state["q_values"][arm]
        count = state["counts"][arm]
        latency = result["latency_sec"]
        shots = variant.get("shots", 0)

        append_csv(
            CSV_LOG,
            [
                t,
                timestamp,
                example_index,
                example_id,
                variant["id"],
                variant["category"],
                shots,
                reward,
                q_val,
                count,
                latency,
                result["output"],
                result["error"],
            ],
        )

        print(
            f"[{t:03d}] ex={example_index} id={example_id} "
            f"arm={arm} ({variant['id']}, shots={shots}) "
            f"reward={reward:.3f}, Q={q_val:.3f}, n={count}, latency={latency:.3f}s"
        )

    print("\nFinal Q-values:")
    for i, v in enumerate(variants):
        print(
            f"  Arm {i} ({v['id']} | {v['category']} | shots={v.get('shots', 0)}): "
            f"Q={state['q_values'][i]:.3f}, n={state['counts'][i]}"
        )


if __name__ == "__main__":
    main()

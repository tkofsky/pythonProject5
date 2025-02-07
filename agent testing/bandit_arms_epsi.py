"""
simple_shot_bandit_f1_epsilon.py

Epsilon-greedy bandit over 0/1/2/3-shot prompting using a small FEW_SHOT set.
Computes F1 for intent, entities, constraints, urgency, and steps.
"""

import os
import json
import math
import random
import time
from typing import List, Dict, Any
from datetime import datetime
from openai import OpenAI


# ---------------------------------------------------------
# OpenAI client
# ---------------------------------------------------------

api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    raise EnvironmentError("OPENAI_API_KEY is not set")

client = OpenAI(api_key=api_key)


# ---------------------------------------------------------
# Config
# ---------------------------------------------------------

SHOT_VALUES = [0, 1, 2, 3]   # arms
N_TRIALS = 100               # number of bandit pulls
EPSILON = 0.25               # exploration rate


# ---------------------------------------------------------
# Base prompt and few-shot examples
# ---------------------------------------------------------

BASE_PROMPT = (
    "Extract a structured JSON plan from the user request.\n"
    "Return strict JSON only with keys: intent, entities, constraints, urgency, steps.\n"
    "No extra text, no commentary."
)

FEW_SHOT = [
    {
        "input": "Book a train from Boston to New York tomorrow morning under $120.",
        "output": {
            "intent": "book_train",
            "entities": {
                "from": "Boston",
                "to": "New York",
                "date": "tomorrow morning",
                "budget": "120"
            },
            "constraints": ["budget<=120"],
            "urgency": "normal",
            "steps": [
                "search_trains",
                "filter_by_price_and_time",
                "propose_top_options"
            ]
        },
    },
    {
        "input": "Schedule a 30 minute Zoom check-in with Maya next Tuesday after 3pm.",
        "output": {
            "intent": "schedule_meeting",
            "entities": {
                "participants": ["Maya"],
                "duration": "30 minutes",
                "time_window": "next Tuesday after 3pm",
                "location": "Zoom"
            },
            "constraints": ["include_zoom_link"],
            "urgency": "normal",
            "steps": [
                "find_common_slot",
                "create_zoom",
                "send_invites"
            ]
        },
    },
    {
        "input": "Order 4 vegan lunches for pickup at 1pm at 9 King St.",
        "output": {
            "intent": "order_food",
            "entities": {
                "headcount": 4,
                "diet": "vegan",
                "pickup_time": "1pm",
                "address": "9 King St"
            },
            "constraints": ["vegan_only"],
            "urgency": "time_sensitive",
            "steps": [
                "choose_restaurants",
                "filter_menu",
                "place_order"
            ]
        },
    },
    {
        "input": "Write a brief thank-you email to the interviewer and ask for feedback.",
        "output": {
            "intent": "draft_email",
            "entities": {
                "recipient": "interviewer",
                "topic": "thank_you",
                "extra_request": "feedback"
            },
            "constraints": ["polite_tone", "brief"],
            "urgency": "normal",
            "steps": [
                "draft_email",
                "review_tone",
                "send_or_copy"
            ]
        },
    },
]


# ---------------------------------------------------------
# Bandit state (epsilon-greedy)
# ---------------------------------------------------------

def init_bandit_state(n_arms: int) -> Dict[str, Any]:
    return {
        "q_values": [0.0] * n_arms,
        "counts": [0] * n_arms,
        "total_pulls": 0,
    }


def select_arm_epsilon(state: Dict[str, Any], epsilon: float) -> int:
    q = state["q_values"]
    n_arms = len(q)

    if random.random() < epsilon:
        return random.randrange(n_arms)

    best_val = max(q)
    best_indices = [i for i, v in enumerate(q) if v == best_val]
    return random.choice(best_indices)


def update_bandit_state(state: Dict[str, Any], arm: int, reward: float) -> Dict[str, Any]:
    q = state["q_values"]
    c = state["counts"]

    state["total_pulls"] += 1
    c[arm] += 1

    k = c[arm]
    old = q[arm]
    new = old + (reward - old) / k
    q[arm] = new

    return state


# ---------------------------------------------------------
# Few-shot construction
# ---------------------------------------------------------

def build_few_shot_block(task_index: int, k: int) -> str:
    if k <= 0:
        return ""

    indices = [i for i in range(len(FEW_SHOT)) if i != task_index]
    random.shuffle(indices)

    blocks = []
    for idx in indices[:k]:
        ex = FEW_SHOT[idx]
        block = (
            "Example:\n"
            f"User: {ex['input']}\n"
            f"Ideal JSON:\n{json.dumps(ex['output'], ensure_ascii=False)}"
        )
        blocks.append(block)

    return "\n\n".join(blocks)


# ---------------------------------------------------------
# F1 helpers
# ---------------------------------------------------------

def flatten_to_items(value: Any) -> List[str]:
    items: List[str] = []

    def _walk(v: Any):
        if v is None:
            return
        if isinstance(v, (str, int, float, bool)):
            items.append(str(v).strip().lower())
        elif isinstance(v, list):
            for e in v:
                _walk(e)
        elif isinstance(v, dict):
            for val in v.values():
                _walk(val)
        else:
            items.append(str(v).strip().lower())

    _walk(value)
    return [x for x in set(items) if x]


def f1_from_sets(pred_items: List[str], true_items: List[str]) -> float:
    pred_set = set(pred_items)
    true_set = set(true_items)

    if not true_set and not pred_set:
        return 1.0
    if not true_set or not pred_set:
        return 0.0

    inter = len(pred_set & true_set)
    if inter == 0:
        return 0.0

    precision = inter / len(pred_set)
    recall = inter / len(true_set)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


# ---------------------------------------------------------
# Scoring
# ---------------------------------------------------------

def score_output(true_json: Dict[str, Any], text: str) -> Dict[str, float]:
    try:
        pred_json = json.loads(text)
        if not isinstance(pred_json, dict):
            raise ValueError("Non-dict JSON")
    except Exception:
        base = simple_structure_score(text)
        return {
            "reward_overall": base,
            "F1_intent": 0.0,
            "F1_entities": 0.0,
            "F1_constraints": 0.0,
            "F1_urgency": 0.0,
            "F1_steps": 0.0,
        }

    fields = ["intent", "entities", "constraints", "urgency", "steps"]
    f1_scores: Dict[str, float] = {}
    f1_values: List[float] = []

    for field in fields:
        true_val = true_json.get(field, None)
        pred_val = pred_json.get(field, None)

        true_items = flatten_to_items(true_val)
        pred_items = flatten_to_items(pred_val)

        f1 = f1_from_sets(pred_items, true_items)
        f1_scores[field] = f1
        if true_items:
            f1_values.append(f1)

    if f1_values:
        reward_overall = sum(f1_values) / len(f1_values)
    else:
        reward_overall = simple_structure_score(text)

    return {
        "reward_overall": reward_overall,
        "F1_intent": f1_scores["intent"],
        "F1_entities": f1_scores["entities"],
        "F1_constraints": f1_scores["constraints"],
        "F1_urgency": f1_scores["urgency"],
        "F1_steps": f1_scores["steps"],
    }


def simple_structure_score(text: str) -> float:
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
# Prompt + trial
# ---------------------------------------------------------

def build_prompt(task_index: int, shots: int) -> str:
    task = FEW_SHOT[task_index]
    few_shot_block = build_few_shot_block(task_index, shots)

    parts = [BASE_PROMPT]
    if few_shot_block:
        parts.append(few_shot_block)
    parts.append(f"Task:\n{task['input']}\nReturn JSON only.")

    return "\n\n".join(parts)


def run_trial(shots: int, task_index: int) -> Dict[str, Any]:
    task = FEW_SHOT[task_index]
    true_json = task["output"]

    prompt = build_prompt(task_index, shots)
    start = time.time()

    try:
        resp = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=500,
        )
        latency = time.time() - start
        output = resp.choices[0].message.content or ""

        scores = score_output(true_json, output)
        return {
            "output": output,
            "latency_sec": latency,
            **scores,
            "error": "",
        }

    except Exception as e:
        latency = time.time() - start
        scores = {
            "reward_overall": 0.0,
            "F1_intent": 0.0,
            "F1_entities": 0.0,
            "F1_constraints": 0.0,
            "F1_urgency": 0.0,
            "F1_steps": 0.0,
        }
        return {
            "output": "",
            "latency_sec": latency,
            **scores,
            "error": str(e),
        }


# ---------------------------------------------------------
# Main loop
# ---------------------------------------------------------

def main():
    n_arms = len(SHOT_VALUES)
    state = init_bandit_state(n_arms)

    for t in range(N_TRIALS):
        task_index = t % len(FEW_SHOT)

        arm = select_arm_epsilon(state, EPSILON)
        shots = SHOT_VALUES[arm]

        ts = datetime.utcnow().isoformat()
        result = run_trial(shots, task_index)
        reward = result["reward_overall"]

        state = update_bandit_state(state, arm, reward)

        q_val = state["q_values"][arm]
        count = state["counts"][arm]

        print(
            f"[{t:03d}] ts={ts} task={task_index} shots={shots} "
            f"reward={reward:.3f} "
            f"F1_intent={result['F1_intent']:.3f} "
            f"F1_entities={result['F1_entities']:.3f} "
            f"F1_constraints={result['F1_constraints']:.3f} "
            f"F1_urgency={result['F1_urgency']:.3f} "
            f"F1_steps={result['F1_steps']:.3f} "
            f"Q={q_val:.3f} n={count} "
            f"latency={result['latency_sec']:.3f}s "
            f"error={result['error']}"
        )

    print("\nFinal Q-values by shots:")
    for idx, shots in enumerate(SHOT_VALUES):
        print(
            f"  shots={shots}: Q={state['q_values'][idx]:.3f}, "
            f"n={state['counts'][idx]}"
        )


if __name__ == "__main__":
    main()

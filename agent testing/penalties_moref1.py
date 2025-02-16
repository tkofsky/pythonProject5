"""

Improved F1_steps scoring
Penalties for missing top-level keys in the predicted JSON
"""

import os
import json
import random
import time
import csv
import uuid
import re
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
# CONFIG
# ---------------------------------------------------------

SHOT_VALUES = [0, 1, 2, 3]
N_TRIALS = 80
MODEL_NAME = "gpt-4.1-mini"
CSV_PATH = "shot_bandit_log_ucb.csv"

RANDOM_SEED = 123
RUN_ID = str(uuid.uuid4())
UCB_C = 1.0  # exploration strength for UCB1

random.seed(RANDOM_SEED)


# ---------------------------------------------------------
# BASE PROMPT + FEW_SHOT DATA
# ---------------------------------------------------------

BASE_PROMPT = (
    "Extract a structured JSON plan from the user request.\n"
    "Return strict JSON only with keys: intent, entities, constraints, urgency, steps.\n"
    "No extra text."
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
        "task_type": "travel",
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
            "steps": ["find_common_slot", "create_zoom", "send_invites"]
        },
        "task_type": "meeting",
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
            "steps": ["choose_restaurants", "filter_menu", "place_order"]
        },
        "task_type": "food",
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
            "steps": ["draft_email", "review_tone", "send_or_copy"]
        },
        "task_type": "email",
    },
]


# ---------------------------------------------------------
# CSV INITIALIZATION
# ---------------------------------------------------------

def init_csv(path: str):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "run_id",
            "trial",
            "timestamp",
            "model",
            "ucb_c",
            "random_seed",
            "task_index",
            "task_type",
            "shots",
            "prompt_length",
            "reward_overall",
            "F1_intent",
            "F1_entities",
            "F1_constraints",
            "F1_urgency",
            "F1_steps",
            "latency_sec",
            "error",
            "raw_output",
        ])


def append_csv(path: str, row: List[Any]):
    with open(path, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(row)


# ---------------------------------------------------------
# F1 SCORING HELPERS
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

    if not pred_set and not true_set:
        return 1.0
    if not pred_set or not true_set:
        return 0.0

    inter = len(pred_set & true_set)
    if inter == 0:
        return 0.0

    precision = inter / len(pred_set)
    recall = inter / len(true_set)
    if precision + recall == 0:
        return 0.0

    return 2 * precision * recall / (precision + recall)


# --- improved step scoring ---

STEP_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def normalize_step_label(step: str) -> List[str]:
    if not isinstance(step, str):
        step = str(step)
    step = step.lower()
    tokens = STEP_TOKEN_RE.findall(step)
    return tokens


def jaccard_similarity(a_tokens: List[str], b_tokens: List[str]) -> float:
    a_set = set(a_tokens)
    b_set = set(b_tokens)
    if not a_set and not b_set:
        return 1.0
    if not a_set or not b_set:
        return 0.0
    inter = len(a_set & b_set)
    union = len(a_set | b_set)
    return inter / union if union > 0 else 0.0


def f1_for_steps(pred_steps_val: Any, true_steps_val: Any, threshold: float = 0.6) -> float:
    if isinstance(true_steps_val, list):
        true_steps = [s for s in true_steps_val]
    else:
        true_steps = [true_steps_val] if true_steps_val is not None else []

    if isinstance(pred_steps_val, list):
        pred_steps = [s for s in pred_steps_val]
    else:
        pred_steps = [pred_steps_val] if pred_steps_val is not None else []

    if not true_steps and not pred_steps:
        return 1.0
    if not true_steps or not pred_steps:
        return 0.0

    true_tokens = [normalize_step_label(s) for s in true_steps]
    pred_tokens = [normalize_step_label(s) for s in pred_steps]

    matched_true = set()
    matches = 0

    for p_tok in pred_tokens:
        best_score = 0.0
        best_index = None
        for i, t_tok in enumerate(true_tokens):
            if i in matched_true:
                continue
            sim = jaccard_similarity(p_tok, t_tok)
            if sim > best_score:
                best_score = sim
                best_index = i
        if best_index is not None and best_score >= threshold:
            matches += 1
            matched_true.add(best_index)

    precision = matches / len(pred_tokens) if pred_tokens else 0.0
    recall = matches / len(true_tokens) if true_tokens else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


# ---------------------------------------------------------
# SCORING WITH MISSING-KEY PENALTIES
# ---------------------------------------------------------

MISSING_KEY_PENALTIES = {
    "intent": 0.20,
    "entities": 0.15,
    "constraints": 0.10,
    "urgency": 0.10,
    "steps": 0.20,
}


def score_json(pred: Dict[str, Any], true: Dict[str, Any]) -> Dict[str, float]:
    fields = ["intent", "entities", "constraints", "urgency"]
    f1s: Dict[str, float] = {}

    for f in fields:
        true_items = flatten_to_items(true.get(f))
        pred_items = flatten_to_items(pred.get(f))
        f1s[f] = f1_from_sets(pred_items, true_items)

    f1_steps = f1_for_steps(pred.get("steps"), true.get("steps"))

    non_empty = [f1s[f] for f in fields if flatten_to_items(true.get(f))]
    if true.get("steps") is not None:
        non_empty.append(f1_steps)

    base_reward = sum(non_empty) / len(non_empty) if non_empty else 0.0

    penalty = 0.0
    for key, w in MISSING_KEY_PENALTIES.items():
        if key not in pred:
            penalty += w

    reward_overall = max(0.0, base_reward - penalty)

    return {
        "reward_overall": reward_overall,
        "F1_intent": f1s["intent"],
        "F1_entities": f1s["entities"],
        "F1_constraints": f1s["constraints"],
        "F1_urgency": f1s["urgency"],
        "F1_steps": f1_steps,
    }


# ---------------------------------------------------------
# PROMPT + MODEL CALL
# ---------------------------------------------------------

def build_prompt(task_index: int, shots: int) -> str:
    task = FEW_SHOT[task_index]

    indices = [i for i in range(len(FEW_SHOT)) if i != task_index]
    random.shuffle(indices)
    demos = indices[:shots]

    blocks: List[str] = []
    for idx in demos:
        ex = FEW_SHOT[idx]
        blocks.append(
            "Example:\n"
            f"User: {ex['input']}\n"
            f"Ideal JSON:\n{json.dumps(ex['output'], ensure_ascii=False)}"
        )

    prompt = BASE_PROMPT
    if blocks:
        prompt += "\n\n" + "\n\n".join(blocks)

    prompt += f"\n\nTask:\n{task['input']}\nReturn JSON only."
    return prompt


def run_trial(shots: int, task_index: int) -> Dict[str, Any]:
    task = FEW_SHOT[task_index]
    true_json = task["output"]

    prompt = build_prompt(task_index, shots)
    start = time.time()

    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=500,
        )
        latency = time.time() - start
        output = resp.choices[0].message.content or ""

        try:
            pred_json = json.loads(output)
            if not isinstance(pred_json, dict):
                raise ValueError("Predicted JSON is not a dict")
            scores = score_json(pred_json, true_json)
        except Exception:
            scores = {
                "reward_overall": 0.0,
                "F1_intent": 0.0,
                "F1_entities": 0.0,
                "F1_constraints": 0.0,
                "F1_urgency": 0.0,
                "F1_steps": 0.0,
            }

        return {
            **scores,
            "latency_sec": latency,
            "error": "",
            "raw_output": output,
            "prompt_length": len(prompt),
        }

    except Exception as e:
        latency = time.time() - start
        return {
            "reward_overall": 0.0,
            "F1_intent": 0.0,
            "F1_entities": 0.0,
            "F1_constraints": 0.0,
            "F1_urgency": 0.0,
            "F1_steps": 0.0,
            "latency_sec": latency,
            "error": str(e),
            "raw_output": "",
            "prompt_length": len(prompt),
        }


# ---------------------------------------------------------
# UCB1 BANDIT
# ---------------------------------------------------------

def select_arm_ucb1(q_values: List[float], counts: List[int], total_pulls: int, c: float) -> int:
    n_arms = len(q_values)

    untried = [i for i in range(n_arms) if counts[i] == 0]
    if untried:
        return random.choice(untried)

    scores = []
    for i in range(n_arms):
        bonus = c * ( (2.0 * (math.log(total_pulls))) / counts[i] ) ** 0.5
        scores.append(q_values[i] + bonus)

    best_val = max(scores)
    best_indices = [i for i, v in enumerate(scores) if v == best_val]
    return random.choice(best_indices)


import math  # placed here to keep all imports at top logically grouped


def main():
    n_arms = len(SHOT_VALUES)
    q_values = [0.0] * n_arms
    counts = [0] * n_arms
    total_pulls = 0

    init_csv(CSV_PATH)

    for t in range(N_TRIALS):
        task_index = t % len(FEW_SHOT)

        total_pulls += 1
        arm = select_arm_ucb1(q_values, counts, total_pulls, UCB_C)
        shots = SHOT_VALUES[arm]

        timestamp = datetime.utcnow().isoformat()
        result = run_trial(shots, task_index)

        counts[arm] += 1
        lr = 1 / counts[arm]
        q_values[arm] = q_values[arm] + lr * (result["reward_overall"] - q_values[arm])

        append_csv(CSV_PATH, [
            RUN_ID,
            t,
            timestamp,
            MODEL_NAME,
            UCB_C,
            RANDOM_SEED,
            task_index,
            FEW_SHOT[task_index]["task_type"],
            shots,
            result["prompt_length"],
            result["reward_overall"],

        ])

        print(
            f"[{t:03d}] task={task_index} type={FEW_SHOT[task_index]['task_type']} "
            f"shots={shots} R={result['reward_overall']:.3f} "
            f"F1i={result['F1_intent']:.2f} F1e={result['F1_entities']:.2f} "
            f"F1c={result['F1_constraints']:.2f} F1u={result['F1_urgency']:.2f} "
            f"F1s={result['F1_steps']:.2f} "
            f"Q={q_values[arm]:.3f} n={counts[arm]} "
            f"lat={result['latency_sec']:.2f}s"
        )

    print("\nFinal Q-values:")
    for i, shots in enumerate(SHOT_VALUES):
        print(f"  shots={shots}: Q={q_values[i]:.3f}  n={counts[i]}")


if __name__ == "__main__":
    main()

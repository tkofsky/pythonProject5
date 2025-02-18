"""

Logs every trial to CSV with config + metrics.
"""

import os
import json
import random
import time
import csv
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

SHOT_VALUES = [0, 1, 2, 3]      # Arms
N_TRIALS = 80                  # Total number of trials
EPSILON = 0.25                 # Exploration probability
MODEL_NAME = "gpt-4.1-mini"
CSV_PATH = "shot_bandit_log.csv"

random.seed(123)


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
        "task_type": "travel"
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
        "task_type": "meeting"
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
        "task_type": "food"
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
        "task_type": "email"
    },
]


# ---------------------------------------------------------
# CSV INITIALIZATION
# ---------------------------------------------------------

def init_csv(path: str):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            # CONFIG
            "trial", "timestamp", "model", "epsilon",
            "task_index", "task_type", "shots",
            "prompt_length",

            # METRICS
            "reward_overall",
            "F1_intent", "F1_entities", "F1_constraints",
            "F1_urgency", "F1_steps",

            "latency_sec",
            "error",
            "raw_output"
        ])


def append_csv(path: str, row: List[Any]):
    with open(path, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(row)


# ---------------------------------------------------------
# F1 SCORING HELPERS
# ---------------------------------------------------------

def flatten_to_items(value: Any) -> List[str]:
    items = []

    def _walk(v):
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
    pred_set, true_set = set(pred_items), set(true_items)

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


def score_json(pred: Dict[str, Any], true: Dict[str, Any]) -> Dict[str, float]:
    fields = ["intent", "entities", "constraints", "urgency", "steps"]
    f1s = {}

    for f in fields:
        t_items = flatten_to_items(true.get(f))
        p_items = flatten_to_items(pred.get(f))
        f1s[f] = f1_from_sets(p_items, t_items)

    # Average across fields that have ground truth items
    non_empty = [f1s[f] for f in fields if flatten_to_items(true.get(f))]
    reward = sum(non_empty) / len(non_empty) if non_empty else 0.0

    return {"reward_overall": reward, **{f"F1_{f}": f1s[f] for f in fields}}


# ---------------------------------------------------------
# PROMPT + MODEL CALL
# ---------------------------------------------------------

def build_prompt(task_index: int, shots: int) -> str:
    task = FEW_SHOT[task_index]

    # few-shot demonstrations
    indices = [i for i in range(len(FEW_SHOT)) if i != task_index]
    random.shuffle(indices)
    demos = indices[:shots]

    blocks = []
    for idx in demos:
        ex = FEW_SHOT[idx]
        blocks.append(
            f"Example:\nUser: {ex['input']}\nIdeal JSON:\n{json.dumps(ex['output'], ensure_ascii=False)}"
        )

    body = "\n\n".join(blocks)

    prompt = BASE_PROMPT
    if body:
        prompt += "\n\n" + body

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
            scores = score_json(pred_json, true_json)
        except Exception:
            scores = {
                "reward_overall": 0.0,
                "F1_intent": 0.0, "F1_entities": 0.0, "F1_constraints": 0.0,
                "F1_urgency": 0.0, "F1_steps": 0.0,
            }

        return {
            **scores,
            "latency_sec": latency,
            "error": "",
            "raw_output": output,
            "prompt_length": len(prompt),
        }

    except Exception as e:
        return {
            "reward_overall": 0.0,
            "F1_intent": 0.0, "F1_entities": 0.0, "F1_constraints": 0.0,
            "F1_urgency": 0.0, "F1_steps": 0.0,
            "latency_sec": time.time() - start,
            "error": str(e),
            "raw_output": "",
            "prompt_length": len(prompt),
        }


# ---------------------------------------------------------
# EPSILON-GREEDY BANDIT
# ---------------------------------------------------------

def main():
    n_arms = len(SHOT_VALUES)
    q_values = [0.0] * n_arms
    counts = [0] * n_arms

    init_csv(CSV_PATH)

    for t in range(N_TRIALS):
        task_index = t % len(FEW_SHOT)

        # epsilon-greedy
        if random.random() < EPSILON:
            arm = random.randrange(n_arms)
        else:
            best = max(q_values)
            arm = random.choice([i for i, v in enumerate(q_values) if v == best])

        shots = SHOT_VALUES[arm]

        timestamp = datetime.utcnow().isoformat()
        result = run_trial(shots, task_index)

        # incremental Q update
        counts[arm] += 1
        lr = 1 / counts[arm]
        q_values[arm] = q_values[arm] + lr * (result["reward_overall"] - q_values[arm])

        # log row
        append_csv(CSV_PATH, [
            t,
            timestamp,
            MODEL_NAME,
            EPSILON,
            task_index,
            FEW_SHOT[task_index]["task_type"],
            shots,
            result["prompt_length"],

            result["reward_overall"],
            result["F1_intent"],
            result["F1_entities"],
            result["F1_constraints"],
            result["F1_urgency"],
            result["F1_steps"],

            result["latency_sec"],
            result["error"],
            result["raw_output"],
        ])

        print(
            f"[{t:03d}] task={task_index} type={FEW_SHOT[task_index]['task_type']} shots={shots} "
            f"R={result['reward_overall']:.3f} Q={q_values[arm]:.3f} "
            f"F1i={result['F1_intent']:.2f} F1e={result['F1_entities']:.2f} "
            f"lat={result['latency_sec']:.2f}s"
        )

    print("\nFinal Q-values:")
    for i, shots in enumerate(SHOT_VALUES):
        print(f"  shots={shots}: Q={q_values[i]:.3f}  n={counts[i]}")


if __name__ == "__main__":
    main()

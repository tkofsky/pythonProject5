"""
bandit_cost_aware_ucb_temp_multi_prompt.py

UCB1 bandit over (prompt_template x shots x passes x temperature) combinations.
Cost-aware: bandit optimizes adjusted_reward = reward_overall - COST_LAMBDA * latency_sec.
"""

import os, json, random, time, csv, uuid, re, math
from datetime import datetime
from typing import Any, Dict, List, Optional
from openai import OpenAI


# --------------------- OpenAI client ---------------------

api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    raise EnvironmentError("OPENAI_API_KEY is not set")
client = OpenAI(api_key=api_key)


# --------------------- Config ----------------------------

MODEL_NAME   = "gpt-4.1-mini"
RANDOM_SEED  = 123
RUN_ID       = str(uuid.uuid4())

UCB_C        = 1.0
N_TRIALS     = 200

# cost tradeoff per second of latency
COST_LAMBDA  = 0.3

random.seed(RANDOM_SEED)


# ------------------ Prompt templates ---------------------

PROMPT_TEMPLATES = [
    {
        "id": "schema_strict",
        "text": (
            "Extract a structured JSON plan from the request.\n"
            "Return strict JSON only with keys: intent, entities, constraints, urgency, steps.\n"
            "No prose, no explanations."
        ),
    },
    {
        "id": "schema_guided",
        "text": (
            "Plan the task as JSON.\n"
            "Keys: intent, entities, constraints, urgency, steps.\n"
            "Fill missing fields with null, keep JSON valid, no commentary."
        ),
    },
    {
        "id": "minimal",
        "text": (
            "Return only JSON with keys: intent, entities, constraints, urgency, steps.\n"
            "Be concise, avoid extra nesting."
        ),
    },
]


def get_template_text(template_id: str) -> str:
    for t in PROMPT_TEMPLATES:
        if t["id"] == template_id:
            return t["text"]
    raise ValueError(f"Unknown template_id: {template_id}")


# ------------------ Arms: template x shots x passes x temp -----------------

SHOT_VALUES        = [0, 1, 2, 3]
PASS_VALUES        = [1, 2]           # 1-pass or 2-pass
TEMPERATURE_VALUES = [0.0, 0.3, 0.7]

ARMS: List[Dict[str, Any]] = [
    {"template_id": tpl["id"], "shots": s, "passes": p, "temperature": t}
    for tpl in PROMPT_TEMPLATES
    for p in PASS_VALUES
    for s in SHOT_VALUES
    for t in TEMPERATURE_VALUES
]
N_ARMS = len(ARMS)


# ---------------- FEW-SHOT DATA --------------------------

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


# -------------------- CSV helpers ------------------------

CSV_PATH_TRIALS  = "bandit_multi_prompt_trials.csv"
CSV_PATH_SUMMARY = "bandit_multi_prompt_summary.csv"


def init_csvs():
    with open(CSV_PATH_TRIALS, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([
            "run_id", "trial", "timestamp",
            "model", "ucb_c", "cost_lambda", "seed",
            "task_index", "task_type",
            "arm",
            "template_id", "shots", "passes", "temperature",
            "prompt_length",
            "reward_overall",
            "adj_reward",
            "F1_intent", "F1_entities", "F1_constraints", "F1_urgency", "F1_steps",
            "latency_sec",
            "error",
            "raw_output",
        ])

    with open(CSV_PATH_SUMMARY, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([
            "run_id", "model", "ucb_c", "cost_lambda", "seed",
            "arm",
            "template_id", "shots", "passes", "temperature",
            "final_q_adj",      # mean adjusted reward
            "mean_raw_reward",  # mean raw reward_overall
            "mean_latency_sec",
            "pulls", "pull_freq",
            "lcb", "ucb", "bandwidth", "converged",
        ])


def append(path: str, row: list):
    with open(path, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(row)


# ------------------- F1 scoring --------------------------

STEP_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def flatten(v: Any) -> List[str]:
    out: List[str] = []

    def walk(x: Any):
        if x is None:
            return
        if isinstance(x, (str, int, float, bool)):
            out.append(str(x).lower().strip())
        elif isinstance(x, list):
            for e in x:
                walk(e)
        elif isinstance(x, dict):
            for e in x.values():
                walk(e)
        else:
            out.append(str(x).lower().strip())

    walk(v)
    return list(set(out))


def f1_items(pred: List[str], truth: List[str]) -> float:
    p, t = set(pred), set(truth)
    if not p and not t:
        return 1.0
    if not p or not t:
        return 0.0
    inter = len(p & t)
    if inter == 0:
        return 0.0
    prec = inter / len(p)
    rec = inter / len(t)
    return 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0


def jaccard(a: List[str], b: List[str]) -> float:
    a, b = set(a), set(b)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def norm_step(x: Any) -> List[str]:
    return STEP_TOKEN_RE.findall(str(x).lower())


def f1_steps(pred_steps_val: Any, true_steps_val: Any) -> float:
    if not isinstance(true_steps_val, list):
        true_steps = [true_steps_val] if true_steps_val else []
    else:
        true_steps = true_steps_val

    if not isinstance(pred_steps_val, list):
        pred_steps = [pred_steps_val] if pred_steps_val else []
    else:
        pred_steps = pred_steps_val

    if not true_steps and not pred_steps:
        return 1.0
    if not true_steps or not pred_steps:
        return 0.0

    true_tok = [norm_step(s) for s in true_steps]
    pred_tok = [norm_step(s) for s in pred_steps]

    matched = set()
    hits = 0
    for p in pred_tok:
        best = 0.0
        best_i: Optional[int] = None
        for i, t in enumerate(true_tok):
            if i in matched:
                continue
            sim = jaccard(p, t)
            if sim > best:
                best = sim
                best_i = i
        if best_i is not None and best >= 0.6:
            matched.add(best_i)
            hits += 1

    prec = hits / len(pred_tok) if pred_tok else 0.0
    rec = hits / len(true_tok) if true_tok else 0.0
    return 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0


MISSING_PEN = {
    "intent": 0.20,
    "entities": 0.15,
    "constraints": 0.10,
    "urgency": 0.10,
    "steps": 0.20,
}


def score(pred: Dict[str, Any], truth: Dict[str, Any]) -> Dict[str, float]:
    try:
        f_int = f1_items(flatten(pred.get("intent")),      flatten(truth.get("intent")))
        f_ent = f1_items(flatten(pred.get("entities")),    flatten(truth.get("entities")))
        f_con = f1_items(flatten(pred.get("constraints")), flatten(truth.get("constraints")))
        f_urg = f1_items(flatten(pred.get("urgency")),     flatten(truth.get("urgency")))
        f_stp = f1_steps(pred.get("steps"), truth.get("steps"))
        base = (f_int + f_ent + f_con + f_urg + f_stp) / 5
    except Exception:
        return dict(
            reward_overall=0.0,
            F1_intent=0.0, F1_entities=0.0,
            F1_constraints=0.0, F1_urgency=0.0, F1_steps=0.0,
        )

    penalty = sum(w for k, w in MISSING_PEN.items() if k not in pred)
    reward = max(0.0, base - penalty)

    return dict(
        reward_overall=reward,
        F1_intent=f_int, F1_entities=f_ent,
        F1_constraints=f_con, F1_urgency=f_urg, F1_steps=f_stp,
    )


# ------------------ Prompt building ----------------------

def make_prompt(template_id: str, task_i: int, shots: int) -> str:
    task = FEW_SHOT[task_i]
    base_text = get_template_text(template_id)

    others = [i for i in range(len(FEW_SHOT)) if i != task_i]
    random.shuffle(others)
    demos = others[:shots]

    parts = [base_text]
    for j in demos:
        ex = FEW_SHOT[j]
        parts.append(
            "Example:\n"
            f"User: {ex['input']}\n"
            f"Ideal JSON:\n{json.dumps(ex['output'], ensure_ascii=False)}"
        )

    parts.append(f"\nTask:\n{task['input']}\nReturn JSON only.")
    return "\n\n".join(parts)


def make_refine_prompt(template_id: str, task_i: int, draft_json: str) -> str:
    task = FEW_SHOT[task_i]
    base_text = get_template_text(template_id)
    return (
        base_text
        + "\nDraft JSON:\n"
        + draft_json
        + "\nClean up structure, fill missing keys if possible, return JSON only.\n\n"
        + "Original request:\n"
        + task["input"]
    )


# --------------------- Model calls -----------------------

def call_api(prompt: str, temperature: float) -> Dict[str, Any]:
    start = time.time()
    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=500,
        )
        latency = time.time() - start
        output = resp.choices[0].message.content or ""
        return dict(output=output, latency=latency, error="")
    except Exception as e:
        latency = time.time() - start
        return dict(output="", latency=latency, error=str(e))


def run_trial(cfg: Dict[str, Any], task_i: int) -> Dict[str, Any]:
    template_id = cfg["template_id"]
    passes      = cfg["passes"]
    shots       = cfg["shots"]
    temperature = cfg["temperature"]

    true_json = FEW_SHOT[task_i]["output"]

    p1 = make_prompt(template_id, task_i, shots)
    r1 = call_api(p1, temperature)

    if passes == 1 or r1["error"]:
        final_out = r1["output"]
        lat = r1["latency"]
        plen = len(p1)
        err = r1["error"]
    else:
        p2 = make_refine_prompt(template_id, task_i, r1["output"])
        r2 = call_api(p2, temperature)
        final_out = r2["output"]
        lat = r1["latency"] + r2["latency"]
        plen = len(p1) + len(p2)
        err = r1["error"] or r2["error"]

    if err:
        raw_reward = 0.0
        sc = dict(
            reward_overall=0.0,
            F1_intent=0.0, F1_entities=0.0,
            F1_constraints=0.0, F1_urgency=0.0, F1_steps=0.0,
        )
    else:
        try:
            js = json.loads(final_out)
            if not isinstance(js, dict):
                raise ValueError("not dict")
            sc = score(js, true_json)
            raw_reward = sc["reward_overall"]
        except Exception:
            sc = dict(
                reward_overall=0.0,
                F1_intent=0.0, F1_entities=0.0,
                F1_constraints=0.0, F1_urgency=0.0, F1_steps=0.0,
            )
            raw_reward = 0.0

    adj_reward = raw_reward - COST_LAMBDA * lat

    return dict(
        prompt_length=plen,
        latency_sec=lat,
        error=err,
        raw_output=final_out,
        reward_overall=raw_reward,
        adj_reward=adj_reward,
        **{k: sc[k] for k in ["F1_intent", "F1_entities", "F1_constraints", "F1_urgency", "F1_steps"]},
    )


# ----------------------- UCB1 ----------------------------

def pick_arm(q_adj: List[float], pulls: List[int], t: int) -> int:
    for i in range(N_ARMS):
        if pulls[i] == 0:
            return i


    return best_idx


# ------------------------ MAIN ---------------------------

def main():
    init_csvs()

    q_adj = [0.0] * N_ARMS
    pulls = [0] * N_ARMS

    sum_raw_reward = [0.0] * N_ARMS
    sum_latency    = [0.0] * N_ARMS

    for t in range(1, N_TRIALS + 1):
        task_i = (t - 1) % len(FEW_SHOT)

        arm = pick_arm(q_adj, pulls, t)
        cfg = ARMS[arm]

        ts = datetime.utcnow().isoformat()
        res = run_trial(cfg, task_i)

        pulls[arm] += 1
        lr = 1 / pulls[arm]
        q_adj[arm] += lr * (res["adj_reward"] - q_adj[arm])

        sum_raw_reward[arm] += res["reward_overall"]
        sum_latency[arm]    += res["latency_sec"]

        append(CSV_PATH_TRIALS, [
            RUN_ID, t, ts,
            MODEL_NAME, UCB_C, COST_LAMBDA, RANDOM_SEED,
            task_i, FEW_SHOT[task_i]["task_type"],
            arm,
            cfg["template_id"], cfg["shots"], cfg["passes"], cfg["temperature"],
            res["prompt_length"],
            res["reward_overall"],
            res["adj_reward"],
            res["F1_intent"], res["F1_entities"], res["F1_constraints"],
            res["F1_urgency"], res["F1_steps"],
            res["latency_sec"],
            res["error"],
            res["raw_output"],
        ])

        print(
            f"[{t:03d}] arm={arm} tpl={cfg['template_id']} "
            f"shots={cfg['shots']} passes={cfg['passes']} temp={cfg['temperature']} "
            f"rawR={res['reward_overall']:.3f} adjR={res['adj_reward']:.3f} "
            f"Qadj={q_adj[arm]:.3f} n={pulls[arm]} lat={res['latency_sec']:.2f}s"
        )

    for i, cfg in enumerate(ARMS):
        if pulls[i] > 0:
            bonus = UCB_C * math.sqrt(2.0 * math.log(N_TRIALS) / pulls[i])
            lcb = q_adj[i] - bonus
            ucb = q_adj[i] + bonus
            bw = ucb - lcb
            conv = bw < 0.20
            mean_raw = sum_raw_reward[i] / pulls[i]
            mean_lat = sum_latency[i] / pulls[i]
        else:
            lcb = ucb = bw = 0.0
            conv = False
            mean_raw = 0.0
            mean_lat = 0.0

        append(CSV_PATH_SUMMARY, [
            RUN_ID, MODEL_NAME, UCB_C, COST_LAMBDA, RANDOM_SEED,
            i,
            cfg["template_id"], cfg["shots"], cfg["passes"], cfg["temperature"],
            q_adj[i],
            mean_raw,
            mean_lat,
            pulls[i],
            pulls[i] / N_TRIALS,
            lcb,
            ucb,
            bw,
            conv,
        ])

    print("\nFinal cost-aware Q-values (adjusted):")
    for i, cfg in enumerate(ARMS):
        print(
            f" arm={i} tpl={cfg['template_id']} shots={cfg['shots']} "
            f"passes={cfg['passes']} temp={cfg['temperature']} "
            f"Qadj={q_adj[i]:.3f} pulls={pulls[i]}"
        )


if __name__ == "__main__":
    main()

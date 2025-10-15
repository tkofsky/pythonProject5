
# File: bandit_fewshot_agent_experiments.py
#
# Adds experiments:
# - JSON mode (response_format={"type":"json_object"}) switch
# - Two-pass prompting (2nd pass generates only "steps")
# - Intent-matched few-shot retrieval
# - Thompson Sampling bandit (success if reward >= threshold and JSON valid)
# - Weighted reward emphasizing constraints/steps + reward_per_1k logging
# - Extended CSV log columns

import os, csv, json, random, math
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional

from openai import OpenAI

# ========= CONFIG =========
DATA_PATH = "agent_dataset.json"           # eval dataset: [{"input":..., "reference": <json or str>}]
PROMPTS_PATH = "prompt_variants.json"      # prompt templates (id, template, category, intent)
CSV_LOG = "bandit_fewshot_agent_log_experiments.csv"
CSV_LOG ="bandit_fewshot_agent_log_router_hybrid_steps_weighted.csv"

ITERATIONS = 80

# Arm dimensions
FEW_SHOT_LEVELS = [0, 1, 3]
TEMPS = [0.2, 0.5]
JSON_MODE_OPTIONS = [False, True]
TWO_PASS_OPTIONS = [False, True]
FEWSHOT_STRATEGY = ["static", "intent_matched"]

# Thompson Sampling (Beta prior)
SUCCESS_THRESHOLD = 0.75    # reward >= this and JSON valid -> success
PRIOR_ALPHA = 1.0
PRIOR_BETA  = 1.0

# Reward shaping (weights sum to 1.0)
WEIGHTS = {"intent":0.2, "entities":0.2, "constraints":0.3, "urgency":0.1, "steps":0.2}
BONUS_ALL_KEYS = 0.05

GEN_MODEL = "gpt-4o-mini"
#adsds
# Few-shot demos (extendable)
FEW_SHOT = [
    {
      "input": "Book a train from Boston to New York tomorrow morning under $120.",
      "output": {
        "intent":"book_train",
        "entities":{"from":"Boston","to":"New York","date":"tomorrow morning","budget":"120"},
        "constraints":["budget<=120"],
        "urgency":"normal",
        "steps":["search_trains","filter_by_price_and_time","propose_top_options"]
      }
    },
    {
      "input": "Schedule a 30 minute Zoom check-in with Maya next Tuesday after 3pm.",
      "output": {
        "intent":"schedule_meeting",
        "entities":{"participants":["Maya"],"duration":"30 minutes","time_window":"next Tuesday after 3pm","location":"Zoom"},
        "constraints":["include_zoom_link"],
        "urgency":"normal",
        "steps":["find_common_slot","create_zoom","send_invites"]
      }
    },
    {
      "input": "Order 4 vegan lunches for pickup at 1pm at 9 King St.",
      "output": {
        "intent":"order_food",
        "entities":{"headcount":4,"diet":"vegan","pickup_time":"1pm","address":"9 King St"},
        "constraints":["vegan_only"],
        "urgency":"time_sensitive",
        "steps":["choose_restaurants","filter_menu","place_order"]
      }
    }
]

# ========= CLIENT =========
api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    raise EnvironmentError("OPENAI_API_KEY is not set in the environment")
client = OpenAI(api_key=api_key)

# ========= LOAD DATA =========
with open(DATA_PATH, "r", encoding="utf-8") as f:
    DATASET = json.load(f)

with open(PROMPTS_PATH, "r", encoding="utf-8") as f:
    PROMPTS_FILE = json.load(f)

# ========= TYPES =========
@dataclass
class Arm:
    prompt_id: str
    example_count: int
    temperature: float
    json_mode: bool
    two_pass: bool
    fewshot_strategy: str
    def key(self) -> Tuple:
        return (self.prompt_id, self.example_count, self.temperature,
                self.json_mode, self.two_pass, self.fewshot_strategy)

@dataclass
class PromptVariant:
    prompt_id: str
    template: str
    category: str
    intent: str
    parent_id: Optional[str] = None

# Build prompt variants
prompt_variants: Dict[str, PromptVariant] = {
    p["id"]: PromptVariant(
        prompt_id=p["id"],
        template=p["template"],
        category=p.get("category", "uncategorized"),
        intent=p.get("intent", "structured")
    )
    for p in PROMPTS_FILE
}

def enumerate_arms() -> List[Arm]:
    arms = []
    for pid in prompt_variants.keys():
        for k in FEW_SHOT_LEVELS:
            for t in TEMPS:
                for jm in JSON_MODE_OPTIONS:
                    for tp in TWO_PASS_OPTIONS:
                        for fs in FEWSHOT_STRATEGY:
                            arms.append(Arm(pid, k, t, jm, tp, fs))
    return arms

# ========= THOMPSON SAMPLING =========
class ThompsonStats:
    def __init__(self):
        self.alpha: Dict[Tuple, float] = {}
        self.beta: Dict[Tuple, float] = {}

    def sample(self, arm: Arm) -> float:
        # Draw from Beta(alpha, beta) using gamma sampling
        import random
        k = arm.key()
        A = self.alpha.get(k, PRIOR_ALPHA)
        B = self.beta.get(k, PRIOR_BETA)
        x = random.gammavariate(A, 1.0)
        y = random.gammavariate(B, 1.0)
        return x / (x + y) if (x + y) > 0 else 0.0

    def update(self, arm: Arm, success: bool):
        k = arm.key()
        A = self.alpha.get(k, PRIOR_ALPHA)
        B = self.beta.get(k, PRIOR_BETA)
        if success:
            A += 1.0
        else:
            B += 1.0
        self.alpha[k] = A
        self.beta[k]  = B

thompson = ThompsonStats()

# ========= HELPERS =========
FIELDS = ["intent","entities","constraints","urgency","steps"]

def fewshot_block(n: int, strategy: str, target_intent: Optional[str]) -> str:
    if n <= 0:
        return ""
    demos = FEW_SHOT[:]
    if strategy == "intent_matched" and target_intent:
        filtered = [ex for ex in FEW_SHOT if ex["output"].get("intent") == target_intent]
        if filtered:
            demos = filtered
    demos = demos[:min(n, len(demos))]
    lines = []
    for ex in demos:
        lines.append("USER: " + ex["input"])
        lines.append("AGENT_JSON: " + json.dumps(ex["output"], ensure_ascii=False))
    return "\n".join(lines) + "\n"

def build_messages(template: str, user_input: str, example_count: int,
                   strategy: str, target_intent: Optional[str]) -> List[dict]:
    return [
        {"role":"system","content": template},
        {"role":"user","content": fewshot_block(example_count, strategy, target_intent) + "USER: " + user_input + "\nAGENT_JSON:"}
    ]

def call_llm(messages: List[dict], temperature: float, json_mode: bool) -> Tuple[str, int]:
    kwargs = dict(
        model=GEN_MODEL,
        messages=messages,
        temperature=temperature,
        max_tokens=500
    )
    if json_mode:
        # Use structured JSON mode when supported by the model
        kwargs["response_format"] = {"type": "json_object"}
    resp = client.chat.completions.create(**kwargs)
    out = resp.choices[0].message.content.strip()
    toks = resp.usage.total_tokens if getattr(resp, "usage", None) else 0
    return out, toks

def safe_json(s: str):
    # robust-ish JSON parse
    try:
        return json.loads(s)
    except Exception:
        pass
    if isinstance(s, str) and "{" in s and "}" in s:
        chunk = s[s.find("{"):s.rfind("}")+1]
        try:
            return json.loads(chunk)
        except Exception:
            return None
    return None

def tokens(v) -> set:
    if v is None:
        return set()
    if isinstance(v, (list, tuple)):
        bag = []
        for x in v:
            bag += str(x).lower().split()
        return set(bag)
    if isinstance(v, dict):
        bag = []
        for k, val in v.items():
            bag += str(k).lower().split()
            bag += str(val).lower().split()
        return set(bag)
    return set(str(v).lower().split())

def f1(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    inter = len(a & b)
    p = inter / len(a) if len(a) else 0.0
    r = inter / len(b) if len(b) else 0.0
    return 0.0 if (p + r) == 0 else 2 * p * r / (p + r)

def score_json(pred: dict, ref: dict) -> Dict[str, float]:
    scores = {}
    for k in FIELDS:
        scores[k] = f1(tokens(pred.get(k)), tokens(ref.get(k)))
    return scores

def compute_reward(output_text: str, reference_json_str: str) -> Tuple[float, Dict[str,float], bool]:
    ref = json.loads(reference_json_str) if isinstance(reference_json_str, str) else reference_json_str
    pred = safe_json(output_text)
    if pred is None:
        return 0.0, {k:0.0 for k in FIELDS}, False
    field_scores = score_json(pred, ref)
    weighted = sum(WEIGHTS[k] * field_scores[k] for k in FIELDS)
    has_all = all(k in pred for k in FIELDS)
    reward = min(weighted + (BONUS_ALL_KEYS if has_all else 0.0), 1.0)
    return reward, field_scores, True

def ensure_csv_header():
    if not os.path.exists(CSV_LOG):
        with open(CSV_LOG, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([
                "iteration","phase",
                "prompt_id","category","intent",
                "example_count","temperature","json_mode","two_pass","fewshot_strategy",
                "input","reference","output","tokens",
                "reward","reward_per_1k","success",
                "f1_intent","f1_entities","f1_constraints","f1_urgency","f1_steps",
                "prompt_template"
            ])

def log_row(**kw):
    with open(CSV_LOG, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([
            kw.get("iteration"), kw.get("phase"),
            kw.get("prompt_id"), kw.get("category"), kw.get("intent"),
            kw.get("example_count"), kw.get("temperature"), int(kw.get("json_mode",0)), int(kw.get("two_pass",0)), kw.get("fewshot_strategy"),
            kw.get("input"), kw.get("reference"), kw.get("output"), kw.get("tokens"),
            kw.get("reward"), kw.get("reward_per_1k"), int(kw.get("success",0)),
            kw.get("f1_intent"), kw.get("f1_entities"), kw.get("f1_constraints"), kw.get("f1_urgency"), kw.get("f1_steps"),
            kw.get("prompt_template")
        ])

# ========= TWO-PASS (second pass generates only steps) =========
def run_second_pass_steps(core_fields: dict, temperature: float, json_mode: bool) -> Tuple[Optional[List[str]], int]:
    second_system = (
        "You are a planner. Given the core fields (intent, entities, constraints, urgency), "
        "output ONLY a valid JSON object with a single key 'steps' whose value is an array of 3-5 short imperative verbs."
    )
    second_user = json.dumps({
        "intent": core_fields.get("intent", None),
        "entities": core_fields.get("entities", None),
        "constraints": core_fields.get("constraints", None),
        "urgency": core_fields.get("urgency", None)
    }, ensure_ascii=False)
    messages = [
        {"role":"system","content": second_system},
        {"role":"user","content": f"CORE_FIELDS: {second_user}\nReturn JSON with key 'steps' only."}
    ]
    out2, toks2 = call_llm(messages, temperature, json_mode)
    try:
        steps_obj = safe_json(out2) or {}
        if isinstance(steps_obj, dict) and "steps" in steps_obj:
            return steps_obj["steps"], toks2
    except Exception:
        pass
    return None, toks2

# ========= MAIN =========
def select_arm_ts(arms: List[Arm]) -> Arm:
    # Sample from Beta posteriors and pick argmax
    best_arm = None
    best_sample = -1.0
    for a in arms:
        s = thompson.sample(a)
        if s > best_sample:
            best_sample = s
            best_arm = a
    return best_arm

def main():
    ensure_csv_header()
    arms = enumerate_arms()

    for itr in range(1, ITERATIONS + 1):
        arm = select_arm_ts(arms)
        pv = prompt_variants[arm.prompt_id]

        # pick a dataset example
        sample = random.choice(DATASET)
        user_input = sample["input"]
        ref_json_str = sample["reference"]

        # For intent-matched few-shot (offline), we "peek" at the gold intent
        try:
            gold_intent = json.loads(ref_json_str).get("intent")
        except Exception:
            gold_intent = None

        messages = build_messages(pv.template, user_input, arm.example_count, arm.fewshot_strategy, gold_intent)
        out, toks = call_llm(messages, arm.temperature, arm.json_mode)

        # Optional two-pass: re-generate steps based on core fields
        if arm.two_pass:
            core = safe_json(out) or {}
            steps, toks2 = run_second_pass_steps(core, arm.temperature, arm.json_mode)
            toks += toks2
            if steps is not None:
                combined = core if isinstance(core, dict) else {}
                combined["steps"] = steps
                out = json.dumps(combined, ensure_ascii=False)

        reward, field_scores, valid = compute_reward(out, ref_json_str)
        reward_per_1k = reward / max(1.0, (toks / 1000.0))
        success = bool(valid and reward >= SUCCESS_THRESHOLD)

        # Update Thompson posteriors
        thompson.update(arm, success)

        # Log
        log_row(
            iteration=itr, phase="bandit",
            prompt_id=pv.prompt_id, category=pv.category, intent=pv.intent,
            example_count=arm.example_count, temperature=arm.temperature,
            json_mode=arm.json_mode, two_pass=arm.two_pass, fewshot_strategy=arm.fewshot_strategy,
            input=user_input, reference=ref_json_str,
            output=out, tokens=toks,
            reward=round(reward,6), reward_per_1k=round(reward_per_1k,6), success=success,
            f1_intent=field_scores["intent"], f1_entities=field_scores["entities"],
            f1_constraints=field_scores["constraints"], f1_urgency=field_scores["urgency"], f1_steps=field_scores["steps"],
            prompt_template=pv.template
        )

        if itr % 10 == 0:
            print(f"[iter {itr}] reward={reward:.3f} | success={int(success)} | tokens={toks} | arm={arm.key()}")

    print(f"✅ Done. Log → {CSV_LOG}")

if __name__ == "__main__":
    main()

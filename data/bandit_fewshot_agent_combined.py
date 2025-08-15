import os, csv, json, math, random
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional

import numpy as np
from openai import OpenAI

# ========= CONFIG =========
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise EnvironmentError("OPENAI_API_KEY is not set")
client = OpenAI(api_key=OPENAI_API_KEY)

GEN_MODEL = "gpt-4o-mini"

DATA_PATH = "agent_dataset.json"
CSV_LOG = "results/bandit_fewshot_agent_log.csv"
os.makedirs("results", exist_ok=True)

ITERATIONS = 60
EPSILON = 0.25
MUTATION_RATE = 0.18
FEW_SHOT_LEVELS = [0, 1, 3]     # agent tasks usually benefit from a few strong demos
TEMPS = [0.2, 0.5]              # lower temps help schema adherence

# ======== FEW-SHOT EXAMPLES (agent-style) ========
# Each example: user-like INPUT and strict JSON OUTPUT
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
    },
    {
      "input": "Write a brief thank-you email to the interviewer and ask for feedback.",
      "output": {
        "intent":"draft_email",
        "entities":{"recipient":"interviewer","topic":"thank_you","extra_request":"feedback"},
        "constraints":["polite_tone","brief"],
        "urgency":"normal",
        "steps":["draft_email","review_tone","send_or_copy"]
      }
    }
]

# ======== BASE PROMPTS (agent-style, different “families”) ========
BASE_PROMPTS = [
  {
    "id":"p_schema_strict",
    "template": (
      "You are an agent planner. Extract a structured plan as strict JSON only, no extra text.\n"
      "Schema:\n"
      "{\n"
      "  \"intent\": string,\n"
      "  \"entities\": object,\n"
      "  \"constraints\": string[],\n"
      "  \"urgency\": \"normal\" | \"time_sensitive\",\n"
      "  \"steps\": string[]\n"
      "}\n"
      "Requirements: obey schema, fill missing keys with null or omit, NEVER add commentary."
    ),
    "category":"schema_strict","intent":"structured"
  },
  {
    "id":"p_schema_guided",
    "template": (
      "Return ONLY JSON (no prose). Identify intent, entities (from the request), constraints, urgency, and 3-5 steps.\n"
      "Keys: intent, entities, constraints, urgency, steps. If unknown, use null. No extra keys."
    ),
    "category":"schema_guided","intent":"structured"
  },
  {
    "id":"p_reason_then_json",
    "template": (
      "Think silently to extract intent and entities, then output ONLY valid JSON with keys: intent, entities, constraints, urgency, steps. "
      "Do not reveal your reasoning. No text outside JSON."
    ),
    "category":"reason_hidden","intent":"structured"
  },
  {
    "id":"p_minimal_json",
    "template": (
      "Output only minimal valid JSON with keys intent, entities, constraints, urgency, steps. Keep values concise. No commentary."
    ),
    "category":"minimal","intent":"structured"
  }
]

# ========= TYPES & STATE =========
@dataclass
class Arm:
    prompt_id: str
    example_count: int
    temperature: float
    def key(self): return (self.prompt_id, self.example_count, self.temperature)

@dataclass
class PromptVariant:
    prompt_id: str
    template: str
    category: str
    intent: str
    parent_id: Optional[str] = None

prompt_variants: Dict[str, PromptVariant] = {
    p["id"]: PromptVariant(p["id"], p["template"], p["category"], p["intent"]) for p in BASE_PROMPTS
}

def enumerate_arms() -> List[Arm]:
    arms = []
    for pid in prompt_variants.keys():
        for k in FEW_SHOT_LEVELS:
            for t in TEMPS:
                arms.append(Arm(pid, k, t))
    return arms

class BanditStats:
    def __init__(self):
        self.counts: Dict[Tuple[str,int,float], int] = {}
        self.totals: Dict[Tuple[str,int,float], float] = {}
    def update(self, arm: Arm, reward: float):
        k = arm.key()
        self.counts[k] = self.counts.get(k, 0) + 1
        self.totals[k] = self.totals.get(k, 0.0) + reward
    def avg(self, arm: Arm) -> float:
        k = arm.key()
        c = self.counts.get(k, 0)
        return 0.0 if c == 0 else self.totals.get(k, 0.0) / c

bandit = BanditStats()

# ========= HELPERS =========
def build_fewshot_block(n: int) -> str:
    if n <= 0: return ""
    demos = FEW_SHOT[:min(n, len(FEW_SHOT))]
    lines = []
    for ex in demos:
        lines.append("USER: " + ex["input"])
        lines.append("AGENT_JSON: " + json.dumps(ex["output"], ensure_ascii=False))
    return "\n".join(lines) + "\n"

def build_prompt_text(template: str, user_input: str, example_count: int) -> List[dict]:
    fewshot = build_fewshot_block(example_count)
    system_msg = {"role":"system","content": template}
    user_msg = {"role":"user","content": (fewshot + "USER: " + user_input + "\nAGENT_JSON:")}
    return [system_msg, user_msg]

def call_llm(messages: List[dict], temperature: float) -> Tuple[str,int]:
    resp = client.chat.completions.create(
        model=GEN_MODEL, messages=messages, temperature=temperature, max_tokens=300
    )
    out = resp.choices[0].message.content.strip()
    toks = resp.usage.total_tokens if getattr(resp, "usage", None) else 0
    return out, toks

def safe_json(s: str) -> Optional[dict]:
    # try exact, then soften around code fences
    try:
        return json.loads(s)
    except:
        if s.startswith("```"):
            s = s.strip("`")
            idx = s.find("{")
            jdx = s.rfind("}")
            if idx>=0 and jdx>idx:
                try: return json.loads(s[idx:jdx+1])
                except: pass
        # last resort: find first {...}
        idx = s.find("{"); jdx = s.rfind("}")
        if idx>=0 and jdx>idx:
            try: return json.loads(s[idx:jdx+1])
            except: return None
    return None

# ---- reward: field-wise token-F1 + validity bonus ----
FIELDS = ["intent","entities","constraints","urgency","steps"]

def tokens(v) -> set:
    if v is None: return set()
    if isinstance(v, (list, tuple)):
        toks = []
        for x in v:
            toks += str(x).lower().split()
        return set(toks)
    if isinstance(v, dict):
        toks = []
        for k,val in v.items():
            toks += str(k).lower().split()
            toks += str(val).lower().split()
        return set(toks)
    return set(str(v).lower().split())

def f1(a: set, b: set) -> float:
    if not a and not b: return 1.0
    if not a or not b: return 0.0
    inter = len(a & b); p = inter / len(a); r = inter / len(b)
    return 0.0 if (p+r)==0 else 2*p*r/(p+r)

def score_json(pred: dict, ref: dict) -> float:
    scores = []
    for k in FIELDS:
        pa = pred.get(k, None); rb = ref.get(k, None)
        scores.append(f1(tokens(pa), tokens(rb)))
    base = sum(scores)/len(scores)
    # validity bonus if JSON has all required keys
    has_all = all(k in pred for k in FIELDS)
    bonus = 0.05 if has_all else 0.0
    return min(base + bonus, 1.0)

def compute_reward(output_text: str, reference_json_str: str) -> float:
    ref = json.loads(reference_json_str)
    pred = safe_json(output_text)
    if pred is None:
        return 0.0  # invalid JSON = zero reward
    return score_json(pred, ref)

def maybe_mutate(pv: PromptVariant) -> PromptVariant:
    if random.random() > MUTATION_RATE:
        return pv
    tweaks = [
        " Always return exactly the schema keys (intent, entities, constraints, urgency, steps).",
        " Use concise values; avoid filler words.",
        " If unknown, set the key to null rather than guessing.",
        " Ensure steps are 3-5 actionable verbs.",
        " Entities should keep user-provided surface forms."
    ]
    new_template = pv.template + random.choice(tweaks)
    new_id = f"{pv.prompt_id}__m{random.randint(1000,9999)}"
    new_pv = PromptVariant(new_id, new_template, pv.category, pv.intent, parent_id=pv.prompt_id)
    prompt_variants[new_id] = new_pv
    return new_pv

def ensure_csv():
    if not os.path.exists(CSV_LOG):
        with open(CSV_LOG, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([
                "iteration","phase","prompt_id","parent_id","category","intent",
                "example_count","temperature","is_mutation","input","reference",
                "output","tokens","reward","prompt_template"
            ])

def log_row(**kw):
    with open(CSV_LOG, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([
            kw["iteration"], kw["phase"], kw["prompt_id"], kw["parent_id"], kw["category"], kw["intent"],
            kw["example_count"], kw["temperature"], kw["is_mutation"], kw["input"], kw["reference"],
            kw["output"], kw["tokens"], kw["reward"], kw["prompt_template"]
        ])

def select_arm(arms: List[Arm]) -> Arm:
    if random.random() < EPSILON:
        return random.choice(arms)
    best, best_val = None, -1e9
    for a in arms:
        c = bandit.counts.get(a.key(), 0)
        avg = bandit.avg(a)
        bonus = 0.05 if c == 0 else 0.0
        val = avg + bonus
        if val > best_val:
            best, best_val = a, val
    return best

# ========= PRETEST =========
def pretest_all():
    ensure_csv()
    arms = enumerate_arms()
    itr = 0
    for arm in arms:
        pv = prompt_variants[arm.prompt_id]
        sample = random.choice(json.load(open(DATA_PATH,"r",encoding="utf-8")))
        messages = build_prompt_text(pv.template, sample["input"], arm.example_count)
        out, toks = call_llm(messages, arm.temperature)
        reward = compute_reward(out, sample["reference"])
        bandit.update(arm, reward)
        log_row(iteration=itr, phase="pretest", prompt_id=pv.prompt_id, parent_id=pv.parent_id or "",
                category=pv.category, intent=pv.intent, example_count=arm.example_count, temperature=arm.temperature,
                is_mutation=0, input=sample["input"], reference=sample["reference"], output=out, tokens=toks,
                reward=round(reward,6), prompt_template=pv.template)
        itr += 1
    print(f"Pretest complete over {len(arms)} arms.")

# ========= MAIN =========
def main():
    pretest_all()
    data = json.load(open(DATA_PATH,"r",encoding="utf-8"))

    for itr in range(1, ITERATIONS+1):
        arms = enumerate_arms()
        arm = select_arm(arms)
        pv = prompt_variants[arm.prompt_id]

        mutated = 0
        parent_id = pv.parent_id or ""
        if random.random() < MUTATION_RATE:
            new_pv = maybe_mutate(pv)
            if new_pv.prompt_id != pv.prompt_id:
                mutated = 1
                parent_id = pv.prompt_id
                pv = new_pv

        sample = random.choice(data)
        messages = build_prompt_text(pv.template, sample["input"], arm.example_count)
        out, toks = call_llm(messages, arm.temperature)
        reward = compute_reward(out, sample["reference"])

        # update stats on the (possibly new) variant
        effective_arm = Arm(pv.prompt_id, arm.example_count, arm.temperature)
        bandit.update(effective_arm, reward)

        log_row(iteration=itr, phase="bandit", prompt_id=pv.prompt_id, parent_id=parent_id,
                category=pv.category, intent=pv.intent, example_count=arm.example_count, temperature=arm.temperature,
                is_mutation=mutated, input=sample["input"], reference=sample["reference"], output=out, tokens=toks,
                reward=round(reward,6), prompt_template=pv.template)

        if itr % 5 == 0:
            print(f"[iter {itr}] best avg ≈ {max(bandit.totals[k]/max(1,bandit.counts[k]) for k in bandit.totals):.3f}")

    print(f"✅ Done. Log → {CSV_LOG}")

if __name__ == "__main__":
    main()

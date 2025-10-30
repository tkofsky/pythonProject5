# bandit_fewshot_agent_router_hybrid_steps_patch.py
#
# Dynamic-router + Hybrid Two-Pass bandit runner (Steps-Optimized Patch)
# NOW WITH: TWO_PASS_MODE = "plan" | "partial"
#   - "partial": Pass-1 extracts {intent, entities, constraints, urgency}, Pass-2 generates steps (anchored)
#   - "plan":    Pass-1 produces a JSON Schema (planning), Pass-2 emits a JSON object that conforms to that schema
#
# The script logs every iteration to CSV and computes reward/F1s against a reference JSON.
#
# Usage:
#   export OPENAI_API_KEY=YOUR_KEY
#   python bandit_fewshot_agent_router_hybrid_steps_patch.py
#
# Inputs (in working directory):
#   agent_dataset.json     # list of {"input": "...", "reference": "<json-string>"}
#   prompt_variants.json   # list of {"id": "...", "template": "...", "category": "...", "intent": "..."}
#
# Output:
#   bandit_fewshot_agent_log_router_hybrid_steps_patch.csv

import os, csv, json, random, re
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional

from openai import OpenAI

# ========= CONFIG =========
DATA_PATH    = "agent_dataset.json"
PROMPTS_PATH = "prompt_variants.json"
CSV_LOG      = "bandit_fewshot_agent_log_router_hybrid_steps_patch2.csv"

ITERATIONS       = 40
EPSILON          = 0.25              # epsilon-greedy exploration
MUTATION_RATE    = 0.15              # probability of prompt mutation
FEW_SHOT_LEVELS  = [0, 1, 3]         # k-shot options
TEMPS            = [0.2, 0.5]        # decoding temperatures (Pass-2 may clamp lower)
GEN_MODEL        = "gpt-4o-mini"

# ---- New global switch for two-pass behavior ----
# "plan"   : plan → emit (build a JSON schema, then emit JSON instance)
# "partial": partial-schema → steps (extract core fields first, then generate steps)
TWO_PASS_MODE: str = "plan"  # "plan" or "partial"


# Router knobs
ROUTER_TWO_PASS_LEN     = 140        # characters
ROUTER_TWO_PASS_NUM     = True       # numbers/money/date push to 2-pass
ROUTER_TWO_PASS_INTENT  = True       # planning verbs push to 2-pass
ROUTER_TWO_PASS_STEPS   = True       # step-like words push to 2-pass
ROUTER_THRESHOLD        = 2          # score >= threshold => two-pass

# Steps generation knobs (used in partial mode)
PASS2_MAX_TOKENS        = 200
PASS2_TEMP_PRIMARY      = 0.2
PASS2_TEMP_RETRY        = 0.1
PASS2_N_CANDIDATES      = 3          # generate this many step candidates and keep best
STEP_MAX_WORDS          = 6
STEP_MIN_COUNT          = 3
STEP_MAX_COUNT          = 5

# ========= CLIENT =========
api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    raise EnvironmentError("OPENAI_API_KEY is not set in the environment")
client = OpenAI(api_key=api_key)

# ========= DATA LOAD =========
with open(DATA_PATH, "r", encoding="utf-8") as f:
    DATASET = json.load(f)
with open(PROMPTS_PATH, "r", encoding="utf-8") as f:
    PROMPT_VARIANTS_FILE = json.load(f)

# ========= TYPES =========
@dataclass
class Arm:
    prompt_id: str
    example_count: int
    temperature: float
    def key(self) -> Tuple[str, int, float]:
        return (self.prompt_id, self.example_count, self.temperature)

@dataclass
class PromptVariant:
    prompt_id: str
    template: str
    category: str
    intent: str
    parent_id: Optional[str] = None

# ========= PROMPT POOL =========
prompt_variants: Dict[str, PromptVariant] = {
    p["id"]: PromptVariant(
        prompt_id=p["id"],
        template=p["template"],
        category=p.get("category","uncategorized"),
        intent=p.get("intent","structured")
    ) for p in PROMPT_VARIANTS_FILE
}

# ========= BANDIT =========
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
FIELDS = ["intent","entities","constraints","urgency","steps"]

# --- Safe coercers for None/odd structures ---
def _to_listish(x):
    """Coerce possibly-None / scalar / list-like / dict values to a flat list of strings."""
    if x is None:
        return []
    if isinstance(x, (list, tuple, set)):
        return ["" if v is None else str(v) for v in x]
    if isinstance(x, dict):
        return ["" if v is None else str(v) for v in x.values()]
    return [str(x)]

def _to_dictish(x):
    """Coerce possibly-None / non-dict into a dict; if list/tuple, index keys; else scalar under 'value'."""
    if isinstance(x, dict):
        return x
    if x is None:
        return {}
    if isinstance(x, (list, tuple)):
        return {str(i): v for i, v in enumerate(x)}
    return {"value": x}

# Few-shot demos (Pass-1 / one-pass)
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
]

def fewshot_block(n: int) -> str:
    if n <= 0: return ""
    demos = FEW_SHOT[:min(n, len(FEW_SHOT))]
    lines = []
    for ex in demos:
        lines.append("USER: " + ex["input"])
        lines.append("AGENT_JSON: " + json.dumps(ex["output"], ensure_ascii=False))
    return "\n".join(lines) + "\n"

# ==== Messaging builders (shared/one-pass) ====

def build_messages(template: str, user_input: str, example_count: int) -> List[dict]:
    return [
        {"role":"system","content": template},
        {"role":"user","content": fewshot_block(example_count) + "USER: " + user_input + "\nAGENT_JSON:"}
    ]

# ---- Intent-specific style hints to standardize phrasing ----
VERB_WHITELIST = {"search","filter","choose","book","confirm","send","create","draft","compose","schedule","reserve","arrange"}

STEP_STYLE_BY_INTENT = {
  "book_train": [
    "search trains from A to B",
    "filter by price under $X",
    "choose best departure",
    "book ticket",
    "send confirmation"
  ],
  "schedule_meeting": [
    "find common time",
    "create Zoom link",
    "send invites"
  ],
  "order_food": [
    "choose restaurant",
    "filter vegan options",
    "place order",
    "confirm pickup"
  ]
}

STYLE_HINT_PREFIX = "EXAMPLES OF STYLE ONLY (do not copy facts):\n"

def style_hint_for_intent(intent: str) -> str:
    lst = STEP_STYLE_BY_INTENT.get(intent or "", [])
    if not lst: return ""
    ex_json = {"steps": lst}
    return STYLE_HINT_PREFIX + json.dumps(ex_json, ensure_ascii=False) + "\n"

# ---- Pass-2 messages for TWO_PASS_MODE == "partial": anchored + canonical steps ----

def build_pass2_messages_partial(pass1_json: dict) -> List[dict]:
    intent = (pass1_json.get("intent") or "").strip()
    style_hint = style_hint_for_intent(intent)
    sys = (
        style_hint +
        "You will be given a JSON object with keys: intent, entities, constraints, urgency. "
        "Using ONLY those fields, output JSON with a single key 'steps' whose value is an array "
        f"of {STEP_MIN_COUNT} to {STEP_MAX_COUNT} SHORT imperative verb phrases. STRICT RULES:\n"
        "• Reuse surface forms from 'entities' (cities, dates, amounts). Mention them explicitly.\n"
        "• Respect 'constraints' literally (e.g., budget<=120 must appear as 'filter by price under $120').\n"
        "• Do NOT invent new facts or tools. No prose, no numbering, no terminal punctuation.\n"
        f"• Keep each step ≤ {STEP_MAX_WORDS} words; start with a verb from this set: "
        + ", ".join(sorted(VERB_WHITELIST)) + ".\n"
        "• Output valid JSON only: {\"steps\":[\"verb phrase\", ...]}."
    )
    user = "STRUCTURED_FIELDS_JSON:\n" + json.dumps(pass1_json, ensure_ascii=False) + "\nSTEPS_JSON:"
    return [{"role":"system","content": sys}, {"role":"user","content": user}]

# ---- Messages for TWO_PASS_MODE == "plan": Plan (schema) → Emit (instance) ----

PLAN_SCHEMA_INSTRUCTIONS = (
    "You will receive an unstructured task description. Plan a JSON Schema (Draft-07 style is fine) "
    "that captures the task with properties: intent, entities, constraints, urgency, and steps. Rules:\n"
    "• 'intent': short string.\n"
    "• 'entities': object with typed fields extracted from the description (e.g., from, to, date, budget, participants).\n"
    "• 'constraints': array of strings (e.g., 'budget<=120', 'include_zoom_link').\n"
    "• 'urgency': enum like ['low','normal','time_sensitive'] if inferrable, else string.\n"
    f"• 'steps': array of {STEP_MIN_COUNT}-{STEP_MAX_COUNT} short strings (imperative verb phrases).\n"
    "• Provide a valid JSON Schema object with 'type':'object' and 'properties'.\n"
    "• Do not output any prose, only the JSON Schema."
)

EMIT_FROM_SCHEMA_INSTRUCTIONS = (
    "You will receive: (1) a JSON Schema that defines fields, and (2) the original unstructured user description.\n"
    "Emit a single valid JSON object that conforms to the schema. Rules:\n"
    "• Use only facts present in the user description; do not invent.\n"
    f"• If a field is unknown, set it to null or an empty array/object as appropriate.\n"
    f"• Keep 'steps' to {STEP_MIN_COUNT}-{STEP_MAX_COUNT} short imperative verb phrases (≤ {STEP_MAX_WORDS} words).\n"
    "• Output only the JSON object; no prose."
)

def build_plan_pass1_messages(user_input: str) -> List[dict]:
    sys = PLAN_SCHEMA_INSTRUCTIONS
    user = "TASK_DESCRIPTION:\n" + user_input + "\nJSON_SCHEMA:"
    return [{"role":"system","content": sys}, {"role":"user","content": user}]


def build_plan_pass2_messages(schema_obj: dict, user_input: str) -> List[dict]:
    sys = EMIT_FROM_SCHEMA_INSTRUCTIONS
    user = (
        "JSON_SCHEMA:\n" + json.dumps(schema_obj, ensure_ascii=False) +
        "\nTASK_DESCRIPTION:\n" + user_input +
        "\nJSON_OBJECT:"
    )
    return [{"role":"system","content": sys}, {"role":"user","content": user}]

# ---- LLM call ----

def call_llm(messages: List[dict], temperature: float, max_tokens: int = 350) -> Tuple[str, int]:
    resp = client.chat.completions.create(
        model=GEN_MODEL,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens
    )
    out = resp.choices[0].message.content.strip()
    toks = resp.usage.total_tokens if getattr(resp, "usage", None) else 0
    return out, toks

# ---- JSON & scoring helpers ----

def safe_json(s: str):
    if not isinstance(s, str): return None
    try:
        return json.loads(s)
    except Exception:
        pass
    if "{" in s and "}" in s:
        chunk = s[s.find("{"):s.rfind("}")+1]
        try:
            return json.loads(chunk)
        except Exception:
            return None
    return None


def tokens(v) -> set:
    if v is None: return set()
    if isinstance(v, (list, tuple)):
        bag = []
        for x in v: bag += str(x).lower().split()
        return set(bag)
    if isinstance(v, dict):
        bag = []
        for k,val in v.items():
            bag += str(k).lower().split()
            bag += str(val).lower().split()
        return set(bag)
    return set(str(v).lower().split())


def f1(a: set, b: set) -> float:
    if not a and not b: return 1.0
    if not a or not b: return 0.0
    inter = len(a & b)
    p = inter / len(a) if len(a) else 0.0
    r = inter / len(b) if len(b) else 0.0
    return 0.0 if (p+r)==0 else 2*p*r/(p+r)

# ---- Step normalization & generic blocker (used in scoring + partial mode) ----
CANON_MAP = {
    "search train schedules": "search trains",
    "look up trains": "search trains",
    "search schedules": "search schedule",
    "choose best option": "choose best",
    "select best option": "choose best",
    "finalize booking": "book",
    "send confirmation email": "send confirmation",
    "apply price filter": "filter by price",
    "filter by cost": "filter by price",
    "filter by price and time": "filter by price and time",
    "find best departure": "choose best",
    "confirm booking": "send confirmation"
}


def normalize_phrase(s: str) -> str:
    s = s.lower().strip()
    s = s.replace("schedules", "schedule").replace("trains", "train")
    for k, v in CANON_MAP.items():
        s = s.replace(k, v)
    for junk in [" the ", " a ", " an "]:
        s = s.replace(junk, " ")
    return " ".join(s.split())


def normalize_steps(x):
    if isinstance(x, list):
        return [normalize_phrase(str(t)) for t in x]
    return x

GENERIC_PATTERNS = [
  r"\b(select|choose) (best|an) option\b",
  r"\bfinalize (the )?booking\b",
  r"\bstart (the )?process\b",
  r"\bcomplete (the )?task\b"
]


def too_generic(steps):
    import re as _re
    txt = " ".join(map(str, steps)).lower()
    return any(_re.search(p, txt) for p in GENERIC_PATTERNS)


def steps_anchor_ok(steps, pass1_json) -> bool:
    if not isinstance(steps, list) or not (STEP_MIN_COUNT <= len(steps) <= STEP_MAX_COUNT):
        return False
    txt = " ".join(map(str, steps)).lower()
    ents = _to_dictish(pass1_json.get("entities"))
    ent_txt = " ".join(_to_listish(ents)).lower()
    entity_tokens = [tok for tok in ent_txt.split() if tok.isalpha() and len(tok) > 2]
    return any(tok in txt for tok in entity_tokens) if entity_tokens else True


def steps_score(steps, pass1):
    if not isinstance(steps, list) or not (STEP_MIN_COUNT <= len(steps) <= STEP_MAX_COUNT):
        return -1
    if too_generic(steps):
        return -1
    ents_dict = _to_dictish(pass1.get("entities"))
    ents = " ".join(_to_listish(ents_dict)).lower()
    ent_toks = {t for t in ents.split() if t.isalpha() and len(t)>2}
    txt = " ".join(map(str, steps)).lower()
    # coverage
    has_entity = any(t in txt for t in ent_toks) if ent_toks else True
    # style and brevity
    style = 0
    brevity_pen = 0
    for s in steps:
        words = str(s).strip().split()
        if not words: return -1
        if words[0].lower() in VERB_WHITELIST:
            style += 1
        else:
            style -= 1
        if len(words) > STEP_MAX_WORDS:
            brevity_pen += (len(words) - STEP_MAX_WORDS)
    # constraints/date/number coverage boost
    cons_txt = " ".join(_to_listish(pass1.get("constraints"))).lower()
    cons_boost = 1 if any(x in txt for x in ["under $","before","after","by "]) or any(y in txt for y in cons_txt.split()) else 0
    return (3 if has_entity else 0) + style - brevity_pen + cons_boost


def score_json(pred: dict, ref: dict) -> float:
    # normalize steps before tokenization
    if isinstance(pred, dict) and "steps" in pred:
        pred["steps"] = normalize_steps(pred.get("steps"))
    if isinstance(ref, dict) and "steps" in ref:
        ref["steps"] = normalize_steps(ref.get("steps"))
    scores = []
    for k in FIELDS:
        scores.append(f1(tokens(pred.get(k)), tokens(ref.get(k))))
    base = sum(scores)/len(scores)
    has_all = all(k in pred for k in FIELDS)
    bonus = 0.05 if has_all else 0.0
    return min(base + bonus, 1.0)


def compute_reward(output_text: str, reference_json_str: str) -> float:
    ref = json.loads(reference_json_str) if isinstance(reference_json_str, str) else reference_json_str
    pred = safe_json(output_text)
    if pred is None: return 0.0
    return score_json(pred, ref)

# ========= LOGGING =========

def ensure_csv_header():
    log_dir = os.path.dirname(CSV_LOG)
    if log_dir: os.makedirs(log_dir, exist_ok=True)
    if not os.path.exists(CSV_LOG):
        with open(CSV_LOG, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([
                "iteration","phase","router_mode","router_features","two_pass_mode",
                "prompt_id","parent_id","category","intent",
                "example_count","temperature","two_pass","is_mutation",
                "input","reference","output","output_p1","output_p2",
                "tokens","reward","reward_per_1k",
                "json_valid","f1_intent","f1_entities","f1_constraints","f1_urgency","f1_steps",
                "prompt_template"
            ])


def log_row(**kw):
    with open(CSV_LOG, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([
            kw.get("iteration"), kw.get("phase"), kw.get("router_mode"), kw.get("router_features"), kw.get("two_pass_mode"),
            kw.get("prompt_id"), kw.get("parent_id"), kw.get("category"), kw.get("intent"),
            kw.get("example_count"), kw.get("temperature"), kw.get("two_pass"), kw.get("is_mutation"),
            kw.get("input"), kw.get("reference"), kw.get("output"), kw.get("output_p1",""), kw.get("output_p2",""),
            kw.get("tokens"), kw.get("reward"), kw.get("reward_per_1k"),
            kw.get("json_valid"), kw.get("f1_intent"), kw.get("f1_entities"), kw.get("f1_constraints"), kw.get("f1_urgency"), kw.get("f1_steps"),
            kw.get("prompt_template")
        ])

# ========= ARMS =========

def enumerate_arms() -> List[Arm]:
    arms = []
    for pid in prompt_variants.keys():
        for k in FEW_SHOT_LEVELS:
            for t in TEMPS:
                arms.append(Arm(pid, k, t))
    return arms


def select_arm(arms: List[Arm]) -> Arm:
    if random.random() < EPSILON:
        return random.choice(arms)  # explore
    best, best_val = None, -1e9
    for a in arms:
        c = bandit.counts.get(a.key(), 0)
        avg = bandit.avg(a)
        bonus = 0.05 if c == 0 else 0.0
        val = avg + bonus
        if val > best_val:
            best, best_val = a, val
    return best

# ========= MUTATION =========
TWEAKS = [
    " Always return exactly the schema keys (intent, entities, constraints, urgency, steps).",
    " Keep values concise; avoid filler words.",
    " If unknown, set the key to null rather than guessing.",
    " Ensure steps are 3-5 short action verbs.",
    " Preserve user-provided surface forms in entities."
]


def maybe_mutate(pv: PromptVariant) -> PromptVariant:
    if random.random() > MUTATION_RATE:
        return pv
    new_template = pv.template + random.choice(TWEAKS)
    new_id = f"{pv.prompt_id}__m{random.randint(1000,9999)}"
    new_pv = PromptVariant(new_id, new_template, pv.category, pv.intent, parent_id=pv.prompt_id)
    prompt_variants[new_id] = new_pv
    return new_pv

# ========= ROUTER =========
_num_pat = re.compile(r"\d")
_money_pat = re.compile(r"\$|\bUSD\b|\bCAD\b")
_date_words = re.compile(r"\b(today|tomorrow|monday|tuesday|wednesday|thursday|friday|saturday|sunday|next week|next monday)\b", re.I)
_plan_hints = re.compile(r"\b(book|schedule|order|plan|reserve|arrange|organize|set up|create|compose|draft)\b", re.I)
_step_hints = re.compile(r"\b(plan|draft|sequence|itinerary|checklist|procedure|steps?)\b", re.I)


def extract_router_features(text: str) -> Dict[str, object]:
    length = len(text)
    has_num = bool(_num_pat.search(text))
    has_money = bool(_money_pat.search(text))
    has_date = bool(_date_words.search(text))
    plan_hint = bool(_plan_hints.search(text))
    step_hint = bool(_step_hints.search(text))
    return {
        "len": length,
        "has_num": has_num,
        "has_money": has_money,
        "has_date": has_date,
        "plan_hint": plan_hint,
        "step_hint": step_hint
    }


def decide_two_pass(features: Dict[str, object]) -> bool:
    score = 0
    if features["len"] >= ROUTER_TWO_PASS_LEN: score += 1
    if ROUTER_TWO_PASS_NUM and (features["has_num"] or features["has_money"]): score += 1
    if ROUTER_TWO_PASS_INTENT and features["plan_hint"]: score += 1
    if ROUTER_TWO_PASS_STEPS and features.get("step_hint", False): score += 1
    return score >= ROUTER_THRESHOLD

# ========= MAIN =========

def main():
    ensure_csv_header()
    arms = enumerate_arms()

    for itr in range(1, ITERATIONS+1):
        arm = select_arm(arms)
        pv = prompt_variants[arm.prompt_id]

        # maybe mutate
        mutated = 0
        parent_id = pv.parent_id or ""
        if random.random() < MUTATION_RATE:
            new_pv = maybe_mutate(pv)
            if new_pv.prompt_id != pv.prompt_id:
                mutated = 1
                parent_id = pv.prompt_id
                pv = new_pv

        sample = random.choice(DATASET)
        user_inp = sample["input"]
        ref_str  = sample["reference"]  # JSON string

        # router decision
        feats = extract_router_features(user_inp)
        use_two_pass = decide_two_pass(feats)

        router_mode = "two_pass" if use_two_pass else "one_pass"
        router_features = json.dumps(feats, ensure_ascii=False)

        output_text = ""
        output_p1 = ""
        output_p2 = ""
        toks = 0
        two_pass_flag = 0

        if use_two_pass:
            if TWO_PASS_MODE == "partial":
                # ====== PARTIAL mode ======
                # Pass-1: extraction via selected arm
                msgs1 = build_messages(pv.template, user_inp, arm.example_count)
                out1, toks1 = call_llm(msgs1, arm.temperature, max_tokens=300)
                j1 = safe_json(out1)
                output_p1 = out1
                pass1_valid = isinstance(j1, dict) and all(k in j1 for k in ["intent","entities","constraints","urgency"])

                if pass1_valid:
                    # Pass-2: steps-only, anchored (N-best search)
                    msgs2 = build_pass2_messages_partial(j1)
                    pass2_temp = PASS2_TEMP_PRIMARY if arm.temperature > PASS2_TEMP_PRIMARY else arm.temperature

                    cands = []
                    for _ in range(PASS2_N_CANDIDATES):
                        out2, toks2 = call_llm(msgs2, pass2_temp, max_tokens=PASS2_MAX_TOKENS)
                        j2 = safe_json(out2)
                        if isinstance(j2, dict) and "steps" in j2:
                            steps = normalize_steps(j2["steps"])
                            sc = steps_score(steps, j1)
                            if sc > 0:
                                cands.append((sc, steps, toks2, out2))

                    if not cands:
                        # retry colder once
                        out2b, toks2b = call_llm(msgs2, PASS2_TEMP_RETRY, max_tokens=max(160, PASS2_MAX_TOKENS-40))
                        j2b = safe_json(out2b)
                        if isinstance(j2b, dict) and "steps" in j2b:
                            steps_b = normalize_steps(j2b["steps"])
                            sc_b = steps_score(steps_b, j1)
                            if sc_b > 0:
                                cands.append((sc_b, steps_b, toks2b, out2b))

                    if cands:
                        best = max(cands, key=lambda x: x[0])
                        final = {**j1, "steps": best[1]}
                        output_text = json.dumps(final, ensure_ascii=False)
                        output_p2 = best[3]
                        toks = (toks1 or 0) + (best[2] or 0)
                        two_pass_flag = 1
                    else:
                        # fallback to one-pass if steps invalid
                        msgs = build_messages(pv.template, user_inp, arm.example_count)
                        output_text, toks = call_llm(msgs, arm.temperature, max_tokens=320)
                        two_pass_flag = 0
                else:
                    # fallback to one-pass if Pass-1 invalid
                    msgs = build_messages(pv.template, user_inp, arm.example_count)
                    output_text, toks = call_llm(msgs, arm.temperature, max_tokens=320)
                    two_pass_flag = 0

            else:
                # ====== PLAN mode ======
                # Pass-1: PLAN → produce JSON Schema
                msgs1 = build_plan_pass1_messages(user_inp)
                out1, toks1 = call_llm(msgs1, arm.temperature, max_tokens=350)
                schema = safe_json(out1)
                output_p1 = out1
                pass1_valid = isinstance(schema, dict) and schema.get("type") == "object" and isinstance(schema.get("properties"), dict)

                if pass1_valid:
                    # Pass-2: EMIT → fill instance from schema + text
                    msgs2 = build_plan_pass2_messages(schema, user_inp)
                    out2, toks2 = call_llm(msgs2, min(arm.temperature, 0.3), max_tokens=320)
                    output_p2 = out2
                    # Prefer final output to be the emitted instance
                    output_text = out2
                    toks = (toks1 or 0) + (toks2 or 0)
                    two_pass_flag = 1
                else:
                    # fallback to structured one-pass with prompt template
                    msgs = build_messages(pv.template, user_inp, arm.example_count)
                    output_text, toks = call_llm(msgs, arm.temperature, max_tokens=320)
                    two_pass_flag = 0
        else:
            # one-pass (no router two-pass)
            msgs = build_messages(pv.template, user_inp, arm.example_count)
            output_text, toks = call_llm(msgs, arm.temperature, max_tokens=320)
            two_pass_flag = 0

        # scoring
        reward = compute_reward(output_text, ref_str)
        reward_per_1k = (reward / max(1, toks)) * 1000.0
        pred = safe_json(output_text)
        if pred is None:
            json_valid = 0.0
            f1_intent = f1_entities = f1_constraints = f1_urgency = f1_steps = 0.0
        else:
            json_valid = 1.0
            ref = json.loads(ref_str)
            # normalize steps before per-field F1 too
            if "steps" in pred: pred["steps"] = normalize_steps(pred.get("steps"))
            if "steps" in ref:  ref["steps"]  = normalize_steps(ref.get("steps"))
            f1_intent      = f1(tokens(pred.get("intent")),      tokens(ref.get("intent")))
            f1_entities    = f1(tokens(pred.get("entities")),    tokens(ref.get("entities")))
            f1_constraints = f1(tokens(pred.get("constraints")), tokens(ref.get("constraints")))
            f1_urgency     = f1(tokens(pred.get("urgency")),     tokens(ref.get("urgency")))
            f1_steps       = f1(tokens(pred.get("steps")),       tokens(ref.get("steps")))

        # update bandit on selected arm (independent of router choice)
        bandit.update(arm, reward)

        # log
        log_row(
            iteration=itr, phase="bandit",
            router_mode=router_mode, router_features=router_features, two_pass_mode=TWO_PASS_MODE,
            prompt_id=pv.prompt_id, parent_id=parent_id,
            category=pv.category, intent=pv.intent,
            example_count=arm.example_count, temperature=arm.temperature,
            two_pass=two_pass_flag, is_mutation=mutated,
            input=user_inp, reference=ref_str, output=output_text, output_p1=output_p1, output_p2=output_p2,
            tokens=toks,
            reward=round(reward,6), reward_per_1k=round(reward_per_1k,6),
            json_valid=json_valid,
            f1_intent=round(f1_intent,6), f1_entities=round(f1_entities,6),
            f1_constraints=round(f1_constraints,6), f1_urgency=round(f1_urgency,6),
            f1_steps=round(f1_steps,6),
            prompt_template=pv.template
        )

        if itr % 10 == 0:
            best_avg = max(
                bandit.totals[k] / max(1, bandit.counts[k])
                for k in bandit.totals
            ) if bandit.totals else 0.0
            print(f"[iter {itr}] best_avg≈{best_avg:.3f} | router_mode={router_mode} | mode={TWO_PASS_MODE}")

    print(f"✅ Done. Log → {CSV_LOG}")


if __name__ == "__main__":
    main()

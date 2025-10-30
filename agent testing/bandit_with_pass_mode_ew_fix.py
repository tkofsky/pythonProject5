
import os, csv, json, random, re
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional
from openai import OpenAI

# ======================= CONFIG =======================
DATA_PATH    = "agent_dataset_new.json"
PROMPTS_PATH = "prompt_variants_new.json"
CSV_LOG      = "bandit_fewshot_agent_log_router_hybrid_steps_patch_plan_plus.csv"

ITERATIONS       = 60
EPSILON          = 0.25
MUTATION_RATE    = 0.15
FEW_SHOT_LEVELS  = [0, 1, 3]
TEMPS            = [0.2, 0.5]
GEN_MODEL        = "gpt-4o-mini"

TWO_PASS_MODE: str = "plan"  # "plan" or "partial"

# === Steps quality features ===
ENABLE_STEPS_ENRICHMENT = True        # post-emit repair pass for "steps"
ENABLE_HYBRID_PARTIAL_STEPS = True    # try partial-style steps after plan→emit

# Bound Pass-2 temps for terser, less rambly output
PASS2_TEMP_PRIMARY = 0.2
PASS2_TEMP_RETRY   = 0.1

# Router knobs
ROUTER_TWO_PASS_LEN     = 140
ROUTER_TWO_PASS_NUM     = True
ROUTER_TWO_PASS_INTENT  = True
ROUTER_TWO_PASS_STEPS   = True
ROUTER_THRESHOLD        = 2

# Steps generation knobs
PASS2_MAX_TOKENS        = 200
STEP_MAX_WORDS          = 6
STEP_MIN_COUNT          = 3
STEP_MAX_COUNT          = 5

api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    raise EnvironmentError("OPENAI_API_KEY is not set in the environment")
client = OpenAI(api_key=api_key)

# ======================= DATA LOAD =======================
with open(DATA_PATH, "r", encoding="utf-8") as f:
    DATASET = json.load(f)
with open(PROMPTS_PATH, "r", encoding="utf-8") as f:
    PROMPT_VARIANTS_FILE = json.load(f)

# ======================= TYPES =======================
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

prompt_variants: Dict[str, PromptVariant] = {
    p["id"]: PromptVariant(
        prompt_id=p["id"],
        template=p["template"],
        category=p.get("category","uncategorized"),
        intent=p.get("intent","structured")
    ) for p in PROMPT_VARIANTS_FILE
}

# ======================= BANDIT =======================
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

# ======================= HELPERS =======================
FIELDS = ["intent","entities","constraints","urgency","steps"]

def _to_listish(x):
    if x is None: return []
    if isinstance(x, (list, tuple, set)): return ["" if v is None else str(v) for v in x]
    if isinstance(x, dict): return ["" if v is None else str(v) for v in x.values()]
    return [str(x)]

def _to_dictish(x):
    if isinstance(x, dict): return x
    if x is None: return {}
    if isinstance(x, (list, tuple)): return {str(i): v for i, v in enumerate(x)}
    return {"value": x}

FEW_SHOT = [
    {
      "input": "Book a train from Boston to New York tomorrow morning under $120.",
      "output": {
        "intent":"book_train",
        "entities":{"from":"Boston","to":"New York","date":"tomorrow morning","budget":"120"},
        "constraints":["budget<=120"],
        "urgency":"normal",
        "steps":["search trains","filter by price and time","choose best departure","book ticket","send confirmation"]
      }
    },
    {
      "input": "Schedule a 30 minute Zoom check-in with Maya next Tuesday after 3pm.",
      "output": {
        "intent":"schedule_meeting",
        "entities":{"participants":["Maya"],"duration":"30 minutes","time_window":"next Tuesday after 3pm","location":"Zoom"},
        "constraints":["include_zoom_link"],
        "urgency":"normal",
        "steps":["find common time","create Zoom link","send invites"]
      }
    }
]

def fewshot_block(n: int) -> str:
    if n <= 0: return ""
    demos = FEW_SHOT[:min(n, len(FEW_SHOT))]
    lines = []
    for ex in demos:
        lines.append("USER: " + ex["input"])
        lines.append("AGENT_JSON: " + json.dumps(ex["output"], ensure_ascii=False))
    return "\n".join(lines) + "\n"

def build_messages(template: str, user_input: str, example_count: int) -> List[dict]:
    return [
        {"role":"system","content": template},
        {"role":"user","content": fewshot_block(example_count) + "USER: " + user_input + "\nAGENT_JSON:"}
    ]

VERB_WHITELIST = {"search","filter","choose","book","confirm","send","create","draft","compose","schedule","reserve","arrange"}

STYLE_HINT_PREFIX = "EXAMPLES OF STYLE ONLY (do not copy facts):\n"

def style_hint_for_intent(intent: str) -> str:
    hints = {
      "book_train": ["search trains from A to B","filter by price under $X","choose best departure","book ticket","send confirmation"],
      "schedule_meeting": ["find common time","create Zoom link","send invites"],
      "order_food": ["choose restaurant","filter vegan options","place order","confirm pickup"]
    }.get((intent or "").strip(), [])
    if not hints: return ""
    return STYLE_HINT_PREFIX + json.dumps({"steps": hints}, ensure_ascii=False) + "\n"

def build_pass2_messages_partial(pass1_json: dict) -> List[dict]:
    intent = (pass1_json.get("intent") or "").strip()
    style_hint = style_hint_for_intent(intent)
    sys = (
        style_hint +
        "You will be given a JSON object with keys: intent, entities, constraints, urgency. "
        f"Using ONLY those fields, output JSON with a single key 'steps' whose value is an array of {STEP_MIN_COUNT} to {STEP_MAX_COUNT} SHORT imperative verb phrases. RULES:\n"
        "• Reuse surface forms from 'entities' (cities, dates, amounts). Mention them explicitly.\n"
        "• Respect 'constraints' literally (e.g., budget<=120 must appear as 'filter by price under $120').\n"
        "• Do NOT invent new facts or tools. No prose, no numbering, no periods.\n"
        f"• Keep each step ≤ {STEP_MAX_WORDS} words; start with a verb from: " + ", ".join(sorted(VERB_WHITELIST)) + ".\n"
        "• Output valid JSON only: {\"steps\":[\"verb phrase\", ...]}."
    )
    user = "STRUCTURED_FIELDS_JSON:\n" + json.dumps(pass1_json, ensure_ascii=False) + "\nSTEPS_JSON:"
    return [{"role":"system","content": sys}, {"role":"user","content": user}]

PLAN_SCHEMA_INSTRUCTIONS = (
    "You will receive an unstructured task description. Plan a JSON Schema (Draft-07-ish) "
    "for an object with EXACT keys: intent, entities, constraints, urgency, steps. Rules:\n"
    "• intent: short snake_case verb phrase (e.g., 'book_train', 'schedule_meeting').\n"
    "• entities: object of fields seen in the text (from, to, date, budget, participants). Reuse surface forms exactly.\n"
    "• constraints: array[str] in compact canonical form (e.g., 'budget<=120','before 3pm','include_zoom_link').\n"
    "• urgency: one of ['low','normal','time_sensitive'] if inferrable, else short string.\n"
    f"• steps: array of {STEP_MIN_COUNT}-{STEP_MAX_COUNT} SHORT imperative verb phrases.\n"
    "MUST OUTPUT ONLY a valid JSON Schema with 'type':'object','properties', and include examples to anchor style:\n"
    "{\n"
    '  "type":"object","properties": {\n'
    '    "intent":{"type":"string","examples":["book_train","schedule_meeting"]},\n'
    '    "entities":{"type":"object","additionalProperties":true,"examples":[{"from":"Boston","to":"New York","date":"tomorrow","budget":"120"}]},\n'
    '    "constraints":{"type":"array","items":{"type":"string"},"examples":[["budget<=120","before 3pm"]]},\n'
    '    "urgency":{"type":"string","examples":["normal","time_sensitive"]},\n'
    '    "steps":{"type":"array","items":{"type":"string"},"examples":[["search trains","filter by price","choose best departure","book ticket","send confirmation"]], "minItems": 3, "maxItems": 5}\n'
    "  },\"required\":[\"intent\",\"entities\",\"constraints\",\"urgency\",\"steps\"]\n"
    "}\n"
    "No prose. Only the JSON Schema."
)

EMIT_FROM_SCHEMA_INSTRUCTIONS = (
    "You will receive: (1) a JSON Schema defining fields, and (2) the original user description.\n"
    "Emit a single valid JSON object that CONFORMS to the schema. Strict rules:\n"
    "• Use ONLY facts present in the user description; if unknown, set null or empty {} / [].\n"
    "• Reuse surface forms for entities (names, cities, dates, amounts) exactly as written.\n"
    "• constraints: extract numeric/time/logical restrictions as compact strings (e.g., 'budget<=120','before 3pm','include_zoom_link').\n"
    f"• steps: {STEP_MIN_COUNT}–{STEP_MAX_COUNT} SHORT imperative verb phrases (≤ {STEP_MAX_WORDS} words), grounded in entities/constraints. Style examples ONLY (do not copy facts):\n"
    '  ["search trains from A to B","filter by price under $X","choose best departure","book ticket","send confirmation"]\n'
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

def tokenset(v) -> set:
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

CANON_MAP = {
    "search train schedules": "search trains",
    "look up trains": "search trains",
    "search schedules": "search schedule",
    "choose best option": "choose best",
    "select best option": "choose best",
    "finalize booking": "book ticket",
    "send confirmation email": "send confirmation",
    "apply price filter": "filter by price",
    "filter by cost": "filter by price",
    "find best departure": "choose best",
    "confirm booking": "send confirmation"
}

def normalize_phrase(s: str) -> str:
    s = s.lower().strip()
    for k, v in CANON_MAP.items():
        s = s.replace(k, v)
    s = s.replace("  ", " ").strip()
    return " ".join(s.split())

def normalize_steps(x):
    if isinstance(x, list):
        return [normalize_phrase(str(t)) for t in x]
    return x

def too_generic(steps):
    GENERIC_PATTERNS = [
      r"\b(select|choose) (best|an) option\b",
      r"\bfinalize (the )?booking\b",
      r"\bstart (the )?process\b",
      r"\bcomplete (the )?task\b"
    ]
    txt = " ".join(map(str, steps)).lower()
    return any(re.search(p, txt) for p in [re.compile(g, re.I) for g in GENERIC_PATTERNS])

def steps_score(steps, full_json) -> int:
    if not isinstance(steps, list) or not (STEP_MIN_COUNT <= len(steps) <= STEP_MAX_COUNT):
        return -1
    if too_generic(steps):
        return -1
    ents_dict = _to_dictish(full_json.get("entities"))
    ents = " ".join(_to_listish(ents_dict)).lower()
    ent_toks = {t for t in ents.split() if len(t)>2}
    txt = " ".join(map(str, steps)).lower()
    has_entity = any(t in txt for t in ent_toks) if ent_toks else True
    style = 0
    brevity_pen = 0
    for s in steps:
        words = str(s).strip().split()
        if not words: return -1
        if words[0].lower() in {"search","filter","choose","book","confirm","send","create","draft","compose","schedule","reserve","arrange"}:
            style += 1
        else:
            style -= 1
        if len(words) > STEP_MAX_WORDS:
            brevity_pen += (len(words) - STEP_MAX_WORDS)
    cons_txt = " ".join(_to_listish(full_json.get("constraints"))).lower()
    cons_boost = 1 if any(x in txt for x in ["under $","before","after","by "]) or any(y in txt for y in cons_txt.split()) else 0
    return (3 if has_entity else 0) + style - brevity_pen + cons_boost

def score_json(pred: dict, ref: dict) -> float:
    if isinstance(pred, dict) and "steps" in pred:
        pred["steps"] = normalize_steps(pred.get("steps"))
    if isinstance(ref, dict) and "steps" in ref:
        ref["steps"] = normalize_steps(ref.get("steps"))
    scores = []
    for k in FIELDS:
        scores.append(f1(tokenset(pred.get(k)), tokenset(ref.get(k))))
    base = sum(scores)/len(scores)
    has_all = all(k in pred for k in FIELDS)
    bonus = 0.05 if has_all else 0.0
    return min(base + bonus, 1.0)

def compute_reward(output_text: str, reference_json_str: str) -> float:
    ref = json.loads(reference_json_str) if isinstance(reference_json_str, str) else reference_json_str
    pred = safe_json(output_text)
    if pred is None:
        return 0.0
    base = score_json(pred, ref)
    # Reward shaping — small bonus for strong steps
    try:
        bonus = 0.0
        if isinstance(pred, dict):
            s = pred.get("steps")
            if steps_score(normalize_steps(s), pred) > 2:
                bonus += 0.03
        return min(base + bonus, 1.0)
    except Exception:
        return base

def is_steps_weak(steps):
    if not isinstance(steps, list) or not steps:
        return True
    generic = ["step", "do task", "complete task", "process", "procedure"]
    bad = 0
    for s in steps:
        if not isinstance(s, str) or not s.strip():
            bad += 1
            continue
        low = s.lower().strip()
        words = low.split()
        if any(g in low for g in generic): bad += 1
        if not words or len(words) > STEP_MAX_WORDS: bad += 1
        if words and words[0] not in {"search","filter","choose","book","confirm","send","create","draft","compose","schedule","reserve","arrange"}: bad += 1
    return bad >= max(1, len(steps)//2)

def build_steps_enrichment_messages(full_json_obj: dict) -> List[dict]:
    sys = (
        "You will be given a JSON object with keys intent, entities, constraints, urgency, steps.\n"
        "Rewrite ONLY the 'steps' field to be 3–5 SHORT imperative verb phrases (≤ 6 words), "
        "grounded in the entities/constraints. Reuse surface forms (cities, names, amounts, dates). "
        "No numbering, no periods, no extra keys. Output JSON: {\"steps\":[\"...\"]}."
    )
    user = "CURRENT_JSON:\n" + json.dumps(full_json_obj, ensure_ascii=False) + "\nNEW_STEPS_JSON:"
    return [{"role":"system","content":sys},{"role":"user","content":user}]

def ensure_csv_header():
    header = [
        "iteration","phase","router_mode","router_features","two_pass_mode",
        "prompt_id","parent_id","category","intent",
        "example_count","temperature","two_pass","is_mutation",
        "input","reference","output","output_p1","output_p2",
        "tokens","reward","reward_per_1k",
        "json_valid","f1_intent","f1_entities","f1_constraints","f1_urgency","f1_steps",
        "prompt_template"
    ]
    if not os.path.exists(CSV_LOG):
        with open(CSV_LOG, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(header)

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

def enumerate_arms() -> List[Arm]:
    arms = []
    for pid in prompt_variants.keys():
        for k in FEW_SHOT_LEVELS:
            for t in TEMPS:
                arms.append(Arm(pid, k, t))
    return arms

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
    return {"len": length, "has_num": has_num, "has_money": has_money, "has_date": has_date, "plan_hint": plan_hint, "step_hint": step_hint}

def decide_two_pass(features: Dict[str, object]) -> bool:
    score = 0
    if features["len"] >= ROUTER_TWO_PASS_LEN: score += 1
    if ROUTER_TWO_PASS_NUM and (features["has_num"] or features["has_money"]): score += 1
    if ROUTER_TWO_PASS_INTENT and features["plan_hint"]: score += 1
    if ROUTER_TWO_PASS_STEPS and features.get("step_hint", False): score += 1
    return score >= ROUTER_THRESHOLD

def main():
    ensure_csv_header()
    arms = enumerate_arms()
    for itr in range(1, ITERATIONS+1):
        arm = select_arm(arms)
        pv = prompt_variants[arm.prompt_id]
        user_inp, ref_str = (random.choice(DATASET)["input"], random.choice(DATASET)["reference"]) if isinstance(DATASET, list) else (DATASET["input"], DATASET["reference"])
        if isinstance(ref_str, dict): ref_str = json.dumps(ref_str, ensure_ascii=False)

        feats = extract_router_features(user_inp)
        use_two_pass = decide_two_pass(feats)

        router_mode = "two_pass" if use_two_pass else "one_pass"
        router_features = json.dumps(feats, ensure_ascii=False)

        output_text = ""
        output_p1 = ""
        output_p2 = ""
        toks = 0
        two_pass_flag = 0

        # One-pass fallback if router says no two-pass
        if not use_two_pass:
            msgs = build_messages(pv.template, user_inp, arm.example_count)
            output_text, toks = call_llm(msgs, arm.temperature, max_tokens=320)
        else:
            if TWO_PASS_MODE == "partial":
                msgs1 = build_messages(pv.template, user_inp, arm.example_count)
                out1, toks1 = call_llm(msgs1, arm.temperature, max_tokens=300)
                j1 = safe_json(out1)
                output_p1 = out1
                pass1_valid = isinstance(j1, dict) and all(k in j1 for k in ["intent","entities","constraints","urgency"])
                if pass1_valid:
                    msgs2 = build_pass2_messages_partial(j1)
                    cands = []
                    for _ in range(3):
                        out2, toks2 = call_llm(msgs2, min(arm.temperature, 0.3), max_tokens=PASS2_MAX_TOKENS)
                        j2 = safe_json(out2)
                        if isinstance(j2, dict) and "steps" in j2:
                            st = normalize_steps(j2["steps"])
                            sc = steps_score(st, j1)
                            if sc > 0: cands.append((sc, st, toks2, out2))
                    if cands:
                        best = max(cands, key=lambda x: x[0])
                        final = {**j1, "steps": best[1]}
                        output_text = json.dumps(final, ensure_ascii=False)
                        output_p2 = best[3]
                        toks = (toks1 or 0) + (best[2] or 0)
                        two_pass_flag = 1
                    else:
                        msgs = build_messages(pv.template, user_inp, arm.example_count)
                        output_text, toks = call_llm(msgs, arm.temperature, max_tokens=320)
                else:
                    msgs = build_messages(pv.template, user_inp, arm.example_count)
                    output_text, toks = call_llm(msgs, arm.temperature, max_tokens=320)
            else:
                # PLAN
                msgs1 = build_plan_pass1_messages(user_inp)
                out1, toks1 = call_llm(msgs1, arm.temperature, max_tokens=350)
                schema = safe_json(out1)
                output_p1 = out1
                pass1_valid = isinstance(schema, dict) and schema.get("type") == "object" and isinstance(schema.get("properties"), dict)
                if pass1_valid:
                    msgs2 = build_plan_pass2_messages(schema, user_inp)
                    out2, toks2 = call_llm(msgs2, min(arm.temperature, 0.3), max_tokens=320)
                    output_p2 = out2
                    final_obj = safe_json(out2) or {}

                    if ENABLE_STEPS_ENRICHMENT and isinstance(final_obj, dict):
                        curr_steps = final_obj.get("steps")
                        if is_steps_weak(curr_steps):
                            msgs_enrich = build_steps_enrichment_messages(final_obj)
                            out_enrich, toks_enrich = call_llm(msgs_enrich, PASS2_TEMP_PRIMARY, max_tokens=160)
                            j_enrich = safe_json(out_enrich)
                            if isinstance(j_enrich, dict) and isinstance(j_enrich.get("steps"), list):
                                new_steps = normalize_steps(j_enrich["steps"])
                                if steps_score(new_steps, final_obj) > steps_score(curr_steps or [], final_obj):
                                    final_obj["steps"] = new_steps
                                    output_p2 = out2 + "\n\n/* ENRICHED_STEPS */\n" + out_enrich
                                    toks2 = (toks2 or 0) + (toks_enrich or 0)

                    if ENABLE_HYBRID_PARTIAL_STEPS and isinstance(final_obj, dict):
                        base_for_partial = {
                            "intent": final_obj.get("intent"),
                            "entities": final_obj.get("entities"),
                            "constraints": final_obj.get("constraints"),
                            "urgency": final_obj.get("urgency")
                        }
                        msgs2_partial = build_pass2_messages_partial(base_for_partial)
                        cands = []
                        for _ in range(3):
                            out_c, toks_c = call_llm(msgs2_partial, PASS2_TEMP_PRIMARY, max_tokens=PASS2_MAX_TOKENS)
                            jc = safe_json(out_c)
                            if isinstance(jc, dict) and "steps" in jc:
                                st = normalize_steps(jc["steps"])
                                sc = steps_score(st, final_obj)
                                if sc > 0: cands.append((sc, st, toks_c, out_c))
                        if cands:
                            best = max(cands, key=lambda x: x[0])
                            if steps_score(best[1], final_obj) > steps_score(final_obj.get("steps") or [], final_obj):
                                final_obj["steps"] = best[1]
                                output_p2 = output_p2 + "\n\n/* HYBRID_PARTIAL_STEPS */\n" + best[3]
                                toks2 = (toks2 or 0) + (best[2] or 0)

                    output_text = json.dumps(final_obj, ensure_ascii=False)
                    toks = (toks1 or 0) + (toks2 or 0)
                    two_pass_flag = 1
                else:
                    msgs = build_messages(pv.template, user_inp, arm.example_count)
                    output_text, toks = call_llm(msgs, arm.temperature, max_tokens=320)

        reward = compute_reward(output_text, ref_str)
        reward_per_1k = (reward / max(1, toks)) * 1000.0
        pred = safe_json(output_text)
        if pred is None:
            json_valid = 0.0
            f1_intent = f1_entities = f1_constraints = f1_urgency = f1_steps = 0.0
        else:
            json_valid = 1.0
            ref = json.loads(ref_str)
            if "steps" in pred: pred["steps"] = normalize_steps(pred.get("steps"))
            if "steps" in ref:  ref["steps"]  = normalize_steps(ref.get("steps"))
            f1_intent      = f1(tokenset(pred.get("intent")),      tokenset(ref.get("intent")))
            f1_entities    = f1(tokenset(pred.get("entities")),    tokenset(ref.get("entities")))
            f1_constraints = f1(tokenset(pred.get("constraints")), tokenset(ref.get("constraints")))
            f1_urgency     = f1(tokenset(pred.get("urgency")),     tokenset(ref.get("urgency")))
            f1_steps       = f1(tokenset(pred.get("steps")),       tokenset(ref.get("steps")))

        log_row(
            iteration=itr, phase="bandit",
            router_mode=router_mode, router_features=router_features, two_pass_mode=TWO_PASS_MODE,
            prompt_id=pv.prompt_id, parent_id=pv.parent_id or "",
            category=pv.category, intent=pv.intent,
            example_count=arm.example_count, temperature=arm.temperature,
            two_pass=two_pass_flag, is_mutation=0,
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
            print(f"[iter {itr}] reward={reward:.3f} f1_steps={f1_steps:.3f}")

    print(f"✅ Done. Log → {CSV_LOG}")

if __name__ == "__main__":
    main()

# Patch: create an enhanced router-hybrid runner with stronger Pass-2, step normalization, anchor gating, and router tweak.
# File: /mnt/data/bandit_fewshot_agent_router_hybrid_enhanced.py

code = r""
# bandit_fewest_agent_router_hybrid_enhanced.py
#
# Dynamic-router + Hybrid Two-Pass bandit runner (ENHANCED)
# - Stronger Pass-2 prompt (anchored, canonicalized style, 3–5 short imperative steps)
# - Step normalization & anchor gating before accepting Pass-2
# - Router tweaked to detect "step-heavy" requests (plan/sequence/itinerary/etc.)
# - Retry-once for Pass-2 with lower temperature if anchor check fails
#
# Usage:
#   python bandit_fewshot_agent_router_hybrid_enhanced.py
#
# Inputs:
#   agent_dataset.json
#   prompt_variants.json
#
# Output:
#   bandit_fewshot_agent_log_router_hybrid_enhanced.csv
#
import os, csv, json, random, re
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional

from openai import OpenAI

# ========= CONFIG =========
DATA_PATH   = "agent_dataset.json"
PROMPTS_PATH= "prompt_variants.json"
CSV_LOG     = "bandit_fewshot_agent_log_router_hybrid_enhanced.csv"

ITERATIONS     = 60
EPSILON        = 0.25
MUTATION_RATE  = 0.15
FEW_SHOT_LEVELS= [0, 1, 3]
TEMPS          = [0.2, 0.5]
GEN_MODEL      = "gpt-4o-mini"

# Router knobs
ROUTER_TWO_PASS_LEN    = 140     # characters; longer inputs more likely to use 2-pass
ROUTER_TWO_PASS_NUM    = True    # presence of numbers/money/date increases 2-pass chance
ROUTER_TWO_PASS_INTENT = True    # if text hints a 'planning' intent, prefer 2-pass
ROUTER_TWO_PASS_STEPS  = True    # if text hints steps/planning terminology, prefer 2-pass

# Few-shot demos for Pass-1 / 1-pass
FEW_SHOT = [
    {
      'input': 'Book a train from Boston to New York tomorrow morning under $120.',
      'output': {
        'intent':'book_train',
        'entities':{'from':'Boston','to':'New York','date':'tomorrow morning','budget':'120'},
        'constraints':['budget<=120'],
        'urgency':'normal',
        'steps':['search_trains','filter_by_price_and_time','propose_top_options']
      }
    },
    {
      'input': 'Schedule a 30 minute Zoom check-in with Maya next Tuesday after 3pm.',
      'output': {
        'intent':'schedule_meeting',
        'entities':{'participants':['Maya'],'duration':'30 minutes','time_window':'next Tuesday after 3pm','location':'Zoom'},
        'constraints':['include_zoom_link'],
        'urgency':'normal',
        'steps':['find_common_slot','create_zoom','send_invites']
      }
    },
    {
      'input': 'Order 4 vegan lunches for pickup at 1pm at 9 King St.',
      'output': {
        'intent':'order_food',
        'entities':{'headcount':4,'diet':'vegan','pickup_time':'1pm','address':'9 King St'},
        'constraints':['vegan_only'],
        'urgency':'time_sensitive',
        'steps':['choose_restaurants','filter_menu','place_order']
      }
    },
]

# ========= CLIENT =========
api_key = os.environ.get('OPENAI_API_KEY')
if not api_key:
    raise EnvironmentError('OPENAI_API_KEY is not set in the environment')
client = OpenAI(api_key=api_key)

# ========= DATA LOAD =========
with open(DATA_PATH, 'r', encoding='utf-8') as f:
    DATASET = json.load(f)
with open(PROMPTS_PATH, 'r', encoding='utf-8') as f:
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
    p['id']: PromptVariant(
        prompt_id=p['id'],
        template=p['template'],
        category=p.get('category','uncategorized'),
        intent=p.get('intent','structured')
    ) for p in PROMPT_VARIANTS_FILE
}

# ========= BANDIT STATE =========
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
FIELDS = ['intent','entities','constraints','urgency','steps']

def fewshot_block(n: int) -> str:
    if n <= 0: return ''
    demos = FEW_SHOT[:min(n, len(FEW_SHOT))]
    lines = []
    for ex in demos:
        lines.append('USER: ' + ex['input'])
        lines.append('AGENT_JSON: ' + json.dumps(ex['output'], ensure_ascii=False))
    return '\n'.join(lines) + '\n'

def build_messages(template: str, user_input: str, example_count: int) -> List[dict]:
    return [
        {'role':'system','content': template},
        {'role':'user','content': fewshot_block(example_count) + 'USER: ' + user_input + '\nAGENT_JSON:'}
    ]

# ---- Enhanced Pass-2 (anchored, canonical style, 3–5 short imperative steps) ----
STYLE_HINT = (
    "EXAMPLES OF STYLE ONLY (do not copy facts):\n"
    "{\"steps\":[\"search trains from A to B\",\"filter by price under $X\",\"choose best departure\",\"book ticket\",\"send confirmation\"]}\n"
    "{\"steps\":[\"find common time after 3pm\",\"create Zoom link\",\"send invites\"]}\n"
)

def build_pass2_messages(pass1_json: dict, temperature: float) -> List[dict]:
    sys = (
        STYLE_HINT +
        "You will be given a JSON object with keys: intent, entities, constraints, urgency. "
        "Using ONLY those fields, output JSON with a single key 'steps' whose value is an array "
        "of 3 to 5 SHORT imperative verb phrases. STRICT RULES:\n"
        "• Reuse surface forms from 'entities' (cities, dates, amounts). Mention them explicitly.\n"
        "• Respect 'constraints' literally (e.g., budget<=120 must appear as 'filter by price under $120').\n"
        "• Do NOT invent new facts or tools. No prose, no numbering, no terminal punctuation.\n"
        "• Keep each step ≤ 6 words; start with a verb (search, filter, choose, book, confirm, send).\n"
        "• Output valid JSON only: {\"steps\":[\"verb phrase\", ...]}."
    )
    user = 'STRUCTURED_FIELDS_JSON:\n' + json.dumps(pass1_json, ensure_ascii=False) + '\nSTEPS_JSON:'
    return [{'role':'system','content': sys}, {'role':'user','content': user}]

def call_llm(messages: List[dict], temperature: float, max_tokens: int = 350) -> Tuple[str, int]:
    resp = client.chat.completions.create(
        model=GEN_MODEL,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens
    )
    out = resp.choices[0].message.content.strip()
    toks = resp.usage.total_tokens if getattr(resp, 'usage', None) else 0
    return out, toks

def safe_json(s: str):
    if not isinstance(s, str):
        return None
    try:
        return json.loads(s)
    except Exception:
        pass
    if '{' in s and '}' in s:
        chunk = s[s.find('{'):s.rfind('}')+1]
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

# ---- Step normalization & anchor checks ----
CANON_MAP = {
    "search train schedules": "search trains",
    "search schedule": "search",
    "choose best option": "choose best",
    "select best option": "choose best",
    "finalize booking": "book",
    "send confirmation email": "send confirmation",
    "apply price filter": "filter by price",
    "filter by price and time": "filter by price and time"
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

def steps_anchor_ok(steps, pass1_json) -> bool:
    if not isinstance(steps, list) or not (3 <= len(steps) <= 5):
        return False
    txt = " ".join(map(str, steps)).lower()
    ents = pass1_json.get("entities", {})
    ent_txt = " ".join(map(str, ents.values())).lower()
    # require at least one entity token to appear in steps
    entity_tokens = [tok for tok in ent_txt.split() if tok.isalpha() and len(tok) > 2]
    return any(tok in txt for tok in entity_tokens) if entity_tokens else True

def score_json(pred: dict, ref: dict) -> float:
    scores = []
    # normalize steps for both sides before scoring
    if isinstance(pred, dict):
        if "steps" in pred:
            pred["steps"] = normalize_steps(pred.get("steps"))
    if isinstance(ref, dict):
        if "steps" in ref:
            ref["steps"] = normalize_steps(ref.get("steps"))
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

def ensure_csv_header():
    log_dir = os.path.dirname(CSV_LOG)
    if log_dir: os.makedirs(log_dir, exist_ok=True)
    if not os.path.exists(CSV_LOG):
        with open(CSV_LOG, 'w', newline='', encoding='utf-8') as f:
            csv.writer(f).writerow([
                'iteration','phase','router_mode','router_features','prompt_id','parent_id','category','intent',
                'example_count','temperature','two_pass','is_mutation',
                'input','reference','output','tokens','reward','reward_per_1k',
                'json_valid','f1_intent','f1_entities','f1_constraints','f1_urgency','f1_steps',
                'prompt_template'
            ])

def log_row(**kw):
    with open(CSV_LOG, 'a', newline='', encoding='utf-8') as f:
        csv.writer(f).writerow([
            kw['iteration'], kw['phase'], kw['router_mode'], kw['router_features'], kw['prompt_id'], kw['parent_id'], kw['category'], kw['intent'],
            kw['example_count'], kw['temperature'], kw['two_pass'], kw['is_mutation'],
            kw['input'], kw['reference'], kw['output'], kw['tokens'], kw['reward'], kw['reward_per_1k'],
            kw['json_valid'], kw['f1_intent'], kw['f1_entities'], kw['f1_constraints'], kw['f1_urgency'], kw['f1_steps'],
            kw['prompt_template']
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
    ' Always return exactly the schema keys (intent, entities, constraints, urgency, steps).',
    ' Keep values concise; avoid filler words.',
    ' If unknown, set the key to null rather than guessing.',
    ' Ensure steps are 3-5 short action verbs.',
    ' Preserve user-provided surface forms in entities.'
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
_num_pat = re.compile(r'\d')
_money_pat = re.compile(r'\$|\bUSD\b|\bCAD\b')
_date_words = re.compile(r'\b(today|tomorrow|monday|tuesday|wednesday|thursday|friday|saturday|sunday|next week|next monday)\b', re.I)
_plan_hints = re.compile(r'\b(book|schedule|order|plan|reserve|arrange|organize|set up|create|compose|draft)\b', re.I)
_step_hints = re.compile(r'\b(plan|draft|sequence|itinerary|checklist|procedure|steps?)\b', re.I)

def extract_router_features(text: str) -> Dict[str, object]:
    length = len(text)
    has_num = bool(_num_pat.search(text))
    has_money = bool(_money_pat.search(text))
    has_date = bool(_date_words.search(text))
    plan_hint = bool(_plan_hints.search(text))
    step_hint = bool(_step_hints.search(text))
    return {
        'len': length,
        'has_num': has_num,
        'has_money': has_money,
        'has_date': has_date,
        'plan_hint': plan_hint,
        'step_hint': step_hint
    }

def decide_two_pass(features: Dict[str, object]) -> bool:
    score = 0
    if features['len'] >= ROUTER_TWO_PASS_LEN: score += 1
    if ROUTER_TWO_PASS_NUM and (features['has_num'] or features['has_money']): score += 1
    if ROUTER_TWO_PASS_INTENT and features['plan_hint']: score += 1
    if ROUTER_TWO_PASS_STEPS and features.get('step_hint', False): score += 1
    return score >= 2  # threshold

# ========= MAIN =========
def main():
    ensure_csv_header()
    arms = enumerate_arms()

    for itr in range(1, ITERATIONS+1):
        arm = select_arm(arms)
        pv = prompt_variants[arm.prompt_id]

        # maybe mutate
        mutated = 0
        parent_id = pv.parent_id or ''
        if random.random() < MUTATION_RATE:
            new_pv = maybe_mutate(pv)
            if new_pv.prompt_id != pv.prompt_id:
                mutated = 1
                parent_id = pv.prompt_id
                pv = new_pv

        sample = random.choice(DATASET)
        user_inp = sample['input']
        ref_str  = sample['reference']  # stored as JSON string

        # router decision
        feats = extract_router_features(user_inp)
        use_two_pass = decide_two_pass(feats)

        router_mode = 'two_pass' if use_two_pass else 'one_pass'
        router_features = json.dumps(feats, ensure_ascii=False)

        # run
        if use_two_pass:
            # Pass-1: extraction via selected arm
            msgs1 = build_messages(pv.template, user_inp, arm.example_count)
            out1, toks1 = call_llm(msgs1, arm.temperature, max_tokens=300)
            j1 = safe_json(out1)
            pass1_valid = isinstance(j1, dict) and all(k in j1 for k in ['intent','entities','constraints','urgency'])

            if pass1_valid:
                # Pass-2: steps-only, anchored; use a lower temperature and smaller max_tokens
                pass2_temp = 0.2 if arm.temperature > 0.2 else arm.temperature
                msgs2 = build_pass2_messages(j1, pass2_temp)
                out2, toks2 = call_llm(msgs2, pass2_temp, max_tokens=220)
                j2 = safe_json(out2)

                # Anchor check; retry once at even lower temp if needed
                if isinstance(j2, dict) and 'steps' in j2 and steps_anchor_ok(j2['steps'], j1):
                    # merge
                    final = {**j1, 'steps': normalize_steps(j2['steps'])}
                    output_text = json.dumps(final, ensure_ascii=False)
                    toks = (toks1 or 0) + (toks2 or 0)
                    two_pass_flag = 1
                else:
                    # retry once
                    msgs2_retry = build_pass2_messages(j1, 0.1)
                    out2b, toks2b = call_llm(msgs2_retry, 0.1, max_tokens=200)
                    j2b = safe_json(out2b)
                    if isinstance(j2b, dict) and 'steps' in j2b and steps_anchor_ok(j2b['steps'], j1):
                        final = {**j1, 'steps': normalize_steps(j2b['steps'])}
                        output_text = json.dumps(final, ensure_ascii=False)
                        toks = (toks1 or 0) + (toks2b or 0)
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
            # one-pass
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
            f1_intent      = f1(tokens(pred.get('intent')),      tokens(ref.get('intent')))
            f1_entities    = f1(tokens(pred.get('entities')),    tokens(ref.get('entities')))
            f1_constraints = f1(tokens(pred.get('constraints')), tokens(ref.get('constraints')))
            f1_urgency     = f1(tokens(pred.get('urgency')),     tokens(ref.get('urgency')))
            f1_steps       = f1(tokens(pred.get('steps')),       tokens(ref.get('steps')))

        # update bandit on selected arm (independent of router choice)
        bandit.update(arm, reward)

        # log
        log_row(
            iteration=itr, phase='bandit',
            router_mode=router_mode, router_features=router_features,
            prompt_id=pv.prompt_id, parent_id=parent_id,
            category=pv.category, intent=pv.intent,
            example_count=arm.example_count, temperature=arm.temperature,
            two_pass=two_pass_flag, is_mutation=mutated,
            input=user_inp, reference=ref_str, output=output_text, tokens=toks,
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
            print(f"[iter {itr}] best_avg≈{best_avg:.3f} | router_mode={router_mode}")

    print(f"✅ Done. Log → {CSV_LOG}")

if __name__ == '__main__':
    main()
""
#with open("/mnt/data/bandit_fewshot_agent_router_hybrid_enhanced.py", "w", encoding="utf-8") as f:
#    f.write(code)

#print("bandit_fewshot_agent_router_hybrid_enhanced.py")

import os, csv, json, random
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional

from openai import OpenAI

# ========= CONFIG =========
DATA_PATH = "agent_dataset.json"          # user input + reference JSON (gold)
PROMPTS_PATH = "prompt_variants.json"     # prompt variants library (this is new)
CSV_LOG = "bandit_fewshot_agent_log.csv"

ITERATIONS = 60
EPSILON = 0.25                # exploration rate (ε-greedy)
MUTATION_RATE = 0.18          # chance to mutate a chosen prompt template
FEW_SHOT_LEVELS = [0, 1, 3]   # number of demo examples to prepend
TEMPS = [0.2, 0.5]            # decoding temperatures (lower → stricter JSON)
GEN_MODEL = "gpt-4o-mini"

# Few-shot demos (agent-style)
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

# ========= TYPES & STATE =========
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

# Build initial prompt variant pool from the JSON file
prompt_variants: Dict[str, PromptVariant] = {
    p["id"]: PromptVariant(
        prompt_id=p["id"],
        template=p["template"],
        category=p.get("category", "uncategorized"),
        intent=p.get("intent", "structured")
    ) for p in PROMPT_VARIANTS_FILE
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
FIELDS = ["intent","entities","constraints","urgency","steps"]

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

def call_llm(messages: List[dict], temperature: float) -> Tuple[str, int]:
    resp = client.chat.completions.create(
        model=GEN_MODEL,
        messages=messages,
        temperature=temperature,
        max_tokens=300
    )
    out = resp.choices[0].message.content.strip()
    toks = resp.usage.total_tokens if getattr(resp, "usage", None) else 0
    return out, toks

def safe_json(s: str):
    # try raw; then slice out first {...}
    try:
        return json.loads(s)
    except:
        pass
    if isinstance(s, str) and "{" in s and "}" in s:
        chunk = s[s.find("{"):s.rfind("}")+1]
        try:
            return json.loads(chunk)
        except:
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

def score_json(pred: dict, ref: dict) -> float:
    scores = []
    for k in FIELDS:
        scores.append(f1(tokens(pred.get(k)), tokens(ref.get(k))))
    base = sum(scores)/len(scores)
    has_all = all(k in pred for k in FIELDS)
    bonus = 0.05 if has_all else 0.0
    return min(base + bonus, 1.0)

def compute_reward(output_text: str, reference_json_str: str) -> float:
    ref = json.loads(reference_json_str)
    pred = safe_json(output_text)
    if pred is None:
        return 0.0  # invalid JSON → zero reward
    return score_json(pred, ref)

def maybe_mutate(pv: PromptVariant) -> PromptVariant:
    if random.random() > MUTATION_RATE:
        return pv
    tweaks = [
        " Always return exactly the schema keys (intent, entities, constraints, urgency, steps).",
        " Keep values concise; avoid filler words.",
        " If unknown, set the key to null rather than guessing.",
        " Ensure steps are 3-5 short action verbs.",
        " Preserve user-provided surface forms in entities."
    ]
    new_template = pv.template + random.choice(tweaks)
    new_id = f"{pv.prompt_id}__m{random.randint(1000,9999)}"
    new_pv = PromptVariant(new_id, new_template, pv.category, pv.intent, parent_id=pv.prompt_id)
    prompt_variants[new_id] = new_pv
    return new_pv

def ensure_csv_header():
    #os.makedirs(os.path.dirname(CSV_LOG), exist_ok=True)
    log_dir = os.path.dirname(CSV_LOG)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
    if not os.path.exists(CSV_LOG):
        with open(CSV_LOG, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([
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

def select_arm(arms: List[Arm], stats: BanditStats) -> Arm:
    if random.random() < EPSILON:
        return random.choice(arms)  # explore
    # exploit best avg reward so far (tiny bonus for unseen arms)
    best, best_val = None, -1e9
    for a in arms:
        c = stats.counts.get(a.key(), 0)
        avg = stats.avg(a)
        bonus = 0.05 if c == 0 else 0.0
        val = avg + bonus
        if val > best_val:
            best, best_val = a, val
    return best

# ========= PRETEST =========
def pretest_all_arms(stats: BanditStats):
    ensure_csv_header()
    arms = enumerate_arms()
    itr = 0
    for arm in arms:
        pv = prompt_variants[arm.prompt_id]
        sample = random.choice(DATASET)
        msgs = build_messages(pv.template, sample["input"], arm.example_count)
        out, toks = call_llm(msgs, arm.temperature)
        reward = compute_reward(out, sample["reference"])
        stats.update(arm, reward)
        log_row(
            iteration=itr, phase="pretest",
            prompt_id=pv.prompt_id, parent_id=pv.parent_id or "",
            category=pv.category, intent=pv.intent,
            example_count=arm.example_count, temperature=arm.temperature,
            is_mutation=0, input=sample["input"], reference=sample["reference"],
            output=out, tokens=toks, reward=round(reward,6), prompt_template=pv.template
        )
        itr += 1
    print(f"🔍 Pretested {len(arms)} arms (prompt × few-shot × temp).")

# ========= MAIN =========
def main():
    pretest_all_arms(bandit)

    for itr in range(1, ITERATIONS + 1):
        arms = enumerate_arms()
        arm = select_arm(arms, bandit)
        pv = prompt_variants[arm.prompt_id]

        mutated = 0
        parent_id = pv.parent_id or ""
        # optional: mutate the chosen prompt template
        if random.random() < MUTATION_RATE:
            new_pv = maybe_mutate(pv)
            if new_pv.prompt_id != pv.prompt_id:
                mutated = 1
                parent_id = pv.prompt_id
                pv = new_pv

        sample = random.choice(DATASET)
        msgs = build_messages(pv.template, sample["input"], arm.example_count)
        out, toks = call_llm(msgs, arm.temperature)
        reward = compute_reward(out, sample["reference"])

        # update stats for (possibly new) variant arm
        effective_arm = Arm(pv.prompt_id, arm.example_count, arm.temperature)
        bandit.update(effective_arm, reward)

        log_row(
            iteration=itr, phase="bandit",
            prompt_id=pv.prompt_id, parent_id=parent_id,
            category=pv.category, intent=pv.intent,
            example_count=arm.example_count, temperature=arm.temperature,
            is_mutation=mutated, input=sample["input"], reference=sample["reference"],
            output=out, tokens=toks, reward=round(reward,6), prompt_template=pv.template
        )

        if itr % 5 == 0:
            # quick heartbeat
            best_avg = max(
                bandit.totals[k] / max(1, bandit.counts[k])
                for k in bandit.totals
            ) if bandit.totals else 0.0
            print(f"[iter {itr}] best_avg≈{best_avg:.3f}")

    print(f"✅ Done. Log → {CSV_LOG}")

if __name__ == "__main__":
    main()

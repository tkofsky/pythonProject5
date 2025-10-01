import os, csv, json, random
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional

from openai import OpenAI

# ========= CONFIG =========
DATA_PATH = "agent_dataset.json"
PROMPTS_PATH = "prompt_variants.json"
#CSV_LOG = "bandit_fewshot_agent_log_two_pass.csv"
CSV_LOG = "bandit_fewshot_agent_log"

ITERATIONS = 30
EPSILON = 0.25
FEW_SHOT_LEVELS = [0, 1, 3]
TEMPS = [0.2, 0.3]
TWO_PASS_OPTIONS = [False, True]
GEN_MODEL = "gpt-4o-mini"

# Few-shot demos
FEW_SHOT = [
    {"input": "Book a train from Boston to New York tomorrow morning under $120.",
     "output": {"intent":"book_train","entities":{"from":"Boston","to":"New York","date":"tomorrow morning","budget":"120"},"constraints":["budget<=120"],"urgency":"normal","steps":["search_trains","filter_by_price_and_time","propose_top_options"]}},
    {"input": "Schedule a 30 minute Zoom check-in with Maya next Tuesday after 3pm.",
     "output": {"intent":"schedule_meeting","entities":{"participants":["Maya"],"duration":"30 minutes","time_window":"next Tuesday after 3pm","location":"Zoom"},"constraints":["include_zoom_link"],"urgency":"normal","steps":["find_common_slot","create_zoom","send_invites"]}},
    {"input": "Order 4 vegan lunches for pickup at 1pm at 9 King St.",
     "output": {"intent":"order_food","entities":{"headcount":4,"diet":"vegan","pickup_time":"1pm","address":"9 King St"},"constraints":["vegan_only"],"urgency":"time_sensitive","steps":["choose_restaurants","filter_menu","place_order"]}},
    {"input": "Write a brief thank-you email to the interviewer and ask for feedback.",
     "output": {"intent":"draft_email","entities":{"recipient":"interviewer","topic":"thank_you","extra_request":"feedback"},"constraints":["polite_tone","brief"],"urgency":"normal","steps":["draft_email","review_tone","send_or_copy"]}}
]

# ========= CLIENT =========
api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    raise EnvironmentError("OPENAI_API_KEY is not set")
client = OpenAI(api_key=api_key)

# ========= DATA LOAD =========
with open(DATA_PATH, "r", encoding="utf-8") as f:
    DATASET = json.load(f)
with open(PROMPTS_PATH, "r", encoding="utf-8") as f:
    PROMPT_VARIANTS_FILE = json.load(f)

@dataclass
class Arm:
    prompt_id: str
    example_count: int
    temperature: float
    two_pass: bool
    def key(self): return (self.prompt_id, self.example_count, self.temperature, self.two_pass)

@dataclass
class PromptVariant:
    prompt_id: str
    template: str
    category: str
    intent: str
    parent_id: Optional[str] = None

prompt_variants: Dict[str, PromptVariant] = {
    p["id"]: PromptVariant(p["id"], p["template"], p.get("category","uncategorized"), p.get("intent","structured"))
    for p in PROMPT_VARIANTS_FILE
}

def enumerate_arms():
    arms = []
    for pid in prompt_variants:
        for k in FEW_SHOT_LEVELS:
            for t in TEMPS:
                for tp in TWO_PASS_OPTIONS:
                    arms.append(Arm(pid, k, t, tp))
    return arms

class BanditStats:
    def __init__(self): self.counts, self.totals = {}, {}
    def update(self, arm, reward):
        k = arm.key()
        self.counts[k] = self.counts.get(k, 0) + 1
        self.totals[k] = self.totals.get(k, 0.0) + reward
    def avg(self, arm):
        k = arm.key(); c = self.counts.get(k,0)
        return 0.0 if c==0 else self.totals[k]/c

bandit = BanditStats()

FIELDS = ["intent","entities","constraints","urgency","steps"]

def fewshot_block(n):
    if n<=0: return ""
    demos = FEW_SHOT[:min(n, len(FEW_SHOT))]
    lines = []
    for ex in demos:
        lines.append("USER: " + ex["input"])
        lines.append("AGENT_JSON: " + json.dumps(ex["output"], ensure_ascii=False))
    return "\n".join(lines) + "\n"

def build_messages(template, user_input, k):
    return [{"role":"system","content":template},
            {"role":"user","content":fewshot_block(k)+"USER: "+user_input+"\nAGENT_JSON:"}]

def call_llm(messages, temp):
    resp = client.chat.completions.create(model=GEN_MODEL, messages=messages, temperature=temp, max_tokens=300)
    out = resp.choices[0].message.content.strip()
    toks = resp.usage.total_tokens if getattr(resp,"usage",None) else 0
    return out, toks

def safe_json(s):
    try: return json.loads(s)
    except: pass
    if isinstance(s,str) and "{" in s and "}" in s:
        chunk = s[s.find("{"):s.rfind("}")+1]
        try: return json.loads(chunk)
        except: return None
    return None

def tokens(v):
    if v is None: return set()
    if isinstance(v,(list,tuple)):
        bag=[]; [bag.extend(str(x).lower().split()) for x in v]; return set(bag)
    if isinstance(v,dict):
        bag=[]; [bag.extend(str(k).lower().split()+str(val).lower().split()) for k,val in v.items()]; return set(bag)
    return set(str(v).lower().split())

def f1(a,b):
    if not a and not b: return 1.0
    if not a or not b: return 0.0
    inter=len(a&b); p=inter/len(a) if a else 0; r=inter/len(b) if b else 0
    return 0 if (p+r)==0 else 2*p*r/(p+r)

def score_json(pred,ref):
    scores=[f1(tokens(pred.get(k)), tokens(ref.get(k))) for k in FIELDS]
    base=sum(scores)/len(scores)
    bonus=0.05 if all(k in pred for k in FIELDS) else 0.0
    return min(base+bonus,1.0)

def compute_reward(out, ref_str):
    ref=json.loads(ref_str); pred=safe_json(out)
    if pred is None: return 0.0
    return score_json(pred,ref)

def ensure_csv_header():
    if not os.path.exists(CSV_LOG):
        with open(CSV_LOG,"w",newline="",encoding="utf-8") as f:
            csv.writer(f).writerow(["iteration","phase","prompt_id","parent_id","category","intent",
                "example_count","temperature","two_pass","input","reference",
                "output","tokens","reward","prompt_template"])

def log_row(**kw):
    with open(CSV_LOG,"a",newline="",encoding="utf-8") as f:
        csv.writer(f).writerow([kw["iteration"],kw["phase"],kw["prompt_id"],kw["parent_id"],
            kw["category"],kw["intent"],kw["example_count"],kw["temperature"],kw["two_pass"],
            kw["input"],kw["reference"],kw["output"],kw["tokens"],kw["reward"],kw["prompt_template"]])

# Two-pass prompts
PASS1_SYS="Return ONLY JSON with keys: intent, entities, constraints, urgency. No steps. Unknown→null."
PASS2_SYS="Given structured fields, return ONLY JSON with keys: intent, entities, constraints, urgency, steps. Copy fields verbatim; generate 3–5 action steps."

def run_two_pass(user_text,temp):
    msgs1=[{"role":"system","content":PASS1_SYS},{"role":"user","content":user_text}]
    out1,toks1=call_llm(msgs1,temp); slots=safe_json(out1)
    if slots is None: return out1,toks1
    msgs2=[{"role":"system","content":PASS2_SYS},{"role":"user","content":"Fields:\n"+json.dumps(slots,ensure_ascii=False)}]
    out2,toks2=call_llm(msgs2,temp); return out2,toks1+toks2

def select_arm(arms,stats):
    if random.random()<EPSILON: return random.choice(arms)
    best=None; best_val=-1e9
    for a in arms:
        val=stats.avg(a)+(0.05 if stats.counts.get(a.key(),0)==0 else 0.0)
        if val>best_val: best,best_val=a,val
    return best

def main():
    ensure_csv_header()
    arms=enumerate_arms()
    for itr in range(1,ITERATIONS+1):
        arm=select_arm(arms,bandit); pv=prompt_variants[arm.prompt_id]
        sample=random.choice(DATASET)
        if arm.two_pass: out,toks=run_two_pass(sample["input"],arm.temperature)
        else: out,toks=call_llm(build_messages(pv.template,sample["input"],arm.example_count),arm.temperature)
        reward=compute_reward(out,sample["reference"]); bandit.update(arm,reward)
        log_row(iteration=itr,phase="bandit",prompt_id=pv.prompt_id,parent_id=pv.parent_id or "",
            category=pv.category,intent=pv.intent,example_count=arm.example_count,
            temperature=arm.temperature,two_pass=int(arm.two_pass),input=sample["input"],
            reference=sample["reference"],output=out,tokens=toks,reward=round(reward,6),
            prompt_template=pv.template)
        if itr%10==0: print(f"[iter {itr}] best_avg≈{max(v/max(1,c) for (k,v),c in zip(bandit.totals.items(), bandit.counts.values())):.3f}")
    print("✅ Done. Log→",CSV_LOG)

if __name__=="__main__": main()

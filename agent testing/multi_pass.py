"""
simple_shot_bandit_f1_ucb_no_abandon.py

UCB1 bandit over (shots x passes) combinations using 1-pass and 2-pass prompting.
NO ARM ABANDONMENT.
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

MODEL_NAME = "gpt-4.1-mini"
RANDOM_SEED = 123
RUN_ID = str(uuid.uuid4())
UCB_C = 1.0
N_TRIALS = 120

random.seed(RANDOM_SEED)

# ---- Arms: (shots x passes) ----

SHOT_VALUES = [0,1,2,3]
PASS_VALUES = [1,2]
ARMS = [{"shots": s, "passes": p} for p in PASS_VALUES for s in SHOT_VALUES]
N_ARMS = len(ARMS)

# ---- CSV Paths ----

CSV_PATH_TRIALS  = "bandit_no_abandon_trials.csv"
CSV_PATH_SUMMARY = "bandit_no_abandon_summary.csv"

# ---- FEW SHOT EXAMPLES ----

BASE_PROMPT = (
    "Extract a structured JSON plan from the user request.\n"
    "Return strict JSON only with keys: intent, entities, constraints, urgency, steps.\n"
    "No extra text."
)

FEW_SHOT = [
    {
        "input": "Book a train from Boston to New York tomorrow morning under $120.",
        "output": {
            "intent":"book_train",
            "entities":{"from":"Boston","to":"New York","date":"tomorrow morning","budget":"120"},
            "constraints":["budget<=120"],
            "urgency":"normal",
            "steps":["search_trains","filter_by_price","propose_options"]
        },
        "task_type":"travel"
    },
    {
        "input":"Schedule a 30 minute Zoom check-in with Maya next Tuesday after 3pm.",
        "output":{
            "intent":"schedule_meeting",
            "entities":{"participants":["Maya"],"duration":"30 minutes","time_window":"next Tuesday after 3pm","location":"Zoom"},
            "constraints":["include_zoom_link"],
            "urgency":"normal",
            "steps":["find_slot","create_zoom","send_invites"]
        },
        "task_type":"meeting"
    },
    {
        "input":"Order 4 vegan lunches for pickup at 1pm at 9 King St.",
        "output":{
            "intent":"order_food",
            "entities":{"headcount":4,"diet":"vegan","pickup_time":"1pm","address":"9 King St"},
            "constraints":["vegan_only"],
            "urgency":"time_sensitive",
            "steps":["choose_restaurants","filter_menu","place_order"]
        },
        "task_type":"food"
    },
    {
        "input":"Write a brief thank-you email to the interviewer and ask for feedback.",
        "output":{
            "intent":"draft_email",
            "entities":{"recipient":"interviewer","topic":"thank_you","extra_request":"feedback"},
            "constraints":["polite_tone","brief"],
            "urgency":"normal",
            "steps":["draft_email","review_tone","send_or_copy"]
        },
        "task_type":"email"
    },
]

# -------------------- CSV Init ---------------------------

def init_csvs():
    with open(CSV_PATH_TRIALS,"w",newline="",encoding="utf-8") as f:
        csv.writer(f).writerow([
            "run_id","trial","timestamp","model","ucb_c","seed",
            "task_index","task_type","arm","shots","passes",
            "prompt_length","reward_overall",
            "F1_intent","F1_entities","F1_constraints","F1_urgency","F1_steps",
            "latency_sec","error","raw_output",
        ])

    with open(CSV_PATH_SUMMARY,"w",newline="",encoding="utf-8") as f:
        csv.writer(f).writerow([
            "run_id","model","ucb_c","seed",
            "arm","shots","passes",
            "final_q","pulls","pull_freq",
            "lcb","ucb","bandwidth","converged"
        ])

def append(path,row):
    with open(path,"a",newline="",encoding="utf-8") as f:
        csv.writer(f).writerow(row)

# ------------------- F1 scoring --------------------------

STEP_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")

def flatten(v):
    out=[]
    def walk(x):
        if x is None: return
        if isinstance(x,(str,int,float,bool)):
            out.append(str(x).lower().strip())
        elif isinstance(x,list):
            for e in x: walk(e)
        elif isinstance(x,dict):
            for e in x.values(): walk(e)
        else:
            out.append(str(x).lower().strip())
    walk(v)
    return list(set(out))

def f1_items(pred,truth):
    p,t=set(pred),set(truth)
    if not p and not t: return 1
    if not p or not t: return 0
    inter=len(p&t)
    if inter==0: return 0
    prec=inter/len(p)
    rec =inter/len(t)
    return 2*prec*rec/(prec+rec) if prec+rec>0 else 0

def jaccard(a,b):
    a,b=set(a),set(b)
    if not a and not b: return 1
    if not a or not b: return 0
    return len(a&b)/len(a|b)

def norm_step(x): return STEP_TOKEN_RE.findall(str(x).lower())

def f1_steps(pred_list,true_list):
    if not isinstance(true_list,list):  true_list=[true_list] if true_list else []
    if not isinstance(pred_list,list):  pred_list=[pred_list] if pred_list else []
    if not pred_list and not true_list: return 1
    if not pred_list or  not true_list: return 0

    true_tok=[norm_step(s) for s in true_list]
    pred_tok=[norm_step(s) for s in pred_list]

    matched=set()
    hits=0
    for p in pred_tok:
        best=0; best_i=None
        for i,t in enumerate(true_tok):
            if i in matched: continue
            sim=jaccard(p,t)
            if sim>best: best=sim; best_i=i
        if best>=0.6 and best_i is not None:
            matched.add(best_i); hits+=1

    prec = hits/len(pred_tok)
    rec  = hits/len(true_tok)
    return 2*prec*rec/(prec+rec) if prec+rec>0 else 0

MISSING_PEN = {
    "intent":0.20,"entities":0.15,"constraints":0.10,"urgency":0.10,"steps":0.20
}

def score(pred,true):
    try:
        intent = f1_items(flatten(pred.get("intent")), flatten(true.get("intent")))
        ents   = f1_items(flatten(pred.get("entities")), flatten(true.get("entities")))
        cons   = f1_items(flatten(pred.get("constraints")), flatten(true.get("constraints")))
        urg    = f1_items(flatten(pred.get("urgency")), flatten(true.get("urgency")))
        steps  = f1_steps(pred.get("steps"), true.get("steps"))
        base   = (intent+ents+cons+urg+steps)/5
    except:
        return dict(reward_overall=0,F1_intent=0,F1_entities=0,F1_constraints=0,F1_urgency=0,F1_steps=0)

    penalty=sum(w for k,w in MISSING_PEN.items() if k not in pred)
    rw = max(0, base - penalty)

    return dict(
        reward_overall=rw,
        F1_intent=intent, F1_entities=ents,
        F1_constraints=cons, F1_urgency=urg,
        F1_steps=steps,
    )

# ------------------ Prompt building ----------------------

def make_prompt(task_i, shots):
    task=FEW_SHOT[task_i]

    idx=[i for i in range(len(FEW_SHOT)) if i!=task_i]
    random.shuffle(idx)
    demos=idx[:shots]

    parts=[BASE_PROMPT]
    for j in demos:
        ex=FEW_SHOT[j]
        parts.append("Example:\nUser: "+ex["input"]+"\nIdeal JSON:\n"+json.dumps(ex["output"]))
    parts.append("\nTask:\n"+task["input"]+"\nReturn JSON only.")
    return "\n\n".join(parts)

def make_refine_prompt(task_i, draft):
    task=FEW_SHOT[task_i]
    return (
        BASE_PROMPT
        + "\nDraft JSON:\n" + draft +
        "\nClean up and return final JSON only.\n\nOriginal request:\n"
        + task["input"]
    )

# --------------------- Model Calls -----------------------

def call_api(prompt):
    start=time.time()
    try:
        r=client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role":"user","content":prompt}],
            temperature=0.0,
            max_tokens=500,
        )
        return dict(output=r.choices[0].message.content or "",
                    latency=time.time()-start,
                    error="")
    except Exception as e:
        return dict(output="", latency=time.time()-start, error=str(e))

def run_trial(passes, shots, task_i):
    p1 = make_prompt(task_i, shots)
    r1 = call_api(p1)

    if passes==1 or r1["error"]:
        final = r1["output"]
        lat   = r1["latency"]
        plen  = len(p1)
        err   = r1["error"]
    else:
        p2 = make_refine_prompt(task_i, r1["output"])
        r2 = call_api(p2)
        final = r2["output"]
        lat   = r1["latency"] + r2["latency"]
        plen  = len(p1) + len(p2)
        err   = r1["error"] or r2["error"]

    if err:
        return dict(prompt_length=plen, latency_sec=lat, error=err,
                    raw_output="", reward_overall=0,
                    F1_intent=0, F1_entities=0, F1_constraints=0,
                    F1_urgency=0, F1_steps=0)

    try:
        js=json.loads(final)
        if not isinstance(js,dict): raise ValueError("not dict")
        sc=score(js, FEW_SHOT[task_i]["output"])
    except:
        sc=dict(reward_overall=0,F1_intent=0,F1_entities=0,
                F1_constraints=0,F1_urgency=0,F1_steps=0)

    return dict(prompt_length=plen, latency_sec=lat, error="",
                raw_output=final, **sc)

# ----------------------- UCB1 ----------------------------

def pick_arm(q, n, t):
    """ No abandonment → standard UCB1 """
    # untried first
    for i in range(N_ARMS):
        if n[i]==0: return i

    best=-999; best_i=None
    for i in range(N_ARMS):
        bonus = UCB_C * math.sqrt(2*math.log(t)/n[i])
        val = q[i] + bonus
        if val>best:
            best=val; best_i=i
    return best_i

# ------------------------ MAIN ---------------------------

def main():
    init_csvs()

    q=[0.0]*N_ARMS
    n=[0]*N_ARMS

    for t in range(1, N_TRIALS+1):
        task_i=(t-1)%len(FEW_SHOT)

        arm = pick_arm(q,n,t)
        shots  = ARMS[arm]["shots"]
        passes = ARMS[arm]["passes"]

        ts=datetime.utcnow().isoformat()
        res=run_trial(passes, shots, task_i)

        # update q-values
        n[arm]+=1
        lr=1/n[arm]
        q[arm] += lr*(res["reward_overall"] - q[arm])

        append(CSV_PATH_TRIALS,[
            RUN_ID,t,ts,MODEL_NAME,UCB_C,RANDOM_SEED,
            task_i, FEW_SHOT[task_i]["task_type"],
            arm, shots, passes,
            res["prompt_length"],
            res["reward_overall"],
            res["F1_intent"],res["F1_entities"],res["F1_constraints"],
            res["F1_urgency"],res["F1_steps"],
            res["latency_sec"],res["error"],res["raw_output"]
        ])

        print(f"[{t:03d}] arm={arm} shots={shots} passes={passes} "
              f"R={res['reward_overall']:.3f} Q={q[arm]:.3f} n={n[arm]}")

    # Summary
    for i,a in enumerate(ARMS):
        if n[i]>0:
            bonus = UCB_C*math.sqrt(2*math.log(N_TRIALS)/n[i])
            lcb   = q[i]-bonus
            ucb   = q[i]+bonus
            bw    = ucb-lcb
            conv  = bw<0.20
        else:
            lcb=ucb=bw=0; conv=False

        append(CSV_PATH_SUMMARY,[
            RUN_ID,MODEL_NAME,UCB_C,RANDOM_SEED,
            i,a["shots"],a["passes"],
            q[i],n[i],n[i]/N_TRIALS,
            lcb,ucb,bw,conv
        ])

    print("\nFinal Q-values:")
    for i,a in enumerate(ARMS):
        print(f" arm={i} shots={a['shots']} passes={a['passes']} "
              f"Q={q[i]:.3f} n={n[i]}")

if __name__=="__main__":
    main()

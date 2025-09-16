import os, json, numpy as np, pandas as pd, matplotlib.pyplot as plt

LOG="bandit_fewshot_agent_log_two_pass.csv"; OUT="agent_plots_two_pass"; RES="results_two_pass"
os.makedirs(OUT,exist_ok=True); os.makedirs(RES,exist_ok=True)
FIELDS=["intent","entities","constraints","urgency","steps"]

def safe_json(s):
    if not isinstance(s,str): return None
    try: return json.loads(s)
    except: pass
    if "{" in s and "}" in s:
        chunk=s[s.find("{"):s.rfind("}")+1]
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

def main():
    df=pd.read_csv(LOG,encoding="utf-8-sig")
    for c in ["iteration","example_count","temperature","two_pass","reward","tokens"]:
        df[c]=pd.to_numeric(df[c],errors="coerce")
    if "reward_per_1k" not in df: df["reward_per_1k"]=np.where(df["tokens"]>0, df["reward"]/(df["tokens"]/1000.0), np.nan)
    rows=[]
    for _,r in df.iterrows():
        pred,ref=safe_json(str(r["output"])),safe_json(str(r["reference"]))
        if pred is None or ref is None: sc={f"f1_{k}":0.0 for k in FIELDS}; valid=0
        else: sc={f"f1_{k}":f1(tokens(pred.get(k)),tokens(ref.get(k))) for k in FIELDS}; valid=1
        rows.append({"two_pass":r["two_pass"],"reward":r["reward"],"tokens":r["tokens"],"reward_per_1k":r["reward_per_1k"],"json_valid":valid,**sc})
    S=pd.DataFrame(rows); S["f1_mean"]=S[[c for c in S.columns if c.startswith("f1_")]].mean(1)
    by=S.groupby("two_pass")[["reward","reward_per_1k","json_valid","f1_mean"]+[f"f1_{k}" for k in FIELDS]].mean().reset_index()
    by.to_csv(os.path.join(RES,"by_two_pass.csv"),index=False); print(by.round(3))

    def bar(metric,title,fname):
        plt.bar(["1-pass","2-pass"],by[metric]); plt.title(title); plt.ylabel(metric); plt.savefig(os.path.join(OUT,fname)); plt.close()
    bar("reward","Mean Reward","mean_reward.png"); bar("f1_mean","Mean F1","mean_f1.png")
    bar("reward_per_1k","Efficiency","eff.png"); bar("json_valid","Validity","valid.png")

if __name__=="__main__": main()

import os, json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
#new log
LOG = "bandit_fewshot_agent_log.csv"
OUT = "agent_plots"
os.makedirs(OUT, exist_ok=True)

# ---------- load ----------
try:
    df = pd.read_csv(LOG, encoding="utf-8-sig")
except UnicodeDecodeError:
    df = pd.read_csv(LOG, encoding="latin-1")

# hygiene & types
df["iteration"] = pd.to_numeric(df["iteration"], errors="coerce").fillna(0).astype(int)
df["example_count"] = pd.to_numeric(df["example_count"], errors="coerce").fillna(0).astype(int)
df["temperature"] = pd.to_numeric(df["temperature"], errors="coerce")
df["reward"] = pd.to_numeric(df["reward"], errors="coerce")
df["is_mutation"] = pd.to_numeric(df["is_mutation"], errors="coerce").fillna(0).astype(int)



def safe_json(s: str):
    if not isinstance(s, str):
        return None
    # try raw
    try:
        return json.loads(s)
    except:
        pass
    # try code fences or best-effort slice
    if "{" in s and "}" in s:
        chunk = s[s.find("{"):s.rfind("}")+1]
        try:
            return json.loads(chunk)
        except:
            return None
    return None

def tokens(v):
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

# ---------- per-field scoring ----------
rows = []
for _, r in df.iterrows():
    pred = safe_json(str(r["output"]))
    ref = safe_json(str(r["reference"]))
    if pred is None or ref is None:
        scores = {f"f1_{k}": 0.0 for k in FIELDS}
        valid = 0.0
    else:
        valid = 1.0
        scores = {
            "f1_intent":       f1(tokens(pred.get("intent")),       tokens(ref.get("intent"))),
            "f1_entities":     f1(tokens(pred.get("entities")),     tokens(ref.get("entities"))),
            "f1_constraints":  f1(tokens(pred.get("constraints")),  tokens(ref.get("constraints"))),
            "f1_urgency":      f1(tokens(pred.get("urgency")),      tokens(ref.get("urgency"))),
            "f1_steps":        f1(tokens(pred.get("steps")),        tokens(ref.get("steps"))),
        }
    rows.append({
        "iteration": r["iteration"],
        "example_count": r["example_count"],
        "temperature": r["temperature"],
        "category": r.get("category",""),
        "intent": r.get("intent",""),
        "prompt_id": r.get("prompt_id",""),
        "is_mutation": r.get("is_mutation",0),
        "reward": r.get("reward", None),
        "json_valid": valid,
        **scores
    })

score_df = pd.DataFrame(rows)
os.makedirs("results", exist_ok=True)
score_csv = "results/agent_field_scores.csv"
score_df.to_csv(score_csv, index=False)
print(f"✅ Saved per-field scores → {score_csv}")

sns.set_style("whitegrid")

def savefig(path):
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()

# ---------- summary table ----------
summary = score_df[[c for c in score_df.columns if c.startswith("f1_")]].describe().round(3)
print("\n📊 Per-field F1 summary:\n", summary)

# ---------- 1) Box: per-field distributions ----------
plt.figure(figsize=(10,6))
melt = score_df.melt(id_vars=[], value_vars=[c for c in score_df.columns if c.startswith("f1_")],
                     var_name="field", value_name="f1")
sns.boxplot(x="field", y="f1", data=melt)
plt.title("Per-field F1 Distributions")
plt.xlabel("Field"); plt.ylabel("F1")
savefig(os.path.join(OUT, "fields_box_distributions.png"))

# ---------- 2) Trends over iterations (avg per field) ----------
trend = (melt.join(score_df[["iteration"]])
              .groupby(["field","iteration"])["f1"].mean().reset_index())
plt.figure(figsize=(11,6))
sns.lineplot(x="iteration", y="f1", hue="field", data=trend, marker="o")
plt.title("Per-field F1 Trend over Iterations (mean)")
plt.xlabel("Iteration"); plt.ylabel("F1 (mean)")
savefig(os.path.join(OUT, "fields_trend_iterations.png"))

# ---------- 3) Few-shot effect per field ----------
plt.figure(figsize=(11,6))
melt_fs = melt.join(score_df[["example_count"]])
sns.barplot(x="example_count", y="f1", hue="field", data=melt_fs, ci=None)
plt.title("Few-shot Level vs Per-field F1 (mean)")
plt.xlabel("example_count"); plt.ylabel("F1 (mean)")
savefig(os.path.join(OUT, "fields_by_fewshot.png"))

# ---------- 4) Temperature effect per field ----------
plt.figure(figsize=(11,6))
melt_temp = melt.join(score_df[["temperature"]])
sns.barplot(x="temperature", y="f1", hue="field", data=melt_temp, ci=None)
plt.title("Temperature vs Per-field F1 (mean)")
plt.xlabel("temperature"); plt.ylabel("F1 (mean)")
savefig(os.path.join(OUT, "fields_by_temperature.png"))

# ---------- 5) JSON Validity by factors ----------
plt.figure(figsize=(7,5))
sns.barplot(x="example_count", y="json_valid", data=score_df, estimator="mean", ci=None)
plt.title("JSON Validity by Few-shot Level")
plt.xlabel("example_count"); plt.ylabel("Validity rate")
plt.ylim(0,1)
savefig(os.path.join(OUT, "json_valid_by_fewshot.png"))

plt.figure(figsize=(7,5))
sns.barplot(x="temperature", y="json_valid", data=score_df, estimator="mean", ci=None)
plt.title("JSON Validity by Temperature")
plt.ylabel("Validity rate"); plt.ylim(0,1)
savefig(os.path.join(OUT, "json_valid_by_temperature.png"))

# ---------- 6) Correlation: reward vs fields ----------
# (only if reward exists)
if "reward" in score_df.columns and score_df["reward"].notna().any():
    corr_cols = ["reward","json_valid","f1_intent","f1_entities","f1_constraints","f1_urgency","f1_steps"]
    corr = score_df[corr_cols].corr().round(2)
    plt.figure(figsize=(7,5))
    sns.heatmap(corr, annot=True, cmap="coolwarm", vmin=-1, vmax=1)
    plt.title("Correlation: Reward vs Per-field F1 & Validity")
    savefig(os.path.join(OUT, "corr_reward_vs_fields.png"))
    print("\n🔗 Reward correlations:\n", corr["reward"].sort_values(ascending=False))

print(f"\n✅ Plots saved in: {OUT}")

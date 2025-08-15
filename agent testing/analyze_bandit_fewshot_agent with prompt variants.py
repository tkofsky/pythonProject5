import os, json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

LOG = "bandit_fewshot_agent_log.csv"
PROMPTS_PATH = "prompt_variants.json"  # <— NEW
OUT = "agent_plots"
RES = "results"
os.makedirs(OUT, exist_ok=True)
os.makedirs(RES, exist_ok=True)

# ---------- load log ----------
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
# optional fields
for c in ["prompt_id","parent_id","category","intent","prompt_template"]:
    if c not in df.columns:
        df[c] = ""

# ---------- bring in prompt_variants.json metadata (NEW) ----------
if os.path.exists(PROMPTS_PATH):
    with open(PROMPTS_PATH, "r", encoding="utf-8") as f:
        pv_list = json.load(f)
    pv_df = pd.DataFrame(pv_list).rename(columns={
        "id": "prompt_id",
        "template": "prompt_template_base",
    })
    # merge base metadata (category/intent/template) by prompt_id
    df = df.merge(
        pv_df[["prompt_id", "category", "intent", "prompt_template_base"]],
        on="prompt_id", how="left", suffixes=("", "_basefile")
    )
    # prefer non-empty category/intent from log, otherwise from base file
    for col in ["category","intent"]:
        df[col] = df[col].where(df[col].astype(str).str.len() > 0, df[f"{col}_basefile"])
        if f"{col}_basefile" in df.columns:
            df.drop(columns=[f"{col}_basefile"], inplace=True)
else:
    # if JSON missing, still proceed
    df["prompt_template_base"] = ""

# ---------- JSON helpers ----------
FIELDS = ["intent","entities","constraints","urgency","steps"]

def safe_json(s: str):
    if isinstance(s, dict):  # tolerate already-parsed dicts
        return s
    if not isinstance(s, str):
        return None
    try:
        return json.loads(s)
    except:
        pass
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
invalid_rows = []
for _, r in df.iterrows():
    pred = safe_json(r["output"])
    ref  = safe_json(r["reference"])
    if pred is None or ref is None:
        scores = {f"f1_{k}": 0.0 for k in FIELDS}
        valid = 0.0
        invalid_rows.append({
            "iteration": r["iteration"],
            "prompt_id": r.get("prompt_id",""),
            "example_count": r.get("example_count",0),
            "temperature": r.get("temperature",None),
            "output": r.get("output",""),
            "reference": r.get("reference","")
        })
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
        "parent_id": r.get("parent_id",""),
        "is_mutation": r.get("is_mutation",0),
        "reward": r.get("reward", None),
        "json_valid": valid,
        **scores
    })

score_df = pd.DataFrame(rows)
score_csv = os.path.join(RES, "agent_field_scores.csv")
os.makedirs(RES, exist_ok=True)
score_df.to_csv(score_csv, index=False)
print(f"✅ Saved per-field scores → {score_csv}")

# also dump invalid JSON cases for inspection
if invalid_rows:
    bad_csv = os.path.join(RES, "invalid_json_rows.csv")
    pd.DataFrame(invalid_rows).to_csv(bad_csv, index=False)
    print(f"⚠️ Saved invalid-JSON examples → {bad_csv}")

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
if "reward" in score_df.columns and score_df["reward"].notna().any():
    corr_cols = ["reward","json_valid","f1_intent","f1_entities","f1_constraints","f1_urgency","f1_steps"]
    corr = score_df[corr_cols].corr().round(2)
    plt.figure(figsize=(7,5))
    sns.heatmap(corr, annot=True, cmap="coolwarm", vmin=-1, vmax=1)
    plt.title("Correlation: Reward vs Per-field F1 & Validity")
    savefig(os.path.join(OUT, "corr_reward_vs_fields.png"))
    print("\n🔗 Reward correlations:\n", corr["reward"].sort_values(ascending=False))

# ---------- 7) NEW: Arm-, Prompt-, and Category-level summaries ----------
# Arm = (prompt_id × few-shot × temp)
arm_cols = ["prompt_id","category","intent","example_count","temperature"]
arm_perf = (score_df
            .groupby(arm_cols, dropna=False)
            .agg(mean_reward=("reward","mean"),
                 valid_rate=("json_valid","mean"),
                 n=("reward","size"),
                 f1_intent=("f1_intent","mean"),
                 f1_entities=("f1_entities","mean"),
                 f1_constraints=("f1_constraints","mean"),
                 f1_urgency=("f1_urgency","mean"),
                 f1_steps=("f1_steps","mean"))
            .reset_index()
            .sort_values(["mean_reward","valid_rate","n"], ascending=[False,False,False]))
arm_csv = os.path.join(RES, "arm_performance.csv")
arm_perf.to_csv(arm_csv, index=False)
print(f"✅ Saved arm performance → {arm_csv}")

# Prompt-level (aggregate across few-shot/temps)
prompt_perf = (arm_perf
               .groupby(["prompt_id","category","intent"], dropna=False)
               .agg(mean_reward=("mean_reward","mean"),
                    best_reward=("mean_reward","max"),
                    valid_rate=("valid_rate","mean"),
                    runs=("n","sum"))
               .reset_index()
               .sort_values(["best_reward","mean_reward"], ascending=[False,False]))
prompt_csv = os.path.join(RES, "prompt_performance.csv")
prompt_perf.to_csv(prompt_csv, index=False)
print(f"✅ Saved prompt performance → {prompt_csv}")

# Category-level
cat_perf = (prompt_perf
            .groupby(["category"], dropna=False)
            .agg(mean_of_means=("mean_reward","mean"),
                 mean_best=("best_reward","mean"),
                 prompts=("prompt_id","nunique"))
            .reset_index()
            .sort_values("mean_best", ascending=False))
cat_csv = os.path.join(RES, "category_performance.csv")
cat_perf.to_csv(cat_csv, index=False)
print(f"✅ Saved category performance → {cat_csv}")

# ---------- 8) NEW: Template report (base vs mutated) ----------
tpl_report = (df
              .drop_duplicates("prompt_id")
              .sort_values("prompt_id")
              [["prompt_id","category","intent","prompt_template_base","prompt_template"]])
tpl_csv = os.path.join(RES, "prompt_templates_report.csv")
tpl_report.to_csv(tpl_csv, index=False)
print(f"🧾 Saved template report → {tpl_csv}")

# ---------- 9) NEW: quick prompt/category plots ----------
# Top 20 prompts by mean_reward (best arm per prompt)
top20 = (arm_perf.sort_values("mean_reward", ascending=False)
         .drop_duplicates(subset=["prompt_id"])
         .head(20))
if not top20.empty:
    plt.figure(figsize=(10,7))
    sns.barplot(data=top20, y="prompt_id", x="mean_reward", hue="category", dodge=False)
    plt.title("Top Prompt Variants by Mean Reward (best arm per prompt)")
    plt.xlabel("Mean Reward"); plt.ylabel("Prompt ID")
    plt.legend(title="Category", bbox_to_anchor=(1.02,1), loc="upper left")
    savefig(os.path.join(OUT, "top_prompts_mean_reward.png"))

    # heatmap for the single best prompt to visualize few-shot × temp
    best_pid = top20.iloc[0]["prompt_id"]
    hm = (arm_perf[arm_perf["prompt_id"]==best_pid]
          .pivot_table(index="example_count", columns="temperature", values="mean_reward"))
    if hm.size > 0:
        plt.figure(figsize=(6,4))
        sns.heatmap(hm, annot=True, fmt=".2f", vmin=0, vmax=1, cmap="viridis")
        plt.title(f"Arm Heatmap for {best_pid}")
        savefig(os.path.join(OUT, f"heatmap_{best_pid}.png"))

plt.figure(figsize=(7,5))
sns.barplot(data=cat_perf, x="category", y="mean_best")
plt.title("Category Performance (mean of per-prompt best rewards)")
plt.xlabel("Category"); plt.ylabel("Mean of Prompt Best Rewards")
savefig(os.path.join(OUT, "category_performance.png"))

print(f"\n✅ Plots saved in: {OUT}")

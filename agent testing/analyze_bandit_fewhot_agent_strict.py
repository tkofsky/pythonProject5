# analyze_bandit_strict.py
# Strict JSON analyzer for bandit logs:
# - Requires pure JSON only (no extra prose)
# - Requires exact keys: intent, entities, constraints, urgency, steps
# - Basic type checks (entities: dict or null; steps: list or null)
# - Sets f1_* = 0 when prediction is not strictly valid

import os, json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# ---------------- Settings ----------------
LOG_CANDIDATES = [
    "bandit_fewshot_agent_log.csv",
    "bandit_fewshot_agent_log_two_pass.csv"
]
OUT_PLOTS = "agent_plots_strict"
OUT_RESULTS = "results_strict"
os.makedirs(OUT_PLOTS, exist_ok=True)
os.makedirs(OUT_RESULTS, exist_ok=True)

FIELDS = ["intent","entities","constraints","urgency","steps"]

# ---------------- Helpers ----------------
def autodetect_log():
    for p in LOG_CANDIDATES:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"No log file found. Expected one of: {', '.join(LOG_CANDIDATES)}")

def tokens(v):
    if v is None:
        return set()
    if isinstance(v, (list, tuple)):
        bag = []
        for x in v:
            bag += str(x).lower().split()
        return set(bag)
    if isinstance(v, dict):
        bag = []
        for k, val in v.items():
            bag += str(k).lower().split()
            bag += str(val).lower().split()
        return set(bag)
    return set(str(v).lower().split())

def f1(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    inter = len(a & b)
    p = inter / len(a) if len(a) else 0.0
    r = inter / len(b) if len(b) else 0.0
    return 0.0 if (p + r) == 0 else 2 * p * r / (p + r)

def parse_json_strict(s: str):
    """
    Strict: the entire string must be a JSON object with exactly the 5 schema keys,
    and minimal type checks pass. Returns (obj, None) or (None, reason).
    """
    if not isinstance(s, str):
        return None, "not_string"
    s = s.strip()
    if not (s.startswith("{") and s.endswith("}")):
        return None, "not_pure_json"
    try:
        obj = json.loads(s)
    except Exception:
        return None, "json_load_failed"
    if not isinstance(obj, dict):
        return None, "not_an_object"
    if set(obj.keys()) != set(FIELDS):
        return None, "wrong_keys"
    ent = obj.get("entities", None)
    stp = obj.get("steps", None)
    if ent is not None and not isinstance(ent, dict):
        return None, "entities_not_object"
    if stp is not None and not isinstance(stp, list):
        return None, "steps_not_list"
    return obj, None

def parse_json_loose(s: str):
    """
    Loose parser for reference (and optional diagnostics):
    - Try json.loads
    - If fails and braces exist, slice from first '{' to last '}' and try again.
    """
    if not isinstance(s, str):
        return None
    try:
        return json.loads(s)
    except Exception:
        pass
    if "{" in s and "}" in s:
        chunk = s[s.find("{"): s.rfind("}") + 1]
        try:
            return json.loads(chunk)
        except Exception:
            return None
    return None

def savefig(path):
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()

# ---------------- Load ----------------
log_path = autodetect_log()
try:
    df = pd.read_csv(log_path, encoding="utf-8-sig")
except UnicodeDecodeError:
    df = pd.read_csv(log_path, encoding="latin-1")

# Hygiene
for c in ["iteration","example_count","temperature","reward","tokens","is_mutation","two_pass"]:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
if "is_mutation" not in df.columns:
    # infer mutation if prompt_id ends with __m####
    if "prompt_id" in df.columns:
        df["is_mutation"] = df["prompt_id"].astype(str).str.contains(r"__m\d+$").astype(int)
    else:
        df["is_mutation"] = 0

# ---------------- Strict scoring ----------------
rows = []
for _, r in df.iterrows():
    out_text = str(r.get("output", ""))
    ref_text = str(r.get("reference", ""))

    # Reference: loose is fine (your refs should load cleanly)
    ref_obj = parse_json_loose(ref_text)

    # Prediction: strict
    pred_obj, strict_err = parse_json_strict(out_text)

    json_valid_strict = 1.0 if (pred_obj is not None and ref_obj is not None) else 0.0

    if pred_obj is None or ref_obj is None:
        scores = {f"f1_{k}": 0.0 for k in FIELDS}
    else:
        scores = {
            "f1_intent":       f1(tokens(pred_obj.get("intent")),       tokens(ref_obj.get("intent"))),
            "f1_entities":     f1(tokens(pred_obj.get("entities")),     tokens(ref_obj.get("entities"))),
            "f1_constraints":  f1(tokens(pred_obj.get("constraints")),  tokens(ref_obj.get("constraints"))),
            "f1_urgency":      f1(tokens(pred_obj.get("urgency")),      tokens(ref_obj.get("urgency"))),
            "f1_steps":        f1(tokens(pred_obj.get("steps")),        tokens(ref_obj.get("steps"))),
        }

    rows.append({
        "iteration": r.get("iteration", None),
        "example_count": r.get("example_count", None),
        "temperature": r.get("temperature", None),
        "category": r.get("category",""),
        "intent": r.get("intent",""),
        "prompt_id": r.get("prompt_id",""),
        "is_mutation": r.get("is_mutation",0),
        "reward": r.get("reward", None),
        "json_valid_strict": json_valid_strict,
        "strict_error": strict_err if pred_obj is None else "",
        **scores
    })

score_df = pd.DataFrame(rows)
score_csv = os.path.join(OUT_RESULTS, "agent_field_scores_strict.csv")
score_df.to_csv(score_csv, index=False)
print(f"✅ Saved strict per-field scores → {score_csv}")

# ---------------- Summaries ----------------
sns.set_style("whitegrid")

# Headline validity
overall_valid = (score_df["json_valid_strict"].mean() * 100.0) if len(score_df) else float("nan")
print(f"\n=== Overall strict JSON validity: {overall_valid:.1f}% (n={len(score_df)}) ===")

# Validity by factors (with counts)
print("\n=== Strict validity by few-shot (with counts) ===")
fs = (score_df.groupby("example_count")
      .agg(n=("json_valid_strict","size"),
           strict=("json_valid_strict","mean")))
fs["strict%"] = (fs["strict"] * 100).round(1)
print(fs[["n","strict%"]].sort_index())

print("\n=== Strict validity by temperature (with counts) ===")
tt = (score_df.groupby("temperature")
      .agg(n=("json_valid_strict","size"),
           strict=("json_valid_strict","mean")))
tt["strict%"] = (tt["strict"] * 100).round(1)
print(tt[["n","strict%"]].sort_index())

# Common strict failure reasons (if any)
if "strict_error" in score_df.columns:
    err_counts = (score_df["strict_error"]
                  .replace("", pd.NA)
                  .dropna()
                  .value_counts())
    if not err_counts.empty:
        err_csv = os.path.join(OUT_RESULTS, "strict_failure_reasons.csv")
        err_counts.to_csv(err_csv, header=["count"])
        print("\n=== Strict failure reasons (top) ===")
        print(err_counts.head(10))
        print(f"(Full breakdown saved to {err_csv})")

# ---------------- Plots ----------------
# 1) Per-field F1 distributions (strict)
melt = score_df.melt(id_vars=[], value_vars=[c for c in score_df.columns if c.startswith("f1_")],
                     var_name="field", value_name="f1")
plt.figure(figsize=(10,6))
sns.boxplot(x="field", y="f1", data=melt)
plt.title("Per-field F1 Distributions (strict JSON only)")
plt.xlabel("Field"); plt.ylabel("F1")
savefig(os.path.join(OUT_PLOTS, "strict_fields_box_distributions.png"))

# 2) Trend over iterations (avg per field)
if score_df["iteration"].notna().any():
    trend = (melt.join(score_df[["iteration"]])
                .groupby(["field","iteration"])["f1"].mean().reset_index())
    plt.figure(figsize=(11,6))
    sns.lineplot(x="iteration", y="f1", hue="field", data=trend, marker="o")
    plt.title("Per-field F1 Trend over Iterations (strict, mean)")
    plt.xlabel("Iteration"); plt.ylabel("F1 (mean)")
    savefig(os.path.join(OUT_PLOTS, "strict_fields_trend_iterations.png"))

# 3) Few-shot effect per field
plt.figure(figsize=(11,6))
melt_fs = melt.join(score_df[["example_count"]])
sns.barplot(x="example_count", y="f1", hue="field", data=melt_fs, ci=None)
plt.title("Few-shot Level vs Per-field F1 (strict, mean)")
plt.xlabel("example_count"); plt.ylabel("F1 (mean)")
savefig(os.path.join(OUT_PLOTS, "strict_fields_by_fewshot.png"))

# 4) Temperature effect per field
plt.figure(figsize=(11,6))
melt_temp = melt.join(score_df[["temperature"]])
sns.barplot(x="temperature", y="f1", hue="field", data=melt_temp, ci=None)
plt.title("Temperature vs Per-field F1 (strict, mean)")
plt.xlabel("temperature"); plt.ylabel("F1 (mean)")
savefig(os.path.join(OUT_PLOTS, "strict_fields_by_temperature.png"))

# 5) Strict JSON Validity by factors (bars)
plt.figure(figsize=(7,5))
sns.barplot(x="example_count", y="json_valid_strict", data=score_df, estimator="mean", ci=None)
plt.title("Strict JSON Validity by Few-shot Level")
plt.xlabel("example_count"); plt.ylabel("Validity rate")
plt.ylim(0,1)
savefig(os.path.join(OUT_PLOTS, "strict_json_valid_by_fewshot.png"))

plt.figure(figsize=(7,5))
sns.barplot(x="temperature", y="json_valid_strict", data=score_df, estimator="mean", ci=None)
plt.title("Strict JSON Validity by Temperature")
plt.ylabel("Validity rate"); plt.ylim(0,1)
savefig(os.path.join(OUT_PLOTS, "strict_json_valid_by_temperature.png"))

# 6) Correlation: reward vs fields (strict)
if "reward" in score_df.columns and score_df["reward"].notna().any():
    corr_cols = ["reward","json_valid_strict","f1_intent","f1_entities","f1_constraints","f1_urgency","f1_steps"]
    corr = score_df[corr_cols].corr().round(2)
    plt.figure(figsize=(7,5))
    sns.heatmap(corr, annot=True, cmap="coolwarm", vmin=-1, vmax=1)
    plt.title("Correlation: Reward vs Per-field F1 & Strict Validity")
    savefig(os.path.join(OUT_PLOTS, "strict_corr_reward_vs_fields.png"))
    print("\n🔗 Reward correlations (strict):\n", corr["reward"].sort_values(ascending=False))

print(f"\n✅ Plots saved in: {OUT_PLOTS}")

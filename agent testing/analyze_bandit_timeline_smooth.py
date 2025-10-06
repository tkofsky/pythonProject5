# analyze_bandit_timeline_smooth.py
# Analyzer for bandit_fewshot_agent_log_two_pass.csv with:
# - Arm reconstruction
# - Strict & loose JSON validity
# - Per-field F1 (intent/entities/constraints/urgency/steps)
# - Smoothed timelines (reward & reward/1k)
# - Pulls-over-time
# - Summaries + correlation heatmap
# Run: python analyze_bandit_timeline_smooth.py

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ================= Settings =================
LOG_PATH = "bandit_fewshot_agent_log_two_pass.csv"  # fixed (two-pass only)
TOPN = 6                    # max arms to plot
WINDOW = 25                 # rolling window for moving average
EMA_WEIGHT = 0.90           # exponential smoother weight (higher = smoother)
MIN_PULLS = 5               # min pulls per arm to be included
NUM_BINS = 12               # for binned pulls plot
OUT_DIR_PLOTS = "agent_plots_smooth"
OUT_DIR_RESULTS = "results_smooth"

sns.set_style("whitegrid")

FIELDS = ["intent","entities","constraints","urgency","steps"]

# ================= Helpers =================
def ensure_dirs():
    os.makedirs(OUT_DIR_PLOTS, exist_ok=True)
    os.makedirs(OUT_DIR_RESULTS, exist_ok=True)

def rolling_mean(series, window):
    return series.rolling(window=window, min_periods=max(1, window//2)).mean()

def smooth(y, weight=0.9):
    if len(y) == 0:
        return y
    out = [y[0]]
    for v in y[1:]:
        out.append(out[-1] * weight + (1.0 - weight) * v)
    return np.asarray(out)

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

def safe_json(s: str):
    if not isinstance(s, str):
        return None
    try:
        return json.loads(s)
    except Exception:
        pass
    if "{" in s and "}" in s:
        chunk = s[s.find("{"): s.rfind("}")+1]
        try:
            return json.loads(chunk)
        except Exception:
            return None
    return None

def parse_json_strict(s: str):
    # pure JSON object + exact schema + minimal type checks
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

def rebuild_arm(df):
    # prompt_id | k | T | m | 2p
    for col in ["prompt_id","example_count","temperature","is_mutation","two_pass"]:
        if col not in df.columns:
            df[col] = np.nan
    def to_s(x):
        if pd.isna(x):
            return "NA"
        if isinstance(x, float):
            s = f"{x:.2f}".rstrip('0').rstrip('.')
            return s if s else "0"
        if isinstance(x, (np.integer,)):
            return str(int(x))
        try:
            return str(int(x))
        except Exception:
            return str(x)
    df["arm"] = (
        df["prompt_id"].astype(str)
        + " | k=" + df["example_count"].apply(to_s)
        + " | T=" + df["temperature"].apply(to_s)
        + " | m=" + df["is_mutation"].fillna(0).astype(int).astype(str)
        + " | 2p=" + df["two_pass"].fillna(0).astype(int).astype(str)
    )
    return df

def savefig(path):
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()

# ================= Load =================
def load_log():
    if not os.path.exists(LOG_PATH):
        raise FileNotFoundError(f"Log not found: {LOG_PATH}")
    try:
        df = pd.read_csv(LOG_PATH, encoding="utf-8-sig")
    except UnicodeDecodeError:
        df = pd.read_csv(LOG_PATH, encoding="latin-1")
    # Coerce numerics
    for c in ["iteration","example_count","temperature","reward","tokens","is_mutation","two_pass"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

# ================= Main =================
def main():
    ensure_dirs()
    df = load_log()

    # Rebuild arm label
    if "arm" not in df.columns:
        df = rebuild_arm(df)

    # Diagnostics
    print("\n[DIAG] Using log:", LOG_PATH)
    print("[DIAG] Columns:", list(df.columns))
    print("[DIAG] Rows:", len(df))
    for col in ["iteration","reward","tokens","arm","two_pass","example_count","temperature"]:
        if col in df.columns:
            nn = df[col].notna().sum()
            print(f"[DIAG] non-null {col}: {nn}")
        else:
            print(f"[DIAG] MISSING column: {col}")

    # Compute reward_per_1k
    if "reward" in df.columns and "tokens" in df.columns:
        df["reward_per_1k"] = np.where(
            (df["tokens"].notna()) & (df["tokens"] > 0),
            df["reward"] / (df["tokens"] / 1000.0),
            np.nan
        )
    else:
        df["reward_per_1k"] = np.nan

    # Sort by iteration
    if "iteration" in df.columns:
        df = df.sort_values("iteration")
    else:
        df["iteration"] = np.arange(len(df))

    # ========= Validity & F1 (strict validity + loose F1) =========
    rows = []
    for _, r in df.iterrows():
        out_text = str(r.get("output",""))
        ref_text = str(r.get("reference",""))

        ref_obj = safe_json(ref_text)  # refs should be clean strings
        pred_loose = safe_json(out_text)
        pred_strict, err = parse_json_strict(out_text)

        json_valid_loose  = 1.0 if (pred_loose  is not None and ref_obj is not None) else 0.0
        json_valid_strict = 1.0 if (pred_strict is not None and ref_obj is not None) else 0.0

        pred_for_f1 = pred_loose  # keep continuity with prior analyzers
        if pred_for_f1 is None or ref_obj is None:
            scores = {f"f1_{k}": 0.0 for k in FIELDS}
        else:
            scores = {
                "f1_intent":       f1(tokens(pred_for_f1.get("intent")),       tokens(ref_obj.get("intent"))),
                "f1_entities":     f1(tokens(pred_for_f1.get("entities")),     tokens(ref_obj.get("entities"))),
                "f1_constraints":  f1(tokens(pred_for_f1.get("constraints")),  tokens(ref_obj.get("constraints"))),
                "f1_urgency":      f1(tokens(pred_for_f1.get("urgency")),      tokens(ref_obj.get("urgency"))),
                "f1_steps":        f1(tokens(pred_for_f1.get("steps")),        tokens(ref_obj.get("steps"))),
            }

        rows.append({
            "iteration": r.get("iteration", np.nan),
            "arm": r.get("arm",""),
            "prompt_id": r.get("prompt_id",""),
            "example_count": r.get("example_count", np.nan),
            "temperature": r.get("temperature", np.nan),
            "is_mutation": r.get("is_mutation", 0),
            "two_pass": r.get("two_pass", 0),
            "reward": r.get("reward", np.nan),
            "tokens": r.get("tokens", np.nan),
            "reward_per_1k": r.get("reward_per_1k", np.nan),
            "json_valid_loose": json_valid_loose,
            "json_valid_strict": json_valid_strict,
            **scores
        })

    scored = pd.DataFrame(rows)
    scored_path = os.path.join(OUT_DIR_RESULTS, "scored_rows_smooth.csv")
    scored.to_csv(scored_path, index=False)
    print(f"✅ Saved scored rows → {scored_path}")

    # Overall validity headlines
    overall_loose = (scored["json_valid_loose"].mean() * 100.0) if len(scored) else float("nan")
    overall_strict = (scored["json_valid_strict"].mean() * 100.0) if len(scored) else float("nan")
    print(f"\n=== JSON Validity ===")
    print(f"Loose : {overall_loose:.1f}% (n={len(scored)})")
    print(f"Strict: {overall_strict:.1f}% (n={len(scored)})")

    # Validity by few-shot & temperature (with counts)
    def print_group_validity(colname, label):
        if colname not in scored.columns: return
        g = (scored.groupby(colname)
             .agg(n=("json_valid_strict","size"),
                  loose=("json_valid_loose","mean"),
                  strict=("json_valid_strict","mean"))
             .reset_index())
        g["loose%"]  = (g["loose"]  * 100).round(1)
        g["strict%"] = (g["strict"] * 100).round(1)
        print(f"\n=== Validity by {label} ===")
        print(g[[colname,"n","loose%","strict%"]].sort_values(colname))

    print_group_validity("example_count", "few-shot")
    print_group_validity("temperature", "temperature")

    # ================= Arm filtering for plots =================
    # Apply minimum pulls filter
    pull_counts = df.groupby("arm").size().reset_index(name="n_pulls")
    valid_arms = pull_counts[pull_counts["n_pulls"] >= MIN_PULLS]["arm"]
    df_plot = df[df["arm"].isin(valid_arms)].copy()

    if df_plot.empty:
        print(f"⚠️ No arms meet MIN_PULLS={MIN_PULLS}. Try lowering MIN_PULLS or run longer.")
        return

    # Rank arms (prefer reward_per_1k if available)
    if df_plot["reward_per_1k"].notna().any():
        arm_stats = (df_plot.groupby("arm")["reward_per_1k"]
                        .mean()
                        .sort_values(ascending=False)
                        .reset_index())
    elif df_plot["reward"].notna().any():
        arm_stats = (df_plot.groupby("arm")["reward"]
                        .mean()
                        .sort_values(ascending=False)
                        .reset_index())
    else:
        arm_stats = (df_plot.groupby("arm").size()
                        .sort_values(ascending=False)
                        .rename("n_pulls").reset_index())

    topN = min(TOPN, len(arm_stats))
    top_arms = arm_stats.head(topN)["arm"].tolist()
    arm_summary_path = os.path.join(OUT_DIR_RESULTS, "arm_summary_topN.csv")
    arm_stats.head(topN).to_csv(arm_summary_path, index=False)
    print(f"✅ Saved arm summary → {arm_summary_path}")

    # ================= Plots: Reward timeline (rolling + EMA) =================
    if df_plot["reward"].notna().any():
        plt.figure(figsize=(11,6))
        plotted = 0
        for arm in top_arms:
            sub = df_plot[df_plot["arm"] == arm]
            if sub.empty:
                continue
            s = sub.set_index("iteration")["reward"].sort_index()
            rm = rolling_mean(s, WINDOW)
            rm_clean = rm.dropna()
            if rm_clean.empty:
                continue
            y = smooth(rm_clean.values, weight=EMA_WEIGHT)
            x = rm_clean.index[:len(y)]
            plt.plot(x, y, label=arm)
            plotted += 1
        plt.title(f"Rolling mean (window={WINDOW}) + EMA reward for top {plotted} arms (min pulls ≥ {MIN_PULLS})")
        plt.xlabel("iteration"); plt.ylabel("reward"); plt.ylim(0, 1)
        if plotted > 0:
            plt.legend(loc="best", fontsize=8)
        savefig(os.path.join(OUT_DIR_PLOTS, "reward_timeline_topN.png"))
    else:
        print("[WARN] No usable reward data to plot reward timeline; skipping.")

    # ================= Plots: Reward per 1k timeline (rolling + EMA) =================
    if df_plot["reward_per_1k"].notna().any():
        plt.figure(figsize=(11,6))
        plotted = 0
        for arm in top_arms:
            sub = df_plot[df_plot["arm"] == arm]
            if sub.empty:
                continue
            s = sub.set_index("iteration")["reward_per_1k"].sort_index()
            rm = rolling_mean(s, WINDOW)
            rm_clean = rm.dropna()
            if rm_clean.empty:
                continue
            y = smooth(rm_clean.values, weight=EMA_WEIGHT)
            x = rm_clean.index[:len(y)]
            plt.plot(x, y, label=arm)
            plotted += 1
        plt.title(f"Rolling mean (window={WINDOW}) + EMA reward/1k for top {plotted} arms (min pulls ≥ {MIN_PULLS})")
        plt.xlabel("iteration"); plt.ylabel("reward per 1k tokens")
        if plotted > 0:
            plt.legend(loc="best", fontsize=8)
        savefig(os.path.join(OUT_DIR_PLOTS, "reward_per_1k_timeline_topN.png"))
    else:
        print("[WARN] No usable reward_per_1k data to plot efficiency timeline; skipping.")

    # ================= Plots: Pulls over time (binned counts) =================
    max_iter = int(df_plot["iteration"].max()) if df_plot["iteration"].notna().any() else 0
    bins = np.linspace(0, max_iter, num=NUM_BINS+1, dtype=int) if max_iter>0 else np.arange(0, NUM_BINS+1)
    plt.figure(figsize=(11,6))
    for arm in top_arms:
        sub = df_plot[df_plot["arm"] == arm]
        iters = sub["iteration"].dropna().values
        if len(iters) == 0:
            continue
        counts, edges = np.histogram(iters, bins=bins)
        centers = (edges[:-1] + edges[1:]) / 2.0
        plt.plot(centers, counts, marker="o", label=arm)
    plt.title(f"Arm pulls over time (binned counts) — min pulls ≥ {MIN_PULLS}, bins={NUM_BINS}")
    plt.xlabel("iteration bin center"); plt.ylabel("# pulls in bin")
    plt.legend(loc="best", fontsize=8)
    savefig(os.path.join(OUT_DIR_PLOTS, "pulls_over_time_topN.png"))

    # ================= Plots: Per-field F1 distributions =================
    melt = scored.melt(id_vars=[], value_vars=[c for c in scored.columns if c.startswith("f1_")],
                       var_name="field", value_name="f1")
    plt.figure(figsize=(10,6))
    sns.boxplot(x="field", y="f1", data=melt)
    plt.title("Per-field F1 Distributions (loose parse for F1)")
    plt.xlabel("Field"); plt.ylabel("F1")
    savefig(os.path.join(OUT_DIR_PLOTS, "fields_box_distributions.png"))

    # ================= Plots: Per-field F1 trend over iterations =================
    if scored["iteration"].notna().any():
        trend = (melt.join(scored[["iteration"]])
                      .groupby(["field","iteration"])["f1"].mean().reset_index())
        plt.figure(figsize=(11,6))
        sns.lineplot(x="iteration", y="f1", hue="field", data=trend, marker="o")
        plt.title("Per-field F1 Trend over Iterations (mean)")
        plt.xlabel("Iteration"); plt.ylabel("F1 (mean)")
        savefig(os.path.join(OUT_DIR_PLOTS, "fields_trend_iterations.png"))

    # ================= Plots: Validity bars (strict & loose) =================
    plt.figure(figsize=(7,5))
    sns.barplot(x="example_count", y="json_valid_strict", data=scored, estimator="mean", ci=None)
    plt.title("Strict JSON Validity by Few-shot Level")
    plt.xlabel("example_count"); plt.ylabel("Validity rate"); plt.ylim(0,1)
    savefig(os.path.join(OUT_DIR_PLOTS, "strict_json_valid_by_fewshot.png"))

    plt.figure(figsize=(7,5))
    sns.barplot(x="temperature", y="json_valid_strict", data=scored, estimator="mean", ci=None)
    plt.title("Strict JSON Validity by Temperature")
    plt.xlabel("temperature"); plt.ylabel("Validity rate"); plt.ylim(0,1)
    savefig(os.path.join(OUT_DIR_PLOTS, "strict_json_valid_by_temperature.png"))

    # ================= Correlation: Reward vs F1 & Validity =================
    corr_cols = ["reward","reward_per_1k","json_valid_loose","json_valid_strict",
                 "f1_intent","f1_entities","f1_constraints","f1_urgency","f1_steps"]
    corr_cols = [c for c in corr_cols if c in scored.columns]
    if corr_cols:
        corr = scored[corr_cols].corr().round(2)
        plt.figure(figsize=(8,6))
        sns.heatmap(corr, annot=True, cmap="coolwarm", vmin=-1, vmax=1)
        plt.title("Correlation: Reward & Reward/1k vs F1 & Validity")
        savefig(os.path.join(OUT_DIR_PLOTS, "corr_reward_vs_fields.png"))

    print("\n✅ Plots saved in:", OUT_DIR_PLOTS)
    print("✅ Results saved in:", OUT_DIR_RESULTS)

if __name__ == "__main__":
    main()


import os, json, math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

LOG = "bandit_fewshot_agent_log_two_pass.csv"
OUT = "agent_plots_two_pass_v2"
RES = "results_two_pass_v2"
os.makedirs(OUT, exist_ok=True)
os.makedirs(RES, exist_ok=True)

FIELDS = ["intent","entities","constraints","urgency","steps"]

# -------------------------- utils --------------------------
def safe_json(s):
    if not isinstance(s, str):
        return None
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

def tokens(v):
    if v is None: return set()
    if isinstance(v, (list, tuple)):
        bag = []
        for x in v: bag.extend(str(x).lower().split())
        return set(bag)
    if isinstance(v, dict):
        bag = []
        for k,val in v.items():
            bag.extend(str(k).lower().split())
            bag.extend(str(val).lower().split())
        return set(bag)
    return set(str(v).lower().split())

def f1(a, b):
    if not a and not b: return 1.0
    if not a or not b: return 0.0
    inter = len(a & b)
    p = inter/len(a) if a else 0.0
    r = inter/len(b) if b else 0.0
    return 0.0 if (p+r)==0 else 2*p*r/(p+r)

def score_fields(pred, ref):
    return {f"f1_{k}": f1(tokens(pred.get(k)), tokens(ref.get(k))) for k in FIELDS}

def savefig(path):
    plt.tight_layout()
    plt.savefig(path, dpi=140, bbox_inches="tight")
    plt.close()

# -------------------------- main --------------------------
def main():
    # ---------- load ----------
    try:
        df = pd.read_csv(LOG, encoding="utf-8-sig")
    except UnicodeDecodeError:
        df = pd.read_csv(LOG, encoding="latin-1")

    # ---------- hygiene ----------
    for c in ["iteration","example_count","temperature","two_pass","reward","tokens"]:
        if c not in df.columns:
            df[c] = np.nan
        df[c] = pd.to_numeric(df[c], errors="coerce")

    if "reward_per_1k" not in df.columns or df["reward_per_1k"].isna().all():
        df["reward_per_1k"] = np.where(df["tokens"]>0, df["reward"]/(df["tokens"]/1000.0), np.nan)

    # ---------- recompute per-field F1 + validity ----------
    rows = []
    for _, r in df.iterrows():
        pred = safe_json(str(r["output"]))
        ref  = safe_json(str(r["reference"]))
        if pred is None or ref is None:
            sc = {f"f1_{k}": 0.0 for k in FIELDS}
            valid = 0.0
        else:
            sc = score_fields(pred, ref)
            valid = 1.0
        rows.append({
            "iteration": r["iteration"],
            "prompt_id": r.get("prompt_id",""),
            "example_count": r.get("example_count", np.nan),
            "temperature": r.get("temperature", np.nan),
            "two_pass": r.get("two_pass", np.nan),
            "reward": r.get("reward", np.nan),
            "tokens": r.get("tokens", np.nan),
            "reward_per_1k": r.get("reward_per_1k", np.nan),
            "json_valid": valid,
            **sc
        })
    S = pd.DataFrame(rows)
    S["f1_mean"] = S[[c for c in S.columns if c.startswith("f1_")]].mean(axis=1)
    S.to_csv(os.path.join(RES, "scored_rows.csv"), index=False)

    # ---------- overall agg by two_pass ----------
    agg_cols = ["reward","reward_per_1k","json_valid","f1_mean"] + [f"f1_{k}" for k in FIELDS]
    by_tp_mean = S.groupby("two_pass")[agg_cols].mean().reset_index()
    by_tp_median = S.groupby("two_pass")[agg_cols].median().reset_index()
    by_tp_count = S.groupby("two_pass").size().reset_index(name="n")
    by_tp_mean.to_csv(os.path.join(RES, "overall_by_two_pass_mean.csv"), index=False)
    by_tp_median.to_csv(os.path.join(RES, "overall_by_two_pass_median.csv"), index=False)
    by_tp_count.to_csv(os.path.join(RES, "overall_by_two_pass_counts.csv"), index=False)

    # ---------- plots: overall ----------
    def bar_pair(vals, title, ylabel, fname, ylim=None):
        # expects index row for two_pass=0 and two_pass=1
        order = [0.0, 1.0]
        plot_vals = [float(by_tp_mean[by_tp_mean["two_pass"]==k][vals].values.squeeze()) if (by_tp_mean["two_pass"]==k).any() else np.nan for k in order]
        labels = ["1-pass","2-pass"]
        plt.figure(figsize=(6,4))
        plt.bar(labels, plot_vals)
        plt.title(title)
        plt.ylabel(ylabel)
        if ylim: plt.ylim(*ylim)
        savefig(os.path.join(OUT, fname))

    bar_pair("reward", "Mean Reward: 1-pass vs 2-pass", "reward", "mean_reward_by_two_pass.png", ylim=(0,1))
    bar_pair("f1_mean", "Mean F1: 1-pass vs 2-pass", "F1 (mean)", "mean_f1_by_two_pass.png", ylim=(0,1))
    bar_pair("reward_per_1k", "Efficiency (Reward per 1k tokens): 1-pass vs 2-pass", "reward/1k", "eff_reward_per_1k_by_two_pass.png")
    bar_pair("json_valid", "JSON Validity Rate: 1-pass vs 2-pass", "validity", "validity_by_two_pass.png", ylim=(0,1))

    # Per-field bar chart
    fields = [f"f1_{k}" for k in FIELDS]
    plt.figure(figsize=(9,5))
    x = np.arange(len(fields))
    width = 0.35
    row1 = [float(by_tp_mean[by_tp_mean["two_pass"]==0][f].values.squeeze()) if (by_tp_mean["two_pass"]==0).any() else 0.0 for f in fields]
    row2 = [float(by_tp_mean[by_tp_mean["two_pass"]==1][f].values.squeeze()) if (by_tp_mean["two_pass"]==1).any() else 0.0 for f in fields]
    plt.bar(x - width/2, row1, width=width, label="1-pass")
    plt.bar(x + width/2, row2, width=width, label="2-pass")
    plt.xticks(x, fields, rotation=15)
    plt.ylim(0,1)
    plt.title("Per-field F1: 1-pass vs 2-pass (means)")
    plt.legend()
    savefig(os.path.join(OUT, "per_field_f1_by_two_pass.png"))

    # ---------- paired (stratified) comparison holding (prompt_id, k, temp) constant ----------
    keys = ["prompt_id","example_count","temperature"]
    strat = S.groupby(keys + ["two_pass"])[agg_cols].mean().reset_index()

    def paired_delta(metric):
        p = strat.pivot_table(index=keys, columns="two_pass", values=metric)
        # Ensure columns 0.0 and 1.0 exist
        if 0.0 not in p.columns: p[0.0] = np.nan
        if 1.0 not in p.columns: p[1.0] = np.nan
        p = p[[0.0, 1.0]]
        p.columns = ["one_pass","two_pass"]
        p["delta"] = p["two_pass"] - p["one_pass"]
        p = p.reset_index()
        return p

    pivot_maps = {m: paired_delta(m) for m in agg_cols}
    for m, dfp in pivot_maps.items():
        dfp.to_csv(os.path.join(RES, f"paired_{m}.csv"), index=False)

    # Plot mean deltas across metrics
    delta_means = {m: float(pivot_maps[m]["delta"].mean()) for m in agg_cols}
    plt.figure(figsize=(8,4))
    names = list(delta_means.keys())
    vals = [delta_means[k] for k in names]
    plt.bar(range(len(names)), vals)
    plt.xticks(range(len(names)), names, rotation=20)
    plt.axhline(0, color="gray", linewidth=1)
    plt.title("Two-pass minus One-pass (mean delta across strata)")
    savefig(os.path.join(OUT, "paired_mean_deltas.png"))

    # Distribution of deltas for f1_steps
    steps_deltas = pivot_maps["f1_steps"]["delta"].dropna().values.tolist()
    plt.figure(figsize=(7,4))
    plt.hist(steps_deltas, bins=20)
    plt.title("Delta distribution: f1_steps (2-pass - 1-pass)")
    plt.xlabel("delta"); plt.ylabel("count")
    savefig(os.path.join(OUT, "delta_hist_f1_steps.png"))

    # ---------- interactions: few-shot × two_pass; temperature × two_pass; prompt × two_pass ----------
    inter_fs = S.groupby(["two_pass","example_count"])[["reward","reward_per_1k","json_valid","f1_mean"]].mean().reset_index()
    inter_fs.to_csv(os.path.join(RES, "interact_two_pass_by_fewshot.csv"), index=False)

    # plot reward_per_1k by few-shot
    plt.figure(figsize=(8,4))
    for tp, label in [(0.0,"1-pass"), (1.0,"2-pass")]:
        sub = inter_fs[inter_fs["two_pass"]==tp].sort_values("example_count")
        plt.plot(sub["example_count"], sub["reward_per_1k"], marker="o", label=label)
    plt.title("Reward/1k by few-shot level (1-pass vs 2-pass)")
    plt.xlabel("example_count"); plt.ylabel("reward/1k")
    plt.legend()
    savefig(os.path.join(OUT, "interact_reward_per_1k_by_fewshot.png"))

    inter_temp = S.groupby(["two_pass","temperature"])[["reward","reward_per_1k","json_valid","f1_mean"]].mean().reset_index()
    inter_temp.to_csv(os.path.join(RES, "interact_two_pass_by_temperature.csv"), index=False)

    plt.figure(figsize=(8,4))
    for tp, label in [(0.0,"1-pass"), (1.0,"2-pass")]:
        sub = inter_temp[inter_temp["two_pass"]==tp].sort_values("temperature")
        plt.plot(sub["temperature"], sub["json_valid"], marker="o", label=label)
    plt.title("JSON validity by temperature (1-pass vs 2-pass)")
    plt.xlabel("temperature"); plt.ylabel("validity")
    plt.legend()
    savefig(os.path.join(OUT, "interact_validity_by_temperature.png"))

    inter_prompt = S.groupby(["two_pass","prompt_id"])[["reward","reward_per_1k","f1_mean"]].mean().reset_index()
    inter_prompt.to_csv(os.path.join(RES, "interact_two_pass_by_prompt.csv"), index=False)

    # ---------- best arms per mode ----------
    arm_keys = ["prompt_id","example_count","temperature"]
    best_by_mode = (
        S.groupby(["two_pass"] + arm_keys)[["reward","reward_per_1k","f1_mean","json_valid"]]
        .mean()
        .reset_index()
    )

    # top-N by reward_per_1k for each mode
    topN = 10
    tops = []
    for mode in sorted(best_by_mode["two_pass"].dropna().unique()):
        sub = best_by_mode[best_by_mode["two_pass"]==mode].copy()
        sub = sub.sort_values(["reward_per_1k","reward","f1_mean"], ascending=False).head(topN)
        sub.insert(0, "mode", "2-pass" if mode==1.0 else "1-pass")
        tops.append(sub)
    if tops:
        tops_df = pd.concat(tops, ignore_index=True)
        tops_df.to_csv(os.path.join(RES, "best_arms_by_mode.csv"), index=False)

    print("✅ Analysis complete.")
    print(f"Plots  → {OUT}")
    print(f"Tables → {RES}")

if __name__ == "__main__":
    main()

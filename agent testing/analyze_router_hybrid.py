# %%
# analyze_router_hybrid.py
#
# Analyzer for bandit_fewshot_agent_router_hybrid.csv
# - Compares one-pass vs two-pass (router decisions)
# - Summaries: reward, reward/1k, JSON validity, per-field F1
# - Router feature diagnostics (what triggers two-pass)
# - Cohort deltas: within-arm lift when two_pass==1 vs two_pass==0
# - Saves plots and CSV summaries under agent_plots_router_hybrid/
#
# Usage:
#   python analyze_router_hybrid.py
#
import os, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

LOG = "bandit_fewshot_agent_log_router_hybrid.csv"
LOG = "bandit_fewshot_agent_log_router_hybrid_enhanced.csv"
log = "bandit_fewshot_agent_log_router_hybrid_steps_patch.csv"
log = "bandit_fewshot_agent_log_router_hybrid_steps_weighted.csv"
log = "bandit_fewshot_agent_log_router_hybrid_steps_weighted.csv"
OUT = "agent_plots_router_hybrid"
os.makedirs(OUT, exist_ok=True)

# --------- Load ---------
def load_log(path: str) -> pd.DataFrame:
    encodings = ["utf-8", "utf-8-sig", "latin-1"]
    last_err = None
    for enc in encodings:
        try:
            df = pd.read_csv(path, encoding=enc)
            return df
        except Exception as e:
            last_err = e
    raise last_err

df = load_log(LOG)

# Hygiene
num_cols = ["iteration","example_count","temperature","two_pass","is_mutation","tokens",
            "reward","reward_per_1k","json_valid",
            "f1_intent","f1_entities","f1_constraints","f1_urgency","f1_steps"]
for c in num_cols:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

if "router_features" in df.columns:
    # parse JSON dict if present
    def try_json(s):
        try:
            return json.loads(s) if isinstance(s, str) else {}
        except Exception:
            return {}
    feats = df["router_features"].apply(try_json)
    # expand keys we know about
    for k in ["len","has_num","has_money","has_date","plan_hint"]:
        df[f"feat_{k}"] = feats.apply(lambda d: d.get(k, np.nan))
else:
    for k in ["len","has_num","has_money","has_date","plan_hint"]:
        df[f"feat_{k}"] = np.nan

# Filter only bandit rows
if "phase" in df.columns:
    df = df[df["phase"]=="bandit"].copy()

# Derive arm id (prompt × k × T)
df["arm"] = df["prompt_id"].astype(str) + "|k=" + df["example_count"].astype(int).astype(str) + "|T=" + df["temperature"].round(2).astype(str)

# --------- Helpers ---------
sns.set_style("whitegrid")

def savefig(path):
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()

# --------- 1) Global summaries by router mode ---------
summary_cols = ["reward","reward_per_1k","json_valid","f1_intent","f1_entities","f1_constraints","f1_urgency","f1_steps"]
by_mode = df.groupby("router_mode")[summary_cols].agg(["mean","median","std","count"])
by_mode.to_csv(os.path.join(OUT, "summary_by_router_mode.csv"))
print("Saved:", os.path.join(OUT, "summary_by_router_mode.csv"))

# Plot: mean with CI
plt.figure(figsize=(11,5))
melt = df.melt(id_vars=["router_mode"], value_vars=summary_cols, var_name="metric", value_name="value")
sns.barplot(x="metric", y="value", hue="router_mode", data=melt, ci=95, capsize=.1)
plt.title("One-pass vs Two-pass: metric comparison")
plt.xlabel(""); plt.ylabel("value")
plt.xticks(rotation=30, ha="right")
savefig(os.path.join(OUT, "mode_metrics_bar.png"))

# --------- 2) Two-pass flag (actual) vs metrics ---------
plt.figure(figsize=(11,5))
melt2 = df.melt(id_vars=["two_pass"], value_vars=summary_cols, var_name="metric", value_name="value")
sns.barplot(x="metric", y="value", hue="two_pass", data=melt2, ci=95, capsize=.1)
plt.title("Two-pass (actual) vs metrics")
plt.xlabel(""); plt.ylabel("value")
plt.xticks(rotation=30, ha="right")
savefig(os.path.join(OUT, "two_pass_metrics_bar.png"))

# --------- 3) Router feature diagnostics ---------
# Probability of choosing two_pass by feature bucketing
def prob_two_pass(group):
    return pd.Series({
        "count": len(group),
        "two_pass_rate": np.nanmean(group["two_pass"]==1)
    })

# Length bins
df["len_bin"] = pd.cut(df["feat_len"], bins=[-np.inf,80,120,160,220, np.inf], labels=["≤80","81–120","121–160","161–220",">220"])
len_stats = df.groupby("len_bin").apply(prob_two_pass).reset_index()
len_stats.to_csv(os.path.join(OUT, "router_two_pass_by_len.csv"), index=False)

plt.figure(figsize=(7,4))
sns.barplot(x="len_bin", y="two_pass_rate", data=len_stats)
plt.ylim(0,1); plt.title("Router: P(two-pass) by input length")
plt.xlabel("input length (chars)"); plt.ylabel("P(two-pass)")
savefig(os.path.join(OUT, "router_two_pass_by_len.png"))

# Binary features
bin_feats = ["feat_has_num","feat_has_money","feat_has_date","feat_plan_hint"]
for b in bin_feats:
    tmp = df.groupby(b).apply(prob_two_pass).reset_index()
    tmp.to_csv(os.path.join(OUT, f"router_two_pass_by_{b}.csv"), index=False)
    plt.figure(figsize=(5,4))
    sns.barplot(x=b, y="two_pass_rate", data=tmp)
    plt.ylim(0,1); plt.title(f"Router: P(two-pass) by {b}")
    savefig(os.path.join(OUT, f"router_two_pass_by_{b}.png"))

# --------- 4) Within-arm lift: two_pass==1 vs two_pass==0 ---------
# For each arm, compute mean metrics for rows where two_pass==1 and two_pass==0, then delta
arm_metrics = []
for arm, g in df.groupby("arm"):
    g1 = g[g["two_pass"]==1]
    g0 = g[g["two_pass"]==0]
    if len(g1) >= 5 and len(g0) >= 5:  # min pulls on each side
        row = {"arm": arm, "n_two": len(g1), "n_one": len(g0)}
        for m in ["reward","reward_per_1k","json_valid","f1_steps","f1_constraints","f1_entities","f1_urgency","f1_intent"]:
            row[f"{m}_delta"] = g1[m].mean() - g0[m].mean()
        arm_metrics.append(row)
arm_delta = pd.DataFrame(arm_metrics)
arm_delta.to_csv(os.path.join(OUT, "arm_within_delta.csv"), index=False)
print("Saved:", os.path.join(OUT, "arm_within_delta.csv"))

# Plot distribution of deltas
if not arm_delta.empty:
    for m in ["reward","reward_per_1k","f1_steps","f1_constraints"]:
        plt.figure(figsize=(7,4))
        sns.histplot(arm_delta[f"{m}_delta"].dropna(), kde=True)
        plt.axvline(0, color="black", linestyle="--", linewidth=1)
        plt.title(f"Within-arm delta (two-pass minus one-pass): {m}")
        plt.xlabel("delta"); plt.ylabel("arms")
        savefig(os.path.join(OUT, f"delta_hist_{m}.png"))

    # Top arms by f1_steps_delta
    top_steps = arm_delta.sort_values("f1_steps_delta", ascending=False).head(10)
    top_steps.to_csv(os.path.join(OUT, "top_arms_by_f1_steps_delta.csv"), index=False)

# --------- 5) Timeline (bins) by router_mode ---------
NUM_BINS = 30
df_sorted = df.sort_values("iteration").copy()
df_sorted["bin"] = pd.cut(df_sorted["iteration"], bins=NUM_BINS, labels=False)

# mean reward per bin by router_mode
bin_stats = (df_sorted.groupby(["bin","router_mode"])["reward"]
             .mean().reset_index())
plt.figure(figsize=(10,5))
sns.lineplot(x="bin", y="reward", hue="router_mode", data=bin_stats, marker="o")
plt.title("Reward over time by router mode (binned)"); plt.xlabel("iteration bin"); plt.ylabel("mean reward")
savefig(os.path.join(OUT, "timeline_reward_by_mode.png"))

# two_pass rate over time
bin_rate = (df_sorted.groupby("bin")["two_pass"].mean().reset_index())
plt.figure(figsize=(8,4))
sns.lineplot(x="bin", y="two_pass", data=bin_rate, marker="o")
plt.ylim(0,1); plt.title("Two-pass utilization over time"); plt.xlabel("iteration bin"); plt.ylabel("share two-pass")
savefig(os.path.join(OUT, "timeline_two_pass_rate.png"))

# --------- 6) Overall top arms (counts & means) ---------
arm_summary = (df.groupby("arm")
                 .agg(count=("arm","size"),
                      mean_reward=("reward","mean"),
                      mean_reward_1k=("reward_per_1k","mean"),
                      mean_f1_steps=("f1_steps","mean"),
                      mean_valid=("json_valid","mean"))
                 .sort_values("count", ascending=False))
arm_summary.to_csv(os.path.join(OUT, "arm_summary.csv"))
#################################################################
# --------- 7) Confusion-style evaluation of routing quality ---------
# For each arm, compute whether two-pass yielded higher mean reward than one-pass.
conf_rows = []

for arm, g in df.groupby("arm"):
    g1 = g[g["two_pass"]==1]
    g0 = g[g["two_pass"]==0]
    if len(g1) >= 5 and len(g0) >= 5:
        mean_2 = g1["reward"].mean()
        mean_1 = g0["reward"].mean()
        better_two = mean_2 > mean_1
        # Classify each row
        for _, row in g.iterrows():
            if row["two_pass"] == 1:
                if better_two:
                    bucket = "2+"
                else:
                    bucket = "2-"
            else:
                if better_two:
                    bucket = "1-"
                else:
                    bucket = "1+"
            conf_rows.append({"arm": arm, "iteration": row["iteration"], "bucket": bucket})

conf_df = pd.DataFrame(conf_rows)

# Overall confusion-style summary
conf_summary = conf_df["bucket"].value_counts().rename_axis("bucket").reset_index(name="count")
conf_summary["pct"] = conf_summary["count"] / conf_summary["count"].sum() * 100
conf_summary.to_csv(os.path.join(OUT, "router_confusion_table.csv"), index=False)

plt.figure(figsize=(6,4))
sns.barplot(x="bucket", y="pct", data=conf_summary, order=["2+","2-","1+","1-"])
plt.title("Routing decision vs actual improvement")
plt.ylabel("% of pulls")
plt.xlabel("Router bucket")
savefig(os.path.join(OUT, "router_confusion_table.png"))



###################################################################
plt.figure(figsize=(11,6))
top_show = arm_summary.head(12).reset_index()
sns.barplot(y="arm", x="count", data=top_show)
plt.title("Top arms by pulls"); plt.xlabel("pulls"); plt.ylabel("arm")
savefig(os.path.join(OUT, "top_arms_by_pulls.png"))

print(f"✅ Analysis complete. See folder: {OUT}")

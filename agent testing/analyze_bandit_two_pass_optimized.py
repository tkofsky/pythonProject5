# ============================================================
# analyze_bandit_two_pass_optimized.py
# ============================================================
# Analyzes bandit_fewshot_agent_log_two_pass_optimized.csv
# Produces rolling-mean reward charts, validity, F1 trends,
# correlation heatmaps, convergence plots, and a vertical dashboard
# ============================================================

import os, json, datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------- CONFIG ----------------
LOG_FILE = "bandit_fewshot_agent_log_two_pass_optimized.csv"
#LOG_FILE  ="bandit_fewshot_agent_log_router_hybrid_steps_weighted.csv"
OUT_DIR = "plots_optimized"
RES_DIR = "results_optimized"
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(RES_DIR, exist_ok=True)

ROLLING_WINDOW = 20
TOP_N_ARMS = 10
sns.set_style("whitegrid")

# ---------------- LOAD DATA ----------------
print(f"📂 Loading log: {LOG_FILE}")
df = pd.read_csv(LOG_FILE)

df["iteration"] = pd.to_numeric(df["iteration"], errors="coerce").fillna(0).astype(int)
df["reward"] = pd.to_numeric(df["reward"], errors="coerce").fillna(0.0)
df["tokens_total"] = pd.to_numeric(df["tokens_total"], errors="coerce").fillna(1.0)
df["category"] = df["category"].astype(str)
df["valid_strict"] = pd.to_numeric(df["valid_strict"], errors="coerce").fillna(0.0)

# Compute reward per 1k tokens
df["reward_per_1k"] = df["reward"] / (df["tokens_total"] / 1000)

# ---------------- AGGREGATE STATS ----------------
summary = (
    df.groupby("category")[["reward","reward_per_1k","valid_strict",
        "f1_intent","f1_entities","f1_constraints","f1_urgency","f1_steps"]]
    .agg(["mean","std"])
)
summary.columns = ["_".join(c).strip() for c in summary.columns.values]
summary["RSI"] = 1 / (1 + summary["reward_std"])  # Reward Stability Index
summary = summary.sort_values("reward_mean", ascending=False)
summary.to_csv(os.path.join(RES_DIR, "summary_by_category.csv"))
print(f"✅ Summary saved → {RES_DIR}/summary_by_category.csv")

# ---------------- SMOOTHING HELPER ----------------
def rolling_mean_std(data, win=20):
    return data.rolling(win, min_periods=1).mean(), data.rolling(win, min_periods=1).std()

# ---------------- 1) REWARD TREND ----------------
plt.figure(figsize=(10,6))
for cat, grp in df.groupby("category"):
    grp_sorted = grp.sort_values("iteration")
    mean, std = rolling_mean_std(grp_sorted["reward"], ROLLING_WINDOW)
    plt.plot(grp_sorted["iteration"], mean, label=cat)
    plt.fill_between(grp_sorted["iteration"], mean-std, mean+std, alpha=0.15)
plt.title("Reward Trend by Category (Rolling Mean ±1σ)")
plt.xlabel("Iteration"); plt.ylabel("Reward")
plt.legend(); plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "reward_trend.png"), dpi=140); plt.close()

# ---------------- 2) REWARD PER 1K TOKENS ----------------
plt.figure(figsize=(10,6))
for cat, grp in df.groupby("category"):
    grp_sorted = grp.sort_values("iteration")
    mean, std = rolling_mean_std(grp_sorted["reward_per_1k"], ROLLING_WINDOW)
    plt.plot(grp_sorted["iteration"], mean, label=cat)
    plt.fill_between(grp_sorted["iteration"], mean-std, mean+std, alpha=0.15)
plt.title("Reward per 1K Tokens by Category (Rolling Mean ±1σ)")
plt.xlabel("Iteration"); plt.ylabel("Reward / 1K Tokens")
plt.legend(); plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "reward_per_1k.png"), dpi=140); plt.close()

# ---------------- 3) JSON VALIDITY (STRICT) ----------------
plt.figure(figsize=(10,6))
for cat, grp in df.groupby("category"):
    grp_sorted = grp.sort_values("iteration")
    mean, std = rolling_mean_std(grp_sorted["valid_strict"], ROLLING_WINDOW)
    plt.plot(grp_sorted["iteration"], mean, label=cat)
    plt.fill_between(grp_sorted["iteration"], mean-std, mean+std, alpha=0.15)
plt.title("JSON Validity (Strict) by Category (Rolling Mean ±1σ)")
plt.xlabel("Iteration"); plt.ylabel("Validity (strict)")
plt.ylim(0,1)
plt.legend(); plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "json_validity_strict.png"), dpi=140); plt.close()

# ---------------- 4) JSON VALIDITY (LOOSE) ----------------
# derive from JSON parse success in output_p2 field if exists
if "output_p2" in df.columns:
    df["valid_loose"] = df["output_p2"].apply(lambda x: 1.0 if isinstance(x, str) and "{" in x and "}" in x else 0.0)
else:
    df["valid_loose"] = 0.0
plt.figure(figsize=(10,6))
for cat, grp in df.groupby("category"):
    grp_sorted = grp.sort_values("iteration")
    mean, std = rolling_mean_std(grp_sorted["valid_loose"], ROLLING_WINDOW)
    plt.plot(grp_sorted["iteration"], mean, label=cat)
    plt.fill_between(grp_sorted["iteration"], mean-std, mean+std, alpha=0.15)
plt.title("JSON Validity (Loose) by Category (Rolling Mean ±1σ)")
plt.xlabel("Iteration"); plt.ylabel("Validity (loose)")
plt.ylim(0,1)
plt.legend(); plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "json_validity_loose.png"), dpi=140); plt.close()

# ---------------- 5) F1 FIELD TRENDS ----------------
FIELDS = ["f1_intent","f1_entities","f1_constraints","f1_urgency","f1_steps"]
plt.figure(figsize=(10,6))
for f in FIELDS:
    mean, std = rolling_mean_std(df[f], ROLLING_WINDOW)
    plt.plot(df["iteration"], mean, label=f)
    plt.fill_between(df["iteration"], mean-std, mean+std, alpha=0.1)
plt.title("F1 Field Trends (Rolling Mean ±1σ)")
plt.xlabel("Iteration"); plt.ylabel("F1")
plt.legend(); plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "f1_trends.png"), dpi=140); plt.close()

# ---------------- 6) CORRELATION HEATMAP ----------------
corr_cols = ["reward","reward_per_1k","valid_strict"] + FIELDS + ["tokens_total"]
corr = df[corr_cols].corr().round(2)
plt.figure(figsize=(8,6))
sns.heatmap(corr, annot=True, cmap="coolwarm", vmin=-1, vmax=1)
plt.title("Correlation: Reward vs F1s, Validity, Tokens")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "corr_heatmap.png"), dpi=140); plt.close()
corr.to_csv(os.path.join(RES_DIR, "correlations.csv"))

# ---------------- 7) PULLS / CONVERGENCE ----------------
plt.figure(figsize=(10,6))
pulls = df.groupby(["iteration","category"]).size().unstack(fill_value=0)
pulls = pulls.rolling(ROLLING_WINDOW, min_periods=1).mean()
pulls.plot(ax=plt.gca())
plt.title("Arm Pulls / Convergence (Rolling Mean)")
plt.xlabel("Iteration"); plt.ylabel("# of pulls (smoothed)")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "pulls_convergence.png"), dpi=140); plt.close()

# ---------------- 8) DASHBOARD COMPOSITE ----------------
fig, axs = plt.subplots(5, 1, figsize=(10,22))
plots = [
    ("Reward Trend", "reward_trend.png"),
    ("Reward / 1K Tokens", "reward_per_1k.png"),
    ("JSON Validity (Strict)", "json_validity_strict.png"),
    ("F1 Field Trends", "f1_trends.png"),
    ("Correlation Heatmap", "corr_heatmap.png"),
]
for i, (label, fname) in enumerate(plots):
    img = plt.imread(os.path.join(OUT_DIR, fname))
    axs[i].imshow(img); axs[i].axis("off"); axs[i].set_title(label, fontsize=12)
timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
fig.suptitle(f"Bandit Analysis Dashboard – {LOG_FILE}\nGenerated {timestamp}", fontsize=14, weight="bold")
plt.tight_layout(rect=[0, 0, 1, 0.98])
plt.savefig(os.path.join(OUT_DIR, "dashboard_summary.png"), dpi=150)
plt.close()
print(f"✅ Dashboard saved → {OUT_DIR}/dashboard_summary.png")

# ---------------- 9) PRINT SUMMARY ----------------
print("\n📊 === Summary Stats ===")
def topn(col):
    return summary[[c for c in summary.columns if col in c]].sort_values(f"{col}_mean", ascending=False).head(3)

print("\nTop 3 Categories by Reward:")
print(summary["reward_mean"].nlargest(3))
print("\nTop 3 Categories by Reward/1k Tokens:")
print(summary["reward_per_1k_mean"].nlargest(3))
print("\nTop 3 Categories by JSON Validity (Strict):")
print(summary["valid_strict_mean"].nlargest(3))
print("\nTop 3 Categories by Avg F1 (Overall):")
summary["F1_avg"] = summary[[f"f1_{f}_mean" for f in ["intent","entities","constraints","urgency","steps"]]].mean(axis=1)
print(summary["F1_avg"].nlargest(3))
print("\nReward Stability Index (RSI):")
print(summary["RSI"].sort_values(ascending=False).head(5))
print("\n✅ All plots and summaries saved in:", OUT_DIR, "and", RES_DIR)

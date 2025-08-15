import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ---- Locate the log file (tries both common paths) ----
CANDIDATES = ["bandit_fewshot_agent_log.csv", "multi_agent_log.csv"]
LOG = next((p for p in CANDIDATES if os.path.exists(p)), None)
if not LOG:
    raise FileNotFoundError(f"No log found. Looked for: {', '.join(CANDIDATES)}")

OUTDIR = "quick_plots"
os.makedirs(OUTDIR, exist_ok=True)

# ---- Load with encoding fallback ----
try:
    df = pd.read_csv(LOG, encoding="utf-8-sig")
except UnicodeDecodeError:
    df = pd.read_csv(LOG, encoding="latin-1")

# ---- Basic hygiene & types ----
num_before = len(df)
df = df.dropna(subset=["reward"])
if len(df) != num_before:
    print(f"⚠️ Dropped {num_before - len(df)} rows missing reward")

for col, t in [("iteration", int), ("example_count", int), ("is_mutation", int)]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(t)
for col in ["temperature", "reward"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# Create arm_id if not present
if "arm_key" in df.columns:
    df["arm_id"] = df["arm_key"].astype(str)
else:
    need = {"prompt_id","example_count","temperature"}
    if need.issubset(df.columns):
        df["arm_id"] = df.apply(lambda r: f"{r['prompt_id']}|k{r['example_count']}|t{r['temperature']}", axis=1)
    else:
        # Fallback: just use prompt text
        df["arm_id"] = df.get("prompt", df.get("prompt_template", "unknown")).astype(str)

print(f"\nLoaded: {LOG}  | rows={len(df)}")

# ---- 1) Average reward by category ----
if "category" in df.columns:
    cat_avg = df.groupby("category")["reward"].mean().sort_values(ascending=False)
    print("\n🏷️  Average reward by category:")
    print(cat_avg.round(3))
else:
    print("\n(no 'category' column found; skipping category summary)")
    cat_avg = None

# ---- 2) Best overall arm ----
arm_summary = df.groupby("arm_id").agg(
    avg_reward=("reward","mean"),
    n=("reward","count")
).sort_values("avg_reward", ascending=False)
best_arm = arm_summary.iloc[0]
print("\n🏆  Best overall arm:")
print(arm_summary.head(5).round(3))

# Save arm summary CSV
arm_summary_path = "results/quick_arm_summary.csv"
arm_summary.round(4).to_csv(arm_summary_path)
print(f"💾 Saved arm summary → {arm_summary_path}")

# ---- 3) Plots ----
sns.set_style("whitegrid")

# 3a) Overall reward trend (with rolling avg)
df_sorted = df.sort_values("iteration")
df_sorted["rolling_reward"] = df_sorted["reward"].rolling(window=10, min_periods=1).mean()

plt.figure(figsize=(10,5))
plt.plot(df_sorted["iteration"], df_sorted["reward"], alpha=0.3, label="reward (per run)")
plt.plot(df_sorted["iteration"], df_sorted["rolling_reward"], linewidth=2, label="rolling avg (win=10)")
plt.title("Reward over Iterations (overall)")
plt.xlabel("Iteration"); plt.ylabel("Reward"); plt.legend()
plt.tight_layout(); plt.savefig(os.path.join(OUTDIR, "01_reward_over_time.png"), dpi=140); plt.close()

# 3b) Reward trend by category (if available)
if "category" in df.columns:
    trend_cat = (df.groupby(["category","iteration"])["reward"]
                   .mean().reset_index().sort_values(["category","iteration"]))
    plt.figure(figsize=(10,6))
    sns.lineplot(x="iteration", y="reward", hue="category", data=trend_cat, marker="o")
    plt.title("Average Reward over Iterations by Category")
    plt.xlabel("Iteration"); plt.ylabel("Average Reward")
    plt.tight_layout(); plt.savefig(os.path.join(OUTDIR, "02_reward_by_category_over_time.png"), dpi=140); plt.close()

# 3c) Reward distribution by few-shot level
if "example_count" in df.columns:
    plt.figure(figsize=(8,5))
    sns.boxplot(x="example_count", y="reward", data=df)
    plt.title("Reward Distribution by Few-shot Level (example_count)")
    plt.xlabel("Few-shot examples"); plt.ylabel("Reward")
    plt.tight_layout(); plt.savefig(os.path.join(OUTDIR, "03_reward_by_fewshot_box.png"), dpi=140); plt.close()

# 3d) Reward distribution by temperature
if "temperature" in df.columns:
    plt.figure(figsize=(8,5))
    sns.boxplot(x="temperature", y="reward", data=df)
    plt.title("Reward Distribution by Temperature")
    plt.xlabel("Temperature"); plt.ylabel("Reward")
    plt.tight_layout(); plt.savefig(os.path.join(OUTDIR, "04_reward_by_temperature_box.png"), dpi=140); plt.close()

print(f"\n✅ Plots saved to: {OUTDIR}")
print("Done.")

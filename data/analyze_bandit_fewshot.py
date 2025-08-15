import os
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

LOG = "bandit_fewshot_log.csv"
OUTDIR = "plots"
os.makedirs(OUTDIR, exist_ok=True)

# --- Load with encoding fallback ---
try:
    df = pd.read_csv(LOG, encoding="utf-8-sig")
except UnicodeDecodeError:
    df = pd.read_csv(LOG, encoding="latin-1")

# --- Basic hygiene ---
num_rows_before = len(df)
df = df.dropna(subset=["reward"])
if len(df) < num_rows_before:
    print(f"⚠️ Dropped {num_rows_before - len(df)} rows with NaN reward")

# Force types
df["iteration"] = pd.to_numeric(df["iteration"], errors="coerce").fillna(0).astype(int)
df["example_count"] = pd.to_numeric(df["example_count"], errors="coerce").fillna(0).astype(int)
df["temperature"] = pd.to_numeric(df["temperature"], errors="coerce")
df["reward"] = pd.to_numeric(df["reward"], errors="coerce")
df["is_mutation"] = pd.to_numeric(df["is_mutation"], errors="coerce").fillna(0).astype(int)

# Derived columns
df["arm_id"] = df.apply(lambda r: f"{r['prompt_id']}|k{r['example_count']}|t{r['temperature']}", axis=1)

# --- Quick peek ---
print("\nColumns:", list(df.columns))
print("Rows:", len(df))
print("Iterations range:", df["iteration"].min(), "→", df["iteration"].max())

# ========== 0) WINNERS RIGHT AWAY ==========
# Average reward per arm
arm_avg = df.groupby("arm_id").agg(
    avg_reward=("reward","mean"),
    n=("reward","count"),
    example_count=("example_count","first"),
    temperature=("temperature","first"),
    prompt_id=("prompt_id","first"),
    category=("category","first"),
    intent=("intent","first"),
    prompt_template=("prompt_template","first"),
).sort_values("avg_reward", ascending=False)

# Best overall
best_overall = arm_avg.iloc[0]
print("\n🏆 BEST OVERALL ARM")
print(f"Arm ID: {best_overall.name}")
print(f"Avg Reward: {best_overall.avg_reward:.3f} | Chosen {best_overall.n} times")
print(f"Example Count: {best_overall.example_count} | Temp: {best_overall.temperature}")
print(f"Category: {best_overall.category} | Intent: {best_overall.intent}")
print(f"Prompt Template:\n{best_overall.prompt_template}")

# Best per few-shot level
print("\n🏅 BEST PER FEW-SHOT LEVEL")
for k, group in arm_avg.groupby("example_count"):
    top = group.iloc[0]
    print(f"\nExample Count = {k}")
    print(f"Arm ID: {top.name}")
    print(f"Avg Reward: {top.avg_reward:.3f} | Chosen {top.n} times")
    print(f"Temp: {top.temperature} | Category: {top.category} | Intent: {top.intent}")
    print(f"Prompt Template:\n{top.prompt_template}")

# ========== 1) GLOBAL ROLLING REWARD ==========
roll = df.sort_values("iteration").copy()
roll["rolling_reward"] = roll["reward"].rolling(window=10, min_periods=1).mean()

plt.figure(figsize=(10,5))
plt.plot(roll["iteration"], roll["rolling_reward"])
plt.title("Rolling Reward (window=10) — All Arms")
plt.xlabel("Iteration"); plt.ylabel("Reward")
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout(); plt.savefig(os.path.join(OUTDIR, "01_global_rolling_reward.png")); plt.close()

# ========== 2) REWARD DISTRIBUTIONS & TRENDS ==========
# By example_count
plt.figure(figsize=(8,5))
sns.boxplot(x="example_count", y="reward", data=df)
plt.title("Reward Distribution by Few-shot Level")
plt.xlabel("Few-shot examples"); plt.ylabel("Reward")
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout(); plt.savefig(os.path.join(OUTDIR, "02_reward_by_example_count_box.png")); plt.close()

# By temperature
plt.figure(figsize=(8,5))
sns.boxplot(x="temperature", y="reward", data=df)
plt.title("Reward Distribution by Temperature")
plt.xlabel("Temperature"); plt.ylabel("Reward")
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout(); plt.savefig(os.path.join(OUTDIR, "03_reward_by_temperature_box.png")); plt.close()

# By category
if "category" in df.columns:
    plt.figure(figsize=(9,5))
    sns.boxplot(x="category", y="reward", data=df)
    plt.title("Reward by Prompt Category")
    plt.xlabel("Category"); plt.ylabel("Reward")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout(); plt.savefig(os.path.join(OUTDIR, "04_reward_by_category_box.png")); plt.close()

# By intent
if "intent" in df.columns:
    plt.figure(figsize=(9,5))
    sns.boxplot(x="intent", y="reward", data=df)
    plt.title("Reward by Intent")
    plt.xlabel("Intent"); plt.ylabel("Reward")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout(); plt.savefig(os.path.join(OUTDIR, "05_reward_by_intent_box.png")); plt.close()

# Heatmap: example_count × temperature
pivot = df.pivot_table(index="example_count", columns="temperature", values="reward", aggfunc="mean")
plt.figure(figsize=(8,5))
sns.heatmap(pivot, annot=True, fmt=".3f", cmap="coolwarm")
plt.title("Avg Reward Heatmap — Few-shot Level × Temperature")
plt.tight_layout(); plt.savefig(os.path.join(OUTDIR, "06_heatmap_examplecount_temp.png")); plt.close()

# Trends by example_count
roll_ec = (df.groupby(["example_count","iteration"])["reward"]
             .mean().reset_index().sort_values(["example_count","iteration"]))
plt.figure(figsize=(10,6))
sns.lineplot(x="iteration", y="reward", hue="example_count", data=roll_ec, marker="o")
plt.title("Avg Reward over Iterations by Few-shot Level")
plt.xlabel("Iteration"); plt.ylabel("Avg Reward")
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout(); plt.savefig(os.path.join(OUTDIR, "07_trend_by_example_count.png")); plt.close()

# Trends by temperature
roll_temp = (df.groupby(["temperature","iteration"])["reward"]
               .mean().reset_index().sort_values(["temperature","iteration"]))
plt.figure(figsize=(10,6))
sns.lineplot(x="iteration", y="reward", hue="temperature", data=roll_temp, marker="o")
plt.title("Avg Reward over Iterations by Temperature")
plt.xlabel("Iteration"); plt.ylabel("Avg Reward")
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout(); plt.savefig(os.path.join(OUTDIR, "08_trend_by_temperature.png")); plt.close()

# Mutation effect
plt.figure(figsize=(7,5))
sns.boxplot(x=df["is_mutation"].map({0:"original",1:"mutated"}), y="reward", data=df)
plt.title("Mutation Effect on Reward")
plt.xlabel("Prompt Type"); plt.ylabel("Reward")
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout(); plt.savefig(os.path.join(OUTDIR, "09_mutation_effect_box.png")); plt.close()

print("\n🔬 Mutation effect (avg reward):")
print(df.groupby("is_mutation")["reward"].mean().rename(index={0:"original",1:"mutated"}))

print(f"\n✅ Plots saved in: {OUTDIR}")

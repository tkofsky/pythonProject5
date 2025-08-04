import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# =====================
# 1. LOAD DATA
# =====================
try:
    df = pd.read_csv("multi_agent_results.csv", encoding="utf-8")
except UnicodeDecodeError:
    # Fallback if there are encoding errors
    df = pd.read_csv("multi_agent_results.csv", encoding="latin-1")

# Ensure correct data types
df["iteration"] = df["iteration"].astype(int)
df["reward"] = df["reward"].astype(float)

# =====================
# 2. VISUALIZATIONS
# =====================

# Reward Trend (Rolling Average)
df["rolling_reward"] = df["reward"].rolling(window=10, min_periods=1).mean()
plt.figure(figsize=(10, 5))
plt.plot(df["iteration"], df["rolling_reward"], label="Rolling Reward (window=10)")
plt.xlabel("Iteration")
plt.ylabel("Reward")
plt.title("Reward Trend Over Iterations")
plt.legend()
plt.show()

# Reward by Intent
plt.figure(figsize=(8, 4))
sns.barplot(x="intent", y="reward", data=df, estimator="mean", errorbar=None)
plt.title("Average Reward by Intent")
plt.show()

# Reward by Category
plt.figure(figsize=(8, 4))
sns.barplot(x="category", y="reward", data=df, estimator="mean", errorbar=None)
plt.title("Average Reward by Category")
plt.show()

# =====================
# 3. BEST PROMPT ANALYSIS
# =====================

# Overall Best Prompt
best_prompts = df.groupby("prompt")["reward"].mean().sort_values(ascending=False)
print("\n=== TOP 5 BEST OVERALL PROMPTS ===")
print(best_prompts.head(5))

# Best Prompt by Intent
best_prompts_intent = df.groupby(["intent", "prompt"])["reward"].mean().reset_index()
best_per_intent = best_prompts_intent.loc[best_prompts_intent.groupby("intent")["reward"].idxmax()]
print("\n=== BEST PROMPTS BY INTENT ===")
print(best_per_intent)

# Best Prompt by Category
best_prompts_category = df.groupby(["category", "prompt"])["reward"].mean().reset_index()
best_per_category = best_prompts_category.loc[best_prompts_category.groupby("category")["reward"].idxmax()]
print("\n=== BEST PROMPTS BY CATEGORY ===")
print(best_per_category)

# Best Prompt (Recent Performance - last 20% iterations)
last_iterations = df["iteration"].max() * 0.8
recent_df = df[df["iteration"] >= last_iterations]
recent_best_prompts = recent_df.groupby("prompt")["reward"].mean().sort_values(ascending=False)
print("\n=== BEST PROMPTS (RECENT 20% OF ITERATIONS) ===")
print(recent_best_prompts.head(5))

# =====================
# 4. OPTIONAL: VISUALIZE BEST PROMPT TREND
# =====================
best_prompt = best_prompts.index[0]
plt.figure(figsize=(10, 4))
df[df["prompt"] == best_prompt].plot(x="iteration", y="reward", title=f"Reward Trend for Best Prompt:\n{best_prompt}", legend=False)
plt.xlabel("Iteration")
plt.ylabel("Reward")
plt.show()

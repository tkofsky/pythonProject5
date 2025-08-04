import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load bandit log
df = pd.read_csv("results/bandit_log.csv")

# Ensure directory for plots
os.makedirs("results/plots", exist_ok=True)

# Rolling reward (smoothed)
df["rolling_reward"] = df["reward"].rolling(window=5, min_periods=1).mean()

# 1️⃣ Plot Reward Progression
plt.figure(figsize=(12, 6))
plt.plot(df["iteration"], df["rolling_reward"], label="Rolling Reward (window=5)")
plt.xlabel("Iteration")
plt.ylabel("Reward")
plt.title("Reward Progression Over Iterations")
plt.legend()
plt.grid(True)
plt.savefig("results/plots/reward_progression.png")
plt.close()

# 2️⃣ Plot Number of Mutations Over Time
df["is_mutation"] = df["prompt"].duplicated(keep="first").astype(int)  # mark if prompt appeared later (mutation or reuse)
mutation_counts = df.groupby("iteration")["is_mutation"].sum().cumsum()

plt.figure(figsize=(12, 6))
plt.plot(mutation_counts, label="Cumulative Mutations")
plt.xlabel("Iteration")
plt.ylabel("Mutations")
plt.title("Prompt Mutations Over Time")
plt.legend()
plt.grid(True)
plt.savefig("results/plots/mutations_over_time.png")
plt.close()

# 3️⃣ Best Performing Prompt Over Time
best_prompts = df.groupby("iteration").apply(
    lambda x: x.loc[x["reward"].idxmax(), ["prompt", "reward"]]
).reset_index()

plt.figure(figsize=(12, 6))
plt.plot(best_prompts["iteration"], best_prompts["reward"], label="Best Prompt Reward")
plt.xlabel("Iteration")
plt.ylabel("Reward")
plt.title("Best Prompt Reward Over Iterations")
plt.legend()
plt.grid(True)
plt.savefig("results/plots/best_prompt_reward.png")
plt.close()

# 4️⃣ Prompt Frequency Distribution
prompt_counts = df["prompt"].value_counts()

plt.figure(figsize=(12, 6))
sns.barplot(x=prompt_counts.values, y=prompt_counts.index, palette="viridis")
plt.xlabel("Usage Count")
plt.ylabel("Prompt")
plt.title("Prompt Usage Frequency")
plt.savefig("results/plots/prompt_frequency.png")
plt.close()

print("✅ Analysis complete. Plots saved in results/plots/")

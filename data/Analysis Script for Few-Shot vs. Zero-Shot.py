import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

CSV_LOG = "few_shot_experiment_log.csv"

# Load CSV
df = pd.read_csv(CSV_LOG, encoding="utf-8")

# --- Reward by Example Count ---
plt.figure(figsize=(8, 5))
sns.boxplot(x="example_count", y="reward", data=df)
plt.title("Reward Distribution by Example Count")
plt.xlabel("Number of Few-Shot Examples")
plt.ylabel("Reward")
plt.grid(True, linestyle="--", alpha=0.7)
plt.show()

# --- Average Reward by Iteration ---
avg_reward = df.groupby(["iteration", "example_count"])["reward"].mean().reset_index()

plt.figure(figsize=(10, 6))
sns.lineplot(x="iteration", y="reward", hue="example_count", data=avg_reward, marker="o")
plt.title("Average Reward by Iteration (Few-Shot vs Zero-Shot)")
plt.xlabel("Iteration")
plt.ylabel("Average Reward")
plt.grid(True, linestyle="--", alpha=0.7)
plt.legend(title="Example Count")
plt.show()

# --- Correlation Heatmap ---
plt.figure(figsize=(6, 4))
correlation = df[["example_count", "reward"]].corr()
sns.heatmap(correlation, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation between Example Count and Reward")
plt.show()

# --- Best Prompt per Example Count ---
best_prompts = df.groupby("example_count").apply(lambda x: x.loc[x["reward"].idxmax()])
print("\n=== Best Prompt per Example Count ===")
print(best_prompts[["example_count", "reward", "prompt_text"]])

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

CSV_LOG = "few_shot_experiment_log2.csv"

# Load CSV
df = pd.read_csv(CSV_LOG, encoding="utf-8")

# --- Reward Trends by Category ---
plt.figure(figsize=(10, 6))
sns.lineplot(x="iteration", y="reward", hue="category", data=df, marker="o")
plt.title("Reward Trend by Category (Zero-Shot vs Few-Shot)")
plt.xlabel("Iteration")
plt.ylabel("Reward")
plt.grid(True, linestyle="--", alpha=0.7)
plt.legend(title="Prompt Category")
plt.show()

# --- Average Reward by Category ---
plt.figure(figsize=(8, 5))
sns.barplot(x="category", y="reward", data=df, estimator="mean", ci=None)
plt.title("Average Reward by Prompt Category")
plt.xlabel("Prompt Category")
plt.ylabel("Average Reward")
plt.grid(True, linestyle="--", alpha=0.7)
plt.show()

# --- Box Plot for Reward Distribution ---
plt.figure(figsize=(8, 5))
sns.boxplot(x="category", y="reward", data=df)
plt.title("Reward Distribution per Category")
plt.xlabel("Prompt Category")
plt.ylabel("Reward")
plt.grid(True, linestyle="--", alpha=0.7)
plt.show()

# --- Best Prompt per Category ---
best_prompts = df.groupby("category").apply(lambda x: x.loc[x["reward"].idxmax()])
print("\n=== Best Prompt per Category ===")
print(best_prompts[["category", "reward", "prompt_text"]])

# --- Rolling Average Reward ---
rolling_avg = df.groupby(["category", "iteration"])["reward"].mean().reset_index()
plt.figure(figsize=(10, 6))
sns.lineplot(x="iteration", y="reward", hue="category", data=rolling_avg)
plt.title("Rolling Average Reward by Category")
plt.xlabel("Iteration")
plt.ylabel("Rolling Avg Reward")
plt.grid(True, linestyle="--", alpha=0.7)
plt.legend(title="Prompt Category")
plt.show()

# --- Correlation Heatmap ---
correlation = df[["example_count", "reward"]].corr()
plt.figure(figsize=(5, 4))
sns.heatmap(correlation, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation between Example Count and Reward")
plt.show()

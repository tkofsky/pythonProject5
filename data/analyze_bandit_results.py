import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load bandit log
df = pd.read_csv("results/bandit_log.csv")

# Ensure directory for plots
os.makedirs("results/plots", exist_ok=True)

# Extract prompt category from the prompt text (intent labeling)
def categorize_prompt(prompt):
    prompt_lower = prompt.lower()
    if "short" in prompt_lower:
        return "short-summary"
    elif "detailed" in prompt_lower or "long" in prompt_lower:
        return "detailed-summary"
    elif "key points" in prompt_lower or "bullet" in prompt_lower:
        return "bullet-summary"
    else:
        return "general"

df["prompt_category"] = df["prompt"].apply(categorize_prompt)

# Calculate rolling reward
df["rolling_reward"] = df["reward"].rolling(window=5, min_periods=1).mean()

# Add prompt length
df["prompt_length"] = df["prompt"].apply(len)

# 1️⃣ Reward Progression
plt.figure(figsize=(12, 6))
sns.lineplot(x="iteration", y="rolling_reward", data=df, hue="prompt_category")
plt.xlabel("Iteration")
plt.ylabel("Reward")
plt.title("Reward Progression Over Iterations by Prompt Category")
plt.legend(title="Prompt Category")
plt.grid(True)
plt.savefig("results/plots/reward_progression_by_category.png")
plt.close()

# 2️⃣ Mutations Over Time
df["is_mutation"] = df["prompt"].duplicated(keep="first").astype(int)
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
    lambda x: x.loc[x["reward"].idxmax(), ["prompt", "reward", "prompt_category"]]
).reset_index()

plt.figure(figsize=(12, 6))
sns.lineplot(x="iteration", y="reward", hue="prompt_category", data=best_prompts)
plt.xlabel("Iteration")
plt.ylabel("Reward")
plt.title("Best Prompt Reward Over Iterations by Category")
plt.legend(title="Prompt Category")
plt.grid(True)
plt.savefig("results/plots/best_prompt_reward_by_category.png")
plt.close()

# 4️⃣ Prompt Frequency Distribution by Category
prompt_counts = df.groupby("prompt_category")["prompt"].count().sort_values(ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x=prompt_counts.index, y=prompt_counts.values, palette="viridis")
plt.xlabel("Prompt Category")
plt.ylabel("Usage Count")
plt.title("Prompt Usage Frequency by Category")
plt.savefig("results/plots/prompt_frequency_by_category.png")
plt.close()

# 5️⃣ Correlation Heatmap (Reward vs Tokens vs Prompt Length) by Category
for category in df["prompt_category"].unique():
    subset = df[df["prompt_category"] == category]
    correlation_data = subset[["reward", "tokens", "prompt_length"]].corr()

    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation_data, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title(f"Correlation Heatmap ({category})")
    plt.savefig(f"results/plots/correlation_heatmap_{category}.png")
    plt.close()

print("✅ Analysis complete. Plots saved in results/plots/")

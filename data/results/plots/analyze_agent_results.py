import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# -------------------------------
# 1. CONFIGURATION
# -------------------------------
RESULTS_FILE = "summarization_agent_log.csv"

# -------------------------------
# 2. LOAD DATA
# -------------------------------
if not os.path.exists(RESULTS_FILE):
    raise FileNotFoundError(f"{RESULTS_FILE} not found. Run the summarization agent first.")

#df = pd.read_csv(RESULTS_FILE)
df = pd.read_csv(RESULTS_FILE, encoding="latin1")
# Ensure correct data types
df["iteration"] = df["iteration"].astype(int)
df["reward"] = df["reward"].astype(float)
df["tokens"] = df["tokens"].astype(float)
df["mutation"] = df["mutation"].astype(int)

print("✅ Data Loaded:")
print(df.head())

# -------------------------------
# 3. ROLLING REWARD TREND
# -------------------------------
df["rolling_reward"] = df["reward"].rolling(window=5).mean()

plt.figure(figsize=(10, 5))
plt.plot(df["iteration"], df["rolling_reward"], label="Rolling Reward (window=5)")
plt.xlabel("Iteration")
plt.ylabel("Reward")
plt.title("Reward Trend Over Iterations")
plt.legend()
plt.grid(True)
plt.show()

# -------------------------------
# 4. BEST PROMPTS BY REWARD
# -------------------------------
prompt_avg = df.groupby("prompt")["reward"].mean().sort_values(ascending=False).head(10)
print("\n🏆 Top 10 Prompts by Average Reward:")
print(prompt_avg)

plt.figure(figsize=(12, 6))
sns.barplot(x=prompt_avg.values, y=prompt_avg.index, palette="viridis")
plt.xlabel("Average Reward")
plt.ylabel("Prompt")
plt.title("Top 10 Prompts by Reward")
plt.show()

# -------------------------------
# 5. REWARD BY CATEGORY
# -------------------------------
plt.figure(figsize=(8, 5))
sns.boxplot(x="category", y="reward", data=df, palette="Set2")
plt.xlabel("Prompt Category")
plt.ylabel("Reward")
plt.title("Reward Distribution by Prompt Category")
plt.show()

# -------------------------------
# 6. MUTATION EFFECT
# -------------------------------
mutation_stats = df.groupby("mutation")["reward"].mean()
print("\n🔬 Mutation Effect on Reward:")
print(mutation_stats)

plt.figure(figsize=(6, 4))
sns.barplot(x=mutation_stats.index, y=mutation_stats.values, palette="coolwarm")
plt.xticks([0, 1], ["No Mutation", "Mutation"])
plt.ylabel("Average Reward")
plt.title("Effect of Prompt Mutation on Reward")
plt.show()

# -------------------------------
# 7. TOKENS VS REWARD CORRELATION
# -------------------------------
correlation = df["tokens"].corr(df["reward"])
print(f"\n📈 Correlation between Tokens and Reward: {correlation:.3f}")

plt.figure(figsize=(8, 5))
sns.scatterplot(x="tokens", y="reward", data=df)
plt.xlabel("Tokens Used")
plt.ylabel("Reward")
plt.title("Tokens vs Reward")
plt.grid(True)
plt.show()

# -------------------------------
# 8. EXPORT SUMMARY
# -------------------------------
summary_stats = {
    "best_prompt": prompt_avg.index[0],
    "best_prompt_reward": prompt_avg.values[0],
    "mutation_reward_gain": mutation_stats[1] - mutation_stats[0],
    "token_reward_correlation": correlation
}

print("\n📊 Summary Stats:")
print(summary_stats)

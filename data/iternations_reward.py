import pandas as pd
import matplotlib.pyplot as plt

# === Load the log CSV ===
log_path = "log2.csv"
df = pd.read_csv(log_path)

print("\n✅ Loaded log with", len(df), "rows.")
print(df.head())

# === Aggregate stats per prompt ===
agg = df.groupby("prompt").agg(
    times_tested=("reward", "count"),
    avg_quality=("quality", "mean"),
    avg_tokens=("tokens", "mean"),
    avg_reward=("reward", "mean")
).sort_values(by="avg_reward", ascending=False)

print("\n=== 📊 Top prompts by average reward ===")
print(agg)

# === Optional: save aggregated stats ===
agg.to_csv("prompt_stats.csv")
print("\n✅ Saved aggregated stats to results/prompt_stats.csv")

# === Plot reward trends over iterations ===
plt.figure(figsize=(10, 6))
for prompt, subdf in df.groupby("prompt"):
    plt.plot(subdf["iteration"], subdf["reward"], marker='o', linestyle='-', alpha=0.6, label=prompt[:30]+"...")
plt.xlabel("Iteration")
plt.ylabel("Reward")
plt.title("Reward Over Time per Prompt")
plt.legend()
plt.tight_layout()
plt.show()

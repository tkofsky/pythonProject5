import pandas as pd
import matplotlib.pyplot as plt

############## try using csv with more iterations etc

# === Load the log file ===
df = pd.read_csv("log2.csv")

print("✅ Loaded log with", len(df), "rows")
print(df.head())

# === Scatter plot: Tokens vs Quality ===
plt.figure(figsize=(8, 6))
plt.scatter(df["tokens"], df["quality"], alpha=0.6, edgecolors='k')
plt.xlabel("Tokens used")
plt.ylabel("Quality score")
plt.title("Tokens vs Quality")
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# === Optional: Compute correlation ===
corr = df["tokens"].corr(df["quality"])
print(f"\n📈 Correlation between tokens and quality: {corr:.3f}")
import json
import pandas as pd
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer, util

# Load your JSON file
with open("longer_sample_dataset.json", "r") as f:
    dataset = json.load(f)

# Load the model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Compute semantic similarity (reward)
rewards = []
for i, item in enumerate(dataset):
    emb_input = model.encode(item["input"], convert_to_tensor=True)
    emb_ref = model.encode(item["reference"], convert_to_tensor=True)
    sim = float(util.cos_sim(emb_input, emb_ref))
    rewards.append({"iteration": i, "reward": sim})

# Create DataFrame
df = pd.DataFrame(rewards)
df["rolling_reward"] = df["reward"].rolling(window=2).mean()

# Plot
plt.figure(figsize=(10, 6))
plt.plot(df["iteration"], df["reward"], label="Raw Reward", linestyle="--", marker="o", alpha=0.5)
plt.plot(df["iteration"], df["rolling_reward"], label="Rolling Avg (window=2)", linewidth=2)
plt.xlabel("Iteration (Sample Index)")
plt.ylabel("Semantic Similarity (Reward)")
plt.title("Reward Trend from Input vs Reference")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

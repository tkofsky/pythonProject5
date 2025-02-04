import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# ==== CONFIG ====
LOG = "bandit_fewshot_agent_log_router_hybrid_steps_weighted.csv"
OUT_DIR = "plots_fewshot_effects"
os.makedirs(OUT_DIR, exist_ok=True)

# ==== LOAD DATA ====
df = pd.read_csv(LOG, low_memory=False)

# keep relevant fields
fields = ["f1_intent", "f1_entities", "f1_constraints", "f1_urgency", "f1_steps"]
df["example_count"] = pd.to_numeric(df["example_count"], errors="coerce").fillna(0).astype(int)

# ==== SUMMARY BY FEW-SHOT LEVEL ====
summary = (
    df.groupby("example_count")[fields]
    .mean()
    .round(3)
    .reset_index()
    .melt(id_vars=["example_count"], var_name="field", value_name="mean_f1")
)

# ==== PLOT ====
sns.set_style("whitegrid")
plt.figure(figsize=(10, 6))
sns.barplot(data=summary, x="example_count", y="mean_f1", hue="field")
plt.title("Effect of Few-Shot Level on F1 Scores (Per Field)")
plt.xlabel("Few-Shot Example Count (k)")
plt.ylabel("Mean F1 Score")
plt.legend(title="Field")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "fewshot_effects_bar.png"), dpi=160)
plt.close()

# ==== PRINT SUMMARY ====
print("📊 Mean F1 by Few-Shot Level:\n")
print(summary.pivot(index="example_count", columns="field", values="mean_f1").round(3))
print(f"\n✅ Saved plot → {OUT_DIR}/fewshot_effects_bar.png")

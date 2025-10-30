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
)

# melt for plotting
summary_melted = summary.melt(id_vars=["example_count"], var_name="field", value_name="mean_f1")

# ==== PLOT 1: Absolute Mean F1 ====
sns.set_style("whitegrid")
plt.figure(figsize=(10, 6))
sns.barplot(data=summary_melted, x="example_count", y="mean_f1", hue="field")
plt.title("Effect of Few-Shot Level on F1 Scores (Per Field)")
plt.xlabel("Few-Shot Example Count (k)")
plt.ylabel("Mean F1 Score")
plt.legend(title="Field")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "fewshot_effects_absolute.png"), dpi=160)
plt.close()

# ==== COMPUTE RELATIVE IMPROVEMENT (0-shot → 3-shot) ====
improve_rows = []
zero_shot = summary[summary["example_count"] == 0].set_index("example_count").iloc[0]
three_shot = summary[summary["example_count"] == 3].set_index("example_count").iloc[0]

for f in fields:
    base = zero_shot[f]
    new = three_shot[f]
    if base == 0:
        gain = float("nan")
    else:
        gain = 100 * (new - base) / base
    improve_rows.append({"field": f, "percent_gain": gain})

improve_df = pd.DataFrame(improve_rows).round(1)

# ==== PLOT 2: Relative Improvement ====
plt.figure(figsize=(8, 5))
sns.barplot(data=improve_df, x="field", y="percent_gain", palette="crest")
plt.title("Relative Improvement: 3-Shot vs 0-Shot (Percent Gain)")
plt.xlabel("Field")
plt.ylabel("% Gain in Mean F1 (0 → 3-shot)")
plt.axhline(0, color="black", linewidth=0.8)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "fewshot_effects_relative.png"), dpi=160)
plt.close()

# ==== PRINT SUMMARY TABLES ====
print("\n📊 Mean F1 by Few-Shot Level:\n")
print(summary.set_index("example_count").round(3))

print("\n📈 Relative Improvement (0-shot → 3-shot):\n")
print(improve_df)

print(f"\n✅ Plots saved → {OUT_DIR}/fewshot_effects_absolute.png and fewshot_effects_relative.png")

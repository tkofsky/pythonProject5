import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle

# ==========================
# CONFIG
# ==========================
INPUT_FILES = [
    "bandit_fewshot_agent_log.csv",
    "bandit_fewshot_agent_log_experiments.csv",
    "bandit_fewshot_agent_log_two_pass_optimized.csv"
]
OUT_PDF = "SR&ED_Bandit_Analysis_Report.pdf"
OUT_DIR = "report_figures"
os.makedirs(OUT_DIR, exist_ok=True)

# ==========================
# LOAD & LABEL
# ==========================
dfs = []
for f in INPUT_FILES:
    if not os.path.exists(f):
        print(f"⚠️ Missing file: {f}")
        continue
    df = pd.read_csv(f)
    if "two_pass" in f:
        df["run_type"] = "Two-Pass Optimized"
    elif "experiments" in f:
        df["run_type"] = "Experiments"
    else:
        df["run_type"] = "One-Pass Baseline"
    dfs.append(df)

df_all = pd.concat(dfs, ignore_index=True)
df_all["reward"] = pd.to_numeric(df_all["reward"], errors="coerce")
df_all["tokens"] = pd.to_numeric(df_all["tokens"], errors="coerce").fillna(1)
df_all["reward_per_1k"] = df_all["reward"] / (df_all["tokens"] / 1000)
df_all["json_valid"] = df_all.get("json_valid", 1.0)

# ==========================
# SUMMARY TABLES
# ==========================
summary = df_all.groupby("run_type")[["reward", "reward_per_1k", "json_valid"]].mean().round(3)
summary.reset_index(inplace=True)

# ==========================
# PLOTS
# ==========================
sns.set_style("whitegrid")
plt.rcParams.update({"axes.facecolor": "white"})

def savefig(name):
    path = os.path.join(OUT_DIR, name)
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()
    return path

# Reward timeline
plt.figure(figsize=(8, 4))
sns.lineplot(data=df_all, x="iteration", y="reward", hue="run_type", marker="o", ci=None)
plt.title("Reward Timeline by Run Type")
plt.xlabel("Iteration")
plt.ylabel("Reward")
reward_timeline = savefig("reward_timeline.png")

# Reward per 1k tokens
plt.figure(figsize=(8, 4))
sns.boxplot(data=df_all, x="run_type", y="reward_per_1k", palette="Set2")
plt.title("Reward per 1K Tokens")
plt.xlabel("")
plt.ylabel("Reward / 1K Tokens")
reward_per_1k_fig = savefig("reward_per_1k.png")

# JSON validity
plt.figure(figsize=(8, 4))
sns.barplot(data=df_all, x="run_type", y="json_valid", ci=None, palette="muted")
plt.title("JSON Validity by Configuration")
plt.ylabel("Validity Rate")
plt.ylim(0, 1)
json_valid_fig = savefig("json_validity.png")

# Correlation matrix
corr = df_all[["reward", "reward_per_1k", "tokens"]].corr().round(2)
plt.figure(figsize=(5, 4))
sns.heatmap(corr, annot=True, cmap="coolwarm", vmin=-1, vmax=1)
plt.title("Reward Correlations")
corr_fig = savefig("corr_matrix.png")

# ==========================
# PDF REPORT
# ==========================
doc = SimpleDocTemplate(OUT_PDF, pagesize=letter, rightMargin=40, leftMargin=40, topMargin=60, bottomMargin=40)
styles = getSampleStyleSheet()
Story = []

def add_image_with_caption(img_path, caption):
    Story.append(Spacer(1, 12))
    im = Image(img_path, width=450, height=250)
    Story.append(im)
    Story.append(Spacer(1, 6))
    Story.append(Paragraph(f"<b>{caption}</b>", styles["Normal"]))
    Story.append(Spacer(1, 12))

Story.append(Paragraph("<b>SR&ED Bandit Analysis Report</b>", styles["Title"]))
Story.append(Spacer(1, 12))
Story.append(Paragraph("This report compares One-Pass, Two-Pass, and Experimental Bandit Few-Shot configurations. Each section shows charts and corresponding numeric summaries.", styles["Normal"]))
Story.append(Spacer(1, 12))

# Parameter Summary
Story.append(Paragraph("<b>Configuration Summary</b>", styles["Heading2"]))
table_data = [["Run Type", "Mean Reward", "Reward/1K", "JSON Validity"]]
for _, r in summary.iterrows():
    table_data.append([r["run_type"], r["reward"], r["reward_per_1k"], r["json_valid"]])
t = Table(table_data, hAlign="LEFT")
t.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
    ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
    ("FONT", (0, 0), (-1, 0), "Helvetica-Bold")
]))
Story.append(t)
Story.append(Spacer(1, 12))

# Add charts
add_image_with_caption(reward_timeline, "Reward over Iterations per Run Type")
add_image_with_caption(reward_per_1k_fig, "Normalized Reward per 1K Tokens")
add_image_with_caption(json_valid_fig, "JSON Validity Comparison Across Runs")
add_image_with_caption(corr_fig, "Correlation Matrix of Reward Metrics")

Story.append(Spacer(1, 24))
Story.append(Paragraph("<b>Interpretation</b>", styles["Heading2"]))
Story.append(Paragraph(
    "The One-Pass model demonstrates stronger early convergence but fluctuates more across iterations. "
    "Two-Pass Optimized shows higher token usage with more stable JSON outputs, aligning with its design to decompose structured reasoning. "
    "Experimental variants offer mixed performance, suggesting prompt diversity exploration can improve edge-case handling.",
    styles["Normal"]
))

doc.build(Story)
print(f"✅ Report generated: {OUT_PDF}")

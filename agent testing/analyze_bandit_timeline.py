import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------- Settings ----------
LOG_CANDIDATES = [
    "bandit_fewshot_agent_log.csv",
    "bandit_fewshot_agent_log_two_pass.csv"
]
TOPN = 8          # how many arms to plot
WINDOW = 7        # rolling mean window size
OUT_DIR = "agent_plots_bandit_timeline"

# ---------- Helpers ----------
def autodetect_log():
    for p in LOG_CANDIDATES:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"No log file found. Expected one of: {', '.join(LOG_CANDIDATES)}")

def ensure_out(dirpath):
    os.makedirs(dirpath, exist_ok=True)

def rolling_mean(series, window):
    return series.rolling(window=window, min_periods=max(1, window//2)).mean()

# ---------- Main ----------
def main():
    log_path = autodetect_log()
    ensure_out(OUT_DIR)

    # Load
    try:
        df = pd.read_csv(log_path, encoding="utf-8-sig")
    except UnicodeDecodeError:
        df = pd.read_csv(log_path, encoding="latin-1")

    # Hygiene
    for c in ["iteration","example_count","temperature","reward","tokens","is_mutation","two_pass"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if "tokens" not in df.columns:
        df["tokens"] = np.nan
    if "is_mutation" not in df.columns:
        df["is_mutation"] = df["prompt_id"].astype(str).str.contains(r"__m\d+$").astype(int)

    # Arm label
    def arm_label(row):
        pid = str(row.get("prompt_id",""))
        k = row.get("example_count", np.nan)
        t = row.get("temperature", np.nan)
        if "two_pass" in df.columns and not pd.isna(row.get("two_pass", np.nan)):
            tp = int(row.get("two_pass", 0))
            return f"{pid} | k={int(k) if not pd.isna(k) else '?'} | T={t} | 2p={tp}"
        return f"{pid} | k={int(k) if not pd.isna(k) else '?'} | T={t}"

    df["arm"] = df.apply(arm_label, axis=1)
    df = df.sort_values("iteration")

    # Efficiency
    df["reward_per_1k"] = np.where(df["tokens"]>0, df["reward"]/(df["tokens"]/1000.0), np.nan)

    # Rank arms
    arm_stats = (df.groupby("arm")[["reward","reward_per_1k"]]
                   .mean()
                   .sort_values("reward", ascending=False)
                   .reset_index())

    topN = min(TOPN, len(arm_stats))
    top_arms = arm_stats.head(topN)["arm"].tolist()

    # Save summary
    arm_stats.head(topN).to_csv(os.path.join(OUT_DIR, "arm_summary_topN.csv"), index=False)

    # Rolling reward timeline
    plt.figure(figsize=(11,6))
    for arm in top_arms:
        sub = df[df["arm"]==arm]
        rm = rolling_mean(sub.set_index("iteration")["reward"], WINDOW)
        plt.plot(rm.index, rm.values, label=arm)
    plt.title(f"Rolling mean reward (window={WINDOW}) for top {topN} arms")
    plt.xlabel("iteration"); plt.ylabel("reward"); plt.ylim(0,1)
    plt.legend(loc="best", fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "reward_timeline_topN.png"), dpi=140)
    plt.close()

    # Rolling efficiency timeline
    if df["reward_per_1k"].notna().any():
        plt.figure(figsize=(11,6))
        for arm in top_arms:
            sub = df[df["arm"]==arm]
            rm = rolling_mean(sub.set_index("iteration")["reward_per_1k"], WINDOW)
            plt.plot(rm.index, rm.values, label=arm)
        plt.title(f"Rolling mean reward/1k (window={WINDOW}) for top {topN} arms")
        plt.xlabel("iteration"); plt.ylabel("reward per 1k tokens")
        plt.legend(loc="best", fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, "reward_per_1k_timeline_topN.png"), dpi=140)
        plt.close()

    # Pulls over time (binned)
    if df["iteration"].notna().any():
        max_iter = int(df["iteration"].max())
    else:
        max_iter = 0
    bins = np.linspace(0, max_iter, num=11, dtype=int) if max_iter>0 else np.arange(0, 11)
    plt.figure(figsize=(11,6))
    for arm in top_arms:
        sub = df[df["arm"]==arm]
        counts, edges = np.histogram(sub["iteration"].dropna(), bins=bins)
        centers = (edges[:-1] + edges[1:]) / 2.0
        plt.plot(centers, counts, marker="o", label=arm)
    plt.title("Arm pulls over time (binned counts)")
    plt.xlabel("iteration bin center"); plt.ylabel("# pulls in bin")
    plt.legend(loc="best", fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "pulls_over_time_topN.png"), dpi=140)
    plt.close()

    # Mutation markers
    mut = df[df["is_mutation"]==1]
    if not mut.empty:
        plt.figure(figsize=(10,3))
        plt.scatter(mut["iteration"], [1]*len(mut), s=12)
        plt.yticks([]); plt.xlabel("iteration")
        plt.title("Mutated prompt occurrences")
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, "mutation_markers.png"), dpi=140)
        plt.close()

    print("✅ Plots saved to:", OUT_DIR)
    print("   - reward_timeline_topN.png")
    if df["reward_per_1k"].notna().any():
        print("   - reward_per_1k_timeline_topN.png")
    print("   - pulls_over_time_topN.png")
    if not mut.empty:
        print("   - mutation_markers.png")
    print("   - arm_summary_topN.csv")

if __name__ == "__main__":
    main()


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------- Settings ----------
LOG_CANDIDATES = [
    "bandit_fewshot_agent_log_two_pass.csv",
    "bandit_fewshot_agent_log.csv",
]
TOPN =  8           # how many arms to plot
WINDOW = 25  #15        # rolling mean window size (longer smoothing)
MIN_PULLS = 5      # minimum number of pulls required per arm to plot
OUT_DIR = "agent_plots_bandit_timeline_fixed"

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

def has_reward_data(d):
    return ("reward" in d.columns) and d["reward"].notna().any()

def has_efficiency_data(d):
    return ("reward_per_1k" in d.columns) and d["reward_per_1k"].notna().any()



def smooth(y, weight=0.8):
    """Exponential moving average smoother."""
    smoothed = []
    last = y[0]
    for val in y:
        last = last * weight + (1 - weight) * val
        smoothed.append(last)
    return np.array(smoothed)



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

    # Rebuild 'arm' column if missing
    if "arm" not in df.columns:
        for col in ["prompt_id","example_count","temperature","is_mutation","two_pass"]:
            if col not in df.columns:
                df[col] = np.nan

        def to_s(x):
            if pd.isna(x):
                return "NA"
            if isinstance(x, float):
                # Trim float noise
                s = f"{x:.2f}"
                s = s.rstrip('0').rstrip('.')
                return s if s else "0"
            return str(int(x)) if isinstance(x, (np.integer,)) else str(x)

        df["arm"] = (
            df["prompt_id"].astype(str)
            + " | k=" + df["example_count"].apply(to_s)
            + " | T=" + df["temperature"].apply(to_s)
            + " | m=" + df["is_mutation"].fillna(0).astype(int).astype(str)
            + (" | 2p=" + df["two_pass"].fillna(0).astype(int).astype(str) if "two_pass" in df.columns else "")
        )

    # Diagnostics
    print("\n[DIAG] Columns:", list(df.columns))
    print("[DIAG] Rows:", len(df))
    for col in ["iteration","reward","tokens","arm"]:
        if col in df.columns:
            nn = df[col].notna().sum()
            print(f"[DIAG] non-null {col}: {nn}")
        else:
            print(f"[DIAG] MISSING column: {col}")

    # Compute reward_per_1k robustly
    if "reward" in df.columns and "tokens" in df.columns:
        df["reward_per_1k"] = np.where(
            (df["tokens"].notna()) & (df["tokens"] > 0),
            df["reward"] / (df["tokens"] / 1000.0),
            np.nan
        )
    else:
        df["reward_per_1k"] = np.nan

    # Sort by iteration
    if "iteration" in df.columns:
        df = df.sort_values("iteration")
    else:
        df["iteration"] = np.arange(len(df))

    # Apply minimum pulls filter BEFORE ranking
    pull_counts = df.groupby("arm").size().reset_index(name="n_pulls")
    valid_arms = pull_counts[pull_counts["n_pulls"] >= MIN_PULLS]["arm"]
    df = df[df["arm"].isin(valid_arms)].copy()

    if df.empty:
        print(f"⚠️ No arms meet MIN_PULLS={MIN_PULLS}. Try lowering MIN_PULLS or run longer.")
        return

    # Rank arms by mean reward when available; otherwise by pulls
    if has_reward_data(df):
        arm_stats = (df.groupby("arm")[["reward","reward_per_1k"]]
                       .mean()
                       .sort_values("reward", ascending=False)
                       .reset_index())
    else:
        arm_stats = (df.groupby("arm").size()
                       .sort_values(ascending=False)
                       .rename("n_pulls")
                       .reset_index())

    topN = min(TOPN, len(arm_stats))
    top_arms = arm_stats.head(topN)["arm"].tolist()

    # Save summary
    arm_stats.head(topN).to_csv(os.path.join(OUT_DIR, "arm_summary_topN.csv"), index=False)

    # Rolling reward timeline
    if has_reward_data(df):
        plt.figure(figsize=(11,6))
        for arm in top_arms:
            sub = df[df["arm"]==arm]
            if sub.empty:
                continue
            s = sub.set_index("iteration")["reward"]
            rm = rolling_mean(s, WINDOW)
            if not rm.dropna().empty:
                plt.plot(rm.index, rm.values, label=arm)
        plt.title(f"Rolling mean reward (window={WINDOW}) for top {topN} arms (min pulls ≥ {MIN_PULLS})")
        plt.xlabel("iteration"); plt.ylabel("reward"); plt.ylim(0,1)
        if len(plt.gca().lines) > 0:
            plt.legend(loc="best", fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, "reward_timeline_topN.png"), dpi=140)
        plt.close()
    else:
        print("[WARN] No usable reward data to plot reward timeline; skipping.")

    # Rolling efficiency timeline
    if has_efficiency_data(df):
        plt.figure(figsize=(11,6))
        for arm in top_arms:
            sub = df[df["arm"]==arm]
            if sub.empty:
                continue
            s = sub.set_index("iteration")["reward_per_1k"]
            rm = rolling_mean(s, WINDOW)
            if not rm.dropna().empty:
                plt.plot(rm.index, rm.values, label=arm)
        plt.title(f"Rolling mean reward/1k (window={WINDOW}) for top {topN} arms (min pulls ≥ {MIN_PULLS})")
        plt.xlabel("iteration"); plt.ylabel("reward per 1k tokens")
        if len(plt.gca().lines) > 0:
            plt.legend(loc="best", fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, "reward_per_1k_timeline_topN.png"), dpi=140)
        plt.close()
    else:
        print("[WARN] No usable reward_per_1k data to plot efficiency timeline; skipping.")

    # Pulls over time (binned counts)
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
    plt.title(f"Arm pulls over time (binned counts) — min pulls ≥ {MIN_PULLS}")
    plt.xlabel("iteration bin center"); plt.ylabel("# pulls in bin")
    plt.legend(loc="best", fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "pulls_over_time_topN.png"), dpi=140)
    plt.close()

    print("✅ Plots saved to:", OUT_DIR)
    print("   - arm_summary_topN.csv")
    if has_reward_data(df):
        print("   - reward_timeline_topN.png")
    if has_efficiency_data(df):
        print("   - reward_per_1k_timeline_topN.png")
    print("   - pulls_over_time_topN.png")
    print(f"   (MIN_PULLS={MIN_PULLS}, WINDOW={WINDOW}, TOPN={TOPN})")

if __name__ == "__main__":
    main()

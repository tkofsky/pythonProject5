
"""
analyze_bandit_jsonmode_ext.py

Extended analysis for bandit few-shot agent logs with prompt-variant context.
Breaks out metrics by:
- json_mode (True/False)
- two_pass (True/False)
- fewshot_strategy (static/intent_matched)
Also plots interactions: json_mode × temperature, json_mode × example_count.

Inputs (any of the following, if present in working dir or absolute paths):
- bandit_fewshot_agent_log.csv
- bandit_fewshot_agent_log_experiments.csv

Outputs:
- Tables in results_ext/
- Plots in agent_plots_ext/
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

OUT_DIR = "agent_plots_ext"
RES_DIR = "results_ext"
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(RES_DIR, exist_ok=True)

def load_any_logs(paths):
    dfs = []
    for path in paths:
        if not os.path.exists(path):
            continue
        for enc in ["utf-8-sig", "utf-8", "latin-1"]:
            try:
                df = pd.read_csv(path, encoding=enc)
                df["__source_log__"] = os.path.basename(path)
                dfs.append(df)
                break
            except Exception:
                continue
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

def coerce_columns(df_all: pd.DataFrame) -> pd.DataFrame:
    # Numeric coercions
    num_cols = ["iteration","example_count","temperature","reward","tokens",
                "f1_intent","f1_entities","f1_constraints","f1_urgency","f1_steps",
                "reward_per_1k","json_valid"]
    for c in num_cols:
        if c not in df_all.columns:
            df_all[c] = np.nan
        df_all[c] = pd.to_numeric(df_all[c], errors="coerce")

    # Boolean-like/dimensional columns
    for c in ["json_mode","two_pass"]:
        if c not in df_all.columns:
            df_all[c] = np.nan
    if "fewshot_strategy" not in df_all.columns:
        df_all["fewshot_strategy"] = "static"
    if "phase" not in df_all.columns:
        df_all["phase"] = "bandit"
    if "prompt_id" not in df_all.columns:
        df_all["prompt_id"] = ""
    return df_all

def make_helpers(df_all: pd.DataFrame) -> pd.DataFrame:
    fields = ["f1_intent","f1_entities","f1_constraints","f1_urgency","f1_steps"]
    df_all["f1_mean"] = df_all[fields].mean(axis=1, skipna=True)

    if "json_valid" not in df_all.columns:
        # If json_valid wasn't logged, use a loose proxy
        df_all["json_valid"] = np.where(df_all["f1_mean"].notna() & (df_all["f1_mean"] > 0), 1.0, 0.0)
    return df_all

def savefig(path):
    plt.tight_layout()
    plt.savefig(path, dpi=140, bbox_inches="tight")
    plt.close()

def main():
    # Candidate log paths (relative + sandbox paths)
    paths = [
        "bandit_fewshot_agent_log.csv",
        "bandit_fewshot_agent_log_experiments.csv",
        "/mnt/data/bandit_fewshot_agent_log.csv",
        "/mnt/data/bandit_fewshot_agent_log_experiments.csv",
    ]
    df_all = load_any_logs(paths)
    if df_all.empty:
        print("No logs found. Place a CSV log next to this script (see docstring).")
        return

    df_all = coerce_columns(df_all)
    df_all = make_helpers(df_all)

    agg_cols = ["f1_mean","reward","reward_per_1k","json_valid"]

    # ---- Aggregations by json_mode ----
    by_json = (df_all.groupby("json_mode")[agg_cols].mean().reset_index())
    by_json.to_csv(os.path.join(RES_DIR, "by_json_mode.csv"), index=False)

    plt.figure(figsize=(7,5))
    labels = by_json["json_mode"].astype(str).fillna("nan").values
    vals = by_json["f1_mean"].values
    plt.bar(labels, vals)
    plt.ylim(0,1)
    plt.title("Mean F1 by JSON Mode")
    plt.xlabel("json_mode"); plt.ylabel("F1 (mean)")
    savefig(os.path.join(OUT_DIR, "mean_f1_by_json_mode.png"))

    plt.figure(figsize=(7,5))
    vals = by_json["json_valid"].values
    plt.bar(labels, vals)
    plt.ylim(0,1)
    plt.title("JSON Validity by JSON Mode")
    plt.xlabel("json_mode"); plt.ylabel("Validity rate")
    savefig(os.path.join(OUT_DIR, "validity_by_json_mode.png"))

    # ---- Aggregations by two_pass ----
    by_twopass = (df_all.groupby("two_pass")[agg_cols].mean().reset_index())
    by_twopass.to_csv(os.path.join(RES_DIR, "by_two_pass.csv"), index=False)

    plt.figure(figsize=(7,5))
    labels = by_twopass["two_pass"].astype(str).fillna("nan").values
    vals = by_twopass["f1_mean"].values
    plt.bar(labels, vals)
    plt.ylim(0,1)
    plt.title("Mean F1 by Two-Pass")
    plt.xlabel("two_pass"); plt.ylabel("F1 (mean)")
    savefig(os.path.join(OUT_DIR, "mean_f1_by_two_pass.png"))

    # ---- Aggregations by fewshot_strategy ----
    by_fs = (df_all.groupby("fewshot_strategy")[agg_cols].mean().reset_index())
    by_fs.to_csv(os.path.join(RES_DIR, "by_fewshot_strategy.csv"), index=False)

    plt.figure(figsize=(7,5))
    labels = by_fs["fewshot_strategy"].astype(str).values
    vals = by_fs["f1_mean"].values
    plt.bar(labels, vals)
    plt.ylim(0,1)
    plt.title("Mean F1 by Few-shot Strategy")
    plt.xlabel("fewshot_strategy"); plt.ylabel("F1 (mean)")
    savefig(os.path.join(OUT_DIR, "mean_f1_by_fewshot_strategy.png"))

    # ---- Interactions: json_mode × temperature and × example_count ----
    jm_temp = (df_all.groupby(["json_mode","temperature"])[agg_cols].mean().reset_index())
    jm_temp.to_csv(os.path.join(RES_DIR, "by_json_mode_x_temp.csv"), index=False)

    if not jm_temp.empty and jm_temp["temperature"].notna().any():
        pv = jm_temp.pivot(index="json_mode", columns="temperature", values="f1_mean")
        plt.figure(figsize=(6,4))
        im = plt.imshow(pv.values, vmin=0, vmax=1)
        plt.colorbar(im, fraction=0.046, pad=0.04)
        plt.xticks(range(len(pv.columns)), [str(c) for c in pv.columns])
        plt.yticks(range(len(pv.index)), [str(i) for i in pv.index])
        plt.title("Mean F1: json_mode × temperature")
        plt.xlabel("temperature"); plt.ylabel("json_mode")
        savefig(os.path.join(OUT_DIR, "heatmap_f1_jsonmode_by_temp.png"))

    jm_k = (df_all.groupby(["json_mode","example_count"])[agg_cols].mean().reset_index())
    jm_k.to_csv(os.path.join(RES_DIR, "by_json_mode_x_example_count.csv"), index=False)

    if not jm_k.empty and jm_k["example_count"].notna().any():
        pv2 = jm_k.pivot(index="json_mode", columns="example_count", values="f1_mean")
        plt.figure(figsize=(6,4))
        im = plt.imshow(pv2.values, vmin=0, vmax=1)
        plt.colorbar(im, fraction=0.046, pad=0.04)
        plt.xticks(range(len(pv2.columns)), [str(c) for c in pv2.columns])
        plt.yticks(range(len(pv2.index)), [str(i) for i in pv2.index])
        plt.title("Mean F1: json_mode × example_count")
        plt.xlabel("example_count"); plt.ylabel("json_mode")
        savefig(os.path.join(OUT_DIR, "heatmap_f1_jsonmode_by_k.png"))

    # === Efficiency analysis: reward_per_1k ===
    # Guard: some logs may not have reward_per_1k; compute a proxy where possible
    if "reward_per_1k" not in df_all.columns or df_all["reward_per_1k"].isna().all():
        # tokens may be missing in some rows; avoid divide-by-zero
        df_all["reward_per_1k"] = np.where(df_all["tokens"].gt(0),
                                           df_all["reward"] / (df_all["tokens"] / 1000.0),
                                           np.nan)

    eff_cols = ["reward_per_1k", "reward", "f1_mean", "json_valid"]

    # Few-shot level efficiency
    by_k_eff = (df_all.groupby("example_count")[eff_cols].mean().reset_index())
    by_k_eff.to_csv(os.path.join(RES_DIR, "eff_by_example_count.csv"), index=False)

    plt.figure(figsize=(7,5))
    plt.bar(by_k_eff["example_count"].astype(str), by_k_eff["reward_per_1k"].values)
    plt.title("Efficiency: Reward per 1k tokens by Few-shot Level")
    plt.xlabel("example_count"); plt.ylabel("reward_per_1k (mean)")
    savefig(os.path.join(OUT_DIR, "eff_reward_per_1k_by_k.png"))

    # JSON mode efficiency
    by_json_eff = (df_all.groupby("json_mode")[eff_cols].mean().reset_index())
    by_json_eff.to_csv(os.path.join(RES_DIR, "eff_by_json_mode.csv"), index=False)

    plt.figure(figsize=(7,5))
    plt.bar(by_json_eff["json_mode"].astype(str).fillna("nan"), by_json_eff["reward_per_1k"].values)
    plt.title("Efficiency: Reward per 1k tokens by JSON Mode")
    plt.xlabel("json_mode"); plt.ylabel("reward_per_1k (mean)")
    savefig(os.path.join(OUT_DIR, "eff_reward_per_1k_by_json_mode.png"))

    # Two-pass efficiency
    by_two_eff = (df_all.groupby("two_pass")[eff_cols].mean().reset_index())
    by_two_eff.to_csv(os.path.join(RES_DIR, "eff_by_two_pass.csv"), index=False)

    plt.figure(figsize=(7,5))
    plt.bar(by_two_eff["two_pass"].astype(str).fillna("nan"), by_two_eff["reward_per_1k"].values)
    plt.title("Efficiency: Reward per 1k tokens by Two-Pass")
    plt.xlabel("two_pass"); plt.ylabel("reward_per_1k (mean)")
    savefig(os.path.join(OUT_DIR, "eff_reward_per_1k_by_two_pass.png"))

    # Few-shot strategy efficiency
    by_fs_eff = (df_all.groupby("fewshot_strategy")[eff_cols].mean().reset_index())
    by_fs_eff.to_csv(os.path.join(RES_DIR, "eff_by_fewshot_strategy.csv"), index=False)

    plt.figure(figsize=(7,5))
    plt.bar(by_fs_eff["fewshot_strategy"].astype(str), by_fs_eff["reward_per_1k"].values)
    plt.title("Efficiency: Reward per 1k tokens by Few-shot Strategy")
    plt.xlabel("fewshot_strategy"); plt.ylabel("reward_per_1k (mean)")
    savefig(os.path.join(OUT_DIR, "eff_reward_per_1k_by_fewshot_strategy.png"))

    # Interaction: json_mode × example_count (efficiency)
    jm_k_eff = (df_all.groupby(["json_mode","example_count"])[eff_cols].mean().reset_index())
    jm_k_eff.to_csv(os.path.join(RES_DIR, "eff_by_json_mode_x_example_count.csv"), index=False)

    if not jm_k_eff.empty:
        pv_eff = jm_k_eff.pivot(index="json_mode", columns="example_count", values="reward_per_1k")
        if pv_eff.shape[0] > 0 and pv_eff.shape[1] > 0:
            plt.figure(figsize=(6,4))
            im = plt.imshow(pv_eff.values, aspect="auto")
            plt.colorbar(im, fraction=0.046, pad=0.04)
            plt.xticks(range(len(pv_eff.columns)), [str(c) for c in pv_eff.columns])
            plt.yticks(range(len(pv_eff.index)), [str(i) for i in pv_eff.index])
            plt.title("Efficiency (reward/1k): json_mode × example_count")
            plt.xlabel("example_count"); plt.ylabel("json_mode")
            savefig(os.path.join(OUT_DIR, "heatmap_eff_jsonmode_by_k.png"))

    # Bonus: distribution plot (box) of efficiency by prompt_id (top 12 prompts only)
    if "prompt_id" in df_all.columns and df_all["prompt_id"].notna().any():
        top_prompts = (
            df_all.groupby("prompt_id")["reward_per_1k"]
                 .mean()
                 .sort_values(ascending=False)
                 .head(12)
                 .index.tolist()
        )
        sub = df_all[df_all["prompt_id"].isin(top_prompts)]
        # Box plot without seaborn
        data = [sub[sub["prompt_id"]==pid]["reward_per_1k"].dropna().values for pid in top_prompts]
        if len(data) > 0:
            plt.figure(figsize=(10,5))
            plt.boxplot(data, labels=top_prompts, vert=True, showmeans=True)
            plt.xticks(rotation=45, ha="right")
            plt.title("Efficiency distribution (reward/1k) for top prompts")
            plt.ylabel("reward_per_1k")
            savefig(os.path.join(OUT_DIR, "eff_box_by_prompt.png"))

    # ---- Summary text ----
    lines = []
    def add_block(title, df):
        lines.append(f"== {title} ==")
        lines.append(df.round(3).to_string(index=False))
        lines.append("")
    add_block("BY JSON MODE (means)", by_json)
    add_block("BY TWO-PASS (means)", by_twopass)
    add_block("BY FEWSHOT STRATEGY (means)", by_fs)
    add_block("EFFICIENCY: BY FEW-SHOT (means)", by_k_eff)
    add_block("EFFICIENCY: BY JSON MODE (means)", by_json_eff)
    add_block("EFFICIENCY: BY TWO-PASS (means)", by_two_eff)
    add_block("EFFICIENCY: BY FEW-SHOT STRATEGY (means)", by_fs_eff)
    add_block("EFFICIENCY: BY JSON MODE × FEW-SHOT (means)", jm_k_eff)
    with open(os.path.join(RES_DIR, "jsonmode_summary.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("Extended analysis complete. See:")
    print(f" - Plots: {OUT_DIR}")
    print(f" - Tables: {RES_DIR}")

if __name__ == "__main__":
    main()

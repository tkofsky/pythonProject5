
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
    # Candidate log paths (relative)
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

    # ---- Summary text ----
    lines = []
    def add_block(title, df):
        lines.append(f"== {title} ==")
        lines.append(df.round(3).to_string(index=False))
        lines.append("")
    add_block("BY JSON MODE (means)", by_json)
    add_block("BY TWO-PASS (means)", by_twopass)
    add_block("BY FEWSHOT STRATEGY (means)", by_fs)
    with open(os.path.join(RES_DIR, "jsonmode_summary.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("Extended analysis complete. See:")
    print(f" - Plots: {OUT_DIR}")
    print(f" - Tables: {RES_DIR}")

if __name__ == "__main__":
    main()


"""
analyze_bandit_full_suite.py

One-stop analyzer that merges:
1) Classic per-field analysis (F1 distributions, trends, few-shot & temperature effects, validity, correlations)
2) Extended breakouts by json_mode / two_pass / fewshot_strategy, including interactions
3) Efficiency analysis (reward-per-1k tokens)
4) Optional prompt-variant context if prompt_variants.json is present

It will read any of these logs if present:
- bandit_fewshot_agent_log.csv
- bandit_fewshot_agent_log_experiments.csv
(and their /mnt/data counterparts)

Outputs:
- Tables in results_full/
- Plots in agent_plots_full/
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

OUT_DIR = "agent_plots_full"
RES_DIR = "results_full"
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(RES_DIR, exist_ok=True)

FIELDS = ["f1_intent","f1_entities","f1_constraints","f1_urgency","f1_steps"]

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

def coerce_columns(df: pd.DataFrame) -> pd.DataFrame:
    # numeric
    num_cols = ["iteration","example_count","temperature","reward","tokens",
                "f1_intent","f1_entities","f1_constraints","f1_urgency","f1_steps",
                "reward_per_1k","json_valid"]
    for c in num_cols:
        if c not in df.columns:
            df[c] = np.nan
        df[c] = pd.to_numeric(df[c], errors="coerce")
    # categorical/bool-ish
    for c in ["json_mode","two_pass","fewshot_strategy","prompt_id","phase","category","intent"]:
        if c not in df.columns:
            df[c] = np.nan if c in ["json_mode","two_pass"] else ""
    if df["fewshot_strategy"].isna().all():
        df["fewshot_strategy"] = "static"
    if df["phase"].eq("").all() or df["phase"].isna().all():
        df["phase"] = "bandit"
    # helpers
    df["f1_mean"] = df[[col for col in FIELDS if col in df.columns]].mean(axis=1, skipna=True)
    if "json_valid" not in df.columns or df["json_valid"].isna().all():
        # Proxy validity if not logged: f1_mean > 0
        df["json_valid"] = np.where(df["f1_mean"].notna() & (df["f1_mean"] > 0), 1.0, 0.0)
    # efficiency proxy if missing
    if "reward_per_1k" not in df.columns or df["reward_per_1k"].isna().all():
        df["reward_per_1k"] = np.where(df["tokens"].gt(0), df["reward"] / (df["tokens"]/1000.0), np.nan)
    return df

def maybe_merge_prompt_metadata(df: pd.DataFrame) -> pd.DataFrame:
    # If prompt_variants.json is present, merge category/intent/template (if missing) by prompt_id
    pv_paths = [
        "prompt_variants.json",
        "/mnt/data/prompt_variants.json"
    ]
    pv_path = None
    for p in pv_paths:
        if os.path.exists(p):
            pv_path = p
            break
    if pv_path is None:
        return df
    try:
        with open(pv_path, "r", encoding="utf-8") as f:
            variants = json.load(f)
        map_rows = []
        for v in variants:
            map_rows.append({
                "prompt_id": v.get("id",""),
                "pv_category": v.get("category",""),
                "pv_intent": v.get("intent",""),
                "pv_template": v.get("template","")
            })
        meta = pd.DataFrame(map_rows)
        df = df.merge(meta, on="prompt_id", how="left")
        # If category/intent columns exist but are empty, fill from pv_*
        if "category" in df.columns:
            df["category"] = df["category"].replace("", np.nan).fillna(df["pv_category"]).fillna("")
        else:
            df["category"] = df["pv_category"].fillna("")
        if "intent" in df.columns:
            df["intent"] = df["intent"].replace("", np.nan).fillna(df["pv_intent"]).fillna("")
        else:
            df["intent"] = df["pv_intent"].fillna("")
    except Exception:
        pass
    return df

def savefig(path):
    plt.tight_layout()
    plt.savefig(path, dpi=140, bbox_inches="tight")
    plt.close()

def classic_per_field(df):
    # Summary table per-field
    summary = df[[c for c in FIELDS if c in df.columns]].describe().round(3)
    summary.to_csv(os.path.join(RES_DIR, "fields_summary.csv"))
    print("Per-field F1 summary:\n", summary)

    # 1) Box: per-field distributions
    melt = df.melt(id_vars=[], value_vars=[c for c in FIELDS if c in df.columns],
                   var_name="field", value_name="f1")
    if not melt.empty:
        plt.figure(figsize=(10,6))
        fields = [c for c in FIELDS if c in df.columns]
        data = [melt[melt["field"]==f]["f1"].dropna().values for f in fields]
        plt.boxplot(data, labels=fields, vert=True, showmeans=True)
        plt.title("Per-field F1 Distributions")
        plt.xlabel("Field"); plt.ylabel("F1")
        savefig(os.path.join(OUT_DIR, "fields_box_distributions.png"))

    # 2) Trend over iterations (avg per field)
    if "iteration" in df.columns and df["iteration"].notna().any() and not melt.empty:
        trend = (melt.join(df[["iteration"]])
                    .groupby(["field","iteration"])["f1"].mean().reset_index())
        plt.figure(figsize=(11,6))
        for name, sub in trend.groupby("field"):
            plt.plot(sub["iteration"], sub["f1"], marker="o", label=name)
        plt.legend()
        plt.title("Per-field F1 Trend over Iterations (mean)")
        plt.xlabel("Iteration"); plt.ylabel("F1 (mean)")
        savefig(os.path.join(OUT_DIR, "fields_trend_iterations.png"))

    # 3) Few-shot effect per field
    if "example_count" in df.columns and not melt.empty:
        melt_fs = melt.join(df[["example_count"]])
        grouped = (melt_fs.groupby(["example_count","field"])["f1"].mean().reset_index())
        ks = sorted(grouped["example_count"].dropna().unique())
        if len(ks) > 0:
            plt.figure(figsize=(11,6))
            width = 0.12
            fields_u = sorted(grouped["field"].unique())
            for idx, field in enumerate(fields_u):
                vals = [grouped[(grouped["field"]==field) & (grouped["example_count"]==k)]["f1"].mean() for k in ks]
                positions = np.arange(len(ks)) + (idx - len(fields_u)/2)*width*1.5
                plt.bar(positions, vals, width=width, label=field)
            plt.xticks(np.arange(len(ks)), [str(int(k)) for k in ks])
            plt.title("Few-shot Level vs Per-field F1 (mean)")
            plt.xlabel("example_count"); plt.ylabel("F1 (mean)"); plt.legend()
            savefig(os.path.join(OUT_DIR, "fields_by_fewshot.png"))

    # 4) Temperature effect per field
    if "temperature" in df.columns and not melt.empty:
        melt_temp = melt.join(df[["temperature"]])
        grouped = (melt_temp.groupby(["temperature","field"])["f1"].mean().reset_index())
        temps = sorted(grouped["temperature"].dropna().unique())
        if len(temps) > 0:
            plt.figure(figsize=(11,6))
            width = 0.12
            fields_u = sorted(grouped["field"].unique())
            for idx, field in enumerate(fields_u):
                vals = [grouped[(grouped["field"]==field) & (grouped["temperature"]==t)]["f1"].mean() for t in temps]
                positions = np.arange(len(temps)) + (idx - len(fields_u)/2)*width*1.5
                plt.bar(positions, vals, width=width, label=field)
            plt.xticks(np.arange(len(temps)), [str(t) for t in temps])
            plt.title("Temperature vs Per-field F1 (mean)")
            plt.xlabel("temperature"); plt.ylabel("F1 (mean)"); plt.legend()
            savefig(os.path.join(OUT_DIR, "fields_by_temperature.png"))

    # 5) JSON Validity by factors
    if "example_count" in df.columns:
        val_by_k = df.groupby("example_count")["json_valid"].mean().reset_index()
        if not val_by_k.empty:
            plt.figure(figsize=(7,5))
            plt.bar(val_by_k["example_count"].astype(str), val_by_k["json_valid"].values)
            plt.title("JSON Validity by Few-shot Level")
            plt.xlabel("example_count"); plt.ylabel("Validity rate")
            plt.ylim(0,1)
            savefig(os.path.join(OUT_DIR, "json_valid_by_fewshot.png"))
    if "temperature" in df.columns:
        val_by_t = df.groupby("temperature")["json_valid"].mean().reset_index()
        if not val_by_t.empty:
            plt.figure(figsize=(7,5))
            plt.bar(val_by_t["temperature"].astype(str), val_by_t["json_valid"].values)
            plt.title("JSON Validity by Temperature")
            plt.xlabel("temperature"); plt.ylabel("Validity rate")
            plt.ylim(0,1)
            savefig(os.path.join(OUT_DIR, "json_valid_by_temperature.png"))

    # 6) Correlation: reward vs fields (if reward present)
    if "reward" in df.columns and df["reward"].notna().any():
        corr_cols = ["reward","json_valid"] + [c for c in FIELDS if c in df.columns]
        corr = df[corr_cols].corr().round(2)
        corr.to_csv(os.path.join(RES_DIR, "corr_reward_vs_fields.csv"))
        plt.figure(figsize=(7,5))
        im = plt.imshow(corr.values, vmin=-1, vmax=1)
        plt.colorbar(im, fraction=0.046, pad=0.04)
        plt.xticks(range(len(corr.columns)), corr.columns, rotation=45, ha="right")
        plt.yticks(range(len(corr.index)), corr.index)
        plt.title("Correlation: Reward vs Per-field F1 & Validity")
        savefig(os.path.join(OUT_DIR, "corr_reward_vs_fields.png"))

def extended_breakouts(df):
    agg_cols = ["f1_mean","reward","reward_per_1k","json_valid"]

    # JSON mode
    if "json_mode" in df.columns:
        by_json = (df.groupby("json_mode")[agg_cols].mean().reset_index())
        by_json.to_csv(os.path.join(RES_DIR, "by_json_mode.csv"), index=False)
        plt.figure(figsize=(7,5))
        plt.bar(by_json["json_mode"].astype(str).fillna("nan"), by_json["f1_mean"].values)
        plt.ylim(0,1); plt.title("Mean F1 by JSON Mode"); plt.xlabel("json_mode"); plt.ylabel("F1 (mean)")
        savefig(os.path.join(OUT_DIR, "mean_f1_by_json_mode.png"))

        plt.figure(figsize=(7,5))
        plt.bar(by_json["json_mode"].astype(str).fillna("nan"), by_json["json_valid"].values)
        plt.ylim(0,1); plt.title("JSON Validity by JSON Mode"); plt.xlabel("json_mode"); plt.ylabel("Validity rate")
        savefig(os.path.join(OUT_DIR, "validity_by_json_mode.png"))

        # Interactions
        if "temperature" in df.columns:
            jm_temp = (df.groupby(["json_mode","temperature"])[agg_cols].mean().reset_index())
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

        if "example_count" in df.columns:
            jm_k = (df.groupby(["json_mode","example_count"])[agg_cols].mean().reset_index())
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

    # Two-pass
    if "two_pass" in df.columns:
        by_tp = (df.groupby("two_pass")[agg_cols].mean().reset_index())
        by_tp.to_csv(os.path.join(RES_DIR, "by_two_pass.csv"), index=False)
        plt.figure(figsize=(7,5))
        plt.bar(by_tp["two_pass"].astype(str).fillna("nan"), by_tp["f1_mean"].values)
        plt.ylim(0,1); plt.title("Mean F1 by Two-Pass"); plt.xlabel("two_pass"); plt.ylabel("F1 (mean)")
        savefig(os.path.join(OUT_DIR, "mean_f1_by_two_pass.png"))

    # Few-shot strategy
    if "fewshot_strategy" in df.columns:
        by_fs = (df.groupby("fewshot_strategy")[agg_cols].mean().reset_index())
        by_fs.to_csv(os.path.join(RES_DIR, "by_fewshot_strategy.csv"), index=False)
        plt.figure(figsize=(7,5))
        plt.bar(by_fs["fewshot_strategy"].astype(str), by_fs["f1_mean"].values)
        plt.ylim(0,1); plt.title("Mean F1 by Few-shot Strategy"); plt.xlabel("fewshot_strategy"); plt.ylabel("F1 (mean)")
        savefig(os.path.join(OUT_DIR, "mean_f1_by_fewshot_strategy.png"))

    # Optional: performance by prompt category if available
    if "category" in df.columns and df["category"].notna().any():
        by_cat = (df.groupby("category")[["f1_mean","reward","reward_per_1k","json_valid"]].mean().reset_index())
        by_cat.to_csv(os.path.join(RES_DIR, "by_prompt_category.csv"), index=False)
        plt.figure(figsize=(9,5))
        plt.bar(by_cat["category"].astype(str), by_cat["f1_mean"].values)
        plt.xticks(rotation=30, ha="right")
        plt.ylim(0,1); plt.title("Mean F1 by Prompt Category"); plt.xlabel("category"); plt.ylabel("F1 (mean)")
        savefig(os.path.join(OUT_DIR, "mean_f1_by_prompt_category.png"))

def efficiency_views(df):
    eff_cols = ["reward_per_1k","reward","f1_mean","json_valid"]

    # by k
    if "example_count" in df.columns:
        by_k = (df.groupby("example_count")[eff_cols].mean().reset_index())
        by_k.to_csv(os.path.join(RES_DIR, "eff_by_example_count.csv"), index=False)
        plt.figure(figsize=(7,5))
        plt.bar(by_k["example_count"].astype(str), by_k["reward_per_1k"].values)
        plt.title("Efficiency: Reward per 1k tokens by Few-shot Level")
        plt.xlabel("example_count"); plt.ylabel("reward_per_1k (mean)")
        savefig(os.path.join(OUT_DIR, "eff_reward_per_1k_by_k.png"))

    # by json mode
    if "json_mode" in df.columns:
        by_json = (df.groupby("json_mode")[eff_cols].mean().reset_index())
        by_json.to_csv(os.path.join(RES_DIR, "eff_by_json_mode.csv"), index=False)
        plt.figure(figsize=(7,5))
        plt.bar(by_json["json_mode"].astype(str).fillna("nan"), by_json["reward_per_1k"].values)
        plt.title("Efficiency: Reward per 1k tokens by JSON Mode")
        plt.xlabel("json_mode"); plt.ylabel("reward_per_1k (mean)")
        savefig(os.path.join(OUT_DIR, "eff_reward_per_1k_by_json_mode.png"))

    # by two-pass
    if "two_pass" in df.columns:
        by_tp = (df.groupby("two_pass")[eff_cols].mean().reset_index())
        by_tp.to_csv(os.path.join(RES_DIR, "eff_by_two_pass.csv"), index=False)
        plt.figure(figsize=(7,5))
        plt.bar(by_tp["two_pass"].astype(str).fillna("nan"), by_tp["reward_per_1k"].values)
        plt.title("Efficiency: Reward per 1k tokens by Two-Pass")
        plt.xlabel("two_pass"); plt.ylabel("reward_per_1k (mean)")
        savefig(os.path.join(OUT_DIR, "eff_reward_per_1k_by_two_pass.png"))

    # by few-shot strategy
    if "fewshot_strategy" in df.columns:
        by_fs = (df.groupby("fewshot_strategy")[eff_cols].mean().reset_index())
        by_fs.to_csv(os.path.join(RES_DIR, "eff_by_fewshot_strategy.csv"), index=False)
        plt.figure(figsize=(7,5))
        plt.bar(by_fs["fewshot_strategy"].astype(str), by_fs["reward_per_1k"].values)
        plt.title("Efficiency: Reward per 1k tokens by Few-shot Strategy")
        plt.xlabel("fewshot_strategy"); plt.ylabel("reward_per_1k (mean)")
        savefig(os.path.join(OUT_DIR, "eff_reward_per_1k_by_fewshot_strategy.png"))

    # interaction: json_mode × k
    if "json_mode" in df.columns and "example_count" in df.columns:
        jm_k = (df.groupby(["json_mode","example_count"])[eff_cols].mean().reset_index())
        jm_k.to_csv(os.path.join(RES_DIR, "eff_by_json_mode_x_example_count.csv"), index=False)
        if not jm_k.empty:
            pv = jm_k.pivot(index="json_mode", columns="example_count", values="reward_per_1k")
            if pv.shape[0] > 0 and pv.shape[1] > 0:
                plt.figure(figsize=(6,4))
                im = plt.imshow(pv.values, aspect="auto")
                plt.colorbar(im, fraction=0.046, pad=0.04)
                plt.xticks(range(len(pv.columns)), [str(c) for c in pv.columns])
                plt.yticks(range(len(pv.index)), [str(i) for i in pv.index])
                plt.title("Efficiency (reward/1k): json_mode × example_count")
                plt.xlabel("example_count"); plt.ylabel("json_mode")
                savefig(os.path.join(OUT_DIR, "heatmap_eff_jsonmode_by_k.png"))

    # prompt-level distribution
    if "prompt_id" in df.columns and df["prompt_id"].notna().any():
        top_prompts = (df.groupby("prompt_id")["reward_per_1k"].mean().sort_values(ascending=False).head(12).index.tolist())
        if len(top_prompts) > 0:
            sub = df[df["prompt_id"].isin(top_prompts)]
            data = [sub[sub["prompt_id"]==pid]["reward_per_1k"].dropna().values for pid in top_prompts]
            plt.figure(figsize=(10,5))
            plt.boxplot(data, labels=top_prompts, vert=True, showmeans=True)
            plt.xticks(rotation=45, ha="right")
            plt.title("Efficiency distribution (reward/1k) for top prompts")
            plt.ylabel("reward_per_1k")
            savefig(os.path.join(OUT_DIR, "eff_box_by_prompt.png"))

def write_summary_text(df):
    lines = []

    def add_block(title, frame):
        if isinstance(frame, pd.DataFrame):
            show = frame.round(3)
        else:
            show = frame
        lines.append(f"== {title} ==")
        lines.append(str(show))
        lines.append("")

    # Per-field summary
    fields_present = [c for c in FIELDS if c in df.columns]
    if fields_present:
        add_block("Per-field F1 describe()", df[fields_present].describe())
    # By JSON mode, two-pass, fewshot strategy
    agg_cols = ["f1_mean","reward","reward_per_1k","json_valid"]
    if "json_mode" in df.columns:
        add_block("BY JSON MODE (means)", df.groupby("json_mode")[agg_cols].mean())
    if "two_pass" in df.columns:
        add_block("BY TWO-PASS (means)", df.groupby("two_pass")[agg_cols].mean())
    if "fewshot_strategy" in df.columns:
        add_block("BY FEW-SHOT STRATEGY (means)", df.groupby("fewshot_strategy")[agg_cols].mean())
    # By example_count and temperature
    if "example_count" in df.columns:
        add_block("BY FEW-SHOT LEVEL (means)", df.groupby("example_count")[agg_cols].mean())
    if "temperature" in df.columns:
        add_block("BY TEMPERATURE (means)", df.groupby("temperature")[agg_cols].mean())
    # By prompt category if available
    if "category" in df.columns and df["category"].notna().any():
        add_block("BY PROMPT CATEGORY (means)", df.groupby("category")[agg_cols].mean())

    with open(os.path.join(RES_DIR, "full_summary.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

def main():
    # Candidate log paths (relative + sandbox paths)
    paths = [
        "bandit_fewshot_agent_log.csv",
        "bandit_fewshot_agent_log_experiments.csv",
        "/mnt/data/bandit_fewshot_agent_log.csv",
        "/mnt/data/bandit_fewshot_agent_log_experiments.csv",
    ]
    df = load_any_logs(paths)
    if df.empty:
        print("No logs found. Place a CSV log next to this script (see docstring).")
        return

    df = coerce_columns(df)
    df = maybe_merge_prompt_metadata(df)

    # Classic F1-only analytics
    classic_per_field(df)

    # Extended breakouts
    extended_breakouts(df)

    # Efficiency views
    efficiency_views(df)

    # Summary text
    write_summary_text(df)

    print("Full-suite analysis complete.")
    print(f" - Plots: {OUT_DIR}")
    print(f" - Tables: {RES_DIR}")

if __name__ == "__main__":
    main()

# Re-run the analyzer file creation (the environment was reset).

import os, json, math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

LOG = "bandit_fewshot_agent_log_router_hybrid_steps_patch.csv"
#LOG = "bandit_fewshot_agent_log_router_hybrid_steps_weighted.csv"

OUT = "router_hybrid_steps_reports"
os.makedirs(OUT, exist_ok=True)

# ---------- load ----------
try:
    df = pd.read_csv(LOG, encoding="utf-8-sig")
except UnicodeDecodeError:
    df = pd.read_csv(LOG, encoding="latin-1")

# hygiene & types
for col in ["iteration","example_count","is_mutation","two_pass"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
for col in ["temperature","reward","reward_per_1k","json_valid","f1_intent","f1_entities","f1_constraints","f1_urgency","f1_steps","tokens"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# ---------- JSON helpers ----------
FIELDS = ["intent","entities","constraints","urgency","steps"]

def safe_json(s: str):
    if not isinstance(s, str):
        return None
    try:
        return json.loads(s)
    except Exception:
        pass
    if "{" in s and "}" in s:
        chunk = s[s.find("{"):s.rfind("}")+1]
        try:
            return json.loads(chunk)
        except Exception:
            return None
    return None

def _to_listish(x):
    if x is None: return []
    if isinstance(x, (list, tuple, set)):
        return ["" if v is None else str(v) for v in x]
    if isinstance(x, dict):
        return ["" if v is None else str(v) for v in x.values()]
    return [str(x)]

def _to_dictish(x):
    if isinstance(x, dict): return x
    if x is None: return {}
    if isinstance(x, (list, tuple)):
        return {str(i): v for i, v in enumerate(x)}
    return {"value": x}

def normalize_phrase(s: str) -> str:
    CANON_MAP = {
        "search train schedules": "search trains",
        "look up trains": "search trains",
        "search schedules": "search schedule",
        "choose best option": "choose best",
        "select best option": "choose best",
        "finalize booking": "book",
        "send confirmation email": "send confirmation",
        "apply price filter": "filter by price",
        "filter by cost": "filter by price",
        "filter by price and time": "filter by price and time",
        "find best departure": "choose best",
        "confirm booking": "send confirmation"
    }
    s = s.lower().strip()
    s = s.replace("schedules", "schedule").replace("trains", "train")
    for k, v in CANON_MAP.items():
        s = s.replace(k, v)
    for junk in [" the ", " a ", " an "]:
        s = s.replace(junk, " ")
    return " ".join(s.split())

def normalize_steps(x):
    if isinstance(x, list):
        return [normalize_phrase(str(t)) for t in x]
    return x

def steps_anchor_ok(pred_json: dict) -> bool:
    if not isinstance(pred_json, dict): return False
    steps = pred_json.get("steps")
    if not isinstance(steps, list): return False
    if len(steps) < 1: return False
    ents = _to_dictish(pred_json.get("entities"))
    ent_txt = " ".join(_to_listish(ents)).lower()
    entity_tokens = [tok for tok in ent_txt.split() if tok.isalpha() and len(tok) > 2]
    txt = " ".join([str(s) for s in steps]).lower()
    return any(tok in txt for tok in entity_tokens) if entity_tokens else True

# ---------- summary by router mode ----------
keep_cols = ["router_mode","reward","reward_per_1k","json_valid","f1_intent","f1_entities","f1_constraints","f1_urgency","f1_steps"]
summary = (df[keep_cols]
           .groupby("router_mode")
           .agg(["mean","median","std","count"])
           .round(4))
summary.to_csv(os.path.join(OUT, "summary_by_router_mode.csv"))

# ---------- bar chart: reward & F1 by mode ----------
metrics_to_plot = ["reward","reward_per_1k","f1_intent","f1_entities","f1_constraints","f1_urgency","f1_steps"]
means = df.groupby("router_mode")[metrics_to_plot].mean().reindex(["one_pass","two_pass"]).round(3)
plt.figure(figsize=(10,6))
means.plot(kind="bar")
plt.title("Mean Metrics by Router Mode")
plt.ylabel("Mean value")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "mode_metrics_bar.png"), dpi=140)
plt.close()

# ---------- paired deltas (two-pass – one-pass) ----------
# Pair on (input, prompt_id, example_count, temperature). Use latest two rows if duplicates.
pair_keys = ["input","prompt_id","example_count","temperature"]
df_sorted = df.sort_values("iteration")
pairs = []
for key, g in df_sorted.groupby(pair_keys):
    if {"one_pass","two_pass"} <= set(g["router_mode"].unique()):
        # pick last row of each mode to reduce duplicate noise
        g_last = g.groupby("router_mode").tail(1).set_index("router_mode")
        if "one_pass" in g_last.index and "two_pass" in g_last.index:
            r1 = g_last.loc["one_pass"]
            r2 = g_last.loc["two_pass"]
            pairs.append({
                "input": key[0],
                "prompt_id": key[1],
                "example_count": key[2],
                "temperature": key[3],
                "reward_delta": float(r2["reward"]) - float(r1["reward"]),
                "f1_steps_delta": float(r2["f1_steps"]) - float(r1["f1_steps"]),
                "one_pass_reward": float(r1["reward"]),
                "two_pass_reward": float(r2["reward"]),
                "one_pass_f1_steps": float(r1["f1_steps"]),
                "two_pass_f1_steps": float(r2["f1_steps"]),
            })
paired = pd.DataFrame(pairs)
paired.to_csv(os.path.join(OUT, "paired_deltas_steps.csv"), index=False)

# histograms for deltas
if not paired.empty:
    plt.figure(figsize=(7,5))
    plt.hist(paired["f1_steps_delta"].dropna(), bins=20)
    plt.title("Δ F1_steps (two-pass − one-pass)")
    plt.xlabel("delta"); plt.ylabel("# pairs")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "delta_hist_f1_steps.png"), dpi=140)
    plt.close()

    plt.figure(figsize=(7,5))
    plt.hist(paired["reward_delta"].dropna(), bins=20)
    plt.title("Δ Reward (two-pass − one-pass)")
    plt.xlabel("delta"); plt.ylabel("# pairs")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "delta_hist_reward.png"), dpi=140)
    plt.close()

# ---------- anchor rate by router mode ----------
anchor_rows = []
for _, r in df.iterrows():
    pred = safe_json(str(r.get("output","")))
    if isinstance(pred, dict) and "steps" in pred:
        # normalize for analysis
        pred_norm = dict(pred)
        pred_norm["steps"] = normalize_steps(pred_norm.get("steps"))
        anchor_ok = steps_anchor_ok(pred_norm)
    else:
        anchor_ok = False
    anchor_rows.append({
        "router_mode": r.get("router_mode",""),
        "anchor_ok": 1.0 if anchor_ok else 0.0
    })
anchor_df = pd.DataFrame(anchor_rows)
anchor_summary = anchor_df.groupby("router_mode")["anchor_ok"].mean().reset_index().rename(columns={"anchor_ok":"anchor_rate"})
anchor_summary.to_csv(os.path.join(OUT, "steps_anchor_rate_by_mode.csv"), index=False)

# ---------- top failures (very low F1_steps) ----------
fail_mask = (df["f1_steps"].fillna(0) < 0.1)
cols = ["iteration","router_mode","prompt_id","example_count","temperature","input","output","reference","f1_steps","reward"]
fails = df.loc[fail_mask, cols].sort_values(["f1_steps","reward"]).head(50)
fails.to_csv(os.path.join(OUT, "top_steps_failures.csv"), index=False)

# ---------- router confusion-style table ----------
pair_keys = ["input","prompt_id","example_count","temperature"]
df_sorted = df.sort_values("iteration")

conf_rows = []
for key, g in df_sorted.groupby(pair_keys):
    modes_here = set(g["router_mode"].dropna().unique().tolist())
    if not {"one_pass", "two_pass"} <= modes_here:
        continue

    # last observation for each mode
    g_last_by_mode = g.groupby("router_mode").tail(1).set_index("router_mode")
    if not {"one_pass","two_pass"} <= set(g_last_by_mode.index):
        continue

    r1 = g_last_by_mode.loc["one_pass"]
    r2 = g_last_by_mode.loc["two_pass"]

    # which is actually better on reward?
    actual_improve = float(r2["reward"]) - float(r1["reward"])  # > 0 → two-pass better

    # which mode was *actually chosen last* (latest iteration among both)
    latest_idx = g["iteration"].idxmax()
    chosen_mode = str(g.loc[latest_idx, "router_mode"])

    if chosen_mode == "two_pass" and actual_improve > 0:
        bucket = "2+ (chose two-pass, helped)"
    elif chosen_mode == "two_pass" and actual_improve <= 0:
        bucket = "2− (chose two-pass, hurt/no gain)"
    elif chosen_mode == "one_pass" and actual_improve <= 0:
        bucket = "1+ (chose one-pass, helped)"
    else:
        bucket = "1− (chose one-pass, hurt/no gain)"

    conf_rows.append({"bucket": bucket})

# Build table safely even if there are no rows
if len(conf_rows) == 0:
    conf_tab = pd.DataFrame(columns=["bucket","count"])
    conf_tab.to_csv(os.path.join(OUT, "router_confusion_table.csv"), index=False)
    # also avoid plotting when empty
else:
    conf_df = pd.DataFrame(conf_rows)
    conf_tab = conf_df.groupby("bucket").size().reset_index(name="count").sort_values("count", ascending=False)
    conf_tab.to_csv(os.path.join(OUT, "router_confusion_table.csv"), index=False)

    # simple bar plot for confusion
    plt.figure(figsize=(8,5))
    plt.bar(conf_tab["bucket"], conf_tab["count"])
    plt.title("Routing decision vs actual improvement")
    plt.xlabel("bucket"); plt.ylabel("count")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "router_confusion_table.png"), dpi=140)
    plt.close()

print(f"All reports saved to: {OUT}")

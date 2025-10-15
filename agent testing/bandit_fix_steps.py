"""
Bandit runner tuned for improving F1_steps on the provided dataset format.
Now uses **hardcoded paths** for agent_dataset.json and prompt_variants.json.

Key features
- Schema-first validation + targeted repair
- Two-pass plan→emit with dynamic routing (one-pass vs two-pass)
- Discounted-UCB bandit with step-weighted reward
- Context heuristics: predict_steps_difficulty, plan_confidence, adapt_params
- Rich CSV logging incl. routing diagnostics

Run (example):
  python bandit_steps_priority.py --out runs/exp_agent.csv --model gpt-4o-mini
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import hashlib
import json
import math
import os
import random
import string
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

############################
# Hardcoded file locations #
############################
AGENT_DATASET_PATH = "agent_dataset.json"
PROMPT_VARIANTS_PATH = "prompt_variants.json"

############################
# Arm / Prompt definitions #
############################

@dataclass(frozen=True)
class ArmConfig:
    arm_id: str
    prompt_style: str  # 'schema_min', 'schema_verbose', 'reason_hidden'
    few_shot: int      # 0,1,3
    temperature: float # 0.0 - 1.0
    two_pass: bool     # True → plan then emit JSON

DEFAULT_ARMS: List[ArmConfig] = [
    ArmConfig("min_0p_T0_one", "schema_min", 0, 0.0, False),
    ArmConfig("min_1p_T0_one", "schema_min", 1, 0.0, False),
    ArmConfig("verb_1p_T0_two", "schema_verbose", 1, 0.0, True),
    ArmConfig("verb_3p_T0_two", "schema_verbose", 3, 0.0, True),
    ArmConfig("reason_1p_T02_two", "reason_hidden", 1, 0.2, True),
    ArmConfig("reason_3p_T02_two", "reason_hidden", 3, 0.2, True),
]

########################
# Simple JSON "schema" #
########################

REQUIRED_TOP_LEVEL = ["intent", "entities", "urgency", "constraints", "steps"]


def validate_output(obj: Dict[str, Any]) -> Tuple[bool, List[str]]:
    errors: List[str] = []
    if not isinstance(obj, dict):
        return False, ["Output is not a JSON object"]
    for k in REQUIRED_TOP_LEVEL:
        if k not in obj:
            errors.append(f"Missing key: {k}")
    if errors:
        return False, errors

    if not isinstance(obj["intent"], str):
        errors.append("intent must be string")
    if not isinstance(obj["entities"], list) or not all(isinstance(x, str) for x in obj["entities"]):
        errors.append("entities must be list[str]")
    if not isinstance(obj["urgency"], str):
        errors.append("urgency must be string")
    if not isinstance(obj["constraints"], list) or not all(isinstance(x, str) for x in obj["constraints"]):
        errors.append("constraints must be list[str]")
    steps = obj.get("steps")
    if not isinstance(steps, list):
        errors.append("steps must be list")
        return False, errors
    for i, s in enumerate(steps):
        if not isinstance(s, dict):
            errors.append(f"steps[{i}] not object")
            continue
        if not isinstance(s.get("id"), int):
            errors.append(f"steps[{i}].id must be int")
        if not isinstance(s.get("action"), str) or not s.get("action"):
            errors.append(f"steps[{i}].action must be non-empty string")
        obj_field = s.get("object")
        if obj_field is not None and not isinstance(obj_field, str):
            errors.append(f"steps[{i}].object must be string or null")

    return (len(errors) == 0), errors

###########################
# Reward & metric helpers #
###########################

def _tokenize(s: str) -> List[str]:
    return [t for t in s.lower().translate(str.maketrans("", "", string.punctuation)).split() if t]


def simple_f1(pred_tokens: List[str], ref_tokens: List[str]) -> float:
    if not pred_tokens and not ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0
    pred_set, ref_set = set(pred_tokens), set(ref_tokens)
    inter = len(pred_set & ref_set)
    if not pred_set or not ref_set:
        return 0.0
    prec = inter / len(pred_set)
    rec = inter / len(ref_set)
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)

@dataclass
class FieldF1:
    f1_intent: float
    f1_entities: float
    f1_urgency: float
    f1_constraints: float
    f1_steps: float


def normalize_reference(ref: Dict[str, Any]) -> Dict[str, Any]:
    """Map dataset reference into internal comparable shape.
    - entities: take all leaf string values and flatten to list[str]
    - steps: list[str] of labels → treat first token as verb; keep tokens bag
    """
    ent_vals: List[str] = []
    ents = ref.get("entities", {})
    if isinstance(ents, dict):
        for v in ents.values():
            if isinstance(v, str):
                ent_vals.append(v)
            elif isinstance(v, (list, tuple)):
                ent_vals.extend([str(x) for x in v])
            else:
                ent_vals.append(str(v))
    elif isinstance(ents, list):
        ent_vals.extend([str(x) for x in ents])

    constraints = ref.get("constraints") or []
    if not isinstance(constraints, list):
        constraints = [str(constraints)]

    ref_steps = ref.get("steps") or []
    if not isinstance(ref_steps, list):
        ref_steps = [str(ref_steps)]

    return {
        "intent": ref.get("intent", ""),
        "entities": ent_vals,
        "urgency": ref.get("urgency", ""),
        "constraints": constraints,
        "steps": ref_steps,
    }


def compute_field_f1(pred: Dict[str, Any], ref: Dict[str, Any]) -> FieldF1:
    refn = normalize_reference(ref)

    intent = simple_f1(_tokenize(pred.get("intent", "")), _tokenize(refn["intent"]))

    pred_entities = pred.get("entities", [])
    if not isinstance(pred_entities, list):
        pred_entities = [str(pred_entities)]
    entities = simple_f1(_tokenize(" ".join(map(str, pred_entities))), _tokenize(" ".join(refn["entities"])))

    urgency = 1.0 if str(pred.get("urgency", "")).lower() == str(refn["urgency"]).lower() else 0.0

    pred_constraints = pred.get("constraints", [])
    if not isinstance(pred_constraints, list):
        pred_constraints = [str(pred_constraints)]
    constraints = simple_f1(_tokenize(" ".join(map(str, pred_constraints))), _tokenize(" ".join(refn["constraints"])))

    # Steps F1: compare action verbs to the reference step labels (first token heuristic)
    pred_steps = pred.get("steps") or []
    pred_actions = [s.get("action", "") for s in pred_steps if isinstance(s, dict)]
    pred_tokens = _tokenize(" ".join(pred_actions))

    ref_step_verbs = [(_tokenize(s) or [""])[0] for s in refn["steps"]]
    ref_tokens = _tokenize(" ".join(ref_step_verbs))

    f1_steps = simple_f1(pred_tokens, ref_tokens)

    return FieldF1(intent, entities, urgency, constraints, f1_steps)

@dataclass
class RewardWeights:
    w_steps: float = 0.7
    w_constraints: float = 0.15
    w_other: float = 0.15  # split over intent/entities/urgency
    cost_per_1k: float = 0.03
    latency_penalty: float = 0.01


def shaped_reward(f: FieldF1, tokens: int, latency_s: float, w: RewardWeights) -> float:
    other = (f.f1_intent + f.f1_entities + f.f1_urgency) / 3.0
    cost_pen = (tokens / 1000.0) * w.cost_per_1k
    lat_pen = latency_s * w.latency_penalty
    return w.w_steps * f.f1_steps + w.w_constraints * f.f1_constraints + w.w_other * other - cost_pen - lat_pen

########################
# Bandit (Disc. UCB1) #
########################

@dataclass
class ArmStats:
    pulls: int = 0
    value: float = 0.0

class DiscountedUCB:
    def __init__(self, arm_ids: List[str], gamma: float = 0.98, c: float = 2.0, epsilon: float = 0.05):
        self.gamma = gamma
        self.c = c
        self.epsilon = epsilon
        self.t = 0
        self.stats: Dict[str, ArmStats] = {a: ArmStats() for a in arm_ids}

    def select(self) -> str:
        self.t += 1
        if random.random() < self.epsilon:
            return random.choice(list(self.stats.keys()))
        scores = {}
        total = max(1, sum(s.pulls for s in self.stats.values()))
        for aid, s in self.stats.items():
            if s.pulls == 0:
                scores[aid] = float("inf")
            else:
                bonus = self.c * math.sqrt(max(0.0, math.log(total)) / s.pulls)
                scores[aid] = s.value + bonus
        return max(scores.items(), key=lambda kv: kv[1])[0]

    def update(self, arm_id: str, reward: float):
        st = self.stats[arm_id]
        st.pulls += 1
        st.value = self.gamma * st.value + (1 - self.gamma) * reward

################
# LLM plumbing #
################

class LLMClient:
    def __init__(self, model: str = "gpt-4o-mini"):
        self.model = model

    def generate(self, prompt: str, temperature: float = 0.0, max_tokens: int = 1024) -> Tuple[str, int]:
        """Replace with your real LLM call. Stub emits valid JSON with plausible step verbs."""
        fake = {
            "intent": "book_flight",
            "entities": ["Toronto", "Vancouver", "next Friday", "$600", "direct"],
            "urgency": "normal",
            "constraints": ["budget<=600", "direct_only"],
            "steps": [
                {"id": 1, "action": "search", "object": "flights"},
                {"id": 2, "action": "filter", "object": "nonstop and price"},
                {"id": 3, "action": "present", "object": "top options"},
            ],
        }
        text = json.dumps(fake, ensure_ascii=False)
        tokens = max(1, len(prompt.split()) // 1 + len(text) // 4)
        return text, tokens

###########################
# Prompting / two-pass I/O #
###########################

# Default few-shot (will be overridden if PROMPT_VARIANTS_PATH exists)
FEW_SHOT_BLOCKS: Dict[str, Dict[int, str]] = {
    "schema_min": {
        1: (
            "Example (JSON only)
"
            '{"intent":"reset password","entities":["user"],"urgency":"low","constraints":["no phone"],'
            '"steps":[{"id":1,"action":"verify","object":"user"},{"id":2,"action":"send","object":"reset link"},{"id":3,"action":"confirm","object":"reset"}]}'
        ),
        3: "...",
    },
    "schema_verbose": {1: "Example (strict 1..N, verb-first).", 3: "..."},
    "reason_hidden": {1: "Reason privately; output ONLY JSON.", 3: "..."},
}


def _try_load_prompt_variants(path: str) -> None:
    global FEW_SHOT_BLOCKS
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        loaded: Dict[str, Dict[int, str]] = {}
        for style, inner in raw.items():
            loaded[style] = {}
            for k, val in inner.items():
                try:
                    kk = int(k)
                except Exception:
                    continue
                loaded[style][kk] = str(val)
        if loaded:
            FEW_SHOT_BLOCKS = loaded
    except FileNotFoundError:
        # optional file
        pass
    except Exception as e:
        print(f"Warning: failed to load prompt variants from {path}: {e}", file=sys.stderr)


# Attempt to load external few-shot variants at import time
_try_load_prompt_variants(PROMPT_VARIANTS_PATH)


def build_prompt(input_text: str, arm: ArmConfig, plan: Optional[str] = None) -> str:
    base_min = (
        "You must output ONLY strict JSON matching this schema:
"
        '{"intent": str, "entities": [str], "urgency": str, "constraints": [str], '
        '"steps": [{"id": int, "action": str, "object": str|null}]}
'
        "Rules: steps numbered 1..N, verb-first actions, executable order, 3–6 steps, no extra keys."
    )
    base_verbose = base_min + "
Be explicit and minimal: start with initialization if needed; end with a verification step."

    few = FEW_SHOT_BLOCKS.get(arm.prompt_style, {}).get(arm.few_shot, "")

    if arm.two_pass and plan is not None:
        return (
            f"{base_min}
Use this plan to populate steps in order; do not include the plan in output.
{plan}

"
            f"Input:
{input_text}

JSON:"
        )

    head = base_verbose if arm.prompt_style == "schema_verbose" else base_min
    if arm.prompt_style == "reason_hidden":
        head = head + "
Think step-by-step privately, then output ONLY the JSON."
    return f"{head}

{few}

Input:
{input_text}

JSON:"


def build_plan_prompt(input_text: str) -> str:
    return (
        "Plan numbered actions (1..N), each line 'action: object'. 3–6 lines, executable order. Do NOT output JSON.

"
        f"Input:
{input_text}

Plan:"
    )

#############################
# Structured output helpers #
#############################

# --- Dynamic routing preferences (favor steps reliability) ---
RELIABILITY_FIRST: bool = True  # always favor higher F1_steps
DIFF_THR_HIGH: float = 0.40     # difficulty >= this → 2-pass preferred
PLAN_CONF_THR_LOW: float = 0.50 # plan confidence < this → escalate to 2-pass


def try_json(s: str) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(s)
    except Exception:
        return None


def targeted_repair(raw_json_text: str, errors: List[str]) -> Dict[str, Any]:
    obj = try_json(raw_json_text) or {}
    out: Dict[str, Any] = {
        "intent": obj.get("intent") or "",
        "entities": obj.get("entities") if isinstance(obj.get("entities"), list) else [],
        "urgency": obj.get("urgency") if isinstance(obj.get("urgency"), str) else "low",
        "constraints": obj.get("constraints") if isinstance(obj.get("constraints"), list) else [],
        "steps": obj.get("steps") if isinstance(obj.get("steps"), list) else [],
    }
    fixed_steps = []
    for i, s in enumerate(out["steps"]):
        if not isinstance(s, dict):
            continue
        _id = s.get("id") if isinstance(s.get("id"), int) else len(fixed_steps) + 1
        _action = s.get("action") if isinstance(s.get("action"), str) and s.get("action") else "check"
        _object = s.get("object") if (s.get("object") is None or isinstance(s.get("object"), str)) else None
        fixed_steps.append({"id": int(_id), "action": _action, "object": _object})
    if len(fixed_steps) < 3:
        for k in range(len(fixed_steps) + 1, 4):
            fixed_steps.append({"id": k, "action": "validate", "object": None})
    out["steps"] = fixed_steps[:6]
    return out

#########################
# Context extraction     #
#########################

def extract_context_features(input_text: str) -> Dict[str, Any]:
    has_num = any(ch.isdigit() for ch in input_text)
    length = len(input_text.split())
    has_money = ("$" in input_text) or (" usd" in input_text.lower()) or (" eur" in input_text.lower())
    has_dates = any(
        tok.endswith(("/202", "-202", "202")) or any(m in tok.lower() for m in [
            "jan","feb","mar","apr","may","jun","jul","aug","sep","oct","nov","dec"
        ]) for tok in input_text.split()
    )
    bullet_like = any(b in input_text for b in ["- ", "•", "1.", "2."])
    return {"len": length, "has_num": has_num, "has_money": has_money, "has_dates": has_dates, "bullet_like": bullet_like}


def predict_steps_difficulty(ctx: Dict[str, Any]) -> float:
    """Heuristic 0..1 difficulty score for producing reliable steps."""
    score = 0.0
    score += 0.40 if ctx.get("len", 0) > 80 else 0.0
    score += 0.20 if ctx.get("has_num") else 0.0
    score += 0.15 if ctx.get("has_money") else 0.0
    score += 0.10 if ctx.get("has_dates") else 0.0
    score += 0.10 if not ctx.get("bullet_like") else 0.0  # lack of structure → harder
    return max(0.0, min(1.0, score))


def plan_confidence(plan_text: str) -> float:
    """Rough confidence from plan structure & verb diversity (0..1)."""
    lines = [l.strip(" -•	") for l in plan_text.splitlines() if l.strip()]
    if not lines:
        return 0.0
    verbs = []
    for l in lines:
        tok = (l.split()[0].lower() if l.split() else "")
        verbs.append(tok)
    unique_verbs = len(set(v for v in verbs if v))
    n = len(lines)
    len_score = min(1.0, max(0.0, (n - 2) / 4.0))  # 3..6 lines → ~0.25..1.0
    verb_div = min(1.0, unique_verbs / max(1.0, n))
    penalty_vague = 0.0
    vague = {"do","check","handle","fix","make","process"}
    penalty_vague += 0.2 if any(v in vague for v in verbs) else 0.0
    conf = max(0.0, min(1.0, 0.6 * len_score + 0.4 * verb_div - penalty_vague))
    return conf


def adapt_params(arm: ArmConfig, difficulty: float) -> ArmConfig:
    """Return a tweaked ArmConfig with K-shot & temperature adapted to difficulty."""
    K = 3 if difficulty >= 0.6 else (1 if difficulty >= 0.3 else 0)
    T = 0.2 if difficulty >= 0.5 else 0.0
    return ArmConfig(arm.arm_id, arm.prompt_style, K, T, arm.two_pass)


def choose_arm_by_context(arms: List[ArmConfig], ctx: Dict[str, Any]) -> Optional[str]:
    # Prefer higher reliability when requested
    if RELIABILITY_FIRST:
        if predict_steps_difficulty(ctx) >= DIFF_THR_HIGH:
            for a in arms:
                if a.prompt_style in ("schema_verbose", "reason_hidden") and a.two_pass:
                    return a.arm_id
    return None

################
# Trial Result #
################

@dataclass
class TrialResult:
    run_id: str
    iter_idx: int
    arm_id: str
    prompt_style: str
    few_shot: int
    temperature: float
    two_pass: bool
    input_hash: str
    model: str
    tokens: int
    latency_s: float
    f1_intent: float
    f1_entities: float
    f1_urgency: float
    f1_constraints: float
    f1_steps: float
    reward: float
    json_valid: int

################
# Runner logic #
################

def _load_agent_dataset_hardcoded() -> List[Dict[str, Any]]:
    with open(AGENT_DATASET_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    out = []
    for rec in data:
        input_text = rec.get("input", "")
        ref_raw = rec.get("reference", "{}")
        try:
            ref_dict = json.loads(ref_raw)
        except Exception:
            ref_dict = {}
        out.append({"input": input_text, "reference": ref_dict})
    return out


def run_bandit(dataset: List[Dict[str, Any]], arms: List[ArmConfig], model_name: str, out_csv: str,
               gamma: float = 0.98, c: float = 2.0, epsilon: float = 0.05,
               weights: RewardWeights = RewardWeights()) -> None:
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    bandit = DiscountedUCB([a.arm_id for a in arms], gamma=gamma, c=c, epsilon=epsilon)
    arm_by_id = {a.arm_id: a for a in arms}
    llm = LLMClient(model_name)

    fieldnames = [
        "run_id","iter","arm_id","prompt_style","few_shot","temperature","two_pass",
        "input_hash","model","tokens","latency_s",
        # routing diagnostics
        "routing_mode","difficulty","plan_conf","escalated","routing_reason",
        # metrics
        "f1_intent","f1_entities","f1_urgency","f1_constraints","f1_steps","reward","json_valid"
    ]
    run_id = time.strftime("%Y%m%d_%H%M%S")

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()

        for i, item in enumerate(dataset):
            inp = item["input"]
            ref = item["reference"]

            # --- Pre-selection: context & difficulty gating ---
            ctx = extract_context_features(inp)
            difficulty = predict_steps_difficulty(ctx)
            override = choose_arm_by_context(arms, ctx)
            selected_id = override if override is not None else bandit.select()
            arm = arm_by_id[selected_id]

            # Adapt params (few-shot, temperature) based on difficulty
            arm = adapt_params(arm, difficulty)

            t0 = time.time()
            tokens_used = 0

            # --- Two-step toggle: plan → emit if reliability is favored or input is hard ---
            use_two_pass = arm.two_pass or (RELIABILITY_FIRST and difficulty >= DIFF_THR_HIGH)
            routing_mode = "two_pass" if use_two_pass else "one_pass"
            routing_reason = []
            if override:
                routing_reason.append(f"override:{override}")
            routing_reason.append(f"diff={difficulty:.2f}")

            plan_conf_val = None
            escalated = 0

            if use_two_pass:
                plan_prompt = build_plan_prompt(inp)
                plan, tplan = llm.generate(plan_prompt, temperature=min(0.4, arm.temperature + 0.1), max_tokens=512)
                tokens_used += tplan

                plan_conf_val = plan_confidence(plan)
                routing_reason.append(f"plan_conf={plan_conf_val:.2f}")

                # Confidence check: escalate to reasoning style if plan is weak
                if plan_conf_val < PLAN_CONF_THR_LOW:
                    escalated = 1
                    routing_reason.append("escalate:reason_hidden")
                    arm = ArmConfig("reason_1p_T02_two", "reason_hidden", max(1, arm.few_shot), max(arm.temperature, 0.2), True)

                emit_prompt = build_prompt(inp, arm, plan=plan)
                out_text, tout = llm.generate(emit_prompt, temperature=arm.temperature, max_tokens=1024)
                tokens_used += tout
            else:
                # One-pass path
                prompt = build_prompt(inp, arm)
                out_text, tout = llm.generate(prompt, temperature=arm.temperature, max_tokens=1024)
                tokens_used += tout

            latency = time.time() - t0

            parsed = try_json(out_text)
            json_valid = 0
            if parsed is None:
                repaired = targeted_repair(out_text, ["parse_error"])
                ok, _ = validate_output(repaired)
                parsed = repaired
                json_valid = 1 if ok else 0
            else:
                ok, errs = validate_output(parsed)
                if not ok:
                    repaired = targeted_repair(out_text, errs)
                    ok2, _ = validate_output(repaired)
                    parsed = repaired if ok2 else parsed
                    json_valid = 1 if ok2 else 0
                else:
                    json_valid = 1

            fields = compute_field_f1(parsed, ref)
            reward = shaped_reward(fields, tokens_used, latency, weights)
            bandit.update(arm.arm_id, reward)

            row = {
                "run_id": run_id,
                "iter": i,
                "arm_id": arm.arm_id,
                "prompt_style": arm.prompt_style,
                "few_shot": arm.few_shot,
                "temperature": arm.temperature,
                "two_pass": arm.two_pass,
                "input_hash": hashlib.md5(inp.encode("utf-8")).hexdigest(),
                "model": llm.model,
                "tokens": tokens_used,
                "latency_s": latency,
                # routing diag
                "routing_mode": routing_mode,
                "difficulty": round(float(difficulty), 3),
                "plan_conf": (round(float(plan_conf_val), 3) if plan_conf_val is not None else None),
                "escalated": escalated,
                "routing_reason": ";".join(routing_reason) if routing_reason else "",
                # metrics
                "f1_intent": fields.f1_intent,
                "f1_entities": fields.f1_entities,
                "f1_urgency": fields.f1_urgency,
                "f1_constraints": fields.f1_constraints,
                "f1_steps": fields.f1_steps,
                "reward": reward,
                "json_valid": json_valid,
            }
            w.writerow(row)

############
# CLI entry #
############

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=str, required=True, help="CSV output path")
    p.add_argument("--model", type=str, default="gpt-4o-mini")
    p.add_argument("--gamma", type=float, default=0.98)
    p.add_argument("--c", type=float, default=2.0)
    p.add_argument("--epsilon", type=float, default=0.05)
    p.add_argument("--w_steps", type=float, default=0.7)
    p.add_argument("--w_constraints", type=float, default=0.15)
    p.add_argument("--w_other", type=float, default=0.15)
    p.add_argument("--cost_per_1k", type=float, default=0.03)
    p.add_argument("--latency_penalty", type=float, default=0.01)
    p.add_argument("--selftest", action="store_true", help="Run minimal self-tests and exit")
    p.add_argument("--favor_steps", action="store_true", default=True, help="Always favor higher step reliability (defaults True)")
    args = p.parse_args()

    # runtime toggle
    global RELIABILITY_FIRST
    RELIABILITY_FIRST = bool(args.favor_steps)

    if args.selftest:
        # 1) build_prompt/build_plan_prompt basic checks
        _p = build_prompt("test input", DEFAULT_ARMS[0])
        assert "JSON:" in _p and "{\"intent\"" in _p, "build_prompt malformed"
        _pp = build_plan_prompt("test input")
        assert "Plan" in _pp and "1..N" in _pp, "build_plan_prompt malformed"
        # 2) validator minimal object
        ok, errs = validate_output({
            "intent": "x", "entities": ["a"], "urgency": "low",
            "constraints": ["c"],
            "steps": [{"id":1, "action":"do", "object":None}],
        })
        assert ok, f"validator failed: {errs}"
        # 3) compute_field_f1 sanity
        f1 = compute_field_f1({
            "intent":"book flight",
            "entities":["Toronto"],
            "urgency":"low",
            "constraints":["budget"],
            "steps":[{"id":1,"action":"search","object":"flights"}]
        }, {
            "intent":"book flight",
            "entities":{"from":"Toronto"},
            "urgency":"low",
            "constraints":["budget"],
            "steps":["search_flights"]
        })
        assert 0.0 <= f1.f1_steps <= 1.0, "F1 range invalid"
        # 4) difficulty gating
        easy_ctx = {"len": 10, "has_num": False, "has_money": False, "has_dates": False, "bullet_like": True}
        hard_ctx = {"len": 140, "has_num": True, "has_money": True, "has_dates": True, "bullet_like": False}
        assert predict_steps_difficulty(easy_ctx) < DIFF_THR_HIGH, "easy should be below threshold"
        assert predict_steps_difficulty(hard_ctx) >= DIFF_THR_HIGH, "hard should be above threshold"
        # 5) plan confidence
        weak_plan = "1. do thing\\n2. check\\n3. fix"
        strong_plan = "1. search flights\\n2. filter nonstop\\n3. present options\\n4. confirm booking"

        assert plan_confidence(weak_plan) < PLAN_CONF_THR_LOW, "weak plan should be low confidence"
        assert plan_confidence(strong_plan) >= PLAN_CONF_THR_LOW, "strong plan should be high confidence"
        # 6) adapt_params
        a0 = DEFAULT_ARMS[0]
        a_lo = adapt_params(a0, 0.1)
        a_hi = adapt_params(a0, 0.9)
        assert a_lo.few_shot in (0,1) and a_lo.temperature in (0.0, 0.2), "adapt low difficulty"
        assert a_hi.few_shot == 3 and a_hi.temperature == 0.2, "adapt high difficulty"
        print("Self-tests passed.")
        return

    # Hardcoded dataset load
    ds = _load_agent_dataset_hardcoded()

    weights = RewardWeights(
        w_steps=args.w_steps,
        w_constraints=args.w_constraints,
        w_other=args.w_other,
        cost_per_1k=args.cost_per_1k,
        latency_penalty=args.latency_penalty,
    )

    run_bandit(ds, DEFAULT_ARMS, args.model, args.out, gamma=args.gamma, c=args.c, epsilon=args.epsilon, weights=weights)


if __name__ == "__main__":
    main()

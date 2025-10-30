# bandit_fix_steps.py — single-file runner with all settings hardcoded
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import random
import string
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

# ======================================================
# Hardcoded configuration (edit here)
# ======================================================
AGENT_DATASET_PATH = "agent_dataset.json"   # required dataset
PROMPT_VARIANTS_PATH = "prompt_variants.json"  # optional
SCHEMA_TXT_PATH = "schema.txt"              # external text
FEWSHOT_MIN_1_PATH = "fewshot_min_1.txt"    # external text
OUT_CSV_PATH = "exp_fixed.csv"  # results
MODEL_NAME = "gpt-4o-mini"                             # stubbed LLM client

# Bandit & routing preferences
TWO_PASS_MODE: str = "plan"  # "plan" (plan→emit) or "partial" (partial-schema→steps)
RELIABILITY_FIRST: bool = True
DIFF_THR_HIGH: float = 0.40
PLAN_CONF_THR_LOW: float = 0.50

# Discounted-UCB params
GAMMA = 0.98
EXPL_C = 2.0
EPSILON = 0.05

# Reward shaping
W_STEPS = 0.70
W_CONSTRAINTS = 0.15
W_OTHER = 0.15
COST_PER_1K = 0.03
LAT_PENALTY = 0.01

# ======================================================
# Utilities
# ======================================================

def _read_text(path: str, default: str = "") -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except Exception:
        return default


def _read_json(path: str, default: Any) -> Any:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default

SCHEMA_TEXT = _read_text(SCHEMA_TXT_PATH, "You must output ONLY strict JSON matching the required schema.")
FEWSHOT_MIN_1 = _read_text(FEWSHOT_MIN_1_PATH, "Example (JSON only)")

# ======================================================
# Arm / Prompt
# ======================================================

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

FEW_SHOT_BLOCKS: Dict[str, Dict[int, str]] = {
    "schema_min": {1: FEWSHOT_MIN_1, 3: "..."},
    "schema_verbose": {1: "Example (strict 1..N, verb-first).", 3: "..."},
    "reason_hidden": {1: "Reason privately; output ONLY JSON.", 3: "..."},
}

_ext = _read_json(PROMPT_VARIANTS_PATH, default=None)
if isinstance(_ext, dict):
    for style, inner in _ext.items():
        if isinstance(inner, dict):
            FEW_SHOT_BLOCKS.setdefault(style, {})
            for k, v in inner.items():
                try:
                    kk = int(k)
                    FEW_SHOT_BLOCKS[style][kk] = str(v)
                except Exception:
                    pass

# ======================================================
# Validation / Metrics
# ======================================================

REQUIRED_TOP_LEVEL = ["intent", "entities", "urgency", "constraints", "steps"]


def validate_output(obj: Dict[str, Any]) -> Tuple[bool, List[str]]:
    errors: List[str] = []
    if not isinstance(obj, dict):
        return False, ["Output is not a JSON object"]
    for k in REQUIRED_TOP_LEVEL:
        if k not in obj:
            errors.append("Missing key: " + k)
    if errors:
        return False, errors

    if not isinstance(obj.get("intent"), str):
        errors.append("intent must be string")
    if not isinstance(obj.get("entities"), list) or not all(isinstance(x, str) for x in obj.get("entities", [])):
        errors.append("entities must be list[str]")
    if not isinstance(obj.get("urgency"), str):
        errors.append("urgency must be string")
    if not isinstance(obj.get("constraints"), list) or not all(isinstance(x, str) for x in obj.get("constraints", [])):
        errors.append("constraints must be list[str]")
    steps = obj.get("steps")
    if not isinstance(steps, list):
        errors.append("steps must be list")
        return False, errors
    for i, s in enumerate(steps):
        if not isinstance(s, dict):
            errors.append("steps[" + str(i) + "] not object")
            continue
        if not isinstance(s.get("id"), int):
            errors.append("steps[" + str(i) + "].id must be int")
        if not isinstance(s.get("action"), str) or not s.get("action"):
            errors.append("steps[" + str(i) + "].action must be non-empty string")
        obj_field = s.get("object")
        if obj_field is not None and not isinstance(obj_field, str):
            errors.append("steps[" + str(i) + "].object must be string or null")

    return (len(errors) == 0), errors


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

    pred_steps = pred.get("steps") or []
    pred_actions = [s.get("action", "") for s in pred_steps if isinstance(s, dict)]
    pred_tokens = _tokenize(" ".join(pred_actions))

    ref_step_verbs = [(_tokenize(s) or [""])[0] for s in refn["steps"]]
    ref_tokens = _tokenize(" ".join(ref_step_verbs))

    f1_steps = simple_f1(pred_tokens, ref_tokens)

    return FieldF1(intent, entities, urgency, constraints, f1_steps)

@dataclass
class RewardWeights:
    w_steps: float = W_STEPS
    w_constraints: float = W_CONSTRAINTS
    w_other: float = W_OTHER
    cost_per_1k: float = COST_PER_1K
    latency_penalty: float = LAT_PENALTY


def shaped_reward(f: FieldF1, tokens: int, latency_s: float, w: RewardWeights) -> float:
    other = (f.f1_intent + f.f1_entities + f.f1_urgency) / 3.0
    cost_pen = (tokens / 1000.0) * w.cost_per_1k
    lat_pen = latency_s * w.latency_penalty
    return w.w_steps * f.f1_steps + w.w_constraints * f.f1_constraints + w.w_other * other - cost_pen - lat_pen

# ======================================================
# Bandit (Discounted UCB1)
# ======================================================

@dataclass
class ArmStats:
    pulls: int = 0
    value: float = 0.0

class DiscountedUCB:
    def __init__(self, arm_ids: List[str], gamma: float = GAMMA, c: float = EXPL_C, epsilon: float = EPSILON):
        self.gamma = gamma
        self.c = c
        self.epsilon = epsilon
        self.t = 0
        self.stats: Dict[str, ArmStats] = {a: ArmStats() for a in arm_ids}

    def select(self) -> str:
        self.t += 1
        if random.random() < self.epsilon:
            return random.choice(list(self.stats.keys()))
        scores: Dict[str, float] = {}
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

# ======================================================
# LLM stub
# ======================================================

class LLMClient:
    def __init__(self, model: str = MODEL_NAME):
        self.model = model

    def generate(self, prompt: str, temperature: float = 0.0, max_tokens: int = 1024) -> Tuple[str, int]:
        # Replace with your provider call. Stub returns a small valid JSON.
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
        tokens = max(1, len(prompt.split()) + len(text) // 4)
        return text, tokens

# ======================================================
# Prompt building (no multiline literals)
# ======================================================

def build_prompt(input_text: str, arm: ArmConfig, plan: Optional[str] = None) -> str:
    base_min = SCHEMA_TEXT
    base_verbose = base_min + "\nBe explicit and minimal: start with initialization if needed; end with a verification step."
    few = FEW_SHOT_BLOCKS.get(arm.prompt_style, {}).get(arm.few_shot, "")

    if arm.two_pass and plan is not None:
        parts = [
            base_min,
            "Use this plan to populate steps in order; do not include the plan in output.",
            plan,
            "",
            "Input:",
            input_text,
            "",
            "JSON:",
        ]
        return "\n".join(parts)

    head = base_verbose if arm.prompt_style == "schema_verbose" else base_min
    if arm.prompt_style == "reason_hidden":
        head += "\nThink step-by-step privately, then output ONLY the JSON."

    parts = [head]
    if few:
        parts.extend(["", few])
    parts.extend(["", "Input:", input_text, "", "JSON:"])
    return "\n".join(parts)


def build_plan_prompt(input_text: str) -> str:
    parts = [
        "Plan numbered actions (1..N), each line 'action: object'. 3-6 lines, executable order. Do NOT output JSON.",
        "",
        "Input:",
        input_text,
        "",
        "Plan:",
    ]
    return "\n".join(parts)


def build_partial_schema_prompt(input_text: str) -> str:
    parts = [
        "Emit JSON for intent, entities, urgency, and constraints only.",
        "Do NOT include steps yet.",
        "",
        "Input:",
        input_text,
        "",
        "JSON:",
    ]
    return "\n".join(parts)


def build_steps_from_partial_prompt(partial_json: str) -> str:
    parts = [
        "Using the given JSON, generate only the 'steps' field as an ordered list of numbered actions and return the complete JSON including this new field. Ensure schema validity.",
        "",
        "Partial JSON:",
        partial_json,
        "",
        "Full JSON:",
    ]
    return "\n".join(parts)

# ======================================================
# Structured output helpers + routing
# ======================================================

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
    for s in out["steps"]:
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


def extract_context_features(input_text: str) -> Dict[str, Any]:
    has_num = any(ch.isdigit() for ch in input_text)
    length = len(input_text.split())
    lower = input_text.lower()
    has_money = ("$" in input_text) or (" usd" in lower) or (" eur" in lower)
    months = ["jan","feb","mar","apr","may","jun","jul","aug","sep","oct","nov","dec"]
    has_dates = any(tok.endswith(("/202", "-202", "202")) or any(m in tok.lower() for m in months) for tok in input_text.split())
    bullet_like = any(b in input_text for b in ["- ", "•", "1.", "2."])
    return {"len": length, "has_num": has_num, "has_money": has_money, "has_dates": has_dates, "bullet_like": bullet_like}


def predict_steps_difficulty(ctx: Dict[str, Any]) -> float:
    score = 0.0
    score += 0.40 if ctx.get("len", 0) > 80 else 0.0
    score += 0.20 if ctx.get("has_num") else 0.0
    score += 0.15 if ctx.get("has_money") else 0.0
    score += 0.10 if ctx.get("has_dates") else 0.0
    score += 0.10 if not ctx.get("bullet_like") else 0.0
    return max(0.0, min(1.0, score))


def plan_confidence(plan_text: str) -> float:
    lines = [l.strip(" -•\t") for l in plan_text.splitlines() if l.strip()]
    if not lines:
        return 0.0
    verbs: List[str] = []
    for l in lines:
        parts = l.split()
        tok = parts[0].lower() if parts else ""
        verbs.append(tok)
    unique_verbs = len(set(v for v in verbs if v))
    n = len(lines)
    len_score = min(1.0, max(0.0, (n - 2) / 4.0))
    verb_div = min(1.0, unique_verbs / max(1.0, n))
    vague = {"do","check","handle","fix","make","process"}
    penalty_vague = 0.2 if any(v in vague for v in verbs) else 0.0
    conf = max(0.0, min(1.0, 0.6 * len_score + 0.4 * verb_div - penalty_vague))
    return conf


def adapt_params(arm: ArmConfig, difficulty: float) -> ArmConfig:
    K = 3 if difficulty >= 0.6 else (1 if difficulty >= 0.3 else 0)
    T = 0.2 if difficulty >= 0.5 else 0.0
    return ArmConfig(arm.arm_id, arm.prompt_style, K, T, arm.two_pass)


def choose_arm_by_context(arms: List[ArmConfig], ctx: Dict[str, Any]) -> Optional[str]:
    if RELIABILITY_FIRST and predict_steps_difficulty(ctx) >= DIFF_THR_HIGH:
        for a in arms:
            if a.prompt_style in ("schema_verbose", "reason_hidden") and a.two_pass:
                return a.arm_id
    return None

# ======================================================
# Bandit runner
# ======================================================

class DiscountedBanditRunner:
    def __init__(self, arms: List[ArmConfig], model_name: str = MODEL_NAME):
        self.bandit = DiscountedUCB([a.arm_id for a in arms])
        self.arm_by_id = {a.arm_id: a for a in arms}
        self.llm = LLMClient(model_name)
        self.weights = RewardWeights()
        self.two_pass_mode = TWO_PASS_MODE

    def _ensure_outdir(self, path: str) -> None:
        d = os.path.dirname(path)
        if d:
            os.makedirs(d, exist_ok=True)

    def run(self, dataset: List[Dict[str, Any]], out_csv: str) -> None:
        self._ensure_outdir(out_csv)
        fieldnames = [
            "run_id","iter","arm_id","prompt_style","few_shot","temperature","two_pass",
            "input_hash","model","tokens","latency_s",
            "routing_mode","difficulty","plan_conf","escalated","routing_reason","two_pass_mode",
            "f1_intent","f1_entities","f1_urgency","f1_constraints","f1_steps","reward","json_valid"
        ]
        run_id = time.strftime("%Y%m%d_%H%M%S")

        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()

            for i, item in enumerate(dataset):
                inp = item.get("input", "")
                ref = item.get("reference", {})

                ctx = extract_context_features(inp)
                difficulty = predict_steps_difficulty(ctx)
                override = choose_arm_by_context(list(self.arm_by_id.values()), ctx)
                selected_id = override if override is not None else self.bandit.select()
                arm = self.arm_by_id[selected_id]

                arm = adapt_params(arm, difficulty)

                t0 = time.time()
                tokens_used = 0
                use_two_pass = arm.two_pass or (RELIABILITY_FIRST and difficulty >= DIFF_THR_HIGH)
                routing_mode = "two_pass" if use_two_pass else "one_pass"
                routing_reason: List[str] = []
                if override:
                    routing_reason.append("override:" + override)
                routing_reason.append("diff=" + ("%.2f" % difficulty))

                plan_conf_val: Optional[float] = None
                escalated = 0

                if use_two_pass:
                    if self.two_pass_mode == "partial":
                        p1_prompt = build_partial_schema_prompt(inp)
                        partial_json, t1 = self.llm.generate(p1_prompt, temperature=min(0.2, arm.temperature), max_tokens=512)
                        tokens_used += t1
                        p2_prompt = build_steps_from_partial_prompt(partial_json)
                        out_text, t2 = self.llm.generate(p2_prompt, temperature=arm.temperature, max_tokens=1024)
                        tokens_used += t2
                    else:
                        plan_prompt = build_plan_prompt(inp)
                        plan, tplan = self.llm.generate(plan_prompt, temperature=min(0.4, arm.temperature + 0.1), max_tokens=512)
                        tokens_used += tplan

                        plan_conf_val = plan_confidence(plan)
                        routing_reason.append("plan_conf=" + ("%.2f" % plan_conf_val))

                        if plan_conf_val < PLAN_CONF_THR_LOW:
                            escalated = 1
                            routing_reason.append("escalate:reason_hidden")
                            arm = ArmConfig("reason_1p_T02_two", "reason_hidden", max(1, arm.few_shot), max(arm.temperature, 0.2), True)

                        emit_prompt = build_prompt(inp, arm, plan=plan)
                        out_text, tout = self.llm.generate(emit_prompt, temperature=arm.temperature, max_tokens=1024)
                        tokens_used += tout
                else:
                    prompt = build_prompt(inp, arm)
                    out_text, tout = self.llm.generate(prompt, temperature=arm.temperature, max_tokens=1024)
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
                reward = shaped_reward(fields, tokens_used, latency, self.weights)
                self.bandit.update(arm.arm_id, reward)

                row = {
                    "run_id": run_id,
                    "iter": i,
                    "arm_id": arm.arm_id,
                    "prompt_style": arm.prompt_style,
                    "few_shot": arm.few_shot,
                    "temperature": arm.temperature,
                    "two_pass": arm.two_pass,
                    "input_hash": hashlib.md5(inp.encode("utf-8")).hexdigest(),
                    "model": self.llm.model,
                    "tokens": tokens_used,
                    "latency_s": latency,
                    "routing_mode": routing_mode,
                    "difficulty": round(float(difficulty), 3),
                    "plan_conf": (round(float(plan_conf_val), 3) if plan_conf_val is not None else None),
                    "escalated": escalated,
                    "routing_reason": ";".join(routing_reason) if routing_reason else "",
                    "two_pass_mode": self.two_pass_mode,
                    "f1_intent": fields.f1_intent,
                    "f1_entities": fields.f1_entities,
                    "f1_urgency": fields.f1_urgency,
                    "f1_constraints": fields.f1_constraints,
                    "f1_steps": fields.f1_steps,
                    "reward": reward,
                    "json_valid": json_valid,
                }
                w.writerow(row)

# ======================================================
# Dataset loader & main
# ======================================================

def load_agent_dataset() -> List[Dict[str, Any]]:
    data = _read_json(AGENT_DATASET_PATH, default=[])
    out: List[Dict[str, Any]] = []
    for rec in data:
        input_text = rec.get("input", "") if isinstance(rec, dict) else ""
        ref_raw = rec.get("reference", "{}") if isinstance(rec, dict) else "{}"
        try:
            ref_dict = json.loads(ref_raw)
        except Exception:
            ref_dict = {}
        out.append({"input": input_text, "reference": ref_dict})
    return out


def main() -> None:
    ds = load_agent_dataset()
    runner = DiscountedBanditRunner(DEFAULT_ARMS, MODEL_NAME)
    runner.run(ds, OUT_CSV_PATH)
    print("Run complete →", OUT_CSV_PATH)


if __name__ == "__main__":
    main()

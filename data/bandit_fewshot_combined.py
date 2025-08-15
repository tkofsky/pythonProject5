import os
import csv
import json
import math
import random
from dataclasses import dataclass, asdict
from typing import Dict, Tuple, List, Optional

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI

# ========= CONFIG =========
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise EnvironmentError("OPENAI_API_KEY is not set in environment.")
client = OpenAI(api_key=OPENAI_API_KEY)

GEN_MODEL = "gpt-4o-mini"
EMB_MODEL = "text-embedding-3-small"

DATA_PATH = "extended_sample_dataset.json"  # put your dataset JSON here
CSV_LOG = "bandit_fewshot_log.csv"
os.makedirs("results", exist_ok=True)

ITERATIONS = 60                 # after pre-test
EPSILON = 0.25                  # exploration rate for ε-greedy
MUTATION_RATE = 0.20            # chance to mutate chosen prompt into a new variant
FEW_SHOT_LEVELS = [0, 1, 3, 5]  # how many examples to include
TEMPS = [0.3, 0.7]              # test two decoding settings

# Few-shot examples (edit these to fit your domain)
FEW_SHOT_EXAMPLES = [
    {"input": "The sky is blue because of Rayleigh scattering.", "summary": "The sky is blue due to light scattering."},
    {"input": "He went to the market and bought apples.", "summary": "He purchased apples at the market."},
    {"input": "The book was adapted into a movie.", "summary": "The book became a film adaptation."},
    {"input": "The company launched a new product.", "summary": "A new product was launched by the company."},
    {"input": "She won an award for her research.", "summary": "She received recognition for her research."}
]

# Base prompt pool (can add more)
BASE_PROMPTS = [
    {"id": "p_base_1", "template": "Summarize the following text in one paragraph.", "category": "baseline", "intent": "short"},
    {"id": "p_base_2", "template": "Provide a concise overview of the text below.", "category": "baseline", "intent": "short"},
    {"id": "p_struct_1","template": "Summarize the text in 3 bullet points, focusing only on core facts.", "category": "structured","intent":"structured"},
    {"id": "p_simple_1","template": "Explain the text in simple language for a non-expert reader.", "category":"simplified","intent":"simplified"},
]

# ========= DATA =========
with open(DATA_PATH, "r", encoding="utf-8") as f:
    DATASET = json.load(f)
if not isinstance(DATASET, list) or not DATASET:
    raise ValueError("Dataset must be a non-empty list of {input, reference} objects.")

# ========= TYPES =========
@dataclass
class Arm:
    """One bandit arm = (prompt_variant_id, example_count, temperature)."""
    prompt_id: str
    example_count: int
    temperature: float

    def key(self) -> Tuple[str, int, float]:
        return (self.prompt_id, self.example_count, self.temperature)

@dataclass
class PromptVariant:
    prompt_id: str
    template: str
    category: str
    intent: str
    parent_id: Optional[str] = None  # if created via mutation

# ========= STATE =========
# start with base prompt variants
prompt_variants: Dict[str, PromptVariant] = {
    p["id"]: PromptVariant(prompt_id=p["id"], template=p["template"], category=p["category"], intent=p["intent"])
    for p in BASE_PROMPTS
}

# arms = every (prompt_variant × few-shot level × temp)
def enumerate_arms() -> List[Arm]:
    arms = []
    for pv_id in prompt_variants.keys():
        for k in FEW_SHOT_LEVELS:
            for t in TEMPS:
                arms.append(Arm(prompt_id=pv_id, example_count=k, temperature=t))
    return arms

# bandit stats: running avg without storing all history
class BanditStats:
    def __init__(self):
        self.counts: Dict[Tuple[str,int,float], int] = {}
        self.totals: Dict[Tuple[str,int,float], float] = {}

    def update(self, arm: Arm, reward: float):
        k = arm.key()
        self.counts[k] = self.counts.get(k, 0) + 1
        self.totals[k] = self.totals.get(k, 0.0) + reward

    def avg(self, arm: Arm) -> float:
        k = arm.key()
        c = self.counts.get(k, 0)
        if c == 0:
            return 0.0
        return self.totals.get(k, 0.0) / c

bandit_stats = BanditStats()

# ========= HELPERS =========
def build_prompt_text(template: str, input_text: str, example_count: int) -> str:
    if example_count <= 0:
        return f"{template}\n\n{input_text}"
    examples = FEW_SHOT_EXAMPLES[:min(example_count, len(FEW_SHOT_EXAMPLES))]
    ex_str = "\n".join([f"Example — Input: '{e['input']}' → Summary: '{e['summary']}'" for e in examples])
    return f"{ex_str}\n\n{template}\n\n{input_text}"

def call_llm(prompt_text: str, temperature: float) -> Tuple[str, int]:
    resp = client.chat.completions.create(
        model=GEN_MODEL,
        messages=[{"role":"user","content":prompt_text}],
        temperature=temperature,
        max_tokens=300
    )
    out = resp.choices[0].message.content.strip()
    toks = resp.usage.total_tokens if getattr(resp, "usage", None) else 0
    return out, toks

def embed(text: str) -> List[float]:
    emb = client.embeddings.create(model=EMB_MODEL, input=text).data[0].embedding
    return emb

def semantic_reward(output: str, reference: str) -> float:
    v1 = embed(output)
    v2 = embed(reference)
    return float(cosine_similarity([v1], [v2])[0][0])  # ~0..1+

def maybe_mutate(variant: PromptVariant) -> PromptVariant:
    """Lightweight mutation: rephrase instruction or add a constraint."""
    if random.random() > MUTATION_RATE:
        return variant  # no change
    tweaks = [
        "Focus only on the main conclusion.",
        "Avoid filler; keep it tight and factual.",
        "Prefer concrete facts over generalities.",
        "Use plain language and avoid jargon.",
        "Include one key takeaway at the end."
    ]
    tweak = random.choice(tweaks)
    new_template = f"{variant.template} {tweak}"
    new_id = f"{variant.prompt_id}__m{random.randint(1000,9999)}"
    pv = PromptVariant(
        prompt_id=new_id,
        template=new_template,
        category=variant.category,  # keep category to compare families
        intent=variant.intent,
        parent_id=variant.prompt_id
    )
    prompt_variants[new_id] = pv
    return pv

def ensure_csv_header():
    if not os.path.exists(CSV_LOG):
        with open(CSV_LOG, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([
                "iteration","phase","prompt_id","parent_id","category","intent",
                "example_count","temperature","is_mutation","input","reference",
                "output","tokens","reward","prompt_template"
            ])

def log_row(**kwargs):
    with open(CSV_LOG, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            kwargs["iteration"], kwargs["phase"], kwargs["prompt_id"], kwargs["parent_id"], kwargs["category"],
            kwargs["intent"], kwargs["example_count"], kwargs["temperature"], kwargs["is_mutation"],
            kwargs["input"], kwargs["reference"], kwargs["output"], kwargs["tokens"], kwargs["reward"],
            kwargs["prompt_template"]
        ])

def select_arm(arms: List[Arm]) -> Arm:
    # ε-greedy: explore vs exploit
    if random.random() < EPSILON:
        return random.choice(arms)
    # exploit: pick arm with best average reward so far (ties random)
    best = None
    best_val = -1e9
    for a in arms:
        avg = bandit_stats.avg(a)
        # small bonus for arms not yet tried
        tried = bandit_stats.counts.get(a.key(), 0)
        bonus = 0.05 if tried == 0 else 0.0
        val = avg + bonus
        if val > best_val:
            best = a
            best_val = val
    return best

# ========= PRE-TEST PHASE =========
def pretest_all_arms():
    ensure_csv_header()
    arms = enumerate_arms()
    print(f"🔍 Pre-testing {len(arms)} arms (prompt × few-shot × temp)...")
    itr = 0
    for arm in arms:
        pv = prompt_variants[arm.prompt_id]
        sample = random.choice(DATASET)
        prompt_text = build_prompt_text(pv.template, sample["input"], arm.example_count)
        output, tokens = call_llm(prompt_text, arm.temperature)
        reward = semantic_reward(output, sample["reference"])
        bandit_stats.update(arm, reward)
        log_row(
            iteration=itr, phase="pretest",
            prompt_id=pv.prompt_id, parent_id=pv.parent_id or "",
            category=pv.category, intent=pv.intent,
            example_count=arm.example_count, temperature=arm.temperature,
            is_mutation=0, input=sample["input"], reference=sample["reference"],
            output=output, tokens=tokens, reward=round(reward, 6),
            prompt_template=pv.template
        )
        itr += 1
    print("✅ Pre-test complete.")

# ========= MAIN LOOP =========
def main():
    pretest_all_arms()

    print(f"\n🎯 Bandit optimization for {ITERATIONS} iterations...")
    for itr in range(1, ITERATIONS + 1):
        # re-enumerate to include any newly mutated prompt variants
        arms = enumerate_arms()
        arm = select_arm(arms)
        pv = prompt_variants[arm.prompt_id]

        # maybe mutate the chosen prompt (and switch to the new variant for this turn)
        mutated_flag = 0
        parent_id = pv.parent_id or ""
        if random.random() < MUTATION_RATE:
            new_pv = maybe_mutate(pv)
            if new_pv.prompt_id != pv.prompt_id:
                mutated_flag = 1
                parent_id = pv.prompt_id
                pv = new_pv
                # when a new variant is created, arms list won’t include it until next iteration
                # but we can still evaluate it now with same example_count & temperature

        sample = random.choice(DATASET)
        prompt_text = build_prompt_text(pv.template, sample["input"], arm.example_count)
        output, tokens = call_llm(prompt_text, arm.temperature)
        reward = semantic_reward(output, sample["reference"])

        # If this iteration used a brand-new mutated prompt variant, fake an arm to record stats
        effective_arm = Arm(prompt_id=pv.prompt_id, example_count=arm.example_count, temperature=arm.temperature)
        bandit_stats.update(effective_arm, reward)

        log_row(
            iteration=itr, phase="bandit",
            prompt_id=pv.prompt_id, parent_id=parent_id,
            category=pv.category, intent=pv.intent,
            example_count=arm.example_count, temperature=arm.temperature,
            is_mutation=mutated_flag, input=sample["input"],
            reference=sample["reference"], output=output, tokens=tokens,
            reward=round(reward, 6), prompt_template=pv.template
        )

        if itr % 5 == 0:
            print(f"[iter {itr}] avg(best-arm)≈{max(bandit_stats.totals.get(a.key(),0)/max(1,bandit_stats.counts.get(a.key(),0)) for a in enumerate_arms()):.3f}")

    print(f"\n✅ Finished. Log saved to {CSV_LOG}")

if __name__ == "__main__":
    main()

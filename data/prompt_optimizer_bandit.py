import os
import json
import csv
import random
from openai import OpenAI
from sentence_transformers import SentenceTransformer, util

# -----------------------------
# ✅ LOAD API KEY AND INIT CLIENT
# -----------------------------
api_key = os.environ.get("OPENAI_API_KEY")
if api_key is None:
    raise EnvironmentError("❌ OPENAI_API_KEY environment variable is not set!")

client = OpenAI(api_key=api_key)

# -----------------------------
# CONFIG (experiment with these)
# -----------------------------
OPENAI_MODEL = "gpt-3.5-turbo"
ALPHA = 0.8      # weight for quality
BETA = 0.2       # weight for cost
EPSILON = 0.5    # exploration rate # 0.5
ITERATIONS = 200 # number of iterations #

# -----------------------------
# LOAD DATA
# -----------------------------
#with open("sample_dataset.json") as f:
with open("longer_sample_dataset.json") as f:
    dataset = json.load(f)

# -----------------------------
# INITIAL PROMPTS
# -----------------------------
initial_prompts = [
    "Summarize the following text in one paragraph.",
    "Write a concise summary of the following text in 3 sentences.",
    "Please provide a short overview of the text below.",
    "Summarize the text clearly and briefly.",
    "Summarize the text with a focus on key events.",
    "Summarize the text in plain, simple language.",
    "give a proper summary of the text"
]
# Track stats
prompt_stats = {p: {"count": 0, "total_reward": 0.0} for p in initial_prompts}

# -----------------------------
# SCORING MODEL
# -----------------------------
sim_model = SentenceTransformer('all-MiniLM-L6-v2')

def score_output(output, reference):
    emb1 = sim_model.encode(output, convert_to_tensor=True)
    emb2 = sim_model.encode(reference, convert_to_tensor=True)
    return float(util.cos_sim(emb1, emb2))

# -----------------------------
# LLM CALL
# -----------------------------
def call_llm(prompt, text):
    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": text}
        ]
    )
    output = response.choices[0].message.content
    tokens = response.usage.total_tokens if response.usage else 0
    return output, tokens

# -----------------------------
# BANDIT SELECTION
# -----------------------------
def select_prompt(prompt_stats, epsilon=0.2):
    if random.random() < epsilon:
        return random.choice(list(prompt_stats.keys()))
    avg_rewards = {
        p: (prompt_stats[p]["total_reward"] / (prompt_stats[p]["count"] + 1e-6))
        for p in prompt_stats
    }
    return max(avg_rewards, key=avg_rewards.get)

def update_stats(prompt_stats, prompt, reward):
    prompt_stats[prompt]["count"] += 1
    prompt_stats[prompt]["total_reward"] += reward

# -----------------------------
# MUTATION (OPTIONAL)
# -----------------------------
def mutate_prompt(prompt):
    styles = [
        "using simple words",
        "highlighting only main points",
        "written in plain English",
        "in exactly three sentences",
        "in bullet points"
    ]
    style = random.choice(styles)
    return f"{prompt} Please do this {style}."

# -----------------------------
# CSV LOGGING SETUP
# -----------------------------
os.makedirs("results", exist_ok=True)
log_path = "log2.csv"
with open(log_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["iteration", "prompt", "quality", "tokens", "reward"])

# -----------------------------
# MAIN LOOP
# -----------------------------
for iteration in range(ITERATIONS):
    chosen_prompt = select_prompt(prompt_stats, EPSILON)
    sample = random.choice(dataset)

    output, token_count = call_llm(chosen_prompt, sample["input"])
    quality = score_output(output, sample["reference"])
    reward = (ALPHA * quality) - (BETA * (token_count / 1000))

    update_stats(prompt_stats, chosen_prompt, reward)

    # Occasionally add mutated prompt
    if iteration % 5 == 0:
        new_p = mutate_prompt(chosen_prompt)
        if new_p not in prompt_stats:
            prompt_stats[new_p] = {"count": 0, "total_reward": 0.0}

    print(f"[{iteration}] Prompt: {chosen_prompt[:50]}... | Quality={quality:.3f} | Tokens={token_count} | Reward={reward:.3f}")

    # 🔥 Log to CSV
    with open(log_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([iteration, chosen_prompt, quality, token_count, reward])

# -----------------------------
# RESULTS
# -----------------------------
print("\n=== Final Average Rewards ===")
for p, stats in prompt_stats.items():
    if stats["count"] > 0:
        avg = stats["total_reward"] / stats["count"]
        print(f"{avg:.3f} -> {p}")

print(f"\n✅ Results logged to {log_path}")


##############
#reward = (ALPHA * quality) - (BETA * (token_count / 1000)) combined score with quiality AND token count
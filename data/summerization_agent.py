import os
import csv
import json
import random
import logging
from collections import defaultdict
from openai import OpenAI
import numpy as np

# -------------------------------
# 1. CONFIGURATION
# -------------------------------
#os.environ["OPENAI_API_KEY"] = "YOUR_API_KEY"  # or set in environment

api_key = os.environ.get("OPENAI_API_KEY")



client = OpenAI()

RESULTS_FILE = "results/plots/summarization_agent_log.csv"
DATASET_FILE = "extended_sample_dataset.json"
ITERATIONS = 50
MUTATION_PROB = 0.2

# Prompt categories
PROMPT_CATEGORIES = {
    "short": [
        "Summarize the following text in 1–2 sentences.",
        "Provide a very concise summary of the text below."
    ],
    "detailed": [
        "Write a clear and detailed summary in 3–4 sentences.",
        "Summarize the following text while retaining all key details."
    ],
    "bullet": [
        "Summarize the following text into 3 bullet points.",
        "Provide a bullet-point summary of the text."
    ]
}

# -------------------------------
# 2. LOAD DATA
# -------------------------------
def load_dataset(file_path):
    with open(file_path, "r") as f:
        return json.load(f)

dataset = load_dataset(DATASET_FILE)

# -------------------------------
# 3. LOGGING SETUP
# -------------------------------
os.makedirs("results", exist_ok=True)
if not os.path.exists(RESULTS_FILE):
    with open(RESULTS_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["iteration", "prompt", "category", "reward", "tokens", "mutation", "reference", "summary"])

# -------------------------------
# 4. UTILITY FUNCTIONS
# -------------------------------
def call_openai_api(prompt, text):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that summarizes text."},
            {"role": "user", "content": f"{prompt}\n\n{text}"}
        ],
        max_tokens=200
    )
    summary = response.choices[0].message.content.strip()
    tokens_used = response.usage.total_tokens
    return summary, tokens_used

def reward_function(summary, reference):
    """Heuristic reward: semantic overlap (simplified) and length penalty."""
    words_summary = set(summary.lower().split())
    words_reference = set(reference.lower().split())
    overlap = len(words_summary.intersection(words_reference)) / max(len(words_reference), 1)
    length_penalty = 1 - abs(len(summary) - len(reference)) / max(len(reference), 1)
    return round((0.7 * overlap + 0.3 * length_penalty), 3)

def mutate_prompt(prompt):
    mutations = [
        lambda p: p.replace("Summarize", "Provide a summary of"),
        lambda p: p.replace("Provide", "Write"),
        lambda p: p + " Focus on the main topic only.",
        lambda p: p + " Use clear and simple language."
    ]
    return random.choice(mutations)(prompt)

def log_result(iteration, prompt, category, reward, tokens, mutation, reference, summary):
    with open(RESULTS_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([iteration, prompt, category, reward, tokens, mutation, reference, summary])

# -------------------------------
# 5. MAIN LOOP (BANDIT AGENT)
# -------------------------------
prompt_pool = {cat: list(prompts) for cat, prompts in PROMPT_CATEGORIES.items()}
prompt_rewards = defaultdict(list)

for iteration in range(1, ITERATIONS + 1):
    category = random.choice(list(prompt_pool.keys()))
    prompt = random.choice(prompt_pool[category])

    if random.random() < MUTATION_PROB:
        prompt = mutate_prompt(prompt)
        mutation = 1
    else:
        mutation = 0

    data_point = random.choice(dataset)
    input_text = data_point["input"]
    reference_summary = data_point["reference"]

    summary, tokens = call_openai_api(prompt, input_text)
    reward = reward_function(summary, reference_summary)

    log_result(iteration, prompt, category, reward, tokens, mutation, reference_summary, summary)
    prompt_rewards[prompt].append(reward)

    print(f"[Iteration {iteration}] {category.upper()} | Reward: {reward} | Tokens: {tokens}")

print("✅ Experiment completed. Results saved in:", RESULTS_FILE)

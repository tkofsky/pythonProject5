import os
import json
import random
import pandas as pd
from datetime import datetime
from sentence_transformers import SentenceTransformer, util
import openai

openai.api_key = os.environ.get("OPENAI_API_KEY")

# Load dataset
with open("extended_sample_dataset.json", "r") as f:
    dataset = json.load(f)

# Load top prompts from benchmark
prompt_summary = pd.read_csv("results/prompt_summary.csv")
top_prompts = prompt_summary.head(5)["prompt"].tolist()  # Start from top 5

model = SentenceTransformer("all-MiniLM-L6-v2")

epsilon = 0.2        # Exploration rate
mutation_rate = 0.2  # Chance to mutate a prompt
iterations = 50
log = []

def get_reward(prompt, sample):
    """Send prompt to OpenAI, return output, tokens, and reward score."""
    full_prompt = f"{prompt}\n\n{sample['input']}"
    try:
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": full_prompt}],
            temperature=0.7,
            max_tokens=300
        )
        output = response.choices[0].message.content
        tokens_used = response.usage.total_tokens
    except Exception as e:
        print(f"Error: {e}")
        return None, None, None

    emb_output = model.encode(output, convert_to_tensor=True)
    emb_ref = model.encode(sample["reference"], convert_to_tensor=True)
    quality = float(util.cos_sim(emb_output, emb_ref))
    reward = quality - 0.002 * tokens_used
    return output, tokens_used, reward

def mutate_prompt(base_prompt):
    """Generate a mutated prompt using GPT to add variation."""
    mutation_instruction = f"Rewrite this summarization prompt in a different phrasing but keep the same meaning:\n\n{base_prompt}"
    try:
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": mutation_instruction}],
            temperature=0.9,
            max_tokens=100
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Mutation failed: {e}")
        return base_prompt  # fallback to original

# Initialize bandit stats
prompt_stats = {p: {"reward": 0, "count": 0} for p in top_prompts}

for i in range(iterations):
    # ε-greedy: explore or exploit
    if random.random() < epsilon:
        prompt = random.choice(list(prompt_stats.keys()))
    else:
        prompt = max(prompt_stats, key=lambda p: prompt_stats[p]["reward"] / max(1, prompt_stats[p]["count"]))

    # Prompt mutation
    if random.random() < mutation_rate:
        mutated = mutate_prompt(prompt)
        if mutated not in prompt_stats:
            print(f"🧬 Mutation created new prompt: {mutated}")
            prompt_stats[mutated] = {"reward": 0, "count": 0}
            prompt = mutated

    # Pick random input sample
    sample = random.choice(dataset)

    # Evaluate prompt
    output, tokens, reward = get_reward(prompt, sample)
    if reward is None:
        continue

    prompt_stats[prompt]["reward"] += reward
    prompt_stats[prompt]["count"] += 1

    log.append({
        "iteration": i,
        "prompt": prompt,
        "input": sample["input"],
        "reference": sample["reference"],
        "output": output,
        "tokens": tokens,
        "reward": reward,
        "timestamp": datetime.utcnow().isoformat()
    })

# Save log
df = pd.DataFrame(log)
os.makedirs("results", exist_ok=True)
df.to_csv("results/bandit_log.csv", index=False)
print("✅ Bandit optimization with mutation complete. Results saved to results/bandit_log.csv")

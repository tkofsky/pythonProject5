import os
import json
import openai
import pandas as pd
from datetime import datetime
from sentence_transformers import SentenceTransformer, util
import matplotlib.pyplot as plt
import seaborn as sns

openai.api_key = os.environ.get("OPENAI_API_KEY")

with open("categorized_prompts_with_intent.json", "r") as f:
    prompt_list = json.load(f)

with open("extended_sample_dataset.json", "r") as f:
    dataset = json.load(f)

model = SentenceTransformer("all-MiniLM-L6-v2")

log = []

for i, sample in enumerate(dataset):
    for entry in prompt_list:
        prompt_text = entry["prompt"]
        category = entry["category"]
        intent = entry["intent"]
        full_prompt = f"{prompt_text}\n\n{sample['input']}"

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
            continue

        emb_output = model.encode(output, convert_to_tensor=True)
        emb_ref = model.encode(sample["reference"], convert_to_tensor=True)
        quality = float(util.cos_sim(emb_output, emb_ref))
        reward = quality - 0.002 * tokens_used

        log.append({
            "iteration": len(log),
            "input": sample["input"],
            "reference": sample["reference"],
            "prompt": prompt_text,
            "category": category,
            "intent": intent,
            "output": output,
            "quality": quality,
            "tokens": tokens_used,
            "reward": reward,
            "timestamp": datetime.utcnow().isoformat()
        })

os.makedirs("results", exist_ok=True)
df = pd.DataFrame(log)
df.to_csv("results/log_with_intent.csv", index=False)

# Export prompt performance summary for bandit
prompt_summary = df.groupby("prompt")["reward"].mean().reset_index().sort_values("reward", ascending=False)
prompt_summary.to_csv("results/prompt_summary.csv", index=False)
print("✅ Log saved to results/log_with_intent.csv")
print("✅ Prompt summary saved to results/prompt_summary.csv")

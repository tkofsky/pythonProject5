import os
import random
import pandas as pd
from openai import OpenAI
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import numpy as np
import csv

# ========== CONFIG ==========
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

MODEL_NAME = "gpt-4o-mini"
RESULTS_FILE = "multi_agent_results.csv"
NUM_AGENTS = 5
ITERATIONS = 10
MUTATION_RATE = 0.3

# Load embedding model for reward scoring
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Initial agents with intent
INITIAL_AGENTS = [
    {"template": "Summarize the following text in one paragraph.", "category": "baseline", "intent": "short"},
    {"template": "Please provide a short overview of the text below.", "category": "baseline", "intent": "short"},
    {"template": "Create a clear and concise summary with some details.", "category": "concise", "intent": "detailed"},
    {"template": "Explain the text in simple terms for a non-expert.", "category": "simplified", "intent": "simplified"},
    {"template": "Summarize in bullet points, highlighting the key ideas.", "category": "structured", "intent": "structured"}
]

# Sample dataset
DATASET = [
    {
        "input": "Artificial Intelligence is transforming industries such as healthcare, finance, and transportation. It is enabling automation, improving decision-making, and opening new opportunities for innovation.",
        "reference": "AI is revolutionizing industries by automating processes and driving smarter decisions."
    },
    {
        "input": "Climate change is an urgent global issue caused by greenhouse gas emissions. It results in rising sea levels, extreme weather events, and biodiversity loss, demanding immediate action.",
        "reference": "Climate change, driven by emissions, leads to rising seas and severe environmental consequences."
    }
]

# ========== UTILS ==========
def compute_reward(output: str, reference: str) -> float:
    """Reward: Semantic similarity between output and reference."""
    embeddings = embedder.encode([output, reference])
    return float(cosine_similarity([embeddings[0]], [embeddings[1]])[0][0])

def mutate_prompt(agent):
    """Mutate prompt template and optionally change intent."""
    mutations = [
        ("Provide a very concise summary.", "short"),
        ("Give a detailed explanation in one short paragraph.", "detailed"),
        ("Explain this as if to a beginner.", "simplified"),
        ("Summarize in three bullet points.", "structured"),
        ("Offer a clear and factual summary only.", "short")
    ]
    if random.random() < MUTATION_RATE:
        return {"template": random.choice(mutations)[0], "category": agent["category"], "intent": random.choice(mutations)[1]}
    return agent

def log_result(iteration, agent_id, prompt, category, intent, input_text, output, reference, reward):
    """Append results to CSV."""
    file_exists = os.path.isfile(RESULTS_FILE)
    with open(RESULTS_FILE, mode="a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(["timestamp", "iteration", "agent_id", "prompt", "category", "intent", "input", "output", "reference", "reward"])
        writer.writerow([datetime.now(), iteration, agent_id, prompt, category, intent, input_text, output, reference, reward])

# ========== AGENT LOOP ==========
def run_agents():
    agents = INITIAL_AGENTS.copy()

    for iteration in range(1, ITERATIONS + 1):
        print(f"\n=== Iteration {iteration} ===")
        results = []

        for agent_id, agent in enumerate(agents):
            for data in DATASET:
                response = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[{"role": "system", "content": agent["template"]},
                              {"role": "user", "content": data["input"]}],
                    max_tokens=150,
                    temperature=0.7
                )
                output = response.choices[0].message.content.strip()
                reward = compute_reward(output, data["reference"])

                results.append((agent_id, agent, data["input"], output, data["reference"], reward))
                log_result(iteration, agent_id, agent["template"], agent["category"], agent["intent"], data["input"], output, data["reference"], reward)

        # Average rewards per agent
        agent_scores = {}
        for r in results:
            agent_scores.setdefault(r[0], []).append(r[5])
        avg_scores = {aid: sum(scores)/len(scores) for aid, scores in agent_scores.items()}

        # Keep top agents and mutate some
        top_agents = sorted(avg_scores.items(), key=lambda x: x[1], reverse=True)[:NUM_AGENTS]

        agents = []
        for rank, (agent_id, _) in enumerate(top_agents):
            base_agent = results[agent_id][1]
            if rank < NUM_AGENTS // 2:
                agents.append(base_agent)
            else:
                agents.append(mutate_prompt(base_agent))

if __name__ == "__main__":
    run_agents()
    print("✅ Multi-agent optimization with intent complete. Results saved to", RESULTS_FILE)

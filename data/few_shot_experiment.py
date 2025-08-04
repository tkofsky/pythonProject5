import os
import json
import csv
import random
from openai import OpenAI

# Initialize OpenAI client
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# Paths
DATA_PATH = "extended_sample_dataset2.json"
CSV_LOG = "few_shot_experiment_log.csv"

# Few-shot examples
FEW_SHOT_EXAMPLES = [
    {"input": "The sky is blue because of Rayleigh scattering.", "summary": "The sky is blue due to light scattering."},
    {"input": "He went to the market and bought apples.", "summary": "He purchased apples at the market."},
    {"input": "The book was adapted into a movie.", "summary": "The book became a film adaptation."},
    {"input": "The company launched a new product.", "summary": "A new product was launched by the company."},
    {"input": "She won an award for her research.", "summary": "She received recognition for her research."}
]

# Load dataset
with open(DATA_PATH, "r", encoding="utf-8") as f:
    dataset = json.load(f)

# Build few-shot prompt
def build_prompt(input_text, example_count):
    if example_count == 0:
        return f"Summarize the following text in one paragraph:\n\n{input_text}"

    examples = FEW_SHOT_EXAMPLES[:example_count]
    example_str = "\n".join([f"Example: Input: '{e['input']}' → Summary: '{e['summary']}'" for e in examples])
    return f"{example_str}\n\nNow summarize the following text:\n{input_text}"

# Reward scoring (mocked as overlap for testing)
def calculate_reward(output, reference):
    overlap = len(set(output.lower().split()) & set(reference.lower().split()))
    return overlap / max(len(reference.split()), 1)

# Run experiment
def run_experiment(iterations=10):
    with open(CSV_LOG, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["iteration", "example_count", "category", "prompt_text", "input", "reference", "model_output", "reward"])

        for i in range(iterations):
            for example_count in [0, 1, 3, 5]:
                sample = random.choice(dataset)
                prompt = build_prompt(sample["input"], example_count)

                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.7
                )

                output = response.choices[0].message.content.strip()
                reward = calculate_reward(output, sample["reference"])
                category = f"few-shot:{example_count}" if example_count > 0 else "zero-shot"

                writer.writerow([i+1, example_count, category, prompt, sample["input"], sample["reference"], output, reward])

    print(f"Experiment logged to {CSV_LOG}")

# Run the experiment
if __name__ == "__main__":
    run_experiment(iterations=20)

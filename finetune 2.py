import openai
from openai import OpenAI

import os
openai.api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI()

completion = client.chat.completions.create(
  model="ft:gpt-4o-mini-2024-07-18:personal::9wDFZltb",
  messages=[
    {"role": "system", "content": "You are a teaching assistant for Machine Learning. You should help to user to answer on his question."},
    {"role": "user", "content": "What is a loss function?"}
  ]
)
print(completion.choices[0].message)


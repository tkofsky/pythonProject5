import openai
from openai import OpenAI

import os
openai.api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI()


client.fine_tuning.jobs.create(
  training_file="file-rIua39sJX1O64gzxTYfpvJx7",
  model="gpt-3.5-turbo" #change to gpt-4-0613 if you have access
)

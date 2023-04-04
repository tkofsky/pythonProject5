import os
import time
os.environ["OPENAI_API_KEY"]="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"
from langchain.agents import create_pandas_dataframe_agent
from langchain.llms import OpenAI
import pandas as pd
filename = os.path.join("csv files", "organizations-3.csv") # csv file taken from csv table.py
df = pd.read_csv(filename)
pd_agent = create_pandas_dataframe_agent(OpenAI(temperature=0), df, verbose=True)
pd_agent.run("what organization has the fewest of employees?")
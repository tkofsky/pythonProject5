from langchain.llms import OpenAI
import os
import sys


query = input("Enter doc query? ")



from langchain.document_loaders import TextLoader
from langchain.document_loaders import DirectoryLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import WebBaseLoader
os.environ["OPENAI_API_KEY"]="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"
import time
#change 2
#
#query = sys.argv[1]
query ="what team has the most Points Per Game" #
query ="how many wins do the Boston Celtics Have" #
query ="how many losses to the Golden State Worriors have" #
query ="who had more wins the Golden State Worriors or Los Angles Lakers" #
query ="which teams have less than than 35 wins" #
query ="which team has the most wins in the Pacific Division" #

query ="how many wins (W) does Dallas" #
 #

print (query)
filename = os.path.join("csv files", "2.csv") # csv file taken from csv table.py
loader = TextLoader(filename)
#loader = WebBaseLoader("https://www.tsn.ca/nhl/standings")
#loader = TextLoader('3.html')


index = VectorstoreIndexCreator().from_loaders([loader])

print(index.query(query,llm=ChatOpenAI()))




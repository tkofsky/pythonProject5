from langchain.llms import OpenAI
import os
import sys
import time

query = input("Enter doc query? ")
#print("Hello, " + query + "!")


from langchain.document_loaders import TextLoader
from langchain.document_loaders import DirectoryLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI

os.environ["OPENAI_API_KEY"]="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"


#
#query = sys.argv[1]
query ="what team has the most Points Per Game" #
query ="how many wins do the Boston Celtics Have" #
query ="how many losses to the Golden State Worriors have" #
query ="who had more wins the Golden State Worriors or Los Angles Lakers" #
print (query)
filename = os.path.join("csv files", "2.csv") # csv file taken from csv table.py
loader = TextLoader(filename)

#loader = TextLoader('3.html')


index = VectorstoreIndexCreator().from_loaders([loader])

print(index.query(query,llm=ChatOpenAI()))




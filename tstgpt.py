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
query ="what is the birth date for Jrue Holiday " #
print (query)
loader = TextLoader('data.txt')
loader = TextLoader('teams.csv')
loader = TextLoader('3.html')


index = VectorstoreIndexCreator().from_loaders([loader])

print(index.query(query,llm=ChatOpenAI()))




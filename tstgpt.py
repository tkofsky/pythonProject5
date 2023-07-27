from langchain.llms import OpenAI
import os
import sys
import time

from langchain.document_loaders import TextLoader
from langchain.document_loaders import DirectoryLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
os.environ["OPENAI_API_KEY"]=""


#
#query = sys.argv[1]
query ="what percent did home sales increase by" #
print (query)
loader = TextLoader('data.txt')
time.sleep(8)
index = VectorstoreIndexCreator().from_loaders([loader])

print(index.query(query,llm=ChatOpenAI()))




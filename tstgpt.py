from langchain.llms import OpenAI
import os
import sys

from langchain.document_loaders import TextLoader
from langchain.document_loaders import DirectoryLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
os.environ["OPENAI_API_KEY"]=""
#
#query = sys.argv[1]
query ="what day is it"
print (query)
loader = TextLoader('data.txt')
index = VectorstoreIndexCreator().from_loaders([loader])

print(index.query((query)))




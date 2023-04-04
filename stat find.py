import os
from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.llms  import OpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader
from langchain.document_loaders.csv_loader import CSVLoader

from langchain.embeddings import TensorflowHubEmbeddings
import os
import time
os.environ["OPENAI_API_KEY"]="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"
#https://python.langchain.com/docs/use_cases/question_answering/how_to/chat_vector_db
###CHANGE MODEL from gpt-3.5 to gtp-4 and see one is cheaper than the other gpt-4 is slower
### also change chunk size?
#stream ?

#loader = TextLoader("97578674q.txt",encoding='utf-8')
documents = []



filename = os.path.join("csv files", "organizations-3.csv") # csv file taken from csv table.py
#filename = os.path.join("csv files", "all players.csv") # csv file taken from csv table.py
#loader = TextLoader(filename,encoding='utf-8')

#filename = os.path.join("csv files", "all players.csv") # csv file taken from csv table.py
filename = os.path.join("csv files", "organizations-2500.csv") # csv file taken from csv table.py
#path.join("csv files", "output.csv") # csv file taken from csv table.py
loader = CSVLoader(filename,encoding='utf-8')

documents.extend(loader.load())

#text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
start_time = time.time()
############ SPLITTER
chunk = 1500  # for larger file 1500 works but 1000 does not
chunk_overlap = 100
chain_type = "stuff"  # stuff, refine works, map_reduce does not always give answer
text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk, chunk_overlap=chunk_overlap)# working splitter
documents = text_splitter.split_documents(documents)
### larger token size seems to be more accurate


embeddings = OpenAIEmbeddings()
#embeddings = TensorflowHubEmbeddings()   ## also test ebeddings


vectordb = Chroma.from_documents(documents, embeddings)
retriever = vectordb.as_retriever(search_type = "similarity",search_kwargs={"k": 2})
# similarity find the most similar vector or MMR
#https://python.langchain.com/docs/use_cases/question_answering/how_to/chat_vector_db

## error with gpt-4
## condense means use a more efficent model since it is small data


chat_history =[]
#query = "what is Human-caused climate change a consequence of according to the documents only based on the documents"
query = ""
query ="whats is the FG% for Austin Reaves"
query = "whats is the Number of employees that the Name Sellers-Alvarez have"
query = "what is the Industry for Nicholson-Patel"
query = "which has the highest number of employees"
query = "who has the highest 3PA"
query ="what is the age of Blake Griffin"
query = "whats is the Number of employees that the Name Sellers-Alvarez have"
query = "what organization has the fewest of employees"
query = "what organization has the second most employees and how many do they have"
#query = "according to the table what is the GHG per capita (tCO2-eq per person) for Europe"


llm=OpenAI()
qa = RetrievalQA.from_chain_type(llm=llm, chain_type=chain_type, retriever=retriever)

llm_response = qa(query)
print(llm_response["result"])
total = llm_response["result"]

end_time = time.time()
xtime = end_time - start_time

file = open('organizations log', 'a')

file.write(str('organizations-3') + "," + total+","+chain_type+"," +str(chunk)+","+str(chunk_overlap)+","+str(xtime)+"\n")

##########################NEXT CHAT#####################################

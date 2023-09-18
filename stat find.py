import os
from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.llms  import OpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader
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



filename = os.path.join("csv files", "1.csv") # csv file taken from csv table.py
loader = TextLoader(filename,encoding='utf-8')
documents.extend(loader.load())

#text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
############ SPLITTER
chunk = 1000  # for larger file 1500 works but 1000 does not
chunk_overlap = 30
chain_type = "stuff"  # stuff, refine works, map_reduce does not always give answer
text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk, chunk_overlap=chunk_overlap)# working splitter
documents = text_splitter.split_documents(documents)
### larger token size seems to be more accurate


embeddings = OpenAIEmbeddings()
#embeddings = TensorflowHubEmbeddings()   ## also test ebeddings


vectordb = Chroma.from_documents(documents, embeddings)
retriever = vectordb.as_retriever(search_type = "similarity",search_kwargs={"k": 3})
# similarity find the most similar vector or MMR
#https://python.langchain.com/docs/use_cases/question_answering/how_to/chat_vector_db

## error with gpt-4
## condense means use a more efficent model since it is small data

start_time = time.time()
chat_history =[]
#query = "what is Human-caused climate change a consequence of according to the documents only based on the documents"
query = ""
query ="whats is the MP for Austin Reaves have"
llm=OpenAI()
qa = RetrievalQA.from_chain_type(llm=llm, chain_type=chain_type, retriever=retriever)

llm_response = qa(query)
print(llm_response["result"])


#result = qa({"question": query})
#print (result['answer'])

##################################TIMER and Save LOG
end_time = time.time()
xtime = end_time - start_time
xtime = round(xtime, 3)
file = open('log with chain type', 'a')

file.write(str('all docs') + "," + str(xtime)+","+chain_type+"," +str(chunk)+","+str(chunk_overlap)+"\n")

##########################NEXT CHAT#####################################

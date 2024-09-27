from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain.vectorstores import FAISS
from langchain_community.vectorstores.faiss import FAISS
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CohereRerank
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain.document_loaders.pdf import PyPDFDirectoryLoader
import os
from getpass import getpass
import csv
import time
#https://medium.com/@nadikapoudel16/advanced-rag-implementation-using-hybrid-search-reranking-with-zephyr-alpha-llm-4340b55fef22
HF_token = getpass()
os.environ['HUGGINGFACEHUB_API_TOKEN'] = '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'

pdf_folder_path = "./"
loader = PyPDFDirectoryLoader(pdf_folder_path)
documents = loader.load()

text_splitter=RecursiveCharacterTextSplitter(chunk_size=512,chunk_overlap=90)
text_splits=text_splitter.split_documents(documents)
print(len(text_splits))

embeddings=HuggingFaceInferenceAPIEmbeddings(
    api_key=HF_token,
    model_name='BAAI/bge-base-en-v1.5'
)

model_name = "BAAI/bge-small-en-v1.5"
encode_kwargs = {'normalize_embeddings': True} # set True to compute cosine similarity
from langchain.embeddings import HuggingFaceBgeEmbeddings
embeddings = HuggingFaceBgeEmbeddings(
    model_name=model_name,
    model_kwargs={'device': 'cpu'},
    encode_kwargs=encode_kwargs
)



start = time.time()
start_time = time.process_time()


vectorstore = FAISS.from_documents(text_splits, embeddings)

#retriever_vectordb = vectorstore.as_retriever(search_kwargs={"body_search": "school"})
retriever_vectordb = vectorstore.as_retriever(search_kwargs={"k": 5},)
keyword_retriever = BM25Retriever.from_documents(text_splits)
keyword_retriever.k =  5
ensemble_retriever = EnsembleRetriever(retrievers=[retriever_vectordb,keyword_retriever],
                                       weights=[0.5, 0.5])

query="what are some managements practices?"
docs_rel=ensemble_retriever.get_relevant_documents(query)
print (docs_rel)

end = time.time()
end_time = time.process_time()
totaltime = end - start


with open("hybrid3.csv", mode='a', newline='') as file:
    writer = csv.writer(file)
    # If docs_rel[1] exists, add it to the CSV file with time and query
    writer.writerow([docs_rel[2], totaltime, query])


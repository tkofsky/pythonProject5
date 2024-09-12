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
#https://medium.com/@nadikapoudel16/advanced-rag-implementation-using-hybrid-search-reranking-with-zephyr-alpha-llm-4340b55fef22
HF_token = getpass()


pdf_folder_path = "./"
loader = PyPDFDirectoryLoader(pdf_folder_path)
documents = loader.load()

text_splitter=RecursiveCharacterTextSplitter(chunk_size=512,chunk_overlap=50)
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



vectorstore = FAISS.from_documents(text_splits, embeddings)

retriever_vectordb = vectorstore.as_retriever(search_kwargs={"body_search": "school"})   ## use keywords as hybrid
retriever_vectordb = vectorstore.as_retriever(search_kwargs={"k": 5},)
keyword_retriever = BM25Retriever.from_documents(text_splits)
keyword_retriever.k =  5
ensemble_retriever = EnsembleRetriever(retrievers=[retriever_vectordb,keyword_retriever],
                                       weights=[0.5, 0.5])

query="what are some managements practices?"
docs_rel=ensemble_retriever.get_relevant_documents(query)
print (docs_rel)


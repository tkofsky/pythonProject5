import os
import openai
from getpass import getpass
#
import langchain_community
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import TextLoader
from langchain.vectorstores import FAISS
from langchain.document_loaders.pdf import PyPDFDirectoryLoader
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CohereRerank
os.environ["COHERE_API_KEY"] = getpass("Cohere API Key:")
from langchain.retrievers import BM25Retriever, EnsembleRetriever

OPENAI_API_KEY= os.environ.get("OPENAI_API_KEY")
# run in debug mode

#https://medium.aiplanet.com/advanced-rag-cohere-re-ranker-99acc941601c
def pretty_print_docs(docs):
    print(
        f"\n{'-' * 100}\n".join(
            [f"Document {i+1}:\n\n" + d.page_content for i, d in enumerate(docs)]
        )
    )

pdf_folder_path = "./"
loader = PyPDFDirectoryLoader(pdf_folder_path)
docs = loader.load()
#
print(len(docs))
#
print(docs[1].page_content)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
texts = text_splitter.split_documents(docs)
print(len(texts))


model_name = "BAAI/bge-small-en-v1.5"
encode_kwargs = {'normalize_embeddings': True} # set True to compute cosine similarity

embeddings = HuggingFaceBgeEmbeddings(
    model_name=model_name,
    model_kwargs={'device': 'cpu'},
    encode_kwargs=encode_kwargs
)

vectorstore = FAISS.from_documents(texts, embeddings)


retriever_vectordb = vectorstore.as_retriever(search_kwargs={"k": 5})
keyword_retriever = BM25Retriever.from_documents(texts)
keyword_retriever.k =  5
ensemble_retriever = EnsembleRetriever(retrievers=[retriever_vectordb,keyword_retriever],
                                       weights=[0.5, 0.5])







retriever = vectorstore.as_retriever(search_kwargs={"k": 20})

query = "According to Kelly and Williams what is ethics?"
docs = retriever.get_relevant_documents(query)
pretty_print_docs(docs)

#
llm = ChatOpenAI(model_name="gpt-4o-mini",temperature=0.1)
#
qa = RetrievalQA.from_chain_type(llm=llm,
                                 chain_type="stuff",
                                 retriever=retriever)

import time

start = time.time()
start_time = time.process_time()
print(qa.run(query=query))
end = time.time()
end_time = time.process_time()
print(f"CPU time used: {end_time - start_time} seconds")
print(end - start)

compressor = CohereRerank()
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor, base_retriever=retriever
)
#
compressed_docs = compression_retriever.get_relevant_documents(query)
pretty_print_docs(compressed_docs)

qa = RetrievalQA.from_chain_type(llm=llm,
                                 chain_type="stuff",
                                 retriever=compression_retriever )

print(qa.run(query=query))
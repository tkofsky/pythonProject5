from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
import requests
from bs4 import BeautifulSoup
from langchain.schema import Document  # Ensure this import is correct
from langchain_openai import OpenAI
import os
# Define the #reranking function
#########################
from langchain.embeddings import OpenAIEmbeddings
openai_api_key = os.environ.get("OPENAI_API_KEY")
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
test_embedding = embeddings.embed_query("Test query")
print(test_embedding)



#############
def rerank_documents(query, docs, embeddings):
    query_embedding = embeddings.embed_query(query)
    scored_docs = [
        (doc, cosine_similarity([query_embedding], [embeddings.embed_query(doc.page_content)])[0][0])
        for doc in docs
    ]
    return sorted(scored_docs, key=lambda x: x[1], reverse=True)

# Initialize OpenAI LLM and embeddings
openai_api_key = os.environ.get("OPENAI_API_KEY")
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
llm = OpenAI(api_key=openai_api_key)

# Load documents from a URL and split them into chunks

def split_text(text, max_length=500):
    return [text[i:i+max_length] for i in range(0, len(text), max_length)]

def load_webpage_content(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")
    paragraphs = [p.get_text() for p in soup.find_all("p")]
    return " ".join(paragraphs)
url = "https://en.wikipedia.org/wiki/Stephen_Curry"
webpage_content = load_webpage_content(url)
document_chunks = split_text(webpage_content)
documents = [Document(page_content=chunk) for chunk in document_chunks]

# Initialize ChromaDB with embeddings
vectorstore = Chroma.from_documents(documents, embeddings)

# Retrieve and rerank documents
sample_question = "What religion is steph curry"
retrieved_docs = vectorstore.similarity_search(sample_question, k=5)
reranked_docs = rerank_documents(sample_question, retrieved_docs, embeddings)

# Use the top reranked documents as the basis for the retriever
top_reranked_docs = [doc[0] for doc in reranked_docs[:3]]
reranked_retriever = Chroma.from_documents(top_reranked_docs, embeddings)

# Create the RetrievalQA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=reranked_retriever.as_retriever()  # Properly initialize the retriever
)

# Run the QA chain
answer = qa_chain.run(sample_question)
print("Answer:", answer)

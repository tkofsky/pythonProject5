import openai
import json
from typing import List, Dict
import chromadb

# Configure OpenAI API key
openai.api_key = "YOUR_OPENAI_API_KEY"

# Initialize Chroma client and collection
chroma_client = chromadb.Client()  # Using the default configuration without deprecated settings
collection_name = "rag_documents"
collection = chroma_client.get_or_create_collection(name=collection_name)


# Load documents from JSON file in 'data' subfolder
def load_documents(filepath: str) -> List[Dict[str, str]]:
    with open(filepath, 'r') as file:
        return json.load(file)


# Embed documents using OpenAI's embedding model and store in Chroma
def embed_and_store_documents(documents: List[Dict[str, str]]):
    for doc in documents:
        embedding = openai.Embedding.create(input=doc["content"], model="text-embedding-ada-002")["data"][0][
            "embedding"]
        collection.add(
            documents=[doc["content"]],
            metadatas=[doc],
            ids=[doc["title"]]
        )
    print("Documents embedded and stored.")


# Retrieve top-k documents based on query embedding
def retrieve_documents(query: str, top_k=2) -> List[Dict[str, str]]:
    query_embedding = openai.Embedding.create(input=query, model="text-embedding-ada-002")["data"][0]["embedding"]
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )
    return [{"title": res["id"], "content": res["document"], "metadata": res["metadata"]} for res in
            results["documents"]]


# Generate a response using OpenAI's ChatGPT model, incorporating retrieved context
def generate_response(query: str, context_docs: List[Dict[str, str]]) -> str:
    context = " ".join([doc["content"] for doc in context_docs])
    prompt = f"Answer the question based on the context below:\n\nContext: {context}\n\nQuestion: {query}\nAnswer:"
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content


# Main RAG pipeline function
def run_rag_pipeline(query: str, document_path: str):
    # Step 1: Load and store documents in vector database
    documents = load_documents(document_path)
    embed_and_store_documents(documents)

    # Step 2: Retrieve relevant documents
    context_docs = retrieve_documents(query)
    print("Retrieved Documents:")
    for doc in context_docs:
        print(f"- {doc['title']}: {doc['content']}")

    # Step 3: Generate response
    response = generate_response(query, context_docs)
    print("\nGenerated Response:\n", response)


# Example usage
query = "How does RAG enhance LLM responses?"
run_rag_pipeline(query, "data/sample_documents.json")

import json
import csv
import os
from typing import List, Dict
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import numpy as np


# Load documents from JSON file in 'data' subfolder
def load_documents(filepath: str) -> List[Dict[str, str]]:
    with open(filepath, 'r') as file:
        return json.load(file)


# Initialize embedding model (e.g., SBERT for embeddings)
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')


# Convert documents to embeddings (as numpy arrays)
def embed_documents(documents: List[Dict[str, str]]) -> List[Dict[str, np.ndarray]]:
    for doc in documents:
        doc["embedding"] = embedding_model.encode(doc["content"], convert_to_tensor=True).cpu().numpy()
    return documents


# Retrieve documents based on cosine similarity
def retrieve_documents(query: str, documents: List[Dict[str, str]], top_k: int = 2) -> List[Dict[str, str]]:
    query_embedding = embedding_model.encode(query, convert_to_tensor=True).cpu().numpy()
    similarities = [(doc, cosine_similarity([query_embedding], [doc["embedding"]])[0][0]) for doc in documents]
    ranked_docs = sorted(similarities, key=lambda x: x[1], reverse=True)
    return [doc[0] for doc in ranked_docs[:top_k]]


# Mock LLM response generation using retrieved context
def generate_response(query: str, context_docs: List[Dict[str, str]]) -> str:
    context = " ".join([doc["content"] for doc in context_docs])
    response = f"Based on the context, RAG leverages retrieval to ground LLM responses: {context}"
    return response


# Evaluation function based on coherence, relevance, and completeness
def evaluate_response(response: str, context_docs: List[Dict[str, str]]) -> Dict[str, float]:
    # Convert response to embedding as a numpy array
    response_embedding = embedding_model.encode(response, convert_to_tensor=True).cpu().numpy()
    context_embeddings = [doc["embedding"] for doc in context_docs]

    # Relevance: cosine similarity with retrieved context
    relevance_score = np.mean([cosine_similarity([response_embedding], [emb])[0][0] for emb in context_embeddings])

    # Coherence: Heuristic based on word count
    coherence_score = 1.0 if len(response.split()) > 20 else 0.5

    # Completeness: Coverage of key terms
    key_terms = ["RAG", "retrieval", "LLM"]
    completeness_score = sum(1 for term in key_terms if term in response) / len(key_terms)

    return {
        "relevance": relevance_score,
        "coherence": coherence_score,
        "completeness": completeness_score,
        "average_score": (relevance_score + coherence_score + completeness_score) / 3
    }


# Function to save responses and scores to a CSV file
def save_to_csv(filepath: str, query: str, response: str, scores: Dict[str, float]):
    with open(filepath, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([query, response, scores["relevance"], scores["coherence"], scores["completeness"],
                         scores["average_score"]])


# Main RAG process
def run_rag_pipeline(query: str, document_path: str, output_path: str):
    # Step 1: Load and embed documents
    documents = load_documents(document_path)
    documents = embed_documents(documents)

    # Step 2: Retrieve top-k documents
    context_docs = retrieve_documents(query, documents)
    print("Retrieved Documents:")
    for doc in context_docs:
        print(f"- {doc['title']}: {doc['content']}")

    # Step 3: Generate response
    response = generate_response(query, context_docs)
    print("\nGenerated Response:\n", response)

    # Step 4: Evaluate response
    evaluation_scores = evaluate_response(response, context_docs)
    print("\nEvaluation Scores:", evaluation_scores)

    # Step 5: Save response and scores to CSV
    save_to_csv(output_path, query, response, evaluation_scores)


# Initialize CSV file with headers if it doesn't already exist
if not os.path.isfile("noground.csv"):
    with open("noground.csv", mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(
            ["Query", "Response", "Relevance Score", "Coherence Score", "Completeness Score", "Average Score"])

# Example usage
query = "How does RAG enhance LLM responses?"
query = "what do llms struggle with?"
run_rag_pipeline(query, "data/sample_documents.json", "noground.csv")

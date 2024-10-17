from typing import List, Dict
import random


# Mock function to simulate retrieval process
def retrieve_documents(query: str) -> List[Dict[str, str]]:
    # Example documents returned by retrieval system
    documents = [
        {"title": "Document 1",
         "content": "RAG helps improve LLM responses by providing context from relevant documents."},
        {"title": "Document 2",
         "content": "Retrieval-augmented generation combines retrieval and generation models to create informed responses."},
        {"title": "Document 3",
         "content": "LLMs struggle with factual accuracy, which RAG addresses by grounding responses in real data."},
    ]
    return random.sample(documents, k=2)  # Randomly return 2 documents


# Mock LLM response generation with retrieval-based context
def generate_response(query: str, context_docs: List[Dict[str, str]]) -> str:
    context = " ".join([doc["content"] for doc in context_docs])
    response = f"Based on the context, RAG leverages retrieval to ground LLM responses, improving relevance and factual accuracy. {context}"
    return response


# Evaluation function based on coherence, relevance, and completeness
def evaluate_response(response: str, context_docs: List[Dict[str, str]]) -> Dict[str, float]:
    # Check coherence: length and logical flow (simple heuristic)
    coherence_score = 1.0 if len(response.split()) > 20 else 0.5

    # Check relevance: presence of key terms
    key_terms = ["RAG", "retrieval", "generation", "LLM"]
    relevance_score = sum(1 for term in key_terms if term in response) / len(key_terms)

    # Check completeness: covers main points from context docs
    context_points = [doc["content"] for doc in context_docs]
    completeness_score = sum(1 for point in context_points if point in response) / len(context_points)

    # Return scores
    return {
        "coherence": coherence_score,
        "relevance": relevance_score,
        "completeness": completeness_score,
        "average_score": (coherence_score + relevance_score + completeness_score) / 3
    }


# Main RAG process
def run_rag_pipeline(query: str):
    # Step 1: Retrieve documents
    context_docs = retrieve_documents(query)
    print("Retrieved Documents:")
    for doc in context_docs:
        print(f"- {doc['title']}: {doc['content']}")

    # Step 2: Generate response using the retrieved context
    response = generate_response(query, context_docs)
    print("\nGenerated Response:\n", response)

    # Step 3: Evaluate response
    evaluation_scores = evaluate_response(response, context_docs)
    print("\nEvaluation Scores:", evaluation_scores)


# Example usage
query = "How does RAG enhance LLM responses?"
run_rag_pipeline(query)

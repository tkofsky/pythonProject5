import cohere
import numpy as np

# Initialize Cohere client with your API key
cohere_api_key = "YOUR_API_KEY"  # Replace with your actual Cohere API key
co = cohere.Client(cohere_api_key)


def get_embedding(text):
    """
    Get the embedding of a given text using Cohere's embedding model.
    """
    response = co.embed(
        model='embed-english-v2.0',  # Model name; you can choose the latest or best-suited model
        texts=[text]
    )
    return np.array(response.embeddings[0])


def cosine_similarity(embedding1, embedding2):
    """
    Calculate cosine similarity between two embeddings.
    """
    return np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))


def score_answer(question, answer):
    """
    Score the answer based on its similarity to the question using Cohere embeddings.
    """
    # Get embeddings for the question and the answer
    question_embedding = get_embedding(question)
    answer_embedding = get_embedding(answer)

    # Calculate cosine similarity score
    similarity_score = cosine_similarity(question_embedding, answer_embedding)

    # Normalize the score to a 0-1 range
    normalized_score = (similarity_score + 1) / 2
    return normalized_score


# Example usage
def main():
    # Define the question and the answer
    question = "When was the Eiffel Tower completed?"
    answer = "The Eiffel Tower was completed in 1889."

    # Score the answer
    score = score_answer(question, answer)

    # Print the results
    print(f"Question: {question}")
    print(f"Answer: {answer}")
    print(f"Score: {score:.2f}")


# Run the main function
if __name__ == "__main__":
    main()
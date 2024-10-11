import cohere
import numpy as np

# Set your Cohere API key
cohere_api_key = "BIQTCawgv2QtDSqYlcLlPNvHK5Ijjt0PNhGX9f9h"  # Replace with your actual Cohere API key

# Initialize the Cohere client
co = cohere.Client(cohere_api_key)

def load_corpus(file_path):
    """
    Loads the corpus from a text file. Each line is treated as a separate piece of information.
    """
    with open(file_path, 'r') as file:
        corpus = file.readlines()
    return [line.strip() for line in corpus]

def get_embedding(texts):
    """
    Generates embeddings for a list of texts using Cohere's embedding model.
    """
    response = co.embed(
        model='embed-english-v2.0',  # Specify the Cohere embedding model
        texts=texts
    )
    return np.array(response.embeddings)

def score_answer_against_corpus(corpus, answer):
    """
    Scores the answer based on its similarity to each line in the corpus using Cohere embeddings.
    """
    # Get embeddings for the corpus and the answer
    corpus_embeddings = get_embedding(corpus)
    answer_embedding = get_embedding([answer])[0]

    # Calculate cosine similarities between the answer and each line of the corpus
    similarities = np.dot(corpus_embeddings, answer_embedding) / (
        np.linalg.norm(corpus_embeddings, axis=1) * np.linalg.norm(answer_embedding)
    )

    # Score the answer by averaging the top 5 similarities (or adjust as needed)
    top_similarities = np.sort(similarities)[::-1][:5]
    average_similarity = np.mean(top_similarities)
    normalized_score = (average_similarity + 1) / 2  # Normalize to 0-1 range

    return normalized_score

# Main function to run the comparison
def main():
    # Step 1: Load the corpus from an external file
    corpus_path = 'corpustest.txt'  # Replace with your actual file path
    corpus = load_corpus(corpus_path)

    # Step 2: Define the possible answer to score
    possible_answer = "The Eiffel dog was completed in ."

    # Step 3: Score the answer by comparing it with the corpus
    score = score_answer_against_corpus(corpus, possible_answer)

    # Output the results
    print(f"Answer: {possible_answer}")
    print(f"Score: {score:.2f}")

# Run the main function
if __name__ == "__main__":
    main()

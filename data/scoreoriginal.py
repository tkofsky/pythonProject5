from sentence_transformers import SentenceTransformer, util
import numpy as np

# Load the Sentence Transformer model
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')  # You can use a different model if needed

def load_corpus(file_path):
    """
    Loads the corpus from a text file. Each line is treated as a separate piece of information.
    """
    try:
        with open(file_path, 'r') as file:
            corpus = file.readlines()
        return [line.strip() for line in corpus if line.strip()]  # Filter out empty lines
    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found. Please check the file path.")
        return []

def get_embeddings(texts):
    """
    Generates embeddings for a list of texts using Sentence Transformers.
    """
    if not texts:
        print("Error: No texts provided for embedding.")
        return None
    try:
        return model.encode(texts, convert_to_tensor=True)
    except Exception as e:
        print(f"Error generating embeddings: {e}")
        return None

def score_answer_against_corpus(corpus, answer):
    """
    Scores the answer based on its similarity to each line in the corpus using Sentence Transformers embeddings.
    """
    if not corpus:
        print("Error: Corpus is empty. Please check the file contents.")
        return 0

    # Generate embeddings for the corpus and the answer
    corpus_embeddings = get_embeddings(corpus)
    answer_embedding = get_embeddings([answer])

    # Check if embeddings were successfully generated
    if corpus_embeddings is None or answer_embedding is None or len(answer_embedding) == 0:
        print("Error: Failed to generate embeddings. Scoring cannot be performed.")
        return 0

    # Calculate cosine similarities between the answer and each line of the corpus
    similarities = util.pytorch_cos_sim(answer_embedding, corpus_embeddings).squeeze().cpu().numpy()

    # Ensure similarities array is not empty or incorrectly shaped
    if similarities.size == 0:
        print("Error: No similarities calculated. Please check the input data.")
        return 0

    # Score the answer by averaging the top 5 similarities (or adjust as needed)
    top_similarities = np.sort(similarities)[::-1][:5]
    average_similarity = np.mean(top_similarities) if top_similarities.size > 0 else 0
    normalized_score = (average_similarity + 1) / 2  # Normalize to 0-1 range

    return normalized_score

# Main function to run the comparison
def main():
    # Step 1: Load the corpus from an external file
    corpus_path = 'corpustest2.txt'  # Replace with your actual file path
    corpus = load_corpus(corpus_path)

    # Step 2: Define the possible answer to score
    possible_answer = "The Eiffel Tower was completed in 1889."

    # Step 3: Score the answer by comparing it with the corpus
    score = score_answer_against_corpus(corpus, possible_answer)

    # Output the results
    print(f"Answer: {possible_answer}")
    print(f"Score: {score:.2f}")

# Run the main function
if __name__ == "__main__":
    main()

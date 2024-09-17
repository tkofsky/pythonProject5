from sentence_transformers import SentenceTransformer, util
import numpy as np

# Load the Sentence Transformer model
model = SentenceTransformer('all-mpnet-base-v2')  # You can change the model to experiment with different ones

def normalize_text(text):
    """
    Normalize the text by converting to lowercase, stripping whitespace,
    removing extra spaces, and normalizing punctuation.
    """
    return text.lower().strip()

def load_corpus(file_path):
    """
    Loads the corpus from a text file. Each line is treated as a separate piece of information.
    """
    try:
        with open(file_path, 'r') as file:
            corpus = file.readlines()
        return [normalize_text(line) for line in corpus if line.strip()]  # Normalize and filter out empty lines
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
        return model.encode(texts, convert_to_tensor=True, normalize_embeddings=True)
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

    # Normalize the answer text
    normalized_answer = normalize_text(answer)

    # Generate embeddings for the corpus and the answer
    corpus_embeddings = get_embeddings(corpus)
    answer_embedding = get_embeddings([normalized_answer])

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

    # Check if the answer exactly matches any line in the corpus by setting a high threshold
    max_similarity = np.max(similarities)
    exact_match_score = 1.0 if max_similarity > 0.99 else max_similarity  # Treat near-perfect matches as exact

    return exact_match_score

# Main function to run the comparison
def main():
    # Step 1: Load the corpus from an external file
    corpus_path = 'corpustest.txt'  # Replace with your actual file path
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

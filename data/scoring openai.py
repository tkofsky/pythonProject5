import cohere
import numpy as np
from transformers import pipeline

# Set your Cohere API key
cohere_api_key = "1ODOOnJTr2xY7ncwJP0f54wZaA8o5AmB9hJQmHq7"  # Replace with your actual Cohere API key

# Initialize the Cohere client
co = cohere.Client(cohere_api_key)

# Load a text generation model from Hugging Face (e.g., GPT-2)
generator = pipeline("text-generation", model="gpt2", max_length=1000)  # Replace with a larger model if desired


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


def retrieve_relevant_context(corpus, question, top_k=5):
    """
    Retrieves relevant lines from the corpus based on the question using Cohere embeddings.
    """
    # Get embeddings for the question and the corpus
    question_embedding = get_embedding([question])[0]
    corpus_embeddings = get_embedding(corpus)

    # Calculate cosine similarities between the question and all corpus lines
    similarities = np.dot(corpus_embeddings, question_embedding) / (
                np.linalg.norm(corpus_embeddings, axis=1) * np.linalg.norm(question_embedding))

    # Get the indices of the top K most relevant lines (sorted in descending order)
    top_indices = np.argsort(similarities)[::-1][:top_k]
    relevant_context = " ".join([corpus[idx] for idx in top_indices])
    return relevant_context


def generate_answer(context, question):
    """
    Generates an answer using the retrieved context and the question.
    Uses a Hugging Face Transformer model for text generation.
    """
    # Combine the context and question for the generation input
    input_text = f"Context: {context}\nQuestion: {question}\nAnswer:"

    # Generate the answer using the Hugging Face model
    response = generator(input_text, max_length=1000, num_return_sequences=1)[0]['generated_text']

    # Extract the generated answer from the response
    answer = response.split("Answer:")[-1].strip()
    return answer


def score_answer(question, answer):
    """
    Scores the answer based on its similarity to the question using Cohere embeddings.
    """
    # Get embeddings for the question and the answer
    embeddings = get_embedding([question, answer])
    question_embedding = embeddings[0]
    answer_embedding = embeddings[1]

    # Calculate cosine similarity between the question and the answer
    similarity_score = np.dot(question_embedding, answer_embedding) / (
                np.linalg.norm(question_embedding) * np.linalg.norm(answer_embedding))

    # Normalize score to a range if desired (0 to 1)
    normalized_score = (similarity_score + 1) / 2  # This step makes the score between 0 and 1

    return normalized_score


# Main function to run the RAG process
def main():
    # Step 1: Load the corpus from an external file
    corpus_path = 'corpustest.txt'  # Replace with your actual file path
    corpus = load_corpus(corpus_path)

    # Step 2: Define the question
    question = "When was the Eiffel Tower completed?"

    # Step 3: Retrieve relevant context from the corpus
    context = retrieve_relevant_context(corpus, question)

    # Step 4: Generate an answer using the retrieved context
    answer = generate_answer(context, question)

    # Step 5: Score the generated answer based on its similarity to the question
    score = score_answer(question, answer)

    # Output the results
    print(f"Context: {context}")
    print(f"Question: {question}")
    print(f"Answer: {answer}")
    print(f"Score: {score:.2f}")


# Run the main function
if __name__ == "__main__":
    main()

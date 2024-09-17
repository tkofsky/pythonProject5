import openai
from sentence_transformers import SentenceTransformer, util
import numpy as np
import os
# Set your OpenAI API key
api_key = os.environ.get("OPENAI_API_KEY")
openai.api_key = api_key


# Load the Sentence Transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')  # Lightweight, efficient model for embeddings


def load_corpus(file_path):
    """
    Loads the corpus from a text file. Each line is treated as a separate piece of information.
    """
    with open(file_path, 'r') as file:
        corpus = file.readlines()
    return [line.strip() for line in corpus]


def retrieve_relevant_context(corpus, question, top_k=1):
    """
    Retrieves relevant lines from the corpus based on the question.
    Uses cosine similarity between question embedding and corpus embeddings.
    """
    # Generate embeddings for the question and the corpus
    question_embedding = model.encode(question, convert_to_tensor=True)
    corpus_embeddings = model.encode(corpus, convert_to_tensor=True)

    # Calculate cosine similarities between the question and all corpus lines
    similarities = util.pytorch_cos_sim(question_embedding, corpus_embeddings).squeeze()

    # Get the indices of the top K most relevant lines (sorted in descending order)
    top_indices = np.argsort(similarities.numpy())[::-1][:top_k]
    relevant_context = " ".join([corpus[idx] for idx in top_indices])
    return relevant_context

def generate_answer(context, question):
    """
    Generates an answer using the retrieved context and the question.
    Calls OpenAI's ChatCompletion API with the context and question.
    """
    messages = [
        {"role": "system",
         "content": "You are an expert assistant that provides accurate and concise answers based on the given context."},
        {"role": "user", "content": f"Context: {context}\nQuestion: {question}\nAnswer:"}
    ]

    # Call the OpenAI ChatCompletion model
    response = openai.chat.completions.create(
        model="gpt-4",  # You can use "gpt-3.5-turbo" if gpt-4 is not available
        messages=messages,
        max_tokens=10,
        temperature=0.0  # Low temperature to encourage more deterministic output
    )

    # Extract the score from the response
    score = response.choices[0].message.content.strip()
    return score


def score_answer(question, answer):
    """
    Scores the answer based on its similarity to the question using cosine similarity of embeddings.
    """
    # Generate embeddings for the question and answer
    embeddings = model.encode([question, answer], convert_to_tensor=True)

    # Calculate cosine similarity
    similarity_score = util.pytorch_cos_sim(embeddings[0], embeddings[1]).item()

    # Normalize score to a range if desired (0 to 1)
    normalized_score = (similarity_score + 1) / 2  # This step makes the score between 0 and 1

    return normalized_score


# Main function to run the RAG process
def main():
    # Step 1: Load the corpus from an external file
    corpus_path = 'corpustest2.txt'  # Replace with your actual file path
    corpus = load_corpus(corpus_path)

    # Step 2: Define the question
    question = "When was the Eiffel Tower completed?"
    question = "how big was curry in his sophomore season"

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

from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer, util
import openai
import requests
import numpy as np
import re
import csv

import os
OPENAI_API_KEY= os.environ.get("OPENAI_API_KEY")

# Load a Sentence Transformer model fine-tuned for question answering
model = SentenceTransformer('sentence-transformers/msmarco-distilbert-base-v4')  # QA-focused model


def load_webpage_content(url):
    """
    Loads and parses the content of a web page, extracting text from paragraphs,
    lists, and tables to capture key information.
    """
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')

        # Extract content from paragraphs, list items, and table data cells
        content = []

        # Extract text from paragraphs
        content.extend([p.get_text() for p in soup.find_all('p')])

        # Extract text from list items, which may contain relevant data
        content.extend([li.get_text() for li in soup.find_all('li')])

        # Extract text from table cells, particularly useful for infobox data
        content.extend([td.get_text() for td in soup.find_all('td')])

        # Join all content into a single string
        return ' '.join(content)
    except Exception as e:
        print(f"An error occurred while loading the web page: {e}")
        return ""


def split_text_by_sentences(text, max_length=1000):
    """
    Splits the text into smaller chunks of sentences to maintain coherence.
    """
    sentences = re.split(r'(?<=[.!?]) +', text)  # Split by sentence boundaries
    chunks = []
    current_chunk = []

    for sentence in sentences:
        if len(' '.join(current_chunk) + sentence) < max_length:
            current_chunk.append(sentence)
        else:
            chunks.append(' '.join(current_chunk))
            current_chunk = [sentence]

    if current_chunk:
        chunks.append(' '.join(current_chunk))

    return chunks


def normalize_text(text):
    """
    Normalizes text by converting to lowercase, removing punctuation, and extra spaces.
    """
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = text.strip()
    return text


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


def rank_chunks(chunks, question, top_k=5):
    """
    Ranks chunks based on their similarity to the question and returns the top-k most relevant chunks.
    """
    question_embedding = get_embeddings([question])[0]
    chunk_embeddings = get_embeddings(chunks)

    similarities = util.pytorch_cos_sim(question_embedding, chunk_embeddings).squeeze().cpu().numpy()

    # Sort chunks by similarity score
    ranked_chunks = sorted(zip(chunks, similarities), key=lambda x: x[1], reverse=True)
    return [chunk for chunk, _ in ranked_chunks[:top_k]]


def generate_answer(context, question):
    """
    Generates an answer using OpenAI's GPT model based on the provided context and question.
    """
    try:
        prompt = (
            f"Based on the context, please provide the specific information requested.\n\n"
            f"Context: {context}\n\n"
            f"Question: {question}\n\n"
            f"Answer:"
        )

        response = openai.chat.completions.create(
            model="gpt-4",  # Use gpt-3.5-turbo if gpt-4 is not available
            messages=[
                {"role": "system",
                 "content": "You are an expert assistant that extracts specific information from the provided context."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=150,
            temperature=0.2  # Low temperature to make answers more focused and factual
        )

        answer = response.choices[0].message.content.strip()
        return answer

    except Exception as e:
        print(f"An error occurred while generating the answer: {e}")
        return "Error generating answer."


def score_answer_against_chunk(chunk, answer, question):
    """
    Scores the answer based on its similarity to the chunk using cosine similarity,
    penalizes generic answers, and checks for key term alignment with the question.
    """
    # Normalize the chunk, the question, and the answer
    normalized_chunk = normalize_text(chunk)
    normalized_answer = normalize_text(answer)
    normalized_question = normalize_text(question)

    # Generate embeddings for the chunk and the answer
    chunk_embedding = get_embeddings([normalized_chunk])[0]
    answer_embedding = get_embeddings([normalized_answer])[0]

    # Calculate cosine similarity between the chunk and the answer
    similarity_score = util.cos_sim(answer_embedding, chunk_embedding).item()
    normalized_similarity_score = (similarity_score + 1) / 2  # Normalize score to 0-1

    # Penalize responses that indicate failure to find the correct answer
    penalty = 1.0
    if "context does not provide" in normalized_answer or "not listed" in normalized_answer:
        penalty = 0.5  # Penalize non-specific answers

    # Check if critical terms from the question are present in the answer
    question_terms = set(normalized_question.split())
    answer_terms = set(normalized_answer.split())
    key_term_overlap = len(question_terms.intersection(answer_terms)) / len(question_terms) if question_terms else 0

    # Adjust score based on overlap with key terms to prioritize factual alignment
    adjusted_score = (normalized_similarity_score * penalty) * (0.7 + (0.3 * key_term_overlap))

    # Ensure the score does not exceed 1.0 and properly reflects alignment quality
    return min(1.0, adjusted_score)


# Main function to run the process
def main():
    url = "https://en.wikipedia.org/wiki/Stephen_Curry"
    context = load_webpage_content(url)

    if not context:
        print("Failed to retrieve content from the web page.")
        return

    question = "what is steph curry's college coach's name?"
    chunks = split_text_by_sentences(context)

    # Rank chunks and select the top 5 most relevant chunks
    top_chunks = rank_chunks(chunks, question)

    # Initialize variables to keep track of the highest score and the best chunk
    best_score = 0
    best_chunk = ""
    best_answer = ""

    # Attempt to generate an answer from each top-ranked chunk
    for i, chunk in enumerate(top_chunks):
        answer = generate_answer(chunk, question)
        score = score_answer_against_chunk(chunk, answer, question)

        # Update the best chunk if the current score is higher
        if score > best_score:
            best_score = score
            best_chunk = chunk
            best_answer = answer

    # Display the best answer and its corresponding chunk
    print(f"Best Chunk:\n{best_chunk}\n")
    print(f"Best Answer: {best_answer}\n")
    print(f"Best Score: {best_score:.2f}\n")
    save_to_csv(question, best_answer, best_score, filename="scoring.csv")

def save_to_csv(question, best_answer, best_score, filename="scoring.csv"):
    """
    Saves the question, best answer, and best score into a CSV file called scoring.csv.
    """
    try:
        with open(filename, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([question, best_answer, best_score])
        print(f"Results saved to {filename}")
    except Exception as e:
        print(f"Error writing to file: {e}")


if __name__ == "__main__":


    main()

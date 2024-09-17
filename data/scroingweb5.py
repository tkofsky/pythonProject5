from langchain.document_loaders import WebBaseLoader
from sentence_transformers import SentenceTransformer, util
import openai
import numpy as np
import re
import os
OPENAI_API_KEY= os.environ.get("OPENAI_API_KEY")
# Set your OpenAI API key
#openai.api_key = "YOUR_OPENAI_API_KEY"  # Replace with your actual OpenAI API key

# Load the Sentence Transformer model
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')  # This model is fine-tuned for semantic similarity tasks

def load_webpage_content(url):
    """
    Loads the content of a web page using WebBaseLoader from LangChain.
    """
    try:
        loader = WebBaseLoader(url)
        documents = loader.load()
        content_string = "\n\n".join(doc.page_content for doc in documents)
        return content_string
    except Exception as e:
        print(f"An error occurred while loading the web page: {e}")
        return ""

def split_text_by_paragraphs(text):
    """
    Splits the text into paragraphs to maintain more context coherence.
    """
    paragraphs = text.split('\n\n')
    return [para.strip() for para in paragraphs if para.strip()]

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

    # Print each chunk and its similarity score
    ranked_chunks = sorted(zip(chunks, similarities), key=lambda x: x[1], reverse=True)
    return [chunk for chunk, _ in ranked_chunks[:top_k]]

def generate_answer(context, question):
    """
    Generates an answer using OpenAI's GPT model based on the provided context and question.
    """
    try:
        # Enhanced prompt to direct the model to extract specific information
        prompt = (
            f"Based on the context, please provide the specific information requested.\n\n"
            f"Context: {context}\n\n"
            f"Question: {question}\n\n"
            f"Answer:"
        )

        response = openai.chat.completions.create(
            model="gpt-4",  # Use gpt-3.5-turbo if gpt-4 is not available
            messages=[
                {"role": "system", "content": "You are an expert assistant that extracts specific information from the provided context."},
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

def score_answer_against_chunk(chunk, answer):
    """
    Scores the answer based on its similarity to the chunk using cosine similarity.
    """
    # Normalize the chunk and the answer
    normalized_chunk = normalize_text(chunk)
    normalized_answer = normalize_text(answer)

    # Generate embeddings for the chunk and the answer
    chunk_embedding = get_embeddings([normalized_chunk])[0]
    answer_embedding = get_embeddings([normalized_answer])[0]

    # Calculate cosine similarity between the chunk and the answer
    similarity_score = util.cos_sim(answer_embedding, chunk_embedding).item()

    # Normalize the score to a range between 0 and 1
    normalized_score = (similarity_score + 1) / 2  # This step makes the score between 0 and 1

    return normalized_score

# Main function to run the process
def main():
    url = "https://en.wikipedia.org/wiki/Stephen_Curry"
    context = load_webpage_content(url)

    if not context:
        print("Failed to retrieve content from the web page.")
        return

    question = "what was steph curry's first team he played against in college"
    question = "who is steph curry's favorate player"
    chunks = split_text_by_paragraphs(context)

    # Rank chunks and select the top 5 most relevant chunks
    top_chunks = rank_chunks(chunks, question)

    # Initialize variables to keep track of the highest score and the best chunk
    best_score = 0
    best_chunk = ""
    best_answer = ""

    # Attempt to generate an answer from each top-ranked chunk
    for i, chunk in enumerate(top_chunks):
        answer = generate_answer(chunk, question)
        score = score_answer_against_chunk(chunk, answer)

        # Update the best chunk if the current score is higher
        if score > best_score:
            best_score = score
            best_chunk = chunk
            best_answer = answer

    # Display the best answer and its corresponding chunk
    print(f"Best Chunk:\n{best_chunk}\n")
    print(f"Best Answer: {best_answer}\n")
    print(f"Best Score: {best_score:.2f}\n")

if __name__ == "__main__":
    main()

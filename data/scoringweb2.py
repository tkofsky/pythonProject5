from langchain.document_loaders import WebBaseLoader
from sentence_transformers import SentenceTransformer, util
import openai
import numpy as np
import re
import os
# Set your OpenAI API key
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")  # Replace with your actual OpenAI API key

# Load the Sentence Transformer model
model = SentenceTransformer('all-mpnet-base-v2')  # You can change the model to experiment with different ones

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

def split_text(text, max_length=2000):
    """
    Splits the text into smaller chunks of specified maximum length.
    """
    sentences = text.split('. ')
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

def select_relevant_chunks(chunks, question, top_k=5):
    """
    Selects the most relevant chunks of text based on their similarity to the question.
    """
    question_embedding = get_embeddings([question])[0]
    chunk_embeddings = get_embeddings(chunks)

    similarities = util.pytorch_cos_sim(question_embedding, chunk_embeddings).squeeze().cpu().numpy()
    top_indices = np.argsort(similarities)[::-1][:top_k]
    selected_chunks = [chunks[i] for i in top_indices]
    return " ".join(selected_chunks)

def generate_answer(context, question):
    """
    Generates an answer using OpenAI's GPT model based on the provided context and question.
    """
    try:
        prompt = f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"

        response = openai.chat.completions.create(
            model="gpt-4",  # Use gpt-3.5-turbo if gpt-4 is not available
            messages=[
                {"role": "system", "content": "You are an expert assistant that answers questions based on the provided context."},
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

def score_answer_against_context(context, answer):
    """
    Scores the answer based on its similarity to the context using cosine similarity.
    """
    # Normalize the context and the answer
    normalized_context = normalize_text(context)
    normalized_answer = normalize_text(answer)

    # Generate embeddings for the context and the answer
    context_embedding = get_embeddings([normalized_context])[0]
    answer_embedding = get_embeddings([normalized_answer])[0]

    # Calculate cosine similarity between the context and the answer
    similarity_score = util.cos_sim(answer_embedding, context_embedding).item()

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

    question = "When was Stephen Curry born?"
    chunks = split_text(context)

    # Select the most relevant chunks to use as context
    relevant_context = select_relevant_chunks(chunks, question)

    # Generate an answer based on the relevant context and the question
    answer = generate_answer(relevant_context, question)

    # Score the answer based on its similarity to the web page content
    score = score_answer_against_context(relevant_context, answer)

    print(f"Question: {question}")
    print(f"Answer: {answer}")
    print(f"Score: {score:.2f}")

if __name__ == "__main__":
    main()

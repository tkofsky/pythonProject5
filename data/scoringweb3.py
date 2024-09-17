from langchain.document_loaders import WebBaseLoader
from sentence_transformers import SentenceTransformer, util
import openai
import numpy as np
import re
import os
OPENAI_API_KEY= os.environ.get("OPENAI_API_KEY")


# Load the Sentence Transformer model
model = SentenceTransformer('all-mpnet-base-v2')  # Change to another model if needed

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

def select_most_relevant_chunk(chunks, question):
    """
    Selects the single most relevant chunk of text based on its similarity to the question.
    """
    question_embedding = get_embeddings([question])[0]
    chunk_embeddings = get_embeddings(chunks)

    similarities = util.pytorch_cos_sim(question_embedding, chunk_embeddings).squeeze().cpu().numpy()
    most_relevant_index = np.argmax(similarities)
    return chunks[most_relevant_index]

def generate_answer(context, question):
    """
    Generates an answer using OpenAI's GPT model based on the provided context and question.
    """
    try:
        # Enhance the prompt to focus on extracting birth-related information
        prompt = (
            f"Context: {context}\n\n"
            f"Question: {question}\n\nAnswer:"
        )

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

def score_answer_against_chunk(chunk, answer):
    """
    Scores the answer based on its similarity to the most relevant chunk using cosine similarity.
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

    question = "what college did he go to?"
    chunks = split_text(context)

    # Select the single most relevant chunk to use as context
    most_relevant_chunk = select_most_relevant_chunk(chunks, question)

    # Generate an answer based on the most relevant chunk and the question
    answer = generate_answer(most_relevant_chunk, question)

    # Score the answer based on its similarity to the most relevant chunk
    score = score_answer_against_chunk(most_relevant_chunk, answer)

    print(f"Question: {question}")
    print(f"Answer: {answer}")
    print(f"Score: {score:.2f}")

if __name__ == "__main__":
    main()

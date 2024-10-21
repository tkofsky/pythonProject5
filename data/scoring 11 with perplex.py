import csv
import time
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer, util
import openai
import requests
import numpy as np
import re
from concurrent.futures import ThreadPoolExecutor
import os

# Set your OpenAI API key
openai.api_key = os.environ.get("OPENAI_API_KEY")

# Load a Sentence Transformer model fine-tuned for question answering
model = SentenceTransformer('sentence-transformers/msmarco-distilbert-base-v4')  # QA-focused model


import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

def calculate_perplexity(text):
    """
    Calculates the perplexity of the given text using GPT-2.
    """
    try:
        # Load pre-trained GPT-2 tokenizer and model
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        model = GPT2LMHeadModel.from_pretrained("gpt2")

        # Encode input text into token IDs
        input_ids = tokenizer.encode(text, return_tensors='pt')

        # Disable gradient calculations for evaluation
        with torch.no_grad():
            outputs = model(input_ids, labels=input_ids)
            loss = outputs.loss
            perplexity = torch.exp(loss)

        return perplexity.item()

    except Exception as e:
        print(f"An error occurred while calculating perplexity: {e}")
        return None




def load_webpage_content(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')

        content = []
        content.extend([p.get_text() for p in soup.find_all('p')])
        content.extend([li.get_text() for li in soup.find_all('li')])
        content.extend([td.get_text() for td in soup.find_all('td')])

        return ' '.join(content)
    except Exception as e:
        print(f"An error occurred while loading the web page: {e}")
        return ""

def split_text_by_sentences(text, max_length=1000):
    sentences = re.split(r'(?<=[.!?]) +', text)
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
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = text.strip()
    return text

def get_embeddings(texts):
    if not texts:
        print("Error: No texts provided for embedding.")
        return None
    try:
        return model.encode(texts, convert_to_tensor=True)
    except Exception as e:
        print(f"Error generating embeddings: {e}")
        return None

def rank_chunks(chunks, question_embedding, top_k=5):
    chunk_embeddings = get_embeddings(chunks)
    similarities = util.pytorch_cos_sim(question_embedding, chunk_embeddings).squeeze().cpu().numpy()

    ranked_chunks = sorted(zip(chunks, similarities), key=lambda x: x[1], reverse=True)
    return [chunk for chunk, _ in ranked_chunks[:top_k]], [score for _, score in ranked_chunks[:top_k]]

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
                {"role": "system", "content": "You are an expert assistant that extracts specific information from the provided context."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=150,
            temperature=0.9  # Low temperature to make answers more focused and factual
        )

        answer = response.choices[0].message.content.strip()
        return answer

    except Exception as e:
        print(f"An error occurred while generating the answer: {e}")
        return "Error generating answer."

def score_answer_against_chunk(chunk, answer, question):
    normalized_chunk = normalize_text(chunk)
    normalized_answer = normalize_text(answer)
    normalized_question = normalize_text(question)

    chunk_embedding = get_embeddings([normalized_chunk])[0]
    answer_embedding = get_embeddings([normalized_answer])[0]

    similarity_score = util.cos_sim(answer_embedding, chunk_embedding).item()
    normalized_similarity_score = (similarity_score + 1) / 2

    penalty = 1.0
    if "context does not provide" in normalized_answer or "not listed" in normalized_answer:
        penalty = 0.5

    question_terms = set(normalized_question.split())
    answer_terms = set(normalized_answer.split())
    key_term_overlap = len(question_terms.intersection(answer_terms)) / len(question_terms) if question_terms else 0

    adjusted_score = (normalized_similarity_score * penalty) * (0.7 + (0.3 * key_term_overlap))
    faithfulness_score = min(1.0, adjusted_score)

    recLL_score = similarity_score if normalized_answer in normalized_chunk else 0.5 * similarity_score
    coverage_score = len(answer_terms.intersection(question_terms)) / len(question_terms) if question_terms else 0
    diversity_score = len(set(answer.split())) / (len(answer.split()) + 1e-5)

    return {
        "faithfulness_score": faithfulness_score,
        "recLL_score": recLL_score,
        "coverage_score": coverage_score,
        "diversity_score": diversity_score
    }

def save_to_csv(question, best_answer, best_score, time_taken, scores, filename="scoring.csv"):
    """
    Saves the question, best answer, best score, time taken, and additional scores to a CSV file.
    """
    try:
        with open(filename, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([
                question, best_answer, best_score, time_taken,
                scores["faithfulness_score"], scores["recLL_score"], scores["coverage_score"], scores["diversity_score"]
            ])
        print(f"Results saved to {filename}")
    except Exception as e:
        print(f"Error writing to file: {e}")

def save_to_csv_more(question, best_answer, best_score, time_taken, answer_recall_score,combined_relevance_score, weighted_combined_score,filename="scoring2.csv"):
    """
    Saves the question, best answer, best score, time taken, and additional scores to a CSV file.
    """


    try:
        with open(filename, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([
                question, best_answer, best_score, time_taken,
                answer_recall_score, combined_relevance_score, weighted_combined_score
            ])
        print(f"Results saved to {filename}")
    except Exception as e:
        print(f"Error writing to file: {e}")



def process_chunk(chunk, question):
    """Process each chunk: generate answer and score it."""
    answer = generate_answer(chunk, question)
    scores = score_answer_against_chunk(chunk, answer, question)
    return chunk, answer, scores["faithfulness_score"], scores


def calculate_answer_recall(retrieved_chunks, answer):
    """
    Calculates the Answer Recall by checking the coverage of terms in the answer
    relative to terms in the retrieved chunks.
    """
    combined_retrieved_text = " ".join(retrieved_chunks)
    retrieved_terms = set(normalize_text(combined_retrieved_text).split())
    answer_terms = set(normalize_text(answer).split())

    common_terms = retrieved_terms.intersection(answer_terms)
    answer_recall_score = len(common_terms) / len(retrieved_terms) if retrieved_terms else 0.0

    return answer_recall_score


def calculate_combined_relevance_score(query, retrieved_chunks, answer):
    """
    Calculates the Combined Relevance Score by averaging the similarity
    between query and retrieved chunks and retrieved chunks and the answer.
    """
    query_embedding = get_embeddings([query])[0]
    retrieved_embeddings = get_embeddings(retrieved_chunks)
    answer_embedding = get_embeddings([answer])[0]

    # Similarity between query and retrieved chunks
    query_retrieved_similarity = np.mean(
        [util.cos_sim(query_embedding, chunk_emb).item() for chunk_emb in retrieved_embeddings])

    # Similarity between retrieved chunks and the answer
    retrieved_answer_similarity = np.mean(
        [util.cos_sim(chunk_emb, answer_embedding).item() for chunk_emb in retrieved_embeddings])

    combined_relevance_score = (query_retrieved_similarity + retrieved_answer_similarity) / 2
    return combined_relevance_score
# Main function to run the process
def calculate_weighted_combined_score(query, retrieved_chunks, answer, retrieval_weight=0.4, faithfulness_weight=0.4, recall_weight=0.2):
    """
    Calculates a Weighted Combined Score (WCS) that combines retrieval relevance,
    generation faithfulness, and answer recall with specified weights.
    """
    # Calculate Retrieval Relevance (similarity between query and retrieved chunks)
    query_embedding = get_embeddings([query])[0]
    retrieved_embeddings = get_embeddings(retrieved_chunks)
    retrieval_relevance = np.mean([util.cos_sim(query_embedding, chunk_emb).item() for chunk_emb in retrieved_embeddings])

    # Calculate Generation Faithfulness (similarity between answer and retrieved chunks)
    answer_embedding = get_embeddings([answer])[0]
    faithfulness = np.mean([util.cos_sim(answer_embedding, chunk_emb).item() for chunk_emb in retrieved_embeddings])

    # Calculate Answer Recall (coverage of terms in the answer relative to terms in retrieved chunks)
    answer_recall = calculate_answer_recall(retrieved_chunks, answer)

    # Calculate Weighted Combined Score
    wcs = (retrieval_weight * retrieval_relevance) + (faithfulness_weight * faithfulness) + (recall_weight * answer_recall)
    return wcs




def main():
    url = "https://en.wikipedia.org/wiki/Stephen_Curry"
    context = load_webpage_content(url)

    if not context:
        print("Failed to retrieve content from the web page.")
        return

    question = "What religion is steph curry?"
    chunks = split_text_by_sentences(context)

    # Cache the question embedding
    question_embedding = get_embeddings([question])[0]

    # Rank chunks and select the top 5 most relevant chunks
    top_chunks, retrieval_scores = rank_chunks(chunks, question_embedding)
    avg_retrieval_score = np.mean(retrieval_scores)

    # Start the timer before generating answer and scoring
    start_time = time.time()

    # Parallelize chunk processing
    with ThreadPoolExecutor() as executor:
        results = list(executor.map(lambda chunk: process_chunk(chunk, question), top_chunks))

    # Find the best score and answer
    best_score = 0
    best_answer = ""
    best_chunk = ""
    best_scores = None
    for chunk, answer, faithfulness_score, scores in results:
        if faithfulness_score > best_score:
            best_score = faithfulness_score
            best_answer = answer
            best_chunk = chunk
            best_scores = scores

    # Calculate the time taken
    answer_recall_score = calculate_answer_recall(top_chunks, best_answer)
    combined_relevance_score = calculate_combined_relevance_score(question, top_chunks, best_answer)
    weighted_combined_score = calculate_weighted_combined_score(question, top_chunks, best_answer)

    time_taken = time.time() - start_time

    # Save the best question, answer, score, time taken, and additional scores to scoring.csv
    save_to_csv(question, best_answer, best_score, time_taken, best_scores)
    save_to_csv_more(question, best_answer, best_score, time_taken,answer_recall_score,combined_relevance_score,weighted_combined_score )

    # Display the best answer, corresponding chunk, and time taken
    print(f"Best Chunk:\n{best_chunk}\n")
    print(f"Best Answer: {best_answer}\n")
    print(f"Best Score: {best_score:.2f}\n")
    print(f"Average Retrieval Score: {avg_retrieval_score:.2f}\n")
    print(f"Faithfulness Score: {best_scores['faithfulness_score']:.2f}")
    print(f"RecLL Score: {best_scores['recLL_score']:.2f}")
    print(f"Coverage Score: {best_scores['coverage_score']:.2f}")
    print(f"Diversity Score: {best_scores['diversity_score']:.2f}")
    print('Answer recall score:',answer_recall_score)
    print('Combined Relevance Score:' ,combined_relevance_score)
    print('Weighted Combined Score:', weighted_combined_score)
    perplexity = calculate_perplexity(best_answer)
    print (perplexity)

    print(f"Time Taken: {time_taken:.2f} seconds")

if __name__ == "__main__":
    # Add CSV header if file is created for the first time
    with open('scoring.csv', 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([
            "Question", "Best Answer", "Best Score", "Time Taken (seconds)",
            "Faithfulness Score", "RecLL Score", "Coverage Score", "Diversity Score"
        ])
    with open('scoring2.csv', 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([
            "Question", "Best Answer", "Best Score", "Time Taken (seconds)",
            "Answer recall score", "Combined relevance score", "Weighted combined score"
        ])

    main()

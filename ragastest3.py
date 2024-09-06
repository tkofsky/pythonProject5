import os
import openai
import sqlite3
import chromadb
from ragas import evaluate
from datasets import Dataset
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from ragas.metrics.critique import harmfulness
from langchain_community.vectorstores import Chroma
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.document_loaders import SeleniumURLLoader
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall, context_entity_recall, answer_similarity, answer_correctness
import os
import sys
print (os.path.dirname(sys.executable))
load_dotenv()
api_key = os.environ.get("OPENAI_API_KEY")
openai.api_key = api_key

urls = [
    "https://en.wikipedia.org/wiki/New_York_City",
    "https://en.wikipedia.org/wiki/Snow_leopard",
    "https://www.britannica.com/place/Galapagos-Islands",
    "https://www.birdlife.org/birds/penguins/#:~:text=The%20threats%20are%20numerous%2C%20including,is%20melting%20before%20their%20eyes."
]



documentList = []
for doc in documents:
    d = str(doc.page_content).replace("\\n", " ").replace("\\t"," ").replace("\n", " ").replace("\t", " ")
    documentList.append(d)



embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
text_splitter = SemanticChunker(embedding_function)
docs = text_splitter.create_documents(documentList)

# storing embeddings in a folder
vector_store = Chroma.from_documents(docs, embedding_function, persist_directory="./chroma_db")
# use this to load vector database
vector_store = Chroma(persist_directory="./chroma_db", embedding_function=embedding_function)

PROMPT_TEMPLATE = """
Go through the context and answer given question strictly based on context. 
Context: {context}
Question: {question}
Answer:
"""

qa_chain = RetrievalQA.from_chain_type(
        llm = ChatOpenAI(temperature=0),
        # retriever=vector_store.as_retriever(search_kwargs={'k': 3}),
        retriever=vector_store.as_retriever(),
        return_source_documents=True,
        chain_type_kwargs={"prompt": PromptTemplate.from_template(PROMPT_TEMPLATE)}
    )

queries = [
    "Who discovered the Galapagos Islands and how?",
    "What is Brooklyn–Battery Tunnel?",
    "Are Penguins found in the Galapagos Islands?",
    "How many languages are spoken in New York?",
    "In which countries are snow leopards found?",
    "What are the threats to penguin populations?",
    "What is the economic significance of New York City?",
    "How did New York City get its name?",
    "How did Galapagos Islands get its name?",
    "What is the significance of the Statue of Liberty in New York City?",

]

ground_truths = [
    "The Galapagos Islands were discovered in 1535 by the bishop of Panama, Tomás de Berlanga, whose ship had drifted off course while en route to Peru. He named them Las Encantadas (“The Enchanted”), and in his writings he marveled at the thousands of large galápagos (tortoises) found there. Numerous Spanish voyagers stopped at the islands from the 16th century, and the Galapagos also came to be used by pirates and by whale and seal hunters. ",
    "The Brooklyn-Battery Tunnel (officially known as the Hugh L. Carey Tunnel) is the longest continuous underwater vehicular tunnel in North America and runs underneath Battery Park, connecting the Financial District in Lower Manhattan to Red Hook in Brooklyn.[586]",
    "Penguins live on the galapagos islands side by side with tropical animals.",
    "As many as 800 languages are spoken in New York.",
    "Siberia, Tajikistan, Kyrgyzstan, Uzbekistan, Kazakhstan, Afghanistan, Pakistan, India, Nepal, Bhutan, Mongolia, and Tibet.",
    "The threats are numerous, including habitat loss, pollution, disease, and reduced food availability due to commercial fishing. Climate change is of particular concern for many species of penguin, as the sea ice that they depend on to find food or build nests is melting before their eyes.",
    "New York City's economic significance is vast, as it serves as the global financial capital, housing Wall Street and major financial institutions. Its diverse economy spans technology, media, healthcare, education, and more, making it resilient to economic fluctuations. NYC is a hub for international business, attracting global companies, and boasts a large, skilled labor force. Its real estate market, tourism, cultural industries, and educational institutions further fuel its economic prowess. The city's transportation network and global influence amplify its impact on the world stage, solidifying its status as a vital economic player and cultural epicenter.",
    "New York City got its name when it came under British control in 1664. King Charles II of England granted the lands to his brother, the Duke of York, who named the city New York in his own honor.",
    "Tomás de Berlanga, who discovered the islands, named them Las Encantadas (“The Enchanted”), and in his writings he marveled at the thousands of large galápagos (tortoises) found there. Numerous Spanish voyagers stopped at the islands from the 16th century, and the Galapagos also came to be used by pirates and by whale and seal hunters.",
    "The Statue of Liberty in New York City holds great significance as a symbol of the United States and its ideals of liberty and peace. It greeted millions of immigrants who arrived in the U.S. by ship in the late 19th and early 20th centuries, representing hope and freedom for those seeking a better life. It has since become an iconic landmark and a global symbol of cultural diversity and freedom.",

]

results = []
contexts = []
for query in queries:
    result = qa_chain({"query": query})

  #  result.append(result['result'])
    sources = result["source_documents"]
    contents = []
    for i in range(len(sources)):
        contents.append(sources[i].page_content)
    contexts.append(contents)


d = {
    "question": queries,
    "answer": result,
    "contexts": contexts,
    "ground_truth": ground_truths
}

dataset = Dataset.from_dict(d)
score = evaluate(dataset,metrics=[faithfulness, answer_relevancy, context_precision, context_recall, context_entity_recall, answer_similarity, answer_correctness, harmfulness])
score_df = score.to_pandas()
score_df.to_csv("EvaluationScores.csv", encoding="utf-8", index=False)
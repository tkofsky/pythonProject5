import os
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.embeddings import TensorflowHubEmbeddings
from langchain.embeddings import HuggingFaceEmbeddings, SentenceTransformerEmbeddings
os.environ["OPENAI_API_KEY"]="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"

###################################Sentence Transformers Embeddings - done one CPU
text = "This is a test query."
embeddings = HuggingFaceEmbeddings(model_name="all-mpnet-base-v2")
query_result = embeddings.embed_query(text)
print(query_result)
print(len(query_result))

#################################################Tensor Flow
text = "This is a test query."
embeddings_tf = TensorflowHubEmbeddings()
query_result= embeddings_tf.embed_query(text)
print(query_result)
print(len(query_result))

###################################################OPENAI
embeddings = OpenAIEmbeddings(
    model="text-embedding-ada-002",
    openai_api_key=""
)
text = "This is a test query.OpenAI uses 1,536 dimensions for embeddings in the text-embedding-ada-002 model due to multiple design choices driven by the trade-offs between model complexity, computational efficiency, and predictive performance. 1. Model complexity: Higher-dimensional embeddings can capture more complex and nuanced relationships in the data. This allows the model to better understand and represent various linguistic patterns, syntactic properties, and semantic relationships. 2. Computational efficiency: Though higher-dimensional embeddings provide richer"
query_result = embeddings.embed_query(text)
print(query_result)
print(len(query_result))

documents = ["This is a sample document.", "This is another sample document."]
document_embeddings = embeddings.embed_documents(texts=documents, chunk_size=1000)
print(document_embeddings)


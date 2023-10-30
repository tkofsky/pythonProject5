import os
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.embeddings import TensorflowHubEmbeddings
from langchain.embeddings import HuggingFaceEmbeddings, SentenceTransformerEmbeddings

import time
import os
###################################Sentence Transformers Embeddings - done one CPU
#text = "This is a test query."
text_file = open("97874480.txt", "r",encoding='utf-8')
text = text_file.read()


for k in range(10):
    start_time = time.time()

    #embeddings_tf = TensorflowHubEmbeddings()
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002", openai_api_key="")

   # embeddings_tf = TensorflowHubEmbeddings()
    #query_result = embeddings_tf.embed_query(text)
    #embeddings = OpenAIEmbeddings(model="text-embedding-ada-002", openai_api_key="" )

    query_result = embeddings.embed_query(text)

    print(query_result)
    print(len(query_result))

    end_time = time.time()
    xtime = end_time - start_time
    xtime = round(xtime, 3)
    filename = "97874480.txt"
    file = open('embed test.txt', 'a')
    file.write(str(filename) + "," + str(xtime) + ","  + "openai" + "\n")

exit()
#################################################Tensor Flow
#text = "This is a test query."
embeddings_tf = TensorflowHubEmbeddings()
query_result= embeddings_tf.embed_query(text)
print(query_result)
print(len(query_result))

###################################################OPENAI
embeddings = OpenAIEmbeddings(
    model="text-embedding-ada-002",
    openai_api_key=""
)
#text = "This is a test query.OpenAI uses 1,536 dimensions for embeddings in the text-embedding-ada-002 model due to multiple design choices driven by the trade-offs between model complexity, computational efficiency, and predictive performance. 1. Model complexity: Higher-dimensional embeddings can capture more complex and nuanced relationships in the data. This allows the model to better understand and represent various linguistic patterns, syntactic properties, and semantic relationships. 2. Computational efficiency: Though higher-dimensional embeddings provide richer"
query_result = embeddings.embed_query(text)
print(query_result)
print(len(query_result))

documents = ["This is a sample document.", "This is another sample document."]

document_embeddings = embeddings.embed_documents(texts=documents, chunk_size=1000)
print(document_embeddings)


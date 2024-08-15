import os
import openai
from openai import OpenAI
from llama_index.llms import openai
from llama_index.embeddings.openai import OpenAIEmbedding
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from llama_index.core.prompts import PromptTemplate
openai.api_key = os.getenv("OPENAI_API_KEY")


# Import packages
from llama_index.core import (VectorStoreIndex,ServiceContext, download_loader)

# Download the documents from Wikipedia and load them
WikipediaReader = download_loader("WikipediaReader")
loader = WikipediaReader()
pages = ['Emma_Stone', 'La_La_Land', 'Ryan_Gosling']
documents = loader.load_data(pages=pages, auto_suggest=False, redirect = False)

# Initialize the gpt3.5 model
gpt3 = openai.OpenAI(temperature=0, model="gpt-3.5-turbo-instruct", api_key=openai.api_key)
# Initialize the embedding model
embed_model = OpenAIEmbedding(model= "text-embedding-3-large", api_key=openai.api_key)
# Transform chunks into numerical vectors using the embedding model
service_context_gpt3 = ServiceContext.from_defaults(llm=gpt3, chunk_size = 256, chunk_overlap=0, embed_model=embed_model)
index = VectorStoreIndex.from_documents(documents, service_context=service_context_gpt3)
retriever = index.as_retriever(similarity_top_k=3)

# Build a prompt template to only provide answers based on the loaded documents
template = (
"We have provided context information below. \n"
    "---------------------\n"
    "{context_str}"
    "\n---------------------\n"
    "Given this information, please answer the question: {query_str}\n"
    "Don't give an answer unless it is supported by the context above.\n"
)

qa_template = PromptTemplate(template)

# Create a prompt for the model
question = "What is the plot of the film that led Emma Stone to win her first Academy Award?"
question = "Compare the families of Emma Stone and Ryan Gosling"


# Retrieve the context from the model
contexts = retriever.retrieve(question)
context_list = [n.get_content() for n in contexts]
prompt = qa_template.format(context_str="\n\n".join(context_list), query_str=question)

# Generate the response
response = gpt3.complete(prompt)
print(str(response))



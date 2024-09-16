import os
import llama_index.readers

from llama_index.readers.web import ScrapflyReader
from llama_index.core import VectorStoreIndex

scrapfly_reader = ScrapflyReader(
    api_key="Your Scrapfly API key",
    ignore_scrape_failures=True,
)

# Load documents from URLs as markdown
documents = scrapfly_reader.load_data(
    urls=["https://web-scraping.dev/products"]
)

# Set the OpenAI key as a environment variable
OPENAI_API_KEY= os.getenv("OPENAI_API_KEY")

# Create an index store for the documents
index = VectorStoreIndex.from_documents(documents)

# Create the RAG engine with using the index store
query_engine = index.as_query_engine()

# Submit a query
response = query_engine.query("What is the flavor of the dark energy potion?")
print(response)
"The flavor of the dark energy potion is bold cherry cola."
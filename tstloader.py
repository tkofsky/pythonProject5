import os
from langchain_openai import ChatOpenAI #model
from langchain.globals import set_llm_cache #response caching
from langchain.cache import InMemoryCache
from langchain.prompts import HumanMessagePromptTemplate, ChatPromptTemplate #prompt templating
from langchain.document_loaders import WikipediaLoader #document loader
from langchain_community.document_loaders import WebBaseLoader
import nest_asyncio
import os
nest_asyncio.apply()

api_key = os.getenv("OPENAI_API_KEY")


loader = WebBaseLoader(["https://www.espn.com/", "https://www.apple.com"])
loader.requests_per_second = 1
docs = loader.aload()


print (docs)



import cassio
#https://python.langchain.com/v0.2/docs/how_to/hybrid/#code-example
#https://astra.datastax.com/   - reset db here
cassio.init(
    database_id="4f0c112e-bd2d-444b-b4a9-e12f6558b3fa",
    token="AstraCS:OhOKDeAKUXztuJvvwpmnZdjO:ba5b592fe8a3ae188712c6782ccdb644d94feb12d4a9cc5887b1bd4aa736f179",

)

from cassio.table.cql import STANDARD_ANALYZER
from langchain_community.vectorstores import Cassandra
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()
vectorstore = Cassandra(
    embedding=embeddings,
    table_name="test_hybrid",
    body_index_options=[STANDARD_ANALYZER],
    session=None,
    keyspace=None,
)

vectorstore.add_texts(
    [
        "In 2023, I visited Paris",
        "In 2022, I visited New York",
        "In 2021, I visited New Orleans",
    ]
)
vectorstore.as_retriever().invoke("What city did I visit last?")


#vectorstore.as_retriever(search_kwargs={"body_search": "new"}).invoke(
#    "What city did I visit last?"
#)


from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import (
    ConfigurableField,
    RunnablePassthrough,
)
from langchain_openai import ChatOpenAI


template = """Answer the question based only on the following context:
{context}
Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

model = ChatOpenAI()

retriever = vectorstore.as_retriever()

configurable_retriever = retriever.configurable_fields(
    search_kwargs=ConfigurableField(
        id="search_kwargs",
        name="Search Kwargs",
        description="The search kwargs to use",
    )
)

chain = (
    {"context": configurable_retriever, "question": RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)

aa=chain.invoke("What city did I visit last?")
print (aa)

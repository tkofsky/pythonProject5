from langchain.agents import create_pandas_dataframe_agent
from langchain.llms import OpenAI
import pandas as pd
import os
from langchain.document_loaders import CSVLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
import pandas as pd
from langchain.document_loaders import CSVLoader
import os

filename = os.path.join("csv files", "all players.csv") #
loader = CSVLoader(file_path=filename,encoding='utf-8')

index_creator = VectorstoreIndexCreator()
docsearch = index_creator.from_loaders([loader])

chain = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="stuff", retriever=docsearch.vectorstore.as_retriever(), input_key="question")
query = "which organization has the most numeber employees"
response = chain({"question": query})
print(response['result'])
################
df = pd.read_csv(filename)

pd_agent = create_pandas_dataframe_agent(OpenAI(temperature=0), df, verbose=True)

pd_agent.run("which player is mentioned the most")
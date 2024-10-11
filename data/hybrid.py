from langchain.document_loaders import UnstructuredPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma

from langchain.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain.llms import HuggingFaceHub
import torch
from transformers import ( AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline, )
from langchain import HuggingFacePipeline

from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain.document_loaders.pdf import PyPDFDirectoryLoader
import os
from langchain.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
pdf_folder_path = "./"
loader = PyPDFDirectoryLoader(pdf_folder_path)
docs = loader.load()


loader = PyPDFLoader(r"C:\Users\kofsk\PycharmProjects\pythonProject5\pdfsample.pdf")
docs = loader.load_and_split()


# create chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=512,
                                          chunk_overlap=30)
chunks = splitter.split_documents(docs)

embeddings = HuggingFaceInferenceAPIEmbeddings(
    api_key='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', model_name="BAAI/bge-base-en-v1.5"
)
vectorstore = Chroma.from_documents(chunks, embeddings)

vectorstore_retreiver = vectorstore.as_retriever(search_kwargs={"k": 3})
keyword_retriever = BM25Retriever.from_documents(chunks)
keyword_retriever.k =  3

ensemble_retriever = EnsembleRetriever(retrievers=[vectorstore_retreiver,
                                                   keyword_retriever],
                                       weights=[0.3, 0.7])



# function for loading 4-bit quantized model
def load_quantized_model(model_name: str):
    """
    model_name: Name or path of the model to be loaded.
    return: Loaded quantized model.
    """
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        load_in_4bit=True,
        torch_dtype=torch.bfloat16,
        quantization_config=bnb_config,
    )
    return model

# initializing tokenizer
def initialize_tokenizer(model_name: str):
    """
    model_name: Name or path of the model for tokenizer initialization.
    return: Initialized tokenizer.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name, return_token_type_ids=False)
    tokenizer.bos_token_id = 1  # Set beginning of sentence token id
    return tokenizer

tokenizer = initialize_tokenizer(model_name)

model = load_quantized_model(model_name)
# specify stop token ids
stop_token_ids = [0]

# build huggingface pipeline for using zephyr-7b-beta
pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    use_cache=True,
    device_map="auto",
    max_length=2048,
    do_sample=True,
    top_k=5,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.pad_token_id,
)

llm = HuggingFacePipeline(pipeline=pipeline)


from langchain.chains import RetrievalQA

normal_chain = RetrievalQA.from_chain_type(
    llm=llm, chain_type="stuff", retriever=vectorstore_retreiver
)

hybrid_chain = RetrievalQA.from_chain_type(
    llm=llm, chain_type="stuff", retriever=ensemble_retriever
)
response = hybrid_chain.invoke("QUERY")

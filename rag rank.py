import pypdf
import tiktoken
import wikipedia
import langchain
import torch
import transformers
import sentence_transformers
from sentence_transformers import SentenceTransformer
import faiss
from transformers import AutoTokenizer
from langchain.text_splitter import RecursiveCharacterTextSplitter
import numpy as np
from langchain_community.document_loaders import WikipediaLoader
from sentence_transformers import CrossEncoder
import transformers
import sentence_transformers



#############################




###############################


pages = WikipediaLoader(query="Harry Potter", load_max_docs=3, lang="en").load()

from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
    tokenizer=AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L12-v2"),
    chunk_size=256,
    chunk_overlap=16,
    strip_whitespace=True,
)
docs = text_splitter.split_documents(pages)

from langchain_community.embeddings import HuggingFaceEmbeddings
bi_encoder = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L12-v2", model_kwargs={"device": "cpu"}
)
import numpy as np
embedding1 = bi_encoder.embed_query("Harry Potter")
embedding2 = bi_encoder.embed_query(docs[0].page_content)


print (len(embedding1))

from langchain.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy

faiss_db = FAISS.from_documents(
    docs, bi_encoder, distance_strategy=DistanceStrategy.DOT_PRODUCT
)
question = (
    "On what date was the first book in the Harry Potter series originally published?"
)
retrieved_docs = faiss_db.similarity_search(question, k=25)


#######################
from sentence_transformers import CrossEncoder
model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


cross_encoder = CrossEncoder(
    "cross-encoder/ms-marco-TinyBERT-L-2-v2", max_length=512, device="cpu"
)
#?
cross_encoder.rank(
    query="Do I like cats?",
    documents=["I like cats", "I like horses", "I like dogs"],
    return_documents=True,
)



reranked_docs = cross_encoder.rank(
    question,
    [doc.page_content for doc in retrieved_docs],
    top_k=3,
    return_documents=True,
)

context = "".join(doc["text"] + "\n" for doc in reranked_docs)
print(context)
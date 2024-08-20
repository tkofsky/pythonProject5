
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import OpenAIEmbeddings
import openai
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer
from langchain_chroma import Chroma
from sentence_transformers import CrossEncoder


loader = PyPDFLoader("./Document/Harry Potter and the Sorcerers Stone.pdf")



text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
   tokenizer=AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L12-v2"),
   chunk_size=256,
   chunk_overlap=16,
   strip_whitespace=True,
)
docs = text_splitter.split_documents(pages)


embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
vectordb = Chroma.from_documents(documents=docs, embedding=embeddings)


query = "What are the names of four houses in Hogwarts?"
docsnew = vectordb.similarity_search(query)
print(docsnew[0].page_content)


cross_encoder = CrossEncoder(
   "cross-encoder/ms-marco-TinyBERT-L-2-v2", max_length=512, device="cpu"
)



from sentence_transformers import CrossEncoder
cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
document_texts = [doc.page_content for doc in docsnew]
response = [[query, doc_text] for doc_text in document_texts]
scores = cross_encoder.predict(response)

print("Scores:")
for score in scores:
   print(score)
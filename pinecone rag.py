from datasets import load_dataset

data = load_dataset("jamescalam/ai-arxiv-chunked", split="train")
data

data = data.map(lambda x: {
    "id": f'{x["id"]}-{x["chunk-id"]}',
    "text": x["chunk"],
    "metadata": {
        "title": x["title"],
        "url": x["source"],
        "primary_category": x["primary_category"],
        "published": x["published"],
        "updated": x["updated"],
        "text": x["chunk"],
    }
})
# drop uneeded columns
data = data.remove_columns([
    "title", "summary", "source",
    "authors", "categories", "comment",
    "journal_ref", "primary_category",
    "published", "updated", "references",
    "doi", "chunk-id",
    "chunk"
])
data


import openai
import os
# platform.openai.com
# get API key from top-right dropdown on OpenAI website
OPENAI_API_KEY= os.environ.get("OPENAI_API_KEY")
embed_model = "text-embedding-ada-002"

import pinecone

# initialize connection to pinecone (get API key at app.pinecone.io)
api_key = "22295177-fb63-4402-8e86-9e1bd59c6d48"
# find your environment next to the api key in pinecone console
env = "gcp-starter"

pinecone.init(api_key=api_key, environment=env)

import time

index_name = "rerankers"

# check if index already exists (it shouldn't if this is first time)
if index_name not in pinecone.list_indexes():
    # if does not exist, create index
    pinecone.create_index(
        index_name,
        dimension=1536,  # dimensionality of ada 002
        metric='dotproduct'
    )
    # wait for index to be initialized
    while not pinecone.describe_index(index_name).status['ready']:
        time.sleep(1)

# connect to index
index = pinecone.Index(index_name)

from tqdm.auto import tqdm

batch_size = 100  # how many embeddings we create and insert at once

for i in tqdm(range(0, len(data), batch_size)):
    passed = False
    # find end of batch
    i_end = min(len(data), i+batch_size)
    # create batch
    batch = data[i:i_end]
    # create embeddings (exponential backoff to avoid RateLimitError)
    for j in range(5):  # max 5 retries

        res = openai.Embedding.create(input=batch["text"], engine=embed_model)
        passed = True


    if not passed:
        raise RuntimeError("Failed to create embeddings.")
    # get embeddings
    embeds = [record['embedding'] for record in res['data']]
    to_upsert = list(zip(batch["id"], embeds, batch["metadata"]))
    # upsert to Pinecone
    index.upsert(vectors=to_upsert)

def get_docs(query: str, top_k: int):
    # encode query
    xq = embeds([query])[0]
    # search pinecone index
    res = index.query(xq, top_k=top_k, include_metadata=True)
    # get doc text
    docs = {x["metadata"]['text']: i for i, x in enumerate(res["matches"])}
    return docs

query = "can you explain why we would want to do rlhf?"
docs = get_docs(query, top_k=25)
print("\n---\n".join(docs.keys()[:3]))  # print the first 3 docs


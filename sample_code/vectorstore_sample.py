from langchain_core.documents import Document
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from llm_core.llm import create_llm, create_embeddings
from llm_core.llm_config import *
import asyncio


## documents

docs_1 = [
    Document(
        page_content="Dogs are great companions, known for their loyalty and friendliness.",
        metadata={"source": "mammal-pets-doc"},
    ),
    Document(
        page_content="Cats are independent pets that often enjoy their own space.",
        metadata={"source": "mammal-pets-doc"},
    ),
]

#### PDF Loader
loader = PyMuPDFLoader("./qa-doc/3Q24-Earnings-Transcript.pdf")
docs_2 = loader.load()

print(f"""
      len docs_2 = {len(docs_2)}
      docs_2[0] = {docs_2[0].page_content[:10]}
      docs_2[0].metadata = {docs_2[0].metadata}
      """)

## Text Splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    add_start_index=True,
)
chunks = text_splitter.split_documents(docs_2)
print(f"len chunks = {len(chunks)}")


## Embedding   
embeddings = create_embeddings(OPENAI_EMBEDDING)
vector_1 = embeddings.embed_query(chunks[0].page_content)
vector_2 = embeddings.embed_query(chunks[2].page_content)
assert len(vector_1) == len(vector_2)
print(f"Generated vectors of length {len(vector_1)}\n")
print(vector_1[:10])


## vector store
vector_store = FAISS.from_documents(chunks, embeddings)
_ = vector_store.add_documents(chunks)

# sync
results = vector_store.similarity_search(
    query="What's this regarding for?"
)
print(results[0])

# async
async def async_search():
    results = await vector_store.asimilarity_search("When was Nike incorporated?")
    print(results[0])
asyncio.run(async_search())

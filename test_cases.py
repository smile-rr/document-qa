
from vectorstore import VectorStore
import os
from dotenv import load_dotenv, find_dotenv
from langchain_openai import OpenAIEmbeddings
from llm import create_chat_llm


def test_vectorstore():
    load_dotenv(find_dotenv(), override=True)
    
    # llm = create_llm("gpt-3.5-turbo")
    embeddings = OpenAIEmbeddings(base_url=os.environ['EMBEDDINGS_BASE_URL'])
    faiss_db = VectorStore(dir="./test_qa_db", embeddings=embeddings)  # Load the FAISS vector store
    faiss_db.add_file("/Users/pc-rn/Documents/算法/AI/Aceline_IN.pdf")
    
    query = "What's this about"

    docs = faiss_db.similarity_search(query)
    print(f"Number of documents: {len(docs)}")
    print(f"docs = {docs[0].page_content[:200]}")
    
if __name__ == "__main__":
    test_vectorstore()
    
   
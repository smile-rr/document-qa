import os
from dotenv import find_dotenv, load_dotenv
from rich import print as rprint
import logging
import time

from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

from core.vectorstore import VectorStore, indexing_dir
from core.llm import create_llm, create_embeddings, normalize_response
from core.document_reader import to_documents, split_documents
from core.llm_config import (
    GPT_3_5, GEMINI_FLASH, GEMINI_PRO, OPENCHAT, LLAMA_3, LLAMA_3_VISION, DEEPSEEK_R1, QWEN_2_5,
    TEXTEMBEDDING_GECKO, TEXTEMBEDDING_GECKO_MULTILINGUAL, OPENAI_EMBEDDING, OLLAMA_EMBEDDING
)

def create_qa_chain(llm, vectorstore):
    retriever = vectorstore.as_retriever()
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    return qa_chain

def create_conversation_qa_chain(llm, vectorstore):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 10, "include_metadata": True})

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory
    )
    return qa_chain

def run_qa(qa_chain, invoke_params):
    rprint("[blue]Welcome to the QA system! Type 'exit' or 'bye' to quit.\n")
    while True:
        query = input("Q: ")
        if query.lower() in ['exit', 'bye']:
            rprint("[blue]\nExiting the QA system.")
            break
        result = qa_chain.invoke(invoke_params(query))
        rprint(f"[green]\nA: {result['result' if 'result' in result else 'answer']}\n")

def qa(qa_chain):
    run_qa(qa_chain, lambda query: query)

def conversation_qa(qa_chain):
    run_qa(qa_chain, lambda query: {"question": query})



def document_qa(llm, embeddings, db_path="./faiss_db"):
    faiss_db = VectorStore(dir=db_path, embeddings=embeddings)  
    qa_chain = create_qa_chain(llm, faiss_db)
    conversation_qa(qa_chain)
    
def main():
     # Clear the terminal
    os.system('clear' if os.name == 'posix' else 'cls')
    
    logging.basicConfig(level=logging.DEBUG)
    load_dotenv(find_dotenv(), override=True)
    
    llm = create_llm(LLAMA_3)
    embeddings = create_embeddings(OLLAMA_EMBEDDING)
    
    # indexing_dir(embeddings, "qa-doc", db_path="./faiss_db")
    
    
    # document_qa(llm, embeddings)
    
if __name__ == "__main__":
    main()
    
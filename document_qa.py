from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import os
from dotenv import find_dotenv, load_dotenv
from vectorstore import VectorStore
from llm import LLM
from document_reader import DocumentReader

from rich import print as rprint

import logging

def create_qa_chain(llm, vectorstore):
    retriever = vectorstore.as_retriever()
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    return qa_chain

def create_conversation_qa_chain(llm, vectorstore):
    retriever = vectorstore.as_retriever()
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

def initiate_with_files(embeddings, files, db_path="./faiss_db"):
    if not files or len(files) == 0:
        raise ValueError("No files provided")
   
    documents = DocumentReader.to_documents(files[0])
    chunks = DocumentReader.split_documents(documents)
    faiss_db = VectorStore(dir=db_path, embeddings=embeddings, documents=chunks)  
    
    for file in files[1:]:
        documents = DocumentReader.to_documents(file)
        chunks = DocumentReader.split_documents(documents)
        faiss_db.add_documents(chunks)

def list_files_in_dir(directory):
    supported_extensions = ('.pdf', '.csv', '.xlsx', '.xls')
    files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(supported_extensions)]
    if not files:
        raise ValueError(f"No supported files found in directory: {directory}")
    return files

def document_qa(embeddings, db_path="./faiss_db"):
    llm = LLM.create("gpt-3.5-turbo")
    faiss_db = VectorStore(dir=db_path, embeddings=embeddings)  
    qa_chain = create_conversation_qa_chain(llm, faiss_db)
    conversation_qa(qa_chain)

if __name__ == "__main__":
    load_dotenv(find_dotenv(), override=True)
    embeddings = OpenAIEmbeddings(base_url=os.environ['EMBEDDINGS_BASE_URL'])
    
    # directory = "/Users/pc-rn/Documents/算法/AI"
    # files = list_files_in_dir(directory)
    # initiate_with_files(embeddings, files)
    
    logging.basicConfig(level=logging.DEBUG)
    
    document_qa(embeddings)
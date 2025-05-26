import os
from dotenv import find_dotenv, load_dotenv
from rich import print as rprint
import logging
import time

from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

from llm_core.vectorstore import VectorStore, indexing_dir
from llm_core.llm import create_llm, create_embeddings
from llm_core.llm_config import  *

def run_qa(qa_chain, invoke_params):
    rprint("[blue]Welcome to the QA system! Type 'exit' or 'bye' to quit.\n")
    while True:
        query = input("Q: ")
        if query.lower() in ['exit', 'bye']:
            rprint("[blue]\nExiting the QA system.")
            break
        result = qa_chain.invoke(invoke_params(query))
        rprint(f"[green]\nA: {result['result' if 'result' in result else 'answer']}\n")


def conversation_qa(qa_chain):
    run_qa(qa_chain, lambda query: {"question": query})

def document_qa(llm, embeddings, db_path="./faiss_db"):
    vectorstore = VectorStore(dir=db_path, embeddings=embeddings)  
    retriever = vectorstore.as_retriever()

    system_prompt = (
        "Answer the question using the following context: If you don't know the answer, respond with 'I don't know'."
        "Keep your answers concise, using no more than three sentences. Alternatively, engage with the user in a friendly manner."
        "Context: {context}"
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}")
    ])

    qa_chain = create_stuff_documents_chain(llm, prompt)

    chain = create_retrieval_chain(retriever, qa_chain)
    
    run_qa(chain, lambda query: {"input": query})
    
def main():
     # Clear the terminal
    # os.system('clear' if os.name == 'posix' else 'cls')
    
    logging.basicConfig(level=logging.WARNING)  
    load_dotenv(find_dotenv(), override=True)
    
    llm = create_llm(GPT_3_5)
    embeddings = create_embeddings(OPENAI_EMBEDDING)
    
    indexing_dir(embeddings, "qa-doc")
    document_qa(llm, embeddings)
    
if __name__ == "__main__":
    main()

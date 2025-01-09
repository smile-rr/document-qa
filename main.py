

from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI
import os
from dotenv import find_dotenv, load_dotenv
from document_reader import read_pdf, read_excel
from vectorstore import VectorStore



def create_qa_chain(vectorstore):
    retriever = vectorstore.as_retriever()
    llm = ChatOpenAI(model_name="gpt-3.5-turbo",
        openai_api_base=os.environ['CHATGPT_API_ENDPOINT'],
        openai_api_key=os.environ['OPENAI_API_KEY'],
        temperature=0.1,
        verbose=True
        
        )
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    return qa_chain
def create_conversation_qa_chain(vectorstore):
    retriever = vectorstore.as_retriever()
    llm = ChatOpenAI(
        model_name="gpt-3.5-turbo",
        openai_api_base=os.environ['CHATGPT_API_ENDPOINT'],
        openai_api_key=os.environ['OPENAI_API_KEY'],
        temperature=0.8,
        verbose=True
    )
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory
    )
    return qa_chain

def qa(qa_chain):
    print("Welcome to the QA system! Type 'exit' or 'bye' to quit.\n")
    while True:
        query = input("Q: ")
        if query.lower() in ['exit', 'bye']:
            print("\nExiting the QA system.")
            break
        answer = qa_chain.invoke(query)
        print(f"\nA: {answer['result']}\n")
def conversation_qa(qa_chain):
    print("Welcome to the QA system! Type 'exit' or 'bye' to quit.\n")
    while True:
        query = input("Q: ")
        if query.lower() in ['exit', 'bye']:
            print("\nExiting the QA system.")
            break
        result = qa_chain.invoke({"question": query})
        print(f"\nA: {result['answer']}\n")
def main():
    load_dotenv(find_dotenv(), override=True)
    faiss_db = VectorStore(dir="./faiss_db")  # Load the FAISS vector store
    qa_chain = create_conversation_qa_chain(faiss_db)

    conversation_qa(qa_chain)

if __name__ == "__main__":
    main()
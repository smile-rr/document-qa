
from langchain.document_loaders import PyPDFLoader, CSVLoader ,UnstructuredExcelLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
import os
from dotenv import find_dotenv, load_dotenv

def read_pdf(file_path):
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    return documents

def read_csv(file_path):
    loader = CSVLoader(file_path)
    documents = loader.load()
    return documents

def read_excel(file_path, mode="elements"):
    if mode not in ["single", "elements"]:
        raise ValueError("Invalid mode. Choose 'single' or 'elements'.")

    loader = UnstructuredExcelLoader(file_path, mode=mode)
    documents = loader.load()
    return documents


def split_documents(documents, chunk_size=500, chunk_overlap=50):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    split_docs = text_splitter.split_documents(documents)
    return split_docs


def create_faiss_db(documents):
    embeddings = OpenAIEmbeddings(base_url=os.environ['EMBEDDINGS_BASE_URL'])
    vectorstore = FAISS.from_documents(documents, embeddings)
    return vectorstore

def create_qa_chain(vectorstore):
    retriever = vectorstore.as_retriever()
    llm = ChatOpenAI(model_name="gpt-3.5-turbo",
        openai_api_base=os.environ['CHATGPT_API_ENDPOINT'],
        openai_api_key=os.environ['OPENAI_API_KEY'],
        temperature=0.1
        )
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    return qa_chain

def main():
    load_dotenv(find_dotenv(), override=True)
    PATH_EXCEL = "qa-doc/3Q24-SUPP-ForWeb.xlsx"
    PATH_PDF = "qa-doc/3Q24-SUPP-ForWeb.pdf"

    pdf_documents = read_pdf(PATH_EXCEL)
    excel_documents = read_excel(PATH_PDF)

    split_pdf_docs = split_documents(pdf_documents)
    split_excel_docs = split_documents(excel_documents)

    all_split_docs = split_pdf_docs + split_excel_docs
    faiss_db = create_faiss_db(all_split_docs)

    qa_chain = create_qa_chain(faiss_db)

    query = "What is the main topic of the document?"
    answer = qa_chain.run(query)
    print("Answer:", answer)

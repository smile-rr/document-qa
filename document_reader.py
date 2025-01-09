
from langchain_community.document_loaders import PyPDFLoader, CSVLoader ,UnstructuredExcelLoader
from dotenv import load_dotenv, find_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
import tiktoken

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

def split_documents(documents, chunk_size=400, chunk_overlap=20):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=token_count,
        separators=["\n\n", "\n", " ", ""]
    )
    split_docs = text_splitter.split_documents(documents)
    return split_docs
def token_count(text, encoding_model = "gpt-3.5-turbo"):
    tokenizer = tiktoken.encoding_for_model(encoding_model)
    tokens = tokenizer.encode(
        text,
        disallowed_special=()
    )
    return len(tokens) 


if __name__ == "__main__":
    load_dotenv(find_dotenv(), override=True)
    PATH_EXCEL = "qa-doc/3Q24-SUPP-ForWeb.xlsx"
    PATH_PDF = "qa-doc/3Q24-SUPP-ForWeb.pdf"
    
    documents = read_pdf(PATH_PDF)
    # documents = read_excel(PATH_EXCEL)
    print(documents[0].page_content[:500])
    print(documents[0].metadata)
    print()
    print(f"Number of documents: {len(documents)}, \nlength of first document: {len(documents[0].page_content)}")
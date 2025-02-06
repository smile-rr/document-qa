from langchain_community.document_loaders import PyPDFLoader, CSVLoader, UnstructuredExcelLoader
from dotenv import load_dotenv, find_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
import tiktoken
import os

class DocumentReader:
    
    @staticmethod
    def to_documents(file_path):
        if file_path.endswith('.pdf'):
            return DocumentReader.read_pdf(file_path)
        elif file_path.endswith('.csv'):
            return DocumentReader.read_csv(file_path)
        elif file_path.endswith('.xlsx') or file_path.endswith('.xls'):
            return DocumentReader.read_excel(file_path)
        else:
            raise ValueError(f"Unsupported file type for file: {file_path}")

    @staticmethod
    def read_pdf(file_path):
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        return documents

    @staticmethod
    def read_csv(file_path):
        loader = CSVLoader(file_path)
        documents = loader.load()
        return documents

    @staticmethod
    def read_excel(file_path, mode="elements"):
        if mode not in ["single", "elements"]:
            raise ValueError("Invalid mode. Choose 'single' or 'elements'.")

        loader = UnstructuredExcelLoader(file_path, mode=mode)
        documents = loader.load()
        return documents

    @staticmethod
    def split_documents(documents, chunk_size=400, chunk_overlap=20):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=DocumentReader.token_count,
            separators=["\n\n", "\n", " ", ""]
        )
        split_docs = text_splitter.split_documents(documents)
        return split_docs

    @staticmethod
    def token_count(text, encoding_model="gpt-3.5-turbo"):
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
    
    documents = DocumentReader.as_documents(PATH_PDF)
    # documents = DocumentReader.as_documents(PATH_EXCEL)
    print(documents[0].page_content[:500])
    print(documents[0].metadata)
    print()
    print(f"Number of documents: {len(documents)}, \nlength of first document: {len(documents[0].page_content)}")
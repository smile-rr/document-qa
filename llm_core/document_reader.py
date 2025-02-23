from langchain_community.document_loaders import UnstructuredExcelLoader, CSVLoader, PyMuPDFLoader
from langchain_core.documents import Document
import pandas as pd
from dotenv import load_dotenv, find_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
import tiktoken
import os


def to_documents(file_path):
    if file_path.endswith('.pdf'):
        return read_pdf(file_path)
    elif file_path.endswith('.csv'):
        return read_csv(file_path)
    elif file_path.endswith('.xlsx') or file_path.endswith('.xls'):
        return read_excel_pd(file_path)
    else:
        raise ValueError(f"Unsupported file type for file: {file_path}")

def read_pdf(file_path):
    loader = PyMuPDFLoader(file_path)
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



def read_excel_pd(file_path):
    """
    Reads an Excel file with multiple sheets, cleans the data by removing rows and columns with NaN values,
    and strips leading and trailing spaces from column names and cell values. Converts each sheet into LangChain Document objects.

    Args:
        file_path (str): The path to the Excel file.

    Returns:
        list: A list of LangChain Document objects, each representing a sheet in the Excel file.
    """
    # Define a converter function to read all fields as strings
    def convert_to_str(value):
        return str(value)

    # Read all sheets into a dictionary of DataFrames, treating all values as strings
    sheets_dict = pd.read_excel(file_path, sheet_name=None, dtype=str, na_filter=False)

    documents = []
    for sheet_name, df in sheets_dict.items():
         # Remove rows and columns with NaN values
        df_cleaned = df.dropna(how='all', axis=0).dropna(how='all', axis=1)
        
        # Strip leading and trailing spaces from column names
        df_cleaned.columns = df_cleaned.columns.str.strip()
        
        # Strip leading and trailing spaces from all string cells
        df_cleaned = df_cleaned.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
        
        # Replace NaN values with an empty string
        df_cleaned = df_cleaned.fillna('')
        
        # Convert the cleaned DataFrame to a string with SOH separator
        text = df_cleaned.to_string(index=False, header=False)
        
        # Create a Document for each sheet
        document = Document(page_content=text, metadata={"sheet_name": sheet_name})
        documents.append(document)

    return documents

def split_documents(documents, chunk_size=400, chunk_overlap=20):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=token_count,
        separators=["\n\n", "\n", " ", ""]
    )
    split_docs = text_splitter.split_documents(documents)
    return split_docs

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
    
    # documents = to_documents(PATH_PDF)
    documents = to_documents(PATH_EXCEL)
    print(documents[5].page_content[:10000])
    print(documents[5].metadata)
    print()
    print(f"Number of documents: {len(documents)}, \nlength of first document: {len(documents[0].page_content)}")
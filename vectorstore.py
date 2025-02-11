from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
import pandas as pd
import os
from dotenv import load_dotenv, find_dotenv
from typing import Any
from document_reader import read_pdf, read_excel, split_documents, to_documents
from util import print_structure

def indexing_dir(embeddings, directory, db_path="./faiss_db"):
    files = _list_files_in_dir(directory)
    db = indexing_files(embeddings, files)
    return db

def indexing_files(embeddings, files, db_path="./faiss_db"):
    if not files or len(files) == 0:
        raise ValueError("No files provided")
    
    print(f"Initiating with file: {files[0]}")
    documents = to_documents(files[0])
    chunks = split_documents(documents)
    db = VectorStore(dir=db_path, embeddings=embeddings, documents=chunks)  
    
    for file in files[1:]:
        print(f"Adding to faiss db for file: {file}")
        documents = to_documents(file)
        chunks = split_documents(documents)
        db.add_documents(chunks)
    print("Done adding files to faiss db")
    return db

def _list_files_in_dir(directory):
    # supported_extensions = ('.pdf', '.csv', '.xlsx', '.xls')
    supported_extensions = ('.pdf')
    print(f"Looking for supported files in directory: {directory}, \nsupported extensions: {supported_extensions}")
    files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(supported_extensions)]
    if not files:
        raise ValueError(f"No supported files found in directory: {directory}")
    return files

class VectorStore:
    def __init__(self, embeddings=None, dir=None, documents=None):
        if embeddings is None:
            raise ValueError("Embeddings must be provided")
        
        self.embeddings = embeddings
        self.persist_directory = dir
        self._store = None
        
        if documents is not None:
            self._initialize_with_documents(documents)
        elif dir is not None and os.path.exists(dir):
            self._load_existing_store()
        else:
            self._initialize_empty_store()

    
    # Public Methods
    def to_dataframe(self):
        v_dict = self._store.docstore._dict
        data_rows = []
        for k in v_dict.keys():
            doc_name = v_dict[k].metadata["source"].split("/")[-1]
            page_number = v_dict[k].metadata["page"] + 1
            content = v_dict[k].page_content
            data_rows.append({"chunk_id": k, "doc_name": doc_name, "page_number": page_number, "content": content})

        vector_df = pd.DataFrame(data_rows)
        return vector_df

    def delete_document(self, document_name):
        vector_df = self.to_dataframe()
        chunks_list = vector_df.loc[vector_df['doc_name'] == document_name]["chunk_id"].tolist()
        self._store.delete(ids=chunks_list)
        if self.persist_directory is not None:
            self._store.save_local(self.persist_directory)

    def add_documents(self, documents):
        try:
            new_vector_store = FAISS.from_documents(documents, self.embeddings)
            self._store.merge_from(new_vector_store)
            if self.persist_directory is not None:
                self._store.save_local(self.persist_directory)
        except Exception as e:
            print(f"Error merging vector stores: {e}")

    def add_file(self, file_path):
        documents = read_pdf(file_path) if file_path.endswith('.pdf') else read_excel(file_path)
        chunks = split_documents(documents)
        self.add_documents(chunks)
            
    def as_retriever(self, **kwargs: Any):
        return self._store.as_retriever(**kwargs)
    
    def as_faiss(self):
        return self._store
    
    # Private Methods
    def _initialize_with_documents(self, documents):
        self._store = FAISS.from_documents(documents, self.embeddings)
        if self.persist_directory is not None:
            self._cleanup_directory()
            self._store.save_local(self.persist_directory)

    def _load_existing_store(self):
        self._store = FAISS.load_local(self.persist_directory, self.embeddings, allow_dangerous_deserialization=True)

    def _initialize_empty_store(self):
        print("Initializing empty FAISS store")
        self._store = FAISS.from_texts(texts=[], embedding=self.embeddings)

    def _cleanup_directory(self):
        if os.path.exists(self.persist_directory):
            print(f"Directory [{self.persist_directory}] is not empty, cleaning and re-initializing")
            for file in os.listdir(self.persist_directory):
                os.remove(os.path.join(self.persist_directory, file))
            os.rmdir(self.persist_directory)

if __name__ == "__main__":
    load_dotenv(find_dotenv(), override=True)
    
    PATH_PDF = "qa-doc/3Q24-Earnings-Transcript.pdf"

    documents = read_pdf(PATH_PDF)

    chunks = split_documents(documents)
    
    embeddings = OpenAIEmbeddings(base_url=os.environ['EMBEDDINGS_BASE_URL'])
    # db = VectorStore(documents=chunks, dir="./faiss_db_3", embeddings=embeddings)
    db = VectorStore(embeddings=embeddings, dir="./faiss_db_3")
    # db = VectorStore(documents=chunks)
    
    
    print("====================================================\n")
    
    # db.add_documents(chunks)
    df = db.to_dataframe()
    print(df.head())
    
    faiss = db.as_faiss()
    
    query = "What's this about"
    retriever = db.as_retriever(search_type="similarity", k=5)
    docs = retriever.retrieve(query)
    print(f"Number of documents: {len(docs)}")
    print(f"docs = {docs[0].page_content[:200]}")
    print(f"docs metadata = {docs[0].metadata}")

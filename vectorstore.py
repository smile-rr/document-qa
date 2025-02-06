from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
import pandas as pd
import os
from dotenv import load_dotenv, find_dotenv
from document_reader import DocumentReader
from util import print_structure

class VectorStore:
    
    def __init__(self, embeddings=None,dir=None,  documents=None ):
        if embeddings is None:
            raise ValueError("Embeddings must be provided")
        
        self.embeddings = embeddings or OpenAIEmbeddings(base_url=os.environ.get('EMBEDDINGS_BASE_URL'))
        self.persist_directory = dir
        self._store = None
        
        if documents is not None:
            self._store = FAISS.from_documents(documents, self.embeddings)
            if dir is not None and dir is not None:
                if os.path.exists(dir):
                    print(f"dir [{dir}] is not empty, clean and re-initialize")
                    for file in os.listdir(dir):
                        os.remove(os.path.join(dir, file))
                    os.rmdir(dir)
                self._store.save_local(dir)
        elif dir is not None and os.path.exists(dir):
            self._store = FAISS.load_local(dir, self.embeddings, allow_dangerous_deserialization=True)
        else:
            print("Initializing empty FAISS store")
            self._store = FAISS.from_texts(texts=[], embedding=self.embeddings)

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
            
    def as_retriever(self):
        return self._store.as_retriever()
    
    def as_faiss(self):
        return self._store

if __name__ == "__main__":
    load_dotenv(find_dotenv(), override=True)
    
    PATH_PDF = "qa-doc/3Q24-Earnings-Transcript.pdf"

    documents = DocumentReader.read_pdf(PATH_PDF)

    chunks = DocumentReader.split_documents(documents)
    
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

    docs = faiss.similarity_search(query)
    print(f"Number of documents: {len(docs)}")
    print(f"docs = {docs[0].page_content[:200]}")
    
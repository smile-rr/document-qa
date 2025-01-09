
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
import pandas as pd
import os
from dotenv import load_dotenv, find_dotenv
from document_reader import read_pdf, read_excel, token_count, split_documents
from util import print_structure

class VectorStore:
    
    def __init__(self, documents=None, dir=None):
        self.embeddings = OpenAIEmbeddings(base_url=os.environ['EMBEDDINGS_BASE_URL'])
        self._store = None
        self.persist_directory = dir
        
        if documents is not None and dir is not None:
            # remove dir and the files in it
            if os.path.exists(dir):
                print(f"dir [{dir}] is not empty, clean and re-initialize")
                for file in os.listdir(dir):
                    os.remove(os.path.join(dir, file))
                os.rmdir(dir)
                
            self._store = FAISS.from_documents(documents, self.embeddings)
            self._store.save_local(dir)
        elif documents is not None:
            print(f"no dir, running in memory")
            self._store = FAISS.from_documents(documents, self.embeddings)
        elif dir is not None:
            self._store = FAISS.load_local(dir, self.embeddings, allow_dangerous_deserialization = True)
        else:
            raise ValueError("Either documents or dir must be provided.")


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
        vector_df = self.store_to_df(self._store)
        chunks_list = vector_df.loc[vector_df['doc_name']==document_name]["chunk_id"].tolist()
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
            
    def as_retriever(self):
        return self._store.as_retriever()


if __name__ == "__main__":
    load_dotenv(find_dotenv(), override=True)
    
    PATH_PDF = "qa-doc/3Q24-SUPP-ForWeb.pdf"

    documents = read_pdf(PATH_PDF)
    
    print(f"Number of documents: {len(documents)}, \nlength of first document: {len(documents[0].page_content)}")
    
    tokens = [token_count(doc.page_content) for doc in documents]
    print(f"tokens counts: {tokens}")

    chunks = split_documents(documents)
    
    print(f" len chunks: {len(chunks)}")
    tokens2 = [token_count(chunk.page_content) for chunk in chunks]
    print(f"tokens counts: {tokens2}")
    print("")
    for i, doc in enumerate(chunks[:5]):
        print(f"chunks {i+1}:\n {doc}")
        print("====================================================\n")
    
    # db = VectorStore(documents=chunks, dir="./faiss_db")
    db = VectorStore( dir="./faiss_db")
    # db = VectorStore(documents=chunks)
    
    print("====================================================\n")
    
    # db.add_documents(chunks)
    df = db.to_dataframe()
    print(df.head())
import os
import json
import pandas as pd
import logging
from rich import print as rprint
from pydantic import BaseModel, Field
from typing import Optional, List

from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import RetrievalQA, StuffDocumentsChain
from langchain.chains.combine_documents.stuff import create_stuff_documents_chain

from llm_core.document_reader import read_pdf
from llm_core.llm import create_embeddings, create_llm
from llm_core.llm_config import *
from llm_core.vectorstore import VectorStore, indexing_dir, indexing_files, load_vectorstore

from tabulate import tabulate
from dotenv import load_dotenv, find_dotenv

logging.basicConfig(level=logging.DEBUG)


load_dotenv(find_dotenv(), override=True)

# Define the Pydantic model for the extracted data
class MamiProfile(BaseModel):
    name: Optional[str] = Field(default="N/A", description="Applicant's name")
    age: Optional[str] = Field(default="N/A", description="Applicant's age")
    ethnicity: Optional[str] = Field(default="N/A", description="Applicant's ethnicity")
    location: Optional[str] = Field(default="N/A", description="Applicant's location")
    marital_status: Optional[str] = Field(default="N/A", description="Applicant's marital status")
    pregnancies: Optional[str] = Field(default="N/A", description="Number of pregnancies")
    delivery_method: Optional[str] = Field(default="N/A", description="Delivery method (vaginal or c-section)")
    intended_parents: Optional[str] = Field(default="N/A", description="Type of intended parents the applicant is willing to work with")

class ExtractionData(BaseModel):
    profiles: List[MamiProfile]

# Define the extraction prompt
prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an expert at identifying key historic development in text. "
            "Only extract important historic developments. Extract nothing if no important information can be found in the text.",
        ),
        ("human", "{text}"),
    ]
)

# Process PDFs in the folder and extract data
pdf_folder = "/Users/pc-rn/Documents/算法/AI"  # Replace with the path to your folder
embeddings = create_embeddings(OPENAI_EMBEDDING)
# vector_store = indexing_dir(embeddings, pdf_folder)
vector_store = load_vectorstore(embeddings, db_path="./faiss_db")
retriever = vector_store.as_retriever()

# Initialize the LLM and build the extraction chain
llm = create_llm(QWEN_PLUS, temperature=0.0)
structured_llm = llm.with_structured_output(ExtractionData)
extractor = prompt_template | llm.with_structured_output(
    schema=ExtractionData,
    include_raw=False,
)
rag_extractor = {
    "text": retriever | (lambda docs: docs[0].page_content)  # fetch content of top doc
} | extractor

results = rag_extractor.invoke("extract fields for MamiProfile")

print(results)

print(f"results:\n {results}")

rprint(f"Extracted {len(results.profiles)} profiles:")
rprint(tabulate([profile.dict() for profile in results.profiles], headers="keys", tablefmt="grid"))
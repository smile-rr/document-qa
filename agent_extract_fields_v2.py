import os
import json
import pandas as pd
import logging
from rich.pretty import pprint 
from pydantic import BaseModel, Field
from typing import Optional, List

from langchain_core.prompts import ChatPromptTemplate
from llm_core.document_reader import read_pdf
from llm_core.llm import create_llm
from llm_core.llm_config import *

from tabulate import tabulate


logging.basicConfig(level=logging.INFO)

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

class Data(BaseModel):
    profiles: List[MamiProfile]

# Define the extraction prompt
prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an expert extraction algorithm. "
            "Only extract relevant information from the text. "
            "If you do not know the value of an attribute asked to extract, "
            "return null for the attribute's value.",
        ),
        ("human", "{text}"),
    ]
)

# Initialize the LLM and build the extraction chain
llm = create_llm(QWEN_PLUS, temperature=0.0)
structured_llm = llm.with_structured_output(MamiProfile)

results = []
pdf_folder = "/Users/pc-rn/Documents/算法/AI"  # Replace with the path to your folder
pdf_files = [f for f in os.listdir(pdf_folder) if f.lower().endswith(".pdf")]
for filename in pdf_files[:4]:
    filepath = os.path.join(pdf_folder, filename)
    
    docs = read_pdf(filepath)
    pdf_texts = "\n".join([doc.page_content for doc in docs]) + "\n"
    
    prompt = prompt_template.invoke({"text": pdf_texts})
    extraction_output = structured_llm.invoke(prompt)
    results.append(extraction_output)



pprint(f"results:\n {results}")

print(f"Extracted {len(results)} profiles:")
profile_dict = [profile.dict() for profile in results]
print(tabulate(profile_dict, headers="keys", tablefmt="grid"))


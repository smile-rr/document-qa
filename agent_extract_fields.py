import os
import re
import json
import datetime
import time
import pandas as pd
import logging
from rich import print as rprint

from langchain.prompts import PromptTemplate


from llm_core.document_reader import read_pdf, to_documents
from llm_core.llm import create_llm, create_embeddings
from llm_core.llm_config import (QWEN_2_5, OLLAMA_EMBEDDING, GPT_3_5, OPENAI_EMBEDDING, LLAMA_3, LLAMA_3_VISION, 
                                 GEMINI_PRO, DEEPSEEK_R1,GPT_4_O, QWEN_PLUS, QWEN_MAX)



logging.basicConfig(level=logging.WARNING)

# --- Step 1: Helper function to calculate age from birth date ---
def calculate_age_from_dob(text: str) -> str:
    """
    Search for a birth date pattern (e.g., "DOB: MM/DD/YYYY" or "Birth Date: MM/DD/YYYY")
    in the text and calculate the age using today's date.
    """
    dob_match = re.search(r"(?:DOB|Birth\s*Date)[:\s]+(\d{1,2}/\d{1,2}/\d{4})", text)
    if dob_match:
        dob_str = dob_match.group(1)
        try:
            dob = datetime.datetime.strptime(dob_str, "%m/%d/%Y")
            today = datetime.datetime.today()
            age = today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))
            return age
        except Exception:
            return "Error"
    return "N/A"

# --- Step 2: Define the extraction prompt ---
extraction_prompt_template = """
You are an expert data extractor. Given the following document text from a PDF file with a consistent structure, please extract the following fields and output them in valid JSON format without any additional formatting:
- "name": Applicant's name.
- "age": Applicant's age. If the text only provides a birth date, calculate the age using the current date.
- "ethnicity": Applicant's ethnicity.
- "location": Applicant's location.
- "marital_status": Applicant's marital status.
- "pregnancies": How many pregnancies.
- "delivery_method": Whether the delivery was vaginal or by c-section.
- "intended_parents": The type of intended parents the applicant is willing to work with.

If any field is not mentioned in the text, output "N/A" for that field.

Document text:
{text}
"""

prompt = PromptTemplate(
    template=extraction_prompt_template,
    input_variables=["text"]
)

# --- Step 3: Initialize the LLM and build the extraction chain ---
llm = create_llm(QWEN_MAX, temperature=0.0)
# Build the chain using the pipe operator
extraction_chain = prompt | llm

# --- Step 4: Process PDFs in the folder and extract data ---
pdf_folder = "/Users/pc-rn/Documents/算法/AI"  # Replace with the path to your folder
results = []

pdf_count = 0

# get only for pdf files
pdf_files = [f for f in os.listdir(pdf_folder) if f.lower().endswith(".pdf")]

for filename in pdf_files:
    rprint(f"Processing {filename}...")
    filepath = os.path.join(pdf_folder, filename)
    
    # Load the PDF file using PyMuPDFLoader
    docs = read_pdf(filepath)
    # Combine all pages into one text block
    full_text = "\n".join([doc.page_content for doc in docs])
    
    # Run the extraction chain
    extraction_output = extraction_chain.invoke({"text": full_text})
    logging.debug(extraction_output)
    try:
        # Extract JSON content from the 'content' attribute
        json_content = extraction_output.content.strip('```json\n').strip('\n```')
        extraction_data = json.loads(json_content)
        data = {
            "name": extraction_data.get("name", "N/A"),
            "age": extraction_data.get("age", "N/A"),
            "ethnicity": extraction_data.get("ethnicity", "N/A"),
            "location": extraction_data.get("location", "N/A"),
            "marital_status": extraction_data.get("marital_status", "N/A"),
            "pregnancies": extraction_data.get("pregnancies", "N/A"),
            "delivery_method": extraction_data.get("delivery_method", "N/A"),
            "intended_parents": extraction_data.get("intended_parents", "N/A")
        }
    except json.JSONDecodeError:
        data = {
            "name": "Error",
            "age": "Error",
            "ethnicity": "Error",
            "location": "Error",
            "marital_status": "Error",
            "pregnancies": "Error",
            "delivery_method": "Error",
            "intended_parents": "Error"
        }
    
    # Optionally add the filename to the result for traceability
    data["filename"] = filename
    
    results.append(data)
    
    pdf_count += 1  # Increment the counter
    # Sleep for 1 second to avoid overwhelming the server
    time.sleep(3)
    if pdf_count >= 1:  # Stop after processing 1 PDF
        break

# --- Step 5: Summarize the results in a CSV table ---
df = pd.DataFrame(results)
# show the df in the console as table
print("--- Extracted Summary ---")
print(df.to_string(index=False))

# Get the current date
current_date = datetime.datetime.now().strftime("%Y-%m-%d")

# Save the summary to a CSV file
# csv_file_path = f"{pdf_folder}/extracted_summary_{current_date}.csv"
# df.to_csv(csv_file_path, index=False)
# print(f"Extraction completed. The summary is saved in '{csv_file_path}'.")

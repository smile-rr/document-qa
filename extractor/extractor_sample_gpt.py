import os
import json
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain.prompts import PromptTemplate

def create_parser():
    # Define the expected schema for the user profile
    response_schemas = [
        ResponseSchema(name="name", description="The name of the user"),
        ResponseSchema(name="age", description="The age of the user"),
        ResponseSchema(name="gender", description="The gender of the user"),
        ResponseSchema(name="location", description="The location of the user"),
    ]
    parser = StructuredOutputParser.from_response_schemas(response_schemas)
    return parser

def extract_profile(text, llm, parser):
    # Get format instructions from the parser so that the LLM output is in valid JSON format.
    format_instructions = parser.get_format_instructions()
    prompt_template = f"""
Extract the user profile information from the following text.
The profile should include: name, age, gender, and location.
{format_instructions}

Text:
{{text}}
"""
    prompt = PromptTemplate(input_variables=["text"], template=prompt_template)
    chain = LLMChain(llm=llm, prompt=prompt)
    output = chain.run(text)
    try:
        profile = parser.parse(output)
    except Exception as e:
        # In case parsing fails, return the raw output for debugging.
        profile = {"raw_output": output}
    return profile

def process_pdf(file_path, llm, text_splitter, parser):
    # Load the PDF file
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    profiles = []
    # Process each document (e.g., page) from the PDF
    for doc in docs:
        # Split the page content into smaller chunks (adjust chunk_size/chunk_overlap as needed)
        chunks = text_splitter.split_text(doc.page_content)
        for chunk in chunks:
            profile = extract_profile(chunk, llm, parser)
            profiles.append(profile)
    return profiles

if __name__ == "__main__":
    # Path to the folder containing PDF files
    pdf_folder = "path_to_pdf_folder"
    pdf_files = [os.path.join(pdf_folder, f) for f in os.listdir(pdf_folder) if f.endswith(".pdf")]

    # Initialize the LLM (e.g., OpenAI) and text splitter.
    llm = OpenAI(temperature=0)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    parser = create_parser()

    # Process each PDF file
    for pdf_file in pdf_files:
        print(f"Processing {pdf_file}")
        profiles = process_pdf(pdf_file, llm, text_splitter, parser)
        print("Extracted Profiles:")
        for profile in profiles:
            print(json.dumps(profile, indent=2))

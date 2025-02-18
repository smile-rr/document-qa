import os
import openai
from openai import OpenAI
from dotenv import load_dotenv, find_dotenv

# Load environment variables from .env file
load_dotenv(find_dotenv(), override=True)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), base_url=os.getenv("OPENAI_API_BASE_URL"))




# Get API key and base URL from environment variables
# TODO: The 'openai.api_base' option isn't read in the client API. You will need to pass it when you instantiate the client, e.g. 'OpenAI(base_url=os.getenv("OPENAI_API_BASE_URL"))'
# openai.api_base = os.getenv("OPENAI_API_BASE_URL")

# Define conversation messages
messages = [
    {"role": "system", "content": "You are a friendly assistant."},
    {"role": "user", "content": "Hello! Please tell me something about artificial intelligence in Chinese"}
]

# Call OpenAI's chat model
response = client.chat.completions.create(model="gpt-3.5-turbo",
messages=messages,
max_tokens=100,
temperature=0.7)

# Print the model's response
print(response.choices[0].message.content)
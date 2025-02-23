from llm_core.llm import create_llm
from llm_core.llm_config import *

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(), override=True)

messages = [
    {"role": "user", "content": "2 🦜 2"},
    {"role": "assistant", "content": "4"},
    {"role": "user", "content": "2 🦜 3"},
    {"role": "assistant", "content": "5"},
    {"role": "user", "content": "3 🦜 4"},
]

llm = create_llm(GPT_3_5)
response = llm.invoke(messages)
print(response.content)
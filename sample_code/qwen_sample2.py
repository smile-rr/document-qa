from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv, find_dotenv
import json
from llm_core.util import print_structure
from langchain_openai import OpenAIEmbeddings

class QwenEmbeddings(OpenAIEmbeddings):
    def __init__(self, api_key=None, base_url=None, model="qwen-embedding", **kwargs):
        super().__init__(api_key=api_key, base_url=base_url, model=model, **kwargs)


def test_embeddings():
    api_key = os.getenv("DASHSCOPE_API_KEY")
    base_url = os.getenv("DASHSCOPE_API_BASE_URL")
    
    embeddings = QwenEmbeddings(api_key=api_key, base_url=base_url)
    
    # Example text to embed
    text = "这是一个示例文本。"
    
    # Generate embeddings
    embedding_vector = embeddings.embed(text)
    
    print("Embedding vector:", embedding_vector)
def test_qwen_llm():
    chatLLM = ChatOpenAI(
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url=os.getenv("DASHSCOPE_API_BASE_URL"),
        model="qwen-plus",
        # other params...
    )
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "你是谁？"}]
    response = chatLLM.invoke(messages)

    print_structure(response)
    print("------------------")
    print(response.content)

if __name__ == "__main__":
    load_dotenv(find_dotenv(), override=True)
    # test_qwen_llm()
    test_embeddings()
from langchain_community.llms import Ollama
from langchain_google_vertexai import ChatVertexAI
from langchain_openai import ChatOpenAI
import os

class LLM:

    @staticmethod
    def create(model_name="gpt-3.5-turbo"):
        if model_name.startswith("openchat") or model_name.startswith("llama"):
            return LLM.create_ollama(model_name)
        elif model_name.startswith("gpt"):
            return LLM.create_openai(model_name)
        elif model_name.startswith("gemini"):
            return LLM.create_vertexai(model_name)
        else:
            raise ValueError(f"Model name [{model_name}] not supported.") 

    @staticmethod
    def create_openai(model_name):
        return ChatOpenAI(
            model_name=model_name,
            openai_api_base=os.environ['CHATGPT_API_ENDPOINT'],
            openai_api_key=os.environ['OPENAI_API_KEY'],
            temperature=0.8,
            verbose=True
        )

    @staticmethod
    def create_ollama(model_name):
        return Ollama(
            model=model_name,
            temperature=0.8,
            verbose=True
        )

    @staticmethod
    def create_vertexai(model_name):
        return ChatVertexAI(
            model_name=model_name,
            temperature=0.5,
            verbose=True
        )
import logging
from langchain_ollama import ChatOllama, OllamaLLM, OllamaEmbeddings
from langchain_google_vertexai import ChatVertexAI, VertexAI, VertexAIEmbeddings
from langchain_openai import ChatOpenAI, OpenAI, OpenAIEmbeddings
import os
from dotenv import load_dotenv, find_dotenv
import json
from llm_config import (
    GPT_3_5, GEMINI_FLASH, GEMINI_PRO, OPENCHAT, LLAMA_3, LLAMA_3_VISION, DEEPSEEK_R1, QWEN_2_5,
    TEXTEMBEDDING_GECKO, TEXTEMBEDDING_GECKO_MULTILINGUAL, OPENAI_EMBEDDING, OLLAMA_EMBEDDING
)

# Define a mapping of model types to their corresponding classes
MODEL_CLASSES = {
    "chat": {
        "openai": ChatOpenAI,
        "vertexai": ChatVertexAI,
        "ollama": ChatOllama
    },
    "text": {
        "openai": OpenAI,
        "vertexai": VertexAI,
        "ollama": OllamaLLM
    }
}

def create_llm(model_name, model_type="chat", temperature=0.8, verbose=True, **kwargs):
    if model_name in [GPT_3_5]:
        model_class = MODEL_CLASSES[model_type]["openai"]
        return model_class(
            model_name=model_name,
            base_url=os.environ.get('OPENAI_API_BASE_URL'),
            temperature=temperature,
            verbose=verbose,
            **kwargs
        )
    elif model_name in [GEMINI_FLASH, GEMINI_PRO]:
        model_class = MODEL_CLASSES[model_type]["vertexai"]
        return model_class(
            model_name=model_name,
            location=kwargs.get('location', "us-central1"),
            max_output_tokens=kwargs.get('max_output_tokens', 1024),
            top_p=kwargs.get('top_p', 0.95),
            top_k=kwargs.get('top_k', 40),
            temperature=temperature,
            verbose=verbose,
            **kwargs
        )
    else:
        model_class = MODEL_CLASSES[model_type]["ollama"]
        return model_class(
            model=model_name,
            temperature=temperature,
            verbose=verbose,
            **kwargs
        )



# Define a mapping of embedding model names to their corresponding classes
EMBEDDING_MODEL_CLASSES = {
    OPENAI_EMBEDDING: OpenAIEmbeddings,
    TEXTEMBEDDING_GECKO: VertexAIEmbeddings,
    TEXTEMBEDDING_GECKO_MULTILINGUAL: VertexAIEmbeddings,
    OLLAMA_EMBEDDING: OllamaEmbeddings,
}
def create_embeddings(model_name, **kwargs):
    if model_name in EMBEDDING_MODEL_CLASSES:
        model_class = EMBEDDING_MODEL_CLASSES[model_name]
        if model_name == "openai-embedding":
            kwargs.setdefault('base_url', os.environ.get('EMBEDDINGS_BASE_URL'))
        return model_class(
            model=model_name,
            **kwargs
        )
    else:
        raise ValueError(f"Model name [{model_name}] not supported. Supported models are: {', '.join(EMBEDDING_MODEL_CLASSES.keys())}")


def normalize_response(response):
    if hasattr(response, 'content'):
        return {
            "content": response.content,
            "metadata": getattr(response, 'metadata', {})
        }
    return {
        "text": response,
        "metadata": {}
    }


if __name__ == "__main__":
     # Clear the terminal
    os.system('clear' if os.name == 'posix' else 'cls')
    
    load_dotenv(find_dotenv(), override=True)
    # Configure logging to include timestamps
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    # Define a list of all model name variables
    model_names = [
        GPT_3_5,
        GEMINI_FLASH,
        GEMINI_PRO,
        # OPENCHAT,
        LLAMA_3,
        # DEEPSEEK_R1,
        QWEN_2_5
    ]
    
    query = "你的训练数据截止时间是到什么时候?"
    
    for model_name in model_names:
        logging.info(f"\n\n=============== {model_name} =================")
        
        llm = create_llm(model_name)
        response = llm.invoke(query)
        normalized_response = normalize_response(response)
        logging.info(f"User : {query}")
        logging.info(f"{model_name} : {normalized_response['content']}")
        # print(f"Metadata: {normalized_response['metadata']}")
        logging.info("\n================================\n")
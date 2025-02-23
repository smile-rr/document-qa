from langchain import hub
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.documents import Document
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langgraph.graph import  StateGraph, START

import os
import bs4
from typing import TypedDict, List
from dotenv import load_dotenv, find_dotenv

from llm_core.llm import create_llm, create_embeddings
from llm_core.llm_config import *


load_dotenv(find_dotenv(), override=True)

embeddings = create_embeddings(OPENAI_EMBEDDING)
llm = create_llm(QWEN_PLUS)
vector_store = InMemoryVectorStore(embeddings)


# Load and chunk contents of the blog
loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")
        )
    ),
)

docs = loader.load()
text_spliter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_spliter.split_documents(docs)

_ = vector_store.add_documents(chunks)


prompt = hub.pull("rlm/rag-prompt")

class State(TypedDict):
    question: str
    context: List[Document]
    answer: str

def retrieve(state: State):
    retrived_docs = vector_store.similarity_search(state["question"])
    return {"context": retrived_docs}

def generate(state: State):
    docs = "\n\n".join([doc.page_content for doc in state["context"]])
    messages = prompt.invoke({"question": state["question"], "context": docs})
    response = llm.invoke(messages)
    return {"answer": response.content}

graph_builder = StateGraph(State).add_sequence([retrieve, generate])
graph_builder.add_edge(START, "retrieve")
graph = graph_builder.compile()

response = graph.invoke({"question": "What is Task Decomposition??"})
print(response["answer"])
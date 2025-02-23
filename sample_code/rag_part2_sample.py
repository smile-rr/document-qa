from llm_core.llm import create_llm, create_embeddings
from llm_core.llm_config import *

from langchain import hub
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.documents import Document
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langgraph.graph import StateGraph, MessagesState, END
from langgraph.prebuilt import ToolNode, tools_condition, create_react_agent

import os
import bs4
from dotenv import load_dotenv, find_dotenv



load_dotenv(find_dotenv(), override=True)

llm = create_llm(GPT_3_5)
embeddings = create_embeddings(OPENAI_EMBEDDING)
vector_store = InMemoryVectorStore(embeddings)


docs = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")
        )
    ),
).load()

text_spliter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_spliter.split_documents(docs)
_ = vector_store.add_documents(chunks)


@tool(response_format="content_and_artifact")
def retrieve(query:str):
    """Retrieve information related to query."""
    retrived_docs = vector_store.similarity_search(query)
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\n" f"Content: {doc.page_content}")
        for doc in retrived_docs
    )
    return serialized, retrived_docs


def query_or_respond(state: MessagesState):
    """Generate tool call for retrive or response."""
    llm_with_tools = llm.bind_tools([retrieve])
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}

def generate(state: MessagesState) :
    """Generate answer"""
    recent_tool_messages = []
    for message in reversed(state["messages"]):
        if message.type == "tool":
            recent_tool_messages.append(message)
        else:
            break
    tool_messages = recent_tool_messages[::-1]
    
    docs_content = "\n\n".join([doc.content for doc in tool_messages])
    system_message_content = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the "
        "answer concise."
        "\n\n"
        f"{docs_content}"
    )
    conversation_messages = [
        message
        for message in state["messages"]
        if message.type == ["human", "system"]
        or (message.type == "ai" and not message.tool_calls)
    ]
    prompt = [SystemMessage(system_message_content)] + conversation_messages
    response = llm.invoke(prompt)
    return {"messages": [response]}

def build_graph(memory=None):
    ## DEFINE NODE
    tools = ToolNode([retrieve])
    graph_builder = StateGraph(MessagesState)
    graph_builder.add_node(query_or_respond)
    graph_builder.add_node(tools)
    graph_builder.add_node(generate)

    ## DEFINE EDGES
    graph_builder.set_entry_point("query_or_respond")
    graph_builder.add_conditional_edges(
        "query_or_respond",
        tools_condition,
        {END: END, "tools": "tools"}
    )
    graph_builder.add_edge("tools", "generate")
    graph_builder.add_edge("generate", END)
    
    graph = graph_builder.compile(memory) if memory else graph_builder.compile()
    return graph

def test_graph_flow(graph, input_message, config=None):
    for step in graph.stream(
        {"messages": [{"role": "user", "content": input_message}]},
        stream_mode="values",
        config=config
    ):
        step["messages"][-1].pretty_print()

def test1():
    graph = build_graph()
    test_graph_flow(graph, "Hello")
    test_graph_flow(graph, "What is Task Decomposition??")

################ add memory to the agent ################
from langgraph.checkpoint.memory import MemorySaver

def test2_with_memory():
    memory = MemorySaver()
    graph = build_graph(memory)

    # Specify an ID for the thread
    config = {"configurable": {"thread_id": "abc123"}}
    test_graph_flow(graph, "What is Task Decomposition??", config)


def test3_with_react_agent():
    memory = MemorySaver()
    agent_executor = create_react_agent(llm, [retrieve], checkpointer=memory)
    # Specify an ID for the thread
    config = {"configurable": {"thread_id": "abc123"}}
    
    input_message = (
        "What is the standard method for Task Decomposition? \n\n"
        "Once you get the answer, look up common extensions of that method."
        "用中文回答"
    )
    
    for event in agent_executor.stream(
        {"messages": [{"role": "user", "content": input_message}]},
        stream_mode="values",
        config=config,
    ):
        event["messages"][-1].pretty_print()
    
test3_with_react_agent()


# agent_executor = create_react_agent(llm, [retrieve], checkpointer=memory)





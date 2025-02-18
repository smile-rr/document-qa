# Import relevant functionality
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

from llm_core.llm import create_llm
from llm_core.llm_config import *
from llm_core.util import print_structure

from dotenv import load_dotenv, find_dotenv
from pydantic import BaseModel, Field
from typing import List, Optional, Union

load_dotenv(find_dotenv(), override=True)

# Define Pydantic models for parsing response chunks
class AIMessage(BaseModel):
    content: Optional[str] = Field(None, description="The content of the AI message")

class ToolMessage(BaseModel):
    content: Optional[str] = Field(None, description="The content of the tool message")

class AgentResponse(BaseModel):
    messages: List[AIMessage]

class ToolResponse(BaseModel):
    messages: List[ToolMessage]

class ResponseChunk(BaseModel):
    agent: Optional[AgentResponse] = None
    tools: Optional[ToolResponse] = None

# Create the agent
memory = MemorySaver()
model = create_llm(QWEN_PLUS)
search = TavilySearchResults(max_results=2)
tools = [search]
agent_executor = create_react_agent(model, tools, checkpointer=memory)

# Use the agent
config = {"configurable": {"thread_id": "abc123"}}

# Function to print AI response message in a human-readable format
def print_ai_response(response_chunks):
    for chunk in response_chunks:
        print(chunk)
        if 'agent' in chunk:
            for message in chunk['agent']['messages']:
                if 'tool_calls' in message:
                    print(f"toolcall = {message['tool_calls']}")
                elif message['content']:
                    print(f"content = {message['content']}")
                else:
                    print('EMPTY')
        elif 'tools' in chunk:
            print("It's a tool message")
        print("----")

# Print AI response message for the first interaction
print("AI Response Message for 'hi im bob! and i live in Dalian':")
response_chunks = agent_executor.stream(
    {"messages": [HumanMessage(content="hi im bob! and i live in Dalian")]}, config
)
print_ai_response(response_chunks)

# Print AI response message for the second interaction
print("AI Response Message for 'whats the weather where I live?':")
response_chunks = agent_executor.stream(
    {"messages": [HumanMessage(content="whats the weather where I live?")]}, config
)
print_ai_response(response_chunks)
from typing import List, Optional, Annotated
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.utils.function_calling import tool_example_to_messages

from llm_core.llm import create_llm
from llm_core.llm_config import *
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv(), override=True)

class Person(BaseModel):
    """Information about a person"""
    name: Optional[str] = Field(default="Radon", description="The name of the person")
    hair_color: Optional[str] = Field(default="black", description="The hair color of the person")
    height_in_meters: Optional[str] = Field(default="1.65", description="The height of the person in meters")

class Data(BaseModel):
    """extract data about people."""
    people: List[Person]

prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an expert extraction algorithm. "
            "Only extract relevant information from the text. "
            "If you do not know the value of an attribute asked to extract, "
            "return null for the attribute's value.",
        ),
        ("human", "{text}"),
    ]
)

llm = create_llm(GPT_3_5)
structured_llm = llm.with_structured_output(Data)

def test_simple():

    text = "My name is Jeff, my hair is black and i am 6 feet tall. Anna has the same color hair as me."

    prompt = prompt_template.invoke({"text": text})
    response = structured_llm.invoke(prompt)
    print(response)


def test_with_tool_calling_message():
    examples = [
        (
            "The ocean is vast and blue. It's more than 20,000 feet deep.",
            Data(people=[]),
        ),
        (
            "Fiona traveled far from France to Spain.",
            Data(people=[Person(name="Fiona", height_in_meters=None, hair_color=None)]),
        ),
    ]
    messages = []

    for txt, tool_call in examples:
        if tool_call.people:
            ai_response = "Detected people"
        else:
            ai_response = "No people detected"
        messages.extend(tool_example_to_messages(txt, [tool_call], ai_response=ai_response))

    # for message in messages:
    #     message.pretty_print()
        
    message_no_extraction = {
        "role": "user",
        "content": "The solar system is large, but earth has only 1 moon.",
    }

    response = structured_llm.invoke([message_no_extraction])
    print(f"response:\n{response}")
    
    response = structured_llm.invoke(messages + [message_no_extraction])
    print(f"response2:\n{response}")




if __name__ == "__main__":
    test_simple()
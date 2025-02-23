from llm_core import create_llm
from llm_core.llm_config import *

from langchain_core.messages import HumanMessage, SystemMessage

llm = create_llm(GPT_3_5)

messages = [
        SystemMessage(
            "You are a helpful assistant that translates English to Chinese."
        ),
        HumanMessage("I love you"),
    ]
def test_simple_message():

    print("\nresponse 1:")
    llm.invoke(messages).pretty_print()
    print("\nresponse 2:")
    llm.invoke("Hello").pretty_print()
    print("\nresponse 3:")
    llm.invoke([{"role": "user", "content": "Hello"}]).pretty_print()
    print("\nresponse 4:")
    llm.invoke([HumanMessage("Hello")]).pretty_print()

def test_streaming():
    for token in llm.stream(messages):
        print(token.content, end="|")
    print()

if __name__ == "__main__":
    test_streaming()
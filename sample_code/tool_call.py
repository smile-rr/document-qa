from langchain_core.tools import tool
from llm_core.llm import create_llm
from llm_core.llm_config import GPT_3_5
from llm_core.util import print_structure
from typing import TypedDict, Annotated

class MathResult(TypedDict):
    """Result of a mathematical operation."""
    result: Annotated[int, ..., "The result of the operation"]

@tool
def add(a: int, b: int) -> MathResult:
    """Adds a and b."""
    return {"result": a + b}

@tool
def multiply(a: int, b: int) -> MathResult:
    """Multiplies a and b."""
    return {"result": a * b}

tools = [add, multiply]

llm = create_llm(model_name=GPT_3_5)
llm.bind_tools(tools)
structured_llm =llm.with_structured_output(MathResult)

from langchain.globals import set_verbose, set_debug
# set_verbose(True)
# set_debug(True)


result = structured_llm.invoke("What is 2 + 9?")

print(result)

from typing import Optional, Union
from llm_core.llm import create_llm
from llm_core.llm_config import *

from typing_extensions import Annotated, TypedDict


class Joke(TypedDict):
    """Joke to tell user."""

    setup: Annotated[str, ..., "The setup of the joke"]
    punchline: Annotated[str, ..., "The punchline of the joke"]
    rating: Annotated[Optional[int], None, "How funny the joke is, from 1 to 10"]


class ConversationalResponse(TypedDict):
    """Respond in a conversational manner. Be kind and helpful."""

    response: Annotated[str, ..., "A conversational response to the user's query"]


class FinalResponse(TypedDict):
    final_output: Union[Joke, ConversationalResponse]


llm = create_llm(model_name=QWEN_PLUS)
structured_llm = llm.with_structured_output(FinalResponse)

res = structured_llm.invoke("Tell me a joke about cats")

print(res)


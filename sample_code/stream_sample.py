from llm_core.llm import create_llm
from llm_core.llm_config import *

llm = create_llm(model_name=QWEN_PLUS)


# chunks = []
# for chunk in llm.stream("what color is the sky?"):
#     chunks.append(chunk)
#     print(chunk.content, end="|", flush=True)

import asyncio

async def main():
    chunks = []
    async for chunk in llm.astream("what color is the sky?"):
        chunks.append(chunk)
        print(chunk.content, end="", flush=True)

# asyncio.run(main())

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

async def main2():
    prompt = ChatPromptTemplate.from_template("tell me a joke about {topic}")
    parser = StrOutputParser()
    chain = prompt | llm | parser


    async for chunk in chain.astream({"topic": "parrot"}):
        print(chunk, end="|", flush=True)
        
asyncio.run(main2())
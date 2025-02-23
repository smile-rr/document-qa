from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage
from llm_core.llm import create_llm
from llm_core.llm_config import *


system_template = "Transform the following conversation into a {language}"
prompt_template = ChatPromptTemplate.from_messages(
      [
       ("system", system_template), 
       ("user", "{text}")]
)
prompt = prompt_template.invoke({"language":"Japanese", "text":"今天天气如何"})
print(prompt)
print(prompt.to_messages())
print("=====================")

llm = create_llm(GPT_3_5)
response = llm.invoke(prompt)

print(f"response is: \n{response.content}")
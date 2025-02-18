import os
from datetime import datetime
from langchain.agents import initialize_agent, AgentType
from langchain.tools import tool
from llm_core.llm import create_llm
from llm_core.llm_config import *
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv(), override=True)


# 使用 @tool 装饰器定义计算年龄的工具
@tool("AgeCalculator", return_direct=True)
def calculate_age(birth_date_str: str) -> str:
    """
    Useful for calculating a person's age given their birth date in the format YYYY - MM - DD.
    """
    try:
        # 将输入的生日字符串转换为日期对象
        birth_date = datetime.strptime(birth_date_str, "%Y-%m-%d")
        today = datetime.today()
        # 计算年龄
        age = today.year - birth_date.year - ((today.month, today.day) < (birth_date.month, birth_date.day))
        return f"The age is {age} years old."
    except ValueError:
        return "Invalid date format. Please use YYYY - MM - DD."

llm = create_llm(model_name=QWEN_PLUS)

# 定义工具列表
tools = [calculate_age]

# 初始化代理
agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# 运行查询
query = "What is the age of a person born on 1995-03-20?"
try:
    result = agent.invoke(query)
    print(f"Query: {query}")
    print(f"Answer: {result}")
except Exception as e:
    print(f"An error occurred: {e}")
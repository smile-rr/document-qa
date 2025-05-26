import os
from datetime import datetime
from langchain.agents import initialize_agent, AgentType
from langchain.tools import tool
from llm_core.llm import create_llm
from llm_core.llm_config import *
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv(), override=True)


# Use @tool decorator to define a tool for calculating age
@tool("AgeCalculator", return_direct=True)
def calculate_age(birth_date_str: str) -> str:
    """
    Useful for calculating a person's age given their birth date in the format YYYY - MM - DD.
    """
    try:
        # Convert the input birth date string to a date object
        birth_date = datetime.strptime(birth_date_str, "%Y-%m-%d")
        today = datetime.today()
        # Calculate age
        age = today.year - birth_date.year - ((today.month, today.day) < (birth_date.month, birth_date.day))
        return f"The age is {age} years old."
    except ValueError:
        return "Invalid date format. Please use YYYY - MM - DD."

llm = create_llm(model_name=GPT_3_5)

# Define the tool list
tools = [calculate_age]

# Initialize the agent
agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# Run the query
query = "What is the age of a person born on 1995-03-20?"
try:
    result = agent.invoke(query)
    print(f"Query: {query}")
    print(f"Answer: {result}")
except Exception as e:
    print(f"An error occurred: {e}")
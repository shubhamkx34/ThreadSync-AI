# Workflow :
from langgraph.graph import START , StateGraph , END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_tavily import TavilySearch
import math
import requests
import os 
from typing import Any

load_dotenv()
WEATHERSTACK_API_KEY = os.getenv("WEATHERSTACK_API_KEY")
llmModel = ChatGroq(model='openai/gpt-oss-120b')

# Live Search Tool 1
search_tool = TavilySearch(
    max_results=3,
    topic="general",
    search_depth="advanced",
    include_raw_content=False
)

# Live stock Fetch Tool 2:
@tool
def get_stock_price(symbol: str) -> dict:
    """
    Fetch latest stock price for a given symbol (e.g. 'AAPL', 'TSLA') 
    using Alpha Vantage with API key in the URL.
    """
    url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey=D71XPZPBYI82R4LA"
    r = requests.get(url)
    return r.json()

# Live Weather Fetch Tool 3:
@tool
def get_weather_data(city: str) -> str:
    """
    Fetch current weather information for a city.
    """

    url = (
        f"http://api.weatherstack.com/current?"
        f"access_key={WEATHERSTACK_API_KEY}&query={city}"
    )

    response = requests.get(url)

    data = response.json()

    if "current" not in data:
        return f"Could not fetch weather data for {city}"

    return (
        f"City: {city}\n"
        f"Temperature: {data['current']['temperature']}°C\n"
        f"Weather: {data['current']['weather_descriptions'][0]}\n"
        f"Humidity: {data['current']['humidity']}%"
    )    

# LLM + Tools Connection:
tools = [search_tool, get_stock_price,get_weather_data]# tool list
llm_with_tools = llmModel.bind_tools(tools)# Make the LLM tool-aware


from langgraph.graph.message import add_messages

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# Nodes 1 :message history sent to the AI
def chat_node(state: ChatState):
    #take user query from state
    messages = state['messages'][-6:]
    # send to llm
    response = llm_with_tools.invoke(messages)
    # response store state
    return {'messages': [response]}

# Nodes 2 - tool node
tool_node = ToolNode(tools)

# Memory Persistence: A MemorySaver checkpointer (an in-memory storage tool) is attached to the compiled graph, allowing the backend to save and recall conversation snapshots.
Connection = sqlite3.connect(database="threadsync.db", check_same_thread=False)
checkpoint = SqliteSaver(Connection)


# Graph Node Structures:
graph = StateGraph(ChatState)

# add nodes
graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node)

#add edges
graph.add_edge(START, "chat_node")
graph.add_conditional_edges("chat_node", tools_condition)# If the LLM asked for a tool, go to ToolNode; else finish
graph.add_edge("tools", "chat_node")    # Send the contents provided by Tools to LLM So that more refined output user can get

chatbot = graph.compile(checkpointer=checkpoint)    

# Helper function for Streamlit frontend to list all the threads in the Chatlist Sidebar
def get_all_threads():
    all_threads = set()
    for ckpt in checkpoint.list(None):
        all_threads.add(ckpt.config['configurable']['thread_id'])

    return list(all_threads)
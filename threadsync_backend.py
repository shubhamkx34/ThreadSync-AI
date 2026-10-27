# Workflow :
import os 
from langgraph.graph import START , StateGraph , END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3
from tools_feature import search_tool, get_stock_price, get_weather_data
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import trim_messages
from rag_feature import rag_tool
from HITL_feature import purchase_stock

load_dotenv()
llmModel = ChatGroq(model='openai/gpt-oss-120b')

# LLM + Tools Connection:
tools = [search_tool, get_stock_price,get_weather_data, rag_tool, purchase_stock] # tool list
llm_with_tools = llmModel.bind_tools(tools) # Make the LLM tool-aware


from langgraph.graph.message import add_messages

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# Nodes 1 : message history sent to the AI
def chat_node(state: ChatState):
    system_message = SystemMessage(
        content=
        "You are a helpful Agentic Chatbot with access to several tools.\n\n"
            "Tool usage instructions:\n"
            "- ALWAYS call `rag_tool` whenever the user asks about a document, PDF, resume, "
            "report, invoice, payment, transaction, or specific personal/file details. "
            "Never rely on earlier chat messages to guess what is in the PDF, as the user "
            "may have uploaded a new PDF file during the conversation. Always run `rag_tool` fresh.\n"
            "- Use `search_tool` for current events, recent information, or internet searches.\n"
            "- Use `get_stock_price` when the user asks for the current price of a stock.\n"
            "- Use `get_weather_data` when the user asks about current weather for a location.\n\n"
            "Answer general questions directly when no tool is required. "
            "Do not invent information from the uploaded document. "
            "If the user asks about a PDF but no document is available, ask them to upload a PDF. "
            "After receiving a tool result, provide a clear and helpful final answer."
    )
    # Keep recent conversation history within a token limit
    recent_messages = trim_messages(
        state["messages"],
        max_tokens=3000,
        token_counter=llmModel,
        strategy="last",
        start_on="human",
        include_system=False
    )
    #take user query from state
    messages = [system_message, *recent_messages]
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
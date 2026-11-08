# Agent Workflow :
import os 
from langgraph.graph import START , StateGraph , END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3
from tools_feature import search_tool, get_stock_price, get_weather_data
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import trim_messages
from langchain_core.messages.utils import count_tokens_approximately
from rag_feature import rag_tool
from HITL_feature import purchase_stock
from pathlib import Path
import certifi
from langchain_core.runnables import RunnableConfig

os.environ["SSL_CERT_FILE"]=certifi.where()
os.environ["REQUESTS_CA_BUNDLE"]=certifi.where()

load_dotenv()

DEFAULT_MODEL = "openai/gpt-oss-120b"
def get_llm(model_id):
    if "gemini" in model_id.lower(): # If the model string contains "gemini", use the Google provider
        return ChatGoogleGenerativeAI(model=model_id, google_api_key=os.getenv("GOOGLE_API_KEY"))
    return ChatGroq(model=model_id)    # Otherwise, use the Groq provider

ALLOWED_MODELS={
    "GPT-OSS 120B": "openai/gpt-oss-120b",
    "Gemini 3.6 Flash": "gemini-3.6-flash",
    "Gemini 3.5 Flash-Lite": "gemini-3.5-flash-lite",
    "Gemini 3.1 Flash-Lite": "gemini-3.1-flash-lite",
     "GPT-OSS 20B": "openai/gpt-oss-20b"
}

# LLM + Tools Connection:
tools = [search_tool, get_stock_price,get_weather_data, rag_tool, purchase_stock] # tool list

from langgraph.graph.message import add_messages
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# Nodes 1 : message history sent to the AI
def chat_node(state: ChatState, config: RunnableConfig):

    # 1. Read the selected model from the config
    model_id = config.get("configurable", {}).get("model_id", DEFAULT_MODEL)
    active_llm = get_llm(model_id)
    llm_with_tools = active_llm.bind_tools(tools)

    # 2. Inject the model_id into the SystemMessage using an f-string
    system_message = SystemMessage(
        content=
        f"You are ThreadSync, an Agentic Chat System powered by the {model_id} model.\n\n"
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
        token_counter=count_tokens_approximately,
        strategy="last",
        start_on="human",
        include_system=False
    )
    
    # Universal Sanitization Loop
    sanitized_messages = []
    for msg in recent_messages:
        if isinstance(msg, AIMessage):
            # Strips ALL provider-specific metadata (Groq reasoning, Gemini safety ratings)
            # Guarantees no cross-provider crashes during mid-conversation switches
            sanitized_messages.append(AIMessage(content=msg.content, tool_calls=msg.tool_calls))
        else:
            sanitized_messages.append(msg)

    # Use the universally sanitized messages
    messages = [system_message, *sanitized_messages]
    
    # Send to LLM
    response = llm_with_tools.invoke(messages)
    
    return {'messages': [response]}

# Nodes 2 - tool node
tool_node = ToolNode(tools)

# Memory Persistence: A MemorySaver checkpointer (an in-memory storage tool) is attached to the compiled graph, allowing the backend to save and recall conversation snapshots.
db_folder = Path("Data")
db_folder.mkdir(exist_ok=True)
db_path = db_folder / "threadsync.db"
Connection = sqlite3.connect(database=str(db_path), check_same_thread=False)
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
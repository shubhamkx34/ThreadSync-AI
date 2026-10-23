# Workflow :
from langgraph.graph import START , StateGraph , END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()
llmModel = ChatGroq(model='openai/gpt-oss-120b')

from langgraph.graph.message import add_messages

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

def chat_node(state: ChatState):
    messages = state['messages']
    response = llmModel.invoke(messages)

    return {'messages':[response]}

# Memory Persistence: A MemorySaver checkpointer (an in-memory storage tool) is attached to the compiled graph, allowing the backend to save and recall conversation snapshots.
checkpoint = MemorySaver()

graph = StateGraph(ChatState)

graph.add_node('chat_node',chat_node)

graph.add_edge(START,'chat_node')
graph.add_edge('chat_node', END)

chatbot = graph.compile(checkpointer=checkpoint)    
#
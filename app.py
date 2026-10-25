from threadsync_backend import chatbot, get_all_threads
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, AIMessageChunk, ToolMessage
import streamlit as st
import uuid                               

# To generate unique identification label for a chat session/Thread.
def generate_thread_id():
    return str(uuid.uuid4())

# Add new thread ID to chat session and store it in chat_threads dictionary
def add_thread(thread_id):
    if thread_id not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].append(thread_id)

def reset_chat():
    st.session_state["thread_id"]= generate_thread_id()
    st.session_state['message_history'] = [] 
    add_thread(st.session_state["thread_id"])

# Load a previous conversation from the LangGraph checkpointer
def load_conversation(thread_id):
    # Get the saved state for the selected thread
    state = chatbot.get_state(
       config = {"configurable": {"thread_id": thread_id }})       
    return state.values.get("messages", [])

st.title("ThreadSync-AI")

# thread_id generation if not available in session state 
if 'thread_id' not in st.session_state:
    st.session_state['thread_id'] = generate_thread_id()

# Chat Threads saving Dictionary
if 'chat_threads' not in st.session_state:
    st.session_state['chat_threads'] = get_all_threads()

# Add the current thread to the conversation list
add_thread(st.session_state["thread_id"])

# Message-History saving Dictionary
if 'message_history' not in st.session_state:
    st.session_state['message_history'] = [] 


# ========================= Sidebar threading feature =========================
st.sidebar.title("Chat Sessions")
if st.sidebar.button("New Chat"):
    reset_chat()
    st.rerun()     


# Display all conversation threads in reverse order
# This shows the newest conversation first
for thread_id in st.session_state["chat_threads"][::-1]:

    # Create one sidebar button for every conversation
    if st.sidebar.button(
        str(thread_id),
        key=thread_id
    ):
        # Set the selected thread as the current thread
        st.session_state["thread_id"] = thread_id

        # Load the messages saved under the selected thread
        messages = load_conversation(thread_id)

        # Temporary list for converting LangChain messages
        # into Streamlit's required message format
        temp_messages = []

        # Loop through all saved messages
        for message in messages:

            # Check whether the message was sent by the user
            if isinstance(message, HumanMessage):
                role = "user"

            # Check whether the message was sent by the AI
            elif isinstance(message, AIMessage):
                role = "assistant"

            # Ignore other message types, such as ToolMessage
            else:
                continue

            # Convert the LangChain message into a dictionary
            temp_messages.append({
                "role": role,
                "content": message.content
            })


        # Replace the current UI history with the selected conversation
        st.session_state["message_history"] = temp_messages

        # Rerun the application to display the loaded messages
        st.rerun()


# ========================= Main chat interface =========================
# Loading Conversation History in the Chat Interface
for message in st.session_state['message_history']:
    with st.chat_message(message['role']):    # Identify icon using roles like user, assitant
        st.text(message['content'])           # Text history

user_input = st.chat_input('Type Here :')

config = {"configurable": {"thread_id": st.session_state["thread_id"]},
          "metadata":{
              "thread_id":st.session_state["thread_id"]
          },
          "run_name":"Threadsync_Trace"
          }

if user_input:

    # first add the message to the message_history[USER]
    st.session_state['message_history'].append({'role':'user', 'content':user_input})
    with st.chat_message('user'):    # User icon
        st.text(user_input)     # Creates Visual Box of User text 

# Assistant Code with Streaming Feature :
  
    with st.chat_message("assistant"):
           # Use a mutable holder so the generator can set/modify it
           status_holder = {"box": None}
   
           def ai_only_stream():
               for message_chunk, metadata in chatbot.stream(
                   {"messages": [HumanMessage(content=user_input)]},
                   config=config,
                   stream_mode="messages",
               ):
                   # Lazily create & update the SAME status container when any tool runs
                   if isinstance(message_chunk, ToolMessage):
                       tool_name = getattr(message_chunk, "name", "tool")
                       if status_holder["box"] is None:
                           status_holder["box"] = st.status(
                               f"⚙️ Using `{tool_name}` …", expanded=True
                           )
                       else:
                           status_holder["box"].update(
                               label=f"⚙️ Using `{tool_name}` …",
                               state="running",
                               expanded=True,
                           )
   
                   # Stream ONLY assistant tokens
                   if isinstance(message_chunk, AIMessage):
                       yield message_chunk.content
   
           ai_message = st.write_stream(ai_only_stream())
   
           # Finalize only if a tool was actually used
           if status_holder["box"] is not None:
               status_holder["box"].update(
                   label="Tool finished ⚡", state="complete", expanded=False
               )
   
   
    st.session_state['message_history'].append({'role':'assistant', 'content':ai_message}) # add the message to the message_history[AI]

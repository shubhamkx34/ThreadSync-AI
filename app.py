from threadsync_backend import chatbot, get_all_threads
from rag_feature import ingest_rag_document
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, AIMessageChunk, ToolMessage
import streamlit as st
import uuid                
import os
import tempfile               

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
        last_tool = None

        for message in messages:
            if isinstance(message, HumanMessage):
                temp_messages.append({"role": "user", "content": message.content})
            elif isinstance(message, ToolMessage):
                last_tool = getattr(message, "name", "tool")
            elif isinstance(message, AIMessage):
                if not message.content:
                    continue
                content = message.content
                if last_tool:
                    content = f"⚙️ *Using {last_tool}...*\n\n{content}"
                    last_tool = None
                temp_messages.append({"role": "assistant", "content": content}) 


        # Replace the current UI history with the selected conversation
        st.session_state["message_history"] = temp_messages

        # Rerun the application to display the loaded messages
        st.rerun()


# ========================= Main chat interface =========================
# Loading Conversation History in the Chat Interface

for message in st.session_state['message_history']:
    if message['content']:
        with st.chat_message(message['role']):
            st.markdown(message['content'])

user_input = None

config = {
    "configurable": {"thread_id": st.session_state["thread_id"]},
    "metadata": {"thread_id": st.session_state["thread_id"]},
    "run_name": "Threadsync_Trace"
}

# ========================= Fixed chat input with PDF upload ========================
# accept_file=True adds the attachment button inside the chat input.
# file_type=["pdf"] allows PDF files only.
submission = st.chat_input(
    "Type here",
    accept_file=True,
    file_type=["pdf"]
)

if submission:
    user_input = submission.text

    if submission.files:
        uploaded_pdf = submission.files[0]

        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
                f.write(uploaded_pdf.getvalue())
                pdf_path = f.name

            with st.spinner(f"Processing {uploaded_pdf.name}..."):
                ingest_rag_document(pdf_path)

            st.toast(f"{uploaded_pdf.name} processed successfully.", icon="✔️")

        except Exception as error:
            st.error(f"PDF processing failed: {error}")

        finally:
            if "pdf_path" in locals() and os.path.exists(pdf_path):
                os.remove(pdf_path)

if user_input:
    st.session_state['message_history'].append({'role': 'user', 'content': user_input})
    with st.chat_message('user'):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        def ai_only_stream():
            for message_chunk, metadata in chatbot.stream(
                {"messages": [HumanMessage(content=user_input)]},
                config=config,
                stream_mode="messages",
            ):
                # Yield the tool name directly into the stream so st.write_stream captures and saves it
                if isinstance(message_chunk, ToolMessage):
                    tool_name = getattr(message_chunk, "name", "tool")
                    yield f"⚙️ *Using {tool_name}...*\n\n"

                # Only yield AI text if it is not empty
                if isinstance(message_chunk, AIMessage) and message_chunk.content:
                    yield message_chunk.content

        ai_message = st.write_stream(ai_only_stream())

    st.session_state['message_history'].append({'role': 'assistant', 'content': ai_message})
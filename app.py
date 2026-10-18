from chatbot_backend import chatbot
from langchain_core.messages import BaseMessage, HumanMessage
import streamlit as st

st.title("ThreadSync-AI")

config = {'configurable': {'thread_id': '1'}}

# Session-State to retain The history of Conversation btw user and assistant by creating Dictionary
if 'message_history' not in st.session_state:
    st.session_state['message_history'] = [] 

# Loading Conversation History
for message in st.session_state['message_history']:
    with st.chat_message(message['role']):    # Identify icon using roles like user, assitant
        st.text(message['content'])           # Text history

user_input = st.chat_input('Type Here :')

if user_input:

    # first add the message to the message_history
    st.session_state['message_history'].append({'role':'user', 'content':user_input})
    with st.chat_message('user'):    # User icon
        st.text(user_input)     # User text

# Assistant Code with Streaming Feature :
    with st.chat_message('assistant'):    #Assistant icon
          ai_message = st.write_stream(
              message_chunk.content for message_chunk, metadata in chatbot.stream(
                   {'messages': [HumanMessage(content=user_input)]},
                      config=config,
                      stream_mode='messages'
                  )      
          )
    st.session_state['message_history'].append({'role':'assistant', 'content':ai_message}) # add the message to the message_history

# ThreadSync AI

**A stateful, tool-using AI assistant built with LangGraph, LangChain and Streamlit.** ThreadSync AI brings multi-conversation chat, PDF question-answering (RAG), live information tools, and human approval for simulated stock orders into one application.

It is designed to demonstrate practical agentic AI patterns: routing requests to tools, maintaining conversation state across sessions, retrieving information from uploaded documents, and pausing a workflow when a human decision is required.

## Table of contents

- [Project overview](#project-overview)
- [Key features](#key-features)
- [How it works](#how-it-works)
- [Technology stack](#technology-stack)
- [Repository structure](#repository-structure)
- [Getting started](#getting-started)
- [Environment variables](#environment-variables)
- [Using the application](#using-the-application)
- [Project scope and notes](#project-scope-and-notes)

## Project overview

Many chat applications answer questions from a single prompt but do not combine durable conversation state, document retrieval, external tools, and human review in one workflow. ThreadSync AI explores how these capabilities can work together in a conversational application.

### What it can do

- Keep separate chat threads and restore their saved conversation state.
- Search the web for current or recent information.
- Retrieve stock quotes and current weather through external APIs.
- Accept a PDF in the chat interface and retrieve relevant passages to answer document-related questions.
- Pause a simulated stock-purchase action and ask the user to approve or decline it.
- Stream assistant output and expose tool activity in the chat interface.
- Reduce the message history sent to the language model with LangChain's `trim_messages` utility.
- Optionally send traces to LangSmith for workflow inspection and debugging.

## Key features

### 1. Multi-thread conversations and persistence

The Streamlit sidebar lets users start a new chat or switch to a previous thread. LangGraph's SQLite checkpointer stores graph state in the local `threadsync.db` database, allowing saved state to be loaded by thread ID.

### 2. Tool-using AI assistant

The LangGraph agent can answer general questions directly or call a tool when the request needs external or document-specific information.

| Tool | Purpose | Integration |
|---|---|---|
| `search_tool` | Search the web for current or recent information | Tavily |
| `get_stock_price` | Request a stock quote | Alpha Vantage |
| `get_weather_data` | Request current weather for a city | Weatherstack |
| `rag_tool` | Retrieve relevant passages from the indexed PDF | FAISS and OpenRouter embeddings |
| `purchase_stock` | Simulate a stock purchase after a human decision | LangGraph Human-in-the-Loop |

### 3. File question-answering with RAG

Files can be uploaded through the chat input. The application loads the document, splits its text into overlapping chunks, creates embeddings through OpenRouter, and stores them in a local FAISS index. When a document question is asked, `rag_tool` retrieves relevant chunks and returns the content, including source and page metadata where available, for the assistant to use.

### 4. Human-in-the-Loop (HITL)

The `purchase_stock` tool uses LangGraph's `interrupt()` mechanism to pause the workflow before confirming a simulated order. The Streamlit interface presents **Approve** and **Decline** controls; the chosen decision resumes the graph through `Command(resume=...)`.

> **Safety note:** The stock-purchase feature is a simulation of actual working of HITL. It does not execute real trades.

### 5. Context management and observability

The backend trims the messages supplied to the model to a configured token budget. LangSmith tracing can also be enabled through environment variables to inspect runs and help debug tool calls.

## How it works

```text
                         User
                          |
                          v
                 Streamlit application
                       app.py
                          |
                          v
                 LangGraph StateGraph
              (SQLite checkpoint per thread)
                          |
                          v
                     chat_node
                    (Groq LLM)
                          |
              Does the response request a tool?
                     /           \
                   No             Yes
                   |               |
                   v               v
                  END           ToolNode
                                   |
                 +-----------------+-----------------+
                 |                 |                 |
                 v                 v                 v
             Tavily /          FAISS RAG        Stock purchase
           stock / weather     PDF retrieval       interrupt
                 |                 |                 |
                 +-----------------+           Human decision
                                   |                 |
                                   +------<----------+
                          |
                          v
                Tool result returned to LLM
                          |
                          v
                 Final answer in Streamlit
```

In the normal tool cycle, the graph runs the requested tool and returns its result to the LLM to formulate a final answer. For the simulated stock-purchase tool, the graph pauses for approval before it continues.

## Technology stack

| Area | Technologies |
|---|---|
| User interface | Streamlit |
| Agent workflow and tool routing | LangGraph, LangChain, LangChain Core |
| Chat model | Groq — `openai/gpt-oss-120b` |
| Embeddings | OpenRouter — `nvidia/nemotron-3-embed-1b:free` |
| PDF loading and splitting | PyPDF, `RecursiveCharacterTextSplitter` |
| Vector search | FAISS (CPU) |
| Conversation checkpointing | SQLite, `langgraph-checkpoint-sqlite` |
| Web search | Tavily |
| Stock quotes | Alpha Vantage |
| Weather | Weatherstack |
| Optional tracing | LangSmith |

## Repository structure

```text
ThreadSync-AI/
├── app.py                         # Streamlit interface, chat threads, streaming, PDF upload and HITL controls
├── threadsync_backend.py          # LangGraph StateGraph, LLM setup, tool routing and SQLite checkpointer
├── tools_feature.py               # Tavily search, stock quote and weather tools
├── rag_feature.py                 # PDF ingestion, embeddings, FAISS indexing and retrieval tool
├── HITL_feature.py                # Simulated stock-purchase tool with approval interrupt
├── Workflows/
│   ├── HITL_Workflow.ipynb        # HITL workflow exploration
│   ├── LangGraph Workflow.ipynb    # LangGraph workflow exploration
│   └── Rag_Workflow.ipynb         # RAG workflow exploration
├── Res.pdf                        # PDF included with the project
├── requirements.txt               # Python package dependencies
├── .gitignore                     # Excludes local secrets, databases and generated files
└── README.md
```

## Getting started

### Prerequisites

- Python 3.11
- Git

### 1. Clone the repository

```bash
git clone https://github.com/shubhamkx34/ThreadSync-AI.git
cd ThreadSync-AI
```

### 2. Create and Activate a Virtual Environment
```bash
# Using Conda
conda create -n langagent python=3.11 -y
conda activate langagent

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```


### 4. Start the application

```bash
streamlit run app.py
```



### 5. Configure Environment Variables :
Create a `.env` file in the root directory and add your API keys:

```env
# LLM & Embeddings
GROQ_API_KEY="your_groq_api_key"
OPENROUTER_API_KEY="your_openrouter_api_key"
OPENAI_API_KEY="your_openai_api_key"

# External Tools
TAVILY_API_KEY="your_tavily_api_key"
WEATHERSTACK_API_KEY="your_weather_api_key"
ALPHA_VANTAGE_API_KEY="your_alpha_vantage_api_key"

# LangSmith Observability (Optional)
LANGSMITH_TRACING="true"
LANGSMITH_ENDPOINT="https://api.smith.langchain.com"
LANGSMITH_API_KEY="your_langsmith_api_key"
LANGSMITH_PROJECT="ThreadSync-Agentic-AI"
```

## Using the application

1. **Start a conversation.** Enter a question in the chat input. The assistant answers directly or routes the request to a suitable tool.
2. **Use chat threads.** Select **New Chat** to create another conversation, or choose a saved thread from the sidebar to load its state.
3. **Search the web.** Ask for current or recent information to invoke the Tavily search tool.
4. **Ask for stock or weather information.** Provide a stock symbol or city as part of your request.
5. **Ask questions about a PDF.** Attach a PDF using the file control in the chat input, then ask a question about its content. The document is indexed for retrieval.
6. **Review a simulated stock order.** When the assistant invokes `purchase_stock`, inspect the approval prompt and select **Approve (Yes)** or **Decline (No)** to resume the workflow.

## Project scope and notes

- This project demonstrates agentic workflow patterns and integrations; it should not be described as a production financial system.
- External data depends on the availability, API limits, and credentials of Tavily, Alpha Vantage, Weatherstack, Groq, and OpenRouter.
- The SQLite conversation database and FAISS index are local generated files. Their persistence and lifecycle are separate from each other.
- The configured FAISS index is stored in `faiss_DB/`. Keep local generated files and API credentials out of version control.

---

**Built as a hands-on project exploring conversational AI, LangGraph orchestration, RAG, tool calling, state persistence, and human-in-the-loop workflows.**

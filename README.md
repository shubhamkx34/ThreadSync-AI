# ThreadSync AI

**A stateful, multi-model, tool-using AI assistant built with LangGraph, LangChain, FastAPI, and a custom HTML/CSS/JS frontend—fully containerized with Docker and deployed on AWS EC2 and Render via GitHub Actions CI/CD**. ThreadSync AI brings multi-conversation chat, dynamic LLM switching (Groq and Google Gemini), question-answering (RAG) by attaching source files, live information tools, and human approval for simulated stock orders into one production-ready application.

It is designed to demonstrate practical agentic AI patterns: routing requests to tools, dynamically switching between LLM providers mid-conversation, maintaining conversation state across sessions, retrieving information from uploaded documents, pausing a workflow when a human decision is required, and automating cloud deployments.

## Table of contents

- [Project overview](#project-overview)
- [Key features](#key-features)
- [How it works](#how-it-works)
- [Technology stack](#technology-stack)
- [Repository structure](#repository-structure)
- [Getting started (Local Setup)](#getting-started-local-setup)
- [Environment variables](#environment-variables)
- [Docker & CI/CD Deployment (AWS EC2 & Render)](#docker--cicd-deployment-aws-ec2--render)
- [Using the application](#using-the-application)
- [Project scope and notes](#project-scope-and-notes)

## Project overview

Many chat applications answer questions from a single prompt but do not combine durable conversation state, multi-model flexibility, document retrieval, external tools, and human review in one decoupled workflow. ThreadSync AI explores how these capabilities work together using a FastAPI microservice backend paired with a lightweight, dependency-free HTML/JS frontend, shipped via an automated DevOps pipeline.

### What it can do

- Keep separate chat threads and restore their saved conversation state from SQLite.
- Dynamically switch between Groq (`openai/gpt-oss-120b`, `openai/gpt-oss-20b`) and Google Gemini (`gemini-3.6-flash`, `gemini-3.5-flash-lite`, `gemini-3.1-flash-lite`) models directly from the UI.
- Sanitize message metadata across providers so mid-conversation model switching never causes schema errors.
- Search the web for current or recent information.
- Retrieve stock quotes and current weather through external APIs.
- Accept a PDF in the chat interface and retrieve relevant passages to answer document-related questions.
- Pause a simulated stock-purchase action and ask the user to approve or decline it.
- Stream assistant output in real time via Server-Sent Events (SSE) and expose tool activity in the chat interface.
- Reduce the message history sent to the language model with LangChain's `trim_messages` utility.
- Optionally send traces to LangSmith for workflow inspection and debugging.
- Automatically build, push, and deploy updates to an AWS EC2 instance using Docker and GitHub Actions.

## Key features

### 1. Decoupled FastAPI + HTML/JS architecture & persistence

The custom HTML/CSS/JS sidebar lets users start a new chat or switch to a previous thread without full-page reloads. FastAPI serves the frontend and exposes REST and Server-Sent Events (SSE) endpoints (`/chat/stream`, `/chat/resume`, `/upload`, `/threads`, `/history/{thread_id}`). LangGraph's SQLite checkpointer stores graph state in the local `Data/threadsync.db` database, allowing saved state to be loaded by thread ID.

### 2. Dynamic multi-model orchestration

Users can select their preferred model from the composer dropdown. The selection is passed via LangGraph's `RunnableConfig` into `chat_node`, which dynamically initializes either `ChatGroq` or `ChatGoogleGenerativeAI` and binds the tool suite on the fly. A universal message sanitization loop strips provider-specific metadata so threads can seamlessly alternate between Groq and Gemini models.

### 3. Tool-using AI assistant

The LangGraph agent can answer general questions directly or call a tool when the request needs external or document-specific information.

| Tool | Purpose | Integration |
|---|---|---|
| `search_tool` | Search the web for current or recent information | Tavily |
| `get_stock_price` | Request a stock quote | Alpha Vantage |
| `get_weather_data` | Request current weather for a city | Weatherstack |
| `rag_tool` | Retrieve relevant passages from the indexed PDF | FAISS and OpenRouter embeddings |
| `purchase_stock` | Simulate a stock purchase after a human decision | LangGraph Human-in-the-Loop |

### 4. File question-answering with RAG

Files can be uploaded through the `+` attachment button in the chat composer and sent to the `/upload` endpoint. The application loads the PDF, splits its text into overlapping chunks, creates embeddings through OpenRouter, and stores them in a local FAISS index. When a document question is asked, `rag_tool` retrieves relevant chunks and returns the content for the assistant to synthesize.

### 5. Human-in-the-Loop (HITL)

The `purchase_stock` tool uses LangGraph's `interrupt()` mechanism to pause the workflow before confirming a simulated order[cite: 28]. The frontend renders an interactive **Action Required** card with **Approve (Yes)** and **Decline (No)** controls; the chosen decision hits `/chat/resume` and resumes the graph through `Command(resume=...)`.

> **Safety note:** The stock-purchase feature is a simulation of the actual working of HITL. It does not execute real trades.

### 6. Context management and observability

The backend trims the messages supplied to the model to a 3,000-token budget using `trim_messages`. LangSmith tracing can also be enabled through environment variables to inspect runs and debug tool calls.

### 7. Automated Cloud Deployment (CI/CD)

The repository includes a production-optimized `Dockerfile` exposing port `8000` and a GitHub Actions workflow (`.github/workflows/cicd.yaml`) that automates Continuous Integration, Docker Hub image publishing, and Continuous Deployment onto a self-hosted AWS EC2 runner with automated health checks.

## How it works

```text
                         User
                          |
                          v
             Custom HTML/CSS/JS Frontend
                 (Frontend/index.html)
                          |
               FastAPI Backend (main.py:8000)
                          |
                          v
                 LangGraph StateGraph
              (SQLite checkpoint per thread)
                          |
                          v
                     chat_node
         (Dynamic Routing: Groq / Google Gemini)
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
                                   |           (/chat/resume)
                                   +------<----------+
                          |
                          v
                Tool result returned to LLM
                          |
                          v
               SSE Stream to Web UI
```

## Technology stack

| Area | Technologies |
|---|---|
| Frontend UI | HTML5, Modern CSS3, Vanilla JavaScript (SSE Streaming, Custom Markdown Parser) |
| Backend API Server | FastAPI, Uvicorn, Jinja2Templates |
| Agent workflow and tool routing | LangGraph, LangChain, LangChain Core |
| Chat models (Multi-LLM) | Groq (`openai/gpt-oss-120b`, `openai/gpt-oss-20b`), Google Gemini (`gemini-3.6-flash`, `gemini-3.5-flash-lite`, `gemini-3.1-flash-lite`) |
| Embeddings | OpenRouter — `nvidia/nemotron-3-embed-1b` |
| PDF loading and splitting | PyPDF, `RecursiveCharacterTextSplitter` |
| Vector search | FAISS (CPU)|
| Conversation checkpointing | SQLite (`Data/threadsync.db`), `langgraph-checkpoint-sqlite` |
| Web search | Tavily |
| Stock quotes | Alpha Vantage |
| Weather | Weatherstack |
| Observability & Tracing | LangSmith |
| DevOps & Deployment | Docker, Docker Hub, GitHub Actions (CI/CD), AWS EC2, Render |

## Repository structure

```text
ThreadSync-AI/
├── .github/
│   └── workflows/
│       └── cicd.yaml              # GitHub Actions CI/CD pipeline (Build, Push, Deploy to EC2)
├── Frontend/
│   └── index.html                 # Single-page HTML/CSS/JS chat interface with SSE streaming & model picker
├── main.py                        # FastAPI server, SSE streaming endpoints, PDF upload & thread history routes
├── agent.py                       # LangGraph StateGraph, dynamic LLM routing (Groq/Gemini) & SQLite checkpointer
├── tools_feature.py               # Tavily search, stock quote and weather tools
├── rag_feature.py                 # PDF ingestion, embeddings, FAISS indexing and retrieval tool
├── HITL_feature.py                # Simulated stock-purchase tool with approval interrupt
├── app_streamlit.py               # Legacy Streamlit UI (retained for reference)
├── Workflows/
│   ├── HITL_Workflow.ipynb        # HITL workflow exploration
│   ├── LangGraph Workflow.ipynb   # LangGraph workflow exploration
│   └── Rag_Workflow.ipynb         # RAG workflow exploration
├── Dockerfile                     # Production container definition for FastAPI/Uvicorn & FAISS (Port 8000)
├── CICD.md                        # AWS EC2 & GitHub Actions self-hosted runner setup guide
├── .dockerignore                  # Excludes unnecessary local files from Docker build context
├── requirements.txt               # Python package dependencies
├── .gitignore                     # Excludes local secrets, databases and generated files
└── README.md
```

## Getting started (Local Setup)

### Prerequisites

- Python 3.17
- Git

### 1. Clone the repository

```bash
git clone [https://github.com/shubhamkx34/ThreadSync-AI.git](https://github.com/shubhamkx34/ThreadSync-AI.git)
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
### 4. Configure Environment Variables

Create a `.env` file in the root directory (see [Environment variables](#environment-variables) below).

### 5. Start the application

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```
Open `http://localhost:8000` in your browser.

## Environment variables

Create a `.env` file in the root directory for local development, or configure these keys in **GitHub Secrets** (`Settings > Secrets and variables > Actions`) for CI/CD deployment:

```env
# LLM & Embeddings
GROQ_API_KEY="your_groq_api_key"
GOOGLE_API_KEY="your_google_gemini_api_key"
OPENROUTER_API_KEY="your_openrouter_api_key"
OPENAI_API_KEY="your_openai_api_key"

# External Tools
TAVILY_API_KEY="your_tavily_api_key"
WEATHERSTACK_API_KEY="your_weatherstack_api_key"
ALPHA_VANTAGE_API_KEY="your_alpha_vantage_api_key"

# LangSmith Observability
LANGSMITH_TRACING="true"
LANGSMITH_ENDPOINT="[https://api.smith.langchain.com](https://api.smith.langchain.com)"
LANGSMITH_API_KEY="your_langsmith_api_key"
LANGSMITH_AGENT_ID="your_langsmith_agent_id"
LANGSMITH_AGENT_ENVIRONMENT="your_langsmith_environment"
```

## Docker & CI/CD Deployment (AWS EC2 & Render)

### Running locally with Docker

```bash
# Build the Docker image
docker build -t threadsync-ai .

# Run the container on port 8000 using your local .env file
docker run -d --name stapp -p 8000:8000 --env-file .env threadsync-ai
```

### Automated CI/CD Pipeline (AWS EC2)

Every push to the `main` branch triggers the GitHub Actions workflow (`.github/workflows/cicd.yaml`):

1. **Continuous Integration (`integration`):** Checks out the code and executes linting and unit test checks.
2. **Build & Push (`push_to_registry`):** Uses Docker Buildx with GitHub Actions layer caching to build the `python:3.11-slim` image and pushes `:main` and `:sha` tags to **Docker Hub**.
3. **Continuous Deployment (`continuous_deployment`):** Runs on a **self-hosted AWS EC2 runner** (`us-east-1`), pulls the latest image from Docker Hub, injects application secrets (`GROQ_API_KEY`, `GOOGLE_API_KEY`, etc.) as environment variables, restarts the container on port `8000`, and verifies the FastAPI `/` health endpoint.

**Additional GitHub Secrets required for CI/CD:**
- `DOCKER_USERNAME` — Your Docker Hub username
- `DOCKER_PASSWORD` — Your Docker Hub access token
- `IMAGE_NAME` — Target repository/image name (e.g., `threadsync-ai`)

### Render Deployment

To deploy on Render as a Web Service:
- **Build Command:** `pip install -r requirements.txt`
- **Start Command:** `uvicorn main:app --host 0.0.0.0 --port \$PORT`

## Using the application

1. **Select a model.** Use the model picker in the bottom-right corner of the typing bar to switch between Groq (`GPT-OSS 120B`, `GPT-OSS 20B`) and Google (`Gemini 3.6 Flash`, `Gemini 3.5 Flash-Lite`, `Gemini 3.1 Flash-Lite`) models at any point.
2. **Start a conversation.** Enter a question in the chat input. The assistant answers directly or routes the request to a suitable tool.
3. **Use chat threads.** Select **New chat** to create another conversation, or choose a saved thread from the **Recent** list in the sidebar to load its state.
4. **Search the web.** Ask for current or recent information to invoke the Tavily search tool.
5. **Ask for stock or weather information.** Provide a stock symbol as part of your request.
6. **Ask questions about a PDF.** Click the `+` button in the typing bar to attach a PDF, then ask a question about its content. The document is indexed into FAISS for retrieval.
7. **Review a simulated stock order.** When the assistant invokes `purchase_stock`, inspect the approval card and select **Approve (Yes)** or **Decline (No)** to resume the workflow.

## Project scope and notes

- This project demonstrates agentic workflow patterns, multi-LLM routing, tool integrations, decoupled FastAPI/JS architecture, and cloud CI/CD deployment; it is not a production financial trading system.
- External data depends on the availability, API limits, and credentials of Tavily, Alpha Vantage, Weatherstack, Groq, Google AI Studio, and OpenRouter.
- The SQLite conversation database (`Data/threadsync.db`), uploaded temporary PDFs (`uploads/`), and FAISS index (`faiss_DB/`) are generated runtime files and are excluded from version control and Docker build contexts via `.gitignore` and `.dockerignore`.

---

**Built as a hands-on project exploring conversational AI, LangGraph orchestration, multi-LLM routing, FastAPI SSE streaming, RAG, tool calling, state persistence, human-in-the-loop workflows, and automated Docker/AWS CI/CD deployment**.
import os
import re
import json
import uuid
import certifi
from pathlib import Path
from dotenv import load_dotenv

import uvicorn
from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.templating import Jinja2Templates

from langchain_core.messages import HumanMessage, AIMessage, AIMessageChunk, ToolMessage
from langgraph.types import Command

# Import your LangGraph backend functions
from agent import chatbot, checkpoint, get_all_threads
from rag_feature import ingest_rag_document

# Load environment variables and set SSL certs for external APIs
load_dotenv()
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

app = FastAPI()

# Automatically detect templates or Frontend directory
TEMPLATE_DIR = "templates" if Path("templates").exists() else ("Frontend" if Path("Frontend").exists() else "templates")
Path(TEMPLATE_DIR).mkdir(exist_ok=True)
templates = Jinja2Templates(directory=TEMPLATE_DIR)

# Ensure required directories exist for database and file uploads
Path("uploads").mkdir(exist_ok=True)
Path("Data").mkdir(exist_ok=True)

# Regex to strip PDF system notes from chat history so they don't clutter the UI
PDF_NOTE_RE = re.compile(
    r"^\[System Note: A new PDF '(?P<name>.+?)' was just uploaded\..*?\]\s*",
    re.DOTALL,
)

def clean_user_text(text: str) -> str:
    match = PDF_NOTE_RE.match(text)
    if not match:
        return text
    rest = text[match.end():].strip()
    return rest or f"Uploaded `{match.group('name')}`"


def threads_newest_first() -> list:
    """Read the checkpointer directly (newest first) and extract first user message as title."""
    try:
        seen, ordered = set(), []
        for ckpt in checkpoint.list(None):
            tid = ckpt.config["configurable"]["thread_id"]
            if tid not in seen:
                seen.add(tid)
                ordered.append(tid)
        return ordered
    except Exception:
        return list(get_all_threads())


def get_thread_title(thread_id: str):
    """First user message of the thread; None for empty threads."""
    try:
        state = chatbot.get_state(config={"configurable": {"thread_id": thread_id}})
        for message in state.values.get("messages", []):
            if isinstance(message, HumanMessage):
                return clean_user_text(message.content)
    except Exception:
        pass
    return None


@app.get("/")
async def home(request: Request):
    """Serves the main frontend UI."""
    return templates.TemplateResponse(request=request, name="index.html", context={})


@app.get("/threads")
async def get_threads():
    """Returns the ordered list of non-empty chat threads with their titles for the sidebar."""
    items = []
    for tid in threads_newest_first():
        title = get_thread_title(tid)
        if title:
            items.append({"thread_id": tid, "title": title})
    return {"threads": items}


@app.get("/history/{thread_id}")
async def get_history(thread_id: str):
    """Fetches past messages and any active HITL interrupt for a specific thread."""
    config = {"configurable": {"thread_id": thread_id}}
    state = chatbot.get_state(config)
    messages = state.values.get("messages", [])
    
    ui_messages = []
    last_tool = None
    
    for msg in messages:
        if isinstance(msg, HumanMessage):
            ui_messages.append({"role": "user", "content": clean_user_text(msg.content)})
        elif isinstance(msg, ToolMessage):
            last_tool = getattr(msg, "name", None) or "tool"
        elif isinstance(msg, AIMessage):
            if not msg.content:
                continue
            
            # Handle Gemini lists or Groq strings safely
            if isinstance(msg.content, list):
                content = "".join(part.get("text", "") if isinstance(part, dict) else str(part) for part in msg.content)
            else:
                content = str(msg.content)
                
            if not content.strip():
                continue
                
            if last_tool:
                content = f"⚙️ *Using {last_tool}...*\n\n{content}"
                last_tool = None
                
            ui_messages.append({"role": "assistant", "content": content})

    interrupt_prompt = None
    if state.tasks and state.tasks[0].interrupts:
        interrupt_prompt = state.tasks[0].interrupts[0].value
            
    return JSONResponse({"messages": ui_messages, "interrupt": interrupt_prompt})


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """Handles PDF uploads and ingests into FAISS."""
    try:
        if not file.filename.lower().endswith(".pdf"):
            return JSONResponse({"success": False, "message": "Only PDF files are supported."}, status_code=400)

        file_id = str(uuid.uuid4())
        safe_filename = file.filename.replace(" ", "_")
        file_path = f"uploads/{file_id}_{safe_filename}"

        with open(file_path, "wb") as f:
            f.write(await file.read())

        ingest_rag_document(file_path)
        
        if os.path.exists(file_path):
            os.remove(file_path)

        return JSONResponse({"success": True, "filename": file.filename})
    except Exception as e:
        return JSONResponse({"success": False, "message": str(e)}, status_code=500)


def sse_data(payload: dict) -> str:
    """Formats data for Server-Sent Events (streaming chunks)."""
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def stream_graph_execution(payload, config, thread_id):
    """Core generator that streams agent tokens and checks for HITL interrupts."""
    try:
        for chunk, metadata in chatbot.stream(payload, config=config, stream_mode="messages"):
            if isinstance(chunk, ToolMessage):
                tool_name = getattr(chunk, "name", "tool")
                yield sse_data({"token": f"⚙️ *Using {tool_name}...*\n\n"})
                continue

            if getattr(chunk, "tool_calls", None) or (isinstance(chunk, AIMessageChunk) and chunk.additional_kwargs.get("tool_calls")):
                continue

            if isinstance(chunk, (AIMessage, AIMessageChunk)) and chunk.content:
                if isinstance(chunk.content, list):
                    for part in chunk.content:
                        if isinstance(part, dict) and "text" in part:
                            yield sse_data({"token": part["text"]})
                        elif isinstance(part, str):
                            yield sse_data({"token": part})
                elif isinstance(chunk.content, str):
                    yield sse_data({"token": chunk.content})

        current_state = chatbot.get_state(config)
        if current_state.tasks and current_state.tasks[0].interrupts:
            prompt_text = current_state.tasks[0].interrupts[0].value
            yield sse_data({"interrupt": prompt_text, "thread_id": thread_id})
        else:
            yield sse_data({"done": True, "thread_id": thread_id})

    except Exception as e:
        yield sse_data({"error": str(e), "done": True})


@app.post("/chat/stream")
async def chat_stream(request: Request):
    """Initiates a new chat turn."""
    data = await request.json()
    user_message = data.get("message", "").strip()
    thread_id = data.get("thread_id") or str(uuid.uuid4())
    model_id = data.get("model", "openai/gpt-oss-120b")

    if not user_message:
        return JSONResponse({"error": "Message cannot be empty."}, status_code=400)

    config = {
        "configurable": {"thread_id": thread_id, "model_id": model_id},
        "metadata": {"thread_id": thread_id},
        "run_name": "Threadsync_Trace",
    }
    payload = {"messages": [HumanMessage(content=user_message)]}

    return StreamingResponse(
        stream_graph_execution(payload, config, thread_id),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
    )


@app.post("/chat/resume")
async def chat_resume(request: Request):
    """Resumes the graph after a HITL decision (Approve/Decline)."""
    data = await request.json()
    decision = data.get("decision")
    thread_id = data.get("thread_id")
    model_id = data.get("model", "openai/gpt-oss-120b")

    config = {
        "configurable": {"thread_id": thread_id, "model_id": model_id},
        "metadata": {"thread_id": thread_id},
        "run_name": "Threadsync_Trace",
    }
    payload = Command(resume=decision)

    return StreamingResponse(
        stream_graph_execution(payload, config, thread_id),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
    )


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
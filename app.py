import os
import re
import tempfile
import uuid
from datetime import datetime
import streamlit as st

# --------------------------------------------------------------------------
# Page config (must be the first Streamlit call)
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="ThreadSync AI — Autonomous Agent System",
    page_icon="🧩",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------
# Backend imports (fail with a readable message instead of a stack trace)
# --------------------------------------------------------------------------
try:
    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
    from langgraph.types import Command

    from rag_feature import ingest_rag_document
    # Added ALLOWED_MODELS and DEFAULT_MODEL to support model switching
    from agent import chatbot, checkpoint, get_all_threads, ALLOWED_MODELS, DEFAULT_MODEL 
except Exception as exc:  # noqa: BLE001
    st.error(
        f"Could not start the ThreadSync backend: `{exc}`\n\n"
        "Check that your `.env` has GROQ_API_KEY, OPENROUTER_API_KEY and TAVILY_API_KEY "
        "and that `pip install -r requirements.txt` has been run."
    )
    st.stop()

# --------------------------------------------------------------------------
# Design tokens
# --------------------------------------------------------------------------
BG = "#131314"
PANEL = "#1e1f20"
BORDER = "#2d2e30"
BORDER_STRONG = "#37393b"
HOVER = "#282a2c"
TEXT = "#e3e3e3"
MUTED = "#8e918f"
ACCENT = "#a8c7fa"

USER_AVATAR = ":material/person:"
AI_AVATAR = ":material/auto_awesome:"

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

html, body, [class*="css"], .stApp {{
    font-family: 'Plus Jakarta Sans', system-ui, sans-serif;
    background-color: {BG};
    color: {TEXT};
}}
header[data-testid="stHeader"] {{ background: transparent; }}
.block-container {{ max-width: 56rem; padding-top: 1.5rem; padding-bottom: 6rem; }}

/* Sidebar */
section[data-testid="stSidebar"] {{
    background-color: {PANEL};
    border-right: 1px solid {BORDER};
}}
section[data-testid="stSidebar"] .stButton > button {{
    width: 100%;
    justify-content: flex-start;
    background: {BG};
    color: {TEXT};
    border: 1px solid {BORDER_STRONG};
    border-radius: 6px;
    font-size: 0.75rem;
    font-weight: 500;
}}
section[data-testid="stSidebar"] .stButton > button:hover {{
    background: {HOVER};
    border-color: {BORDER_STRONG};
    color: {TEXT};
}}
section[data-testid="stSidebar"] .stButton > button p {{
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}}
.brand {{ display:flex; align-items:center; padding:.25rem .5rem; margin-bottom:1.5rem; }}
.brand span.name {{ color:#fff; font-weight:500; font-size:1rem; letter-spacing:-.01em; }}
.section-label {{
    padding: 0 .5rem .5rem; font-size: 11px; font-weight: 500;
    color: {MUTED}; text-transform: uppercase; letter-spacing: .08em;
}}
.empty-recent {{
    padding:.75rem; color:{MUTED}; font-size:.75rem; text-align:center;
    border:1px dashed {BORDER}; border-radius:6px;
}}

/* Top strip */
.topbar {{
    border-bottom: 1px solid #222426; padding: .25rem 0 .75rem;
    margin-bottom: 1rem;
}}
.topbar h1 {{
    font-size: 1.875rem; font-weight: 700; line-height: 1;
    letter-spacing: -.02em; color: {TEXT}; margin: 0; padding: 0;
}}

/* Hero */
.hero {{ padding: 3rem 0 1.5rem; }}
.hero.centered {{ padding-top: 22vh; }}
.hero h1 {{
    font-size: 2.25rem; font-weight: 600; letter-spacing: -.02em;
    background: linear-gradient(90deg, {ACCENT}, {TEXT}, {MUTED});
    -webkit-background-clip: text; background-clip: text;
    -webkit-text-fill-color: transparent; margin-bottom: .5rem;
}}
.hero p {{ color: {MUTED}; font-size: 1rem; }}

/* Chat */
[data-testid="stChatMessage"] {{ background: transparent; padding: .5rem 0; }}
.meta {{ font-size: .75rem; color: {MUTED}; margin-bottom: .15rem; }}
.disclaimer {{ text-align:center; font-size:11px; color:{MUTED}; }}

/* Chat input */
[data-testid="stChatInput"] {{
    background: {PANEL}; border: 1px solid {BORDER}; border-radius: 6px;
}}
[data-testid="stChatInput"]:focus-within {{ border-color: {ACCENT}; }}

/* Architecture dialog cards */
.card {{
    border: 1px solid {BORDER}; background: rgba(19,19,20,.5);
    border-radius: 8px; padding: .9rem 1rem; margin-bottom: .6rem;
}}
.card .title {{ font-weight: 500; color: {TEXT}; font-size: .8rem; }}
.card .desc {{ font-size: 11px; color: {MUTED}; margin-top: .25rem; }}
.pill {{
    font-family: 'JetBrains Mono', monospace; font-size: 11px; color: {MUTED};
    background: {PANEL}; border: 1px solid {BORDER};
    padding: .1rem .5rem; border-radius: 4px;
}}
.h-label {{
    font-size: 11px; font-weight: 600; color: {MUTED};
    text-transform: uppercase; letter-spacing: .08em; margin: .75rem 0 .5rem;
}}
.accent-label {{
    color: {ACCENT}; font-weight: 600; font-size: .75rem;
    letter-spacing: .05em; text-transform: uppercase;
}}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# --------------------------------------------------------------------------
# Thread helpers (LangGraph checkpointer <-> Streamlit session)
# --------------------------------------------------------------------------
# The agent receives a "[System Note: ...]" prefix when a PDF is uploaded.
# It is saved in the checkpoint, so strip it again when showing old chats.
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


def generate_thread_id() -> str:
    return str(uuid.uuid4())


def threads_newest_first() -> list:
    """get_all_threads() returns an unordered set, so read the checkpointer directly
    (it lists newest checkpoints first) and fall back to the backend helper."""
    try:
        seen, ordered = set(), []
        for ckpt in checkpoint.list(None):
            tid = ckpt.config["configurable"]["thread_id"]
            if tid not in seen:
                seen.add(tid)
                ordered.append(tid)
        return ordered
    except Exception:  # noqa: BLE001
        return list(get_all_threads())


def add_thread(thread_id: str) -> None:
    if thread_id not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].insert(0, thread_id)


def load_conversation(thread_id: str) -> list:
    state = chatbot.get_state(config={"configurable": {"thread_id": thread_id}})
    return state.values.get("messages", [])

def to_ui_messages(messages: list) -> list:
    """Convert saved LangChain messages into the dicts the UI renders."""
    ui, last_tool = [], None
    for message in messages:
        if isinstance(message, HumanMessage):
            ui.append({"role": "user", "content": clean_user_text(message.content), "ts": ""})
        elif isinstance(message, ToolMessage):
            last_tool = getattr(message, "name", None) or "tool"
        elif isinstance(message, AIMessage):
            if not message.content:
                continue
            
            # Handle Gemini's list formatting 
            if isinstance(message.content, list):
                content = "".join(
                    part.get("text", "") if isinstance(part, dict) else str(part)
                    for part in message.content
                )
            else:
                content = str(message.content)
                
            if not content.strip():
                continue
                
            if last_tool:
                content = f"⚙️ *Using {last_tool}...*\n\n{content}"
                last_tool = None
            ui.append({"role": "assistant", "content": content, "ts": ""})
    return ui


def thread_title(thread_id: str):
    """First user message of the thread (cached); None for empty threads."""
    titles = st.session_state["thread_titles"]
    if thread_id not in titles:
        title = None
        for message in load_conversation(thread_id):
            if isinstance(message, HumanMessage):
                title = clean_user_text(message.content)
                break
        titles[thread_id] = title
    return titles[thread_id]


def reset_chat() -> None:
    st.session_state["thread_id"] = generate_thread_id()
    st.session_state["message_history"] = []
    add_thread(st.session_state["thread_id"])


def switch_thread(thread_id: str) -> None:
    st.session_state["thread_id"] = thread_id
    st.session_state["message_history"] = to_ui_messages(load_conversation(thread_id))


# --------------------------------------------------------------------------
# Session state
# --------------------------------------------------------------------------
if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()
if "chat_threads" not in st.session_state:
    st.session_state["chat_threads"] = threads_newest_first()
if "thread_titles" not in st.session_state:
    st.session_state["thread_titles"] = {}
if "message_history" not in st.session_state:
    st.session_state["message_history"] = []
if "model_id" not in st.session_state:
    st.session_state["model_id"] = DEFAULT_MODEL

add_thread(st.session_state["thread_id"])


# --------------------------------------------------------------------------
# Agent streaming helpers
# --------------------------------------------------------------------------
def stream_agent(payload, config):
    """Yield text for st.write_stream: tool notices + AI tokens (no raw tool output)."""
    for chunk, _metadata in chatbot.stream(payload, config=config, stream_mode="messages"):
        if isinstance(chunk, ToolMessage):
            yield f"⚙️ *Using {getattr(chunk, 'name', None) or 'tool'}...*\n\n"
        elif isinstance(chunk, AIMessage) and chunk.content:
            if isinstance(chunk.content, str):
                yield chunk.content
            elif isinstance(chunk.content, list):
                for part in chunk.content:
                    if isinstance(part, dict) and "text" in part:
                        yield part["text"]
                    elif isinstance(part, str):
                        yield part


def render_message(role: str, content: str, ts: str = "") -> None:
    is_user = role == "user"
    who = "You" if is_user else "ThreadSync"
    with st.chat_message(role, avatar=USER_AVATAR if is_user else AI_AVATAR):
        st.markdown(f'<div class="meta">{who}{" · " + ts if ts else ""}</div>', unsafe_allow_html=True)
        st.markdown(content)


# --------------------------------------------------------------------------
# Architecture dialog
# --------------------------------------------------------------------------
@st.dialog("System Architecture & Agent Pipeline", width="large")
def architecture_dialog() -> None:
    st.markdown(
        """
        <div class="card">
          <div style="display:flex;justify-content:space-between;align-items:center;">
            <span class="accent-label">Core Agent Runtime</span>
            <span class="pill">LangGraph StateGraph</span>
          </div>
          <div class="desc" style="font-size:.75rem;margin-top:.5rem;">
            Multi-turn autonomous graph orchestration powered by checkpointed
            conversation memory (SQLite per session thread). Supports cyclical
            reflection, state persistence, and branch execution.
          </div>
        </div>
        <div class="h-label">Equipped Tools &amp; Routing</div>
        """,
        unsafe_allow_html=True,
    )

    tools = [
        (" Tavily Web Search", "Real-time web crawling and context extraction for breaking information."),
        (" Alpha Vantage Market Data", "Real-time equity quotes, historical indicators, and valuation series."),
        (" Weatherstack Telemetry", "Global meteorological telemetry and historical atmospheric data."),
        (" FAISS Vector RAG", "In-memory similarity retrieval over attached PDFs and corporate filings."),
    ]
    cols = st.columns(2)
    for i, (title, desc) in enumerate(tools):
        with cols[i % 2]:
            st.markdown(
                f'<div class="card"><div class="title">{title}</div>'
                f'<div class="desc">{desc}</div></div>',
                unsafe_allow_html=True,
            )

    st.markdown(
        """
        <div class="card" style="border-color:#37393b;background:rgba(19,19,20,.6);">
          <div class="accent-label" style="text-transform:none;">
             Human-in-the-Loop (HITL) Safety Gate
          </div>
          <div class="desc">
            Equipped with execution interrupt triggers on destructive actions
            (e.g., simulated stock purchases). Halts agent workflow at checkpoint
            for user authorization before resuming transaction state.
          </div>
        </div>
        <div class="h-label">CI/CD &amp; Infrastructure</div>
        <div class="card" style="display:flex;justify-content:space-between;align-items:center;">
          <span style="font-size:.8rem;">🖥️ Docker Containerization on AWS EC2</span>
          <span class="pill" style="border:none;background:none;">via GitHub Actions CI/CD</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.button("Dismiss", key="dismiss_arch"):
        st.rerun()


# --------------------------------------------------------------------------
# Sidebar: new chat + saved threads from the SQLite checkpointer
# --------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        '<div class="brand"><span class="name">ThreadSync chats</span></div>',
        unsafe_allow_html=True,
    )

    if st.button("New chat", icon=":material/add:", key="btn_new_chat", use_container_width=True):
        reset_chat()
        st.rerun()

    # --- Model Selection UI ---
    st.markdown('<div class="section-label" style="padding-top:1.5rem;">Model Selection</div>', unsafe_allow_html=True)
    
    model_options = list(ALLOWED_MODELS.keys())
    current_model = st.session_state.get("model_id", DEFAULT_MODEL)
    
    try:
        current_index = list(ALLOWED_MODELS.values()).index(current_model)
    except ValueError:
        current_index = 0
        
    selected_model_label = st.selectbox(
        "Choose a model",
        options=model_options,
        index=current_index,
        label_visibility="collapsed"
    )
    
    # Store the exact model ID expected by LangGraph backend
    if ALLOWED_MODELS[selected_model_label] != current_model:
        st.session_state["model_id"] = ALLOWED_MODELS[selected_model_label]

    st.markdown('<div class="section-label" style="padding-top:1.5rem;">Recent</div>', unsafe_allow_html=True)

    shown = 0
    for tid in st.session_state["chat_threads"]:
        title = thread_title(tid)
        if not title:  # skip empty threads (e.g. the brand-new current one)
            continue
        shown += 1
        label = title if len(title) <= 34 else title[:33] + "…"
        is_active = tid == st.session_state["thread_id"]
        if st.button(
            label,
            key=f"thread_{tid}",
            icon=":material/radio_button_checked:" if is_active else ":material/chat_bubble:",
            help=title,
            use_container_width=True,
        ):
            switch_thread(tid)
            st.rerun()

    if shown == 0:
        st.markdown('<div class="empty-recent">No recent chats</div>', unsafe_allow_html=True)

    # Spacer approximates `mt-auto` from the HTML: keeps the architecture button at the bottom
    spacer_px = 500 + 42 * shown # Increased spacer to account for the model selectbox height
    st.markdown(
        f'<div style="height:calc(100vh - {spacer_px}px);min-height:24px;"></div>',
        unsafe_allow_html=True,
    )
    st.divider()
    if st.button("Architecture & Pipeline", icon=":material/account_tree:", key="btn_arch", use_container_width=True):
        architecture_dialog()

# --------------------------------------------------------------------------
# Main workspace
# --------------------------------------------------------------------------
st.markdown('<div class="topbar"><h1>ThreadSync</h1></div>', unsafe_allow_html=True)

has_messages = bool(st.session_state["message_history"])
st.markdown(
    f"""
    <div class="hero {'' if has_messages else 'centered'}">
      <h1>Hello, I am ThreadSync, your assistant.</h1>
      <p>Tell me what you would like to explore, analyze, or simulate today.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

for msg in st.session_state["message_history"]:
    if msg["content"]:
        render_message(msg["role"], msg["content"], msg.get("ts", ""))

config = {
    "configurable": {
        "thread_id": st.session_state["thread_id"],
        "model_id": st.session_state.get("model_id", DEFAULT_MODEL)
    },
    "metadata": {"thread_id": st.session_state["thread_id"]},
    "run_name": "Threadsync_Trace",
}

# --------------------------------------------------------------------------
# Chat input with PDF attachment (-> FAISS index used by rag_tool)
# --------------------------------------------------------------------------
submission = st.chat_input("Ask ThreadSync anything...", accept_file=True, file_type=["pdf"])

user_input, llm_input, attached_name = None, None, None

if submission:
    user_input = (submission.text or "").strip()
    llm_input = user_input

    if submission.files:
        uploaded_pdf = submission.files[0]
        attached_name = uploaded_pdf.name
        pdf_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_pdf.getvalue())
                pdf_path = tmp.name

            with st.spinner(f"Processing {uploaded_pdf.name}..."):
                ingest_rag_document(pdf_path)
            st.toast(f"{uploaded_pdf.name} processed successfully.", icon="✔️")

            # Tell the agent a new file is active, even if uploaded mid-conversation
            if user_input:
                llm_input = (
                    f"[System Note: A new PDF '{uploaded_pdf.name}' was just uploaded. "
                    f"Call rag_tool to answer questions about it.]\n\n{user_input}"
                )
            else:
                user_input = f"Uploaded `{uploaded_pdf.name}`"
                llm_input = (
                    f"[System Note: A new PDF '{uploaded_pdf.name}' was just uploaded. "
                    "Call rag_tool to summarize or acknowledge it.]"
                )
        except Exception as error:  # noqa: BLE001
            st.error(f"PDF processing failed: {error}")
        finally:
            if pdf_path and os.path.exists(pdf_path):
                os.remove(pdf_path)

if user_input:
    now = datetime.now().strftime("%H:%M")
    thread_id = st.session_state["thread_id"]
    st.session_state["message_history"].append({"role": "user", "content": user_input, "ts": now})
    if not st.session_state["thread_titles"].get(thread_id):
        st.session_state["thread_titles"][thread_id] = user_input

    render_message("user", user_input, now)
    if attached_name:
        st.caption(f":material/description: {attached_name} · Ready to analyze")

    failed = False
    with st.chat_message("assistant", avatar=AI_AVATAR):
        st.markdown(f'<div class="meta">ThreadSync · {now}</div>', unsafe_allow_html=True)
        try:
            ai_message = st.write_stream(
                stream_agent({"messages": [HumanMessage(content=llm_input)]}, config)
            )
        except Exception as error:  # noqa: BLE001
            failed = True
            ai_message = None
            st.error(f"The agent hit an error: {error}")

    if ai_message:
        st.session_state["message_history"].append({"role": "assistant", "content": ai_message, "ts": now})
    if not failed:
        st.rerun()  # refresh sidebar list, hero position, and HITL check

# --------------------------------------------------------------------------
# Human-in-the-loop: graph paused by interrupt() in HITL_feature.purchase_stock
# --------------------------------------------------------------------------
current_state = chatbot.get_state(config)

if current_state.tasks and current_state.tasks[0].interrupts:
    prompt_text = current_state.tasks[0].interrupts[0].value
    with st.chat_message("assistant", avatar=AI_AVATAR):
        st.warning(f"**Action Required:** {prompt_text}")
        col_yes, col_no = st.columns(2)
        approve = col_yes.button("Approve (Yes)", key="hitl_yes", use_container_width=True)
        decline = col_no.button("Decline (No)", key="hitl_no", use_container_width=True)

        if approve or decline:
            decision = "yes" if approve else "no"
            try:
                result = st.write_stream(stream_agent(Command(resume=decision), config))
            except Exception as error:  # noqa: BLE001
                result = None
                st.error(f"The agent hit an error: {error}")
            else:
                if result:
                    st.session_state["message_history"].append(
                        {"role": "assistant", "content": result, "ts": datetime.now().strftime("%H:%M")}
                    )
                st.rerun()

st.markdown(
    '<div class="disclaimer">ThreadSync may produce inaccurate info. Verify important financial data.</div>',
    unsafe_allow_html=True,
)
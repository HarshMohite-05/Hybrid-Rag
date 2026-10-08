"""
app.py
Dynamic RAG Q&A — Streamlit frontend.
Single file, no duplicates, all imports from config.py.
Run: streamlit run app.py
"""

import sys
import os
import traceback

os.environ["ANONYMIZED_TELEMETRY"]   = "False"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
from pipeline import run_ingestion, run_query_stream, get_sources, db_count, reset_db
from config import (
    GROQ_MODEL, CHUNK_SIZE, CHUNK_OVERLAP,
    TOP_K_RETRIEVAL, TOP_K_RERANK, EMBED_MODEL,
    _has_groq_key,
)

# ── Page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="RAG Studio",
    page_icon="⬡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=JetBrains+Mono:wght@300;400;500&display=swap');

*, *::before, *::after { box-sizing: border-box; }
html, body, [class*="css"], .stApp {
    font-family: 'Syne', sans-serif;
    background: #080b10;
    color: #e8eaf0;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: #0c0f18 !important;
    border-right: 1px solid #1a1f2e;
}
.sidebar-logo {
    padding: 28px 24px 20px;
    border-bottom: 1px solid #1a1f2e;
}
.sidebar-logo h1 {
    font-size: 1.4rem; font-weight: 800;
    letter-spacing: -0.02em; color: #fff; line-height: 1;
}
.sidebar-logo span {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem; color: #3d84f5;
    letter-spacing: 0.15em; text-transform: uppercase;
    display: block; margin-top: 4px;
}
.sidebar-section {
    padding: 18px 24px;
    border-bottom: 1px solid #1a1f2e;
}
.sidebar-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.6rem; color: #4a5568;
    letter-spacing: 0.2em; text-transform: uppercase;
    margin-bottom: 12px;
}

/* Stats */
.stats-row { display: flex; gap: 8px; }
.stat-card {
    flex: 1; background: #111520;
    border: 1px solid #1a1f2e; border-radius: 10px;
    padding: 12px; text-align: center;
}
.stat-num {
    font-size: 1.6rem; font-weight: 800; color: #3d84f5;
    line-height: 1; font-family: 'JetBrains Mono', monospace;
}
.stat-lbl {
    font-size: 0.6rem; color: #4a5568;
    text-transform: uppercase; letter-spacing: 0.1em; margin-top: 4px;
}

/* File pills */
.file-pill {
    display: flex; align-items: center; gap: 8px;
    background: #111520; border: 1px solid #1a1f2e;
    border-radius: 8px; padding: 8px 12px; margin-bottom: 6px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem; color: #8892a4;
}
.file-pill.failed { border-color: #ef4444; color: #ef4444; }
.file-dot { width: 6px; height: 6px; background: #22c55e; border-radius: 50%; flex-shrink: 0; }
.file-dot.failed { background: #ef4444; }

/* Config grid */
.config-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 6px; }
.config-item {
    background: #111520; border: 1px solid #1a1f2e;
    border-radius: 8px; padding: 8px 10px;
}
.config-key {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.55rem; color: #4a5568;
    text-transform: uppercase; letter-spacing: 0.1em;
}
.config-val {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem; color: #3d84f5; margin-top: 2px;
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}

/* ── Main ── */
.main-header { padding: 40px 48px 0; margin-bottom: 32px; }
.main-header h2 {
    font-size: 2.2rem; font-weight: 800;
    letter-spacing: -0.03em; color: #fff; line-height: 1.1;
}
.main-header p {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem; color: #4a5568;
    margin-top: 8px; letter-spacing: 0.05em;
}

/* Chat bubbles */
.chat-wrap { padding: 0 48px; max-width: 900px; }
.msg-user { display: flex; justify-content: flex-end; margin-bottom: 20px; }
.msg-user-bubble {
    background: #1a2540; border: 1px solid #2a3a5c;
    border-radius: 16px 16px 4px 16px;
    padding: 14px 18px; max-width: 75%;
    font-size: 0.95rem; line-height: 1.6; color: #c8d4f0;
}
.msg-assistant { display: flex; gap: 12px; margin-bottom: 24px; align-items: flex-start; }
.msg-avatar {
    width: 32px; height: 32px;
    background: linear-gradient(135deg, #3d84f5, #6366f1);
    border-radius: 8px; display: flex;
    align-items: center; justify-content: center;
    font-size: 0.8rem; flex-shrink: 0; margin-top: 2px;
}
.msg-assistant-bubble {
    background: #0f1420; border: 1px solid #1a1f2e;
    border-radius: 4px 16px 16px 16px;
    padding: 16px 20px; flex: 1;
    font-size: 0.92rem; line-height: 1.75; color: #c8d4f0;
}

/* Sources */
.sources-row {
    display: flex; flex-wrap: wrap; gap: 6px;
    margin-top: 12px; padding-top: 12px;
    border-top: 1px solid #1a1f2e;
}
.source-tag {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem; color: #3d84f5;
    background: #0d1829; border: 1px solid #1e3a6e;
    border-radius: 20px; padding: 3px 10px;
}

/* Empty state */
.empty-state { padding: 60px 48px; text-align: center; }
.empty-icon  { font-size: 3rem; margin-bottom: 16px; opacity: 0.3; }
.empty-title { font-size: 1.1rem; font-weight: 700; color: #4a5568; margin-bottom: 8px; }
.empty-sub   { font-family: 'JetBrains Mono', monospace; font-size: 0.7rem; color: #2d3748; letter-spacing: 0.05em; }

/* Streamlit overrides */
.stButton > button {
    background: #111520 !important; border: 1px solid #1a1f2e !important;
    color: #8892a4 !important; border-radius: 8px !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.72rem !important; width: 100%;
    padding: 8px !important; transition: all 0.2s !important;
}
.stButton > button:hover { border-color: #ef4444 !important; color: #ef4444 !important; }
[data-testid="stFileUploaderDropzone"] {
    background: #0c0f18 !important;
    border: 1px dashed #1a1f2e !important; border-radius: 10px !important;
}
[data-testid="stFileUploaderDropzone"] p { font-size: 0.78rem !important; color: #4a5568 !important; }
.stChatInput textarea {
    background: #0f1420 !important; border: 1px solid #1a1f2e !important;
    border-radius: 12px !important; color: #e8eaf0 !important;
    font-family: 'Syne', sans-serif !important;
}
.stChatInput textarea:focus { border-color: #3d84f5 !important; }
[data-testid="stChatInput"] { padding: 0 48px 32px !important; }
.stExpander {
    border: 1px solid #1a1f2e !important;
    border-radius: 10px !important; background: #0c0f18 !important;
}
.stExpander summary { color: #4a5568 !important; font-size: 0.75rem !important; }
.stSpinner > div { border-color: #3d84f5 !important; }
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: #080b10; }
::-webkit-scrollbar-thumb { background: #1a1f2e; border-radius: 2px; }
table { width: 100%; border-collapse: collapse; margin: 12px 0; }
th {
    background: #111520; color: #3d84f5;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem; padding: 10px 14px;
    text-align: left; border: 1px solid #1a1f2e;
}
td { padding: 8px 14px; border: 1px solid #1a1f2e; font-size: 0.85rem; color: #c8d4f0; }
tr:nth-child(even) td { background: #0d1018; }
code {
    font-family: 'JetBrains Mono', monospace;
    background: #111520; padding: 2px 6px;
    border-radius: 4px; font-size: 0.82em; color: #3d84f5;
}
#MainMenu, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ── Session state ─────────────────────────────────────────────────────────────
if "messages"       not in st.session_state: st.session_state.messages       = []
if "ingested_files" not in st.session_state: st.session_state.ingested_files = []
if "failed_files"   not in st.session_state: st.session_state.failed_files   = []


# ── Cached db count ───────────────────────────────────────────────────────────
@st.cache_data(ttl=5)
def cached_db_count() -> int:
    return db_count()


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:

    # Logo
    st.markdown("""
    <div class="sidebar-logo">
        <h1>⬡ RAG Studio</h1>
        <span>Dynamic Q&amp;A Engine</span>
    </div>
    """, unsafe_allow_html=True)

    # ── Stats ─────────────────────────────────────────────────────────────────
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-label">Knowledge Base</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="stats-row">
        <div class="stat-card">
            <div class="stat-num">{cached_db_count()}</div>
            <div class="stat-lbl">Chunks</div>
        </div>
        <div class="stat-card">
            <div class="stat-num">{len(st.session_state.ingested_files)}</div>
            <div class="stat-lbl">Files</div>
        </div>
        <div class="stat-card">
            <div class="stat-num">{len(st.session_state.messages) // 2}</div>
            <div class="stat-lbl">Turns</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # ── API Key ───────────────────────────────────────────────────────────────
    has_key = _has_groq_key()
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-label">Groq API Key</div>', unsafe_allow_html=True)
    if not has_key:
        st.caption("⚠️ Key required to generate answers:")
        st.text_input(
            "Groq API Key",
            type="password",
            placeholder="gsk_...",
            key="user_groq_key",
            label_visibility="collapsed",
        )
    else:
        with st.expander("🔑 Key Configured", expanded=False):
            st.caption("Active key loaded. Override below if needed:")
            st.text_input(
                "Change Key",
                type="password",
                placeholder="gsk_...",
                key="user_groq_key",
                label_visibility="collapsed",
            )
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Upload ────────────────────────────────────────────────────────────────
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-label">Upload Documents</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader(
        "Drop files here",
        type=["pdf", "csv", "txt", "docx", "xlsx"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if uploaded:
        already = set(st.session_state.ingested_files) | set(st.session_state.failed_files)
        new_files = [f for f in uploaded if f.name not in already]
        if new_files:
            any_success = False
            with st.spinner(f"Indexing {len(new_files)} file(s)…"):
                for f in new_files:
                    try:
                        n = run_ingestion(f, f.name)
                        st.session_state.ingested_files.append(f.name)
                        cached_db_count.clear()
                        st.success(f"✓ {f.name} — {n} chunks")
                        any_success = True
                    except Exception as e:
                        st.session_state.failed_files.append(f.name)
                        st.error(f"✗ {f.name}: {e}")
                        with st.expander(f"🔍 Error details — {f.name}"):
                            st.code(traceback.format_exc())
            if any_success:
                st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Indexed files ─────────────────────────────────────────────────────────
    if st.session_state.ingested_files:
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.markdown('<div class="sidebar-label">Indexed Files</div>', unsafe_allow_html=True)
        for fname in st.session_state.ingested_files:
            short = fname[:28] + "…" if len(fname) > 30 else fname
            st.markdown(
                f'<div class="file-pill"><div class="file-dot"></div>{short}</div>',
                unsafe_allow_html=True,
            )
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Failed files ──────────────────────────────────────────────────────────
    if st.session_state.failed_files:
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.markdown('<div class="sidebar-label">Failed Files</div>', unsafe_allow_html=True)
        for fname in st.session_state.failed_files:
            short = fname[:28] + "…" if len(fname) > 30 else fname
            st.markdown(
                f'<div class="file-pill failed"><div class="file-dot failed"></div>{short}</div>',
                unsafe_allow_html=True,
            )
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Pipeline config ───────────────────────────────────────────────────────
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    with st.expander("⚙ Pipeline Config"):
        embed_short = EMBED_MODEL.split("/")[-1]         # bge-medium-en-v1.5
        llm_short   = GROQ_MODEL.split("-")[0:2]          # ['llama', '3.3']
        llm_label   = "-".join(llm_short)               # llama-3.3
        st.markdown(f"""
        <div class="config-grid">
            <div class="config-item">
                <div class="config-key">Embed</div>
                <div class="config-val">{embed_short}</div>
            </div>
            <div class="config-item">
                <div class="config-key">LLM</div>
                <div class="config-val">{llm_label}</div>
            </div>
            <div class="config-item">
                <div class="config-key">Chunk</div>
                <div class="config-val">{CHUNK_SIZE}c / {CHUNK_OVERLAP}c</div>
            </div>
            <div class="config-item">
                <div class="config-key">Retrieve</div>
                <div class="config-val">top-{TOP_K_RETRIEVAL}</div>
            </div>
            <div class="config-item">
                <div class="config-key">Rerank</div>
                <div class="config-val">top-{TOP_K_RERANK}</div>
            </div>
            <div class="config-item">
                <div class="config-key">DB</div>
                <div class="config-val">ChromaDB</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Clear KB ──────────────────────────────────────────────────────────────
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    if st.button("🗑 Clear Knowledge Base"):
        reset_db()
        cached_db_count.clear()
        st.session_state.ingested_files = []
        st.session_state.failed_files   = []
        st.session_state.messages       = []
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN AREA
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="main-header">
    <h2>Ask your documents<br>anything.</h2>
    <p>RETRIEVE → RERANK → GENERATE · BGE-medium + MiniLM-L12 + Llama 3.3</p>
</div>
""", unsafe_allow_html=True)

chunk_count = cached_db_count()

# ── Chat history ──────────────────────────────────────────────────────────────
if st.session_state.messages:
    st.markdown('<div class="chat-wrap">', unsafe_allow_html=True)
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(f"""
            <div class="msg-user">
                <div class="msg-user-bubble">{msg["content"]}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            sources_html = ""
            if msg.get("sources"):
                tags = "".join(
                    f'<span class="source-tag">📄 {s}</span>'
                    for s in msg["sources"]
                )
                sources_html = f'<div class="sources-row">{tags}</div>'
            st.markdown(f"""
            <div class="msg-assistant">
                <div class="msg-avatar">⬡</div>
                <div class="msg-assistant-bubble">
                    {msg["content"]}
                    {sources_html}
                </div>
            </div>
            """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ── Empty state ───────────────────────────────────────────────────────────────
else:
    if chunk_count == 0:
        st.markdown("""
        <div class="empty-state">
            <div class="empty-icon">⬡</div>
            <div class="empty-title">No documents indexed</div>
            <div class="empty-sub">Upload PDF · CSV · TXT · DOCX · XLSX from the sidebar</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="empty-state">
            <div class="empty-icon">💬</div>
            <div class="empty-title">Ready to answer</div>
            <div class="empty-sub">Ask anything about your uploaded documents</div>
        </div>
        """, unsafe_allow_html=True)

if chunk_count > 0 and not _has_groq_key():
    st.warning("⚠️ Groq API key is not set. Enter your key in the sidebar to start asking questions.")

# ── Chat input ────────────────────────────────────────────────────────────────
query = st.chat_input(
    "Ask anything about your documents…",
    disabled=(chunk_count == 0),
)

if query:
    if not _has_groq_key():
        st.session_state.messages.append({"role": "user", "content": query})
        st.session_state.messages.append({
            "role": "assistant",
            "content": "⚠️ **GROQ_API_KEY is required.** Please enter your Groq API key in the sidebar or add it to Streamlit Secrets (`GROQ_API_KEY = \"...\"`) to generate responses.",
            "sources": []
        })
        st.rerun()

    st.session_state.messages.append({"role": "user", "content": query})

    placeholder = st.empty()
    full_answer = ""

    history = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.messages[:-1]
        if m["role"] in ("user", "assistant")
    ]

    # Stream answer tokens
    try:
        for token in run_query_stream(query, history):
            full_answer += token
            placeholder.markdown(f"""
            <div class="chat-wrap">
                <div class="msg-assistant">
                    <div class="msg-avatar">⬡</div>
                    <div class="msg-assistant-bubble">{full_answer}▌</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    except Exception as e:
        full_answer = f"⚠ Error generating response: {e}"

    # Final render without cursor
    placeholder.markdown(f"""
    <div class="chat-wrap">
        <div class="msg-assistant">
            <div class="msg-avatar">⬡</div>
            <div class="msg-assistant-bubble">{full_answer}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Fetch sources
    sources: list[str] = []
    try:
        top_chunks = get_sources(query)
        if top_chunks:
            sources = list(dict.fromkeys(c["source"] for c in top_chunks))
    except Exception:
        pass  # Sources are non-critical

    st.session_state.messages.append({
        "role":    "assistant",
        "content": full_answer,
        "sources": sources,
    })
    st.rerun()
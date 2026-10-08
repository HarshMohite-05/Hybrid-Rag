"""
config.py
Central configuration — all constants live here.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ── Parallelism (safe default) ─────────────────────────────────────────────────
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["ANONYMIZED_TELEMETRY"]   = "False"

# ── Groq ───────────────────────────────────────────────────────────────────────
# Reads from: Streamlit session_state → Streamlit secrets → .env file → env var
def _get_groq_key() -> str:
    # 1. Try Streamlit session state (user entered in UI)
    try:
        import streamlit as st
        if st.session_state.get("user_groq_key"):
            return st.session_state["user_groq_key"].strip()
    except Exception:
        pass
    # 2. Try Streamlit secrets (Streamlit Cloud deployment)
    try:
        import streamlit as st
        if "GROQ_API_KEY" in st.secrets:
            return st.secrets["GROQ_API_KEY"].strip()
    except Exception:
        pass
    # 3. Try environment variable / .env file
    key = os.getenv("GROQ_API_KEY", "").strip()
    if key:
        return key
    raise RuntimeError(
        "GROQ_API_KEY not found. Set it in Streamlit Secrets, enter it in the sidebar, or provide a .env file."
    )


def _has_groq_key() -> bool:
    try:
        return bool(_get_groq_key())
    except Exception:
        return False

GROQ_MODEL       = "openai/gpt-oss-20b"
GROQ_MAX_TOKENS  = 1024
GROQ_TEMPERATURE = 0.5

# ── ChromaDB ───────────────────────────────────────────────────────────────────
CHROMA_PERSIST_DIR = "./chroma_store"
COLLECTION_NAME    = "dynamic_rag_docs"

# ── Embedding ──────────────────────────────────────────────────────────────────
EMBED_MODEL      = "BAAI/bge-base-en-v1.5"
EMBED_DEVICE     = "cpu"
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

# ── Chunking ───────────────────────────────────────────────────────────────────
CHUNK_SIZE    = 600
CHUNK_OVERLAP = 50

# ── Retrieval ──────────────────────────────────────────────────────────────────
TOP_K_RETRIEVAL = 20
TOP_K_RERANK    = 5

# ── Re-ranker ──────────────────────────────────────────────────────────────────
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# ── Supported uploads ──────────────────────────────────────────────────────────
SUPPORTED_EXTENSIONS = [".pdf", ".csv", ".txt", ".docx", ".xlsx"]

# ── Resolved API key (evaluated lazily at call time) ──────────────────────────
# Use GROQ_API_KEY = _get_groq_key() only where needed to avoid import-time errors
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
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
# Reads from: Streamlit secrets → .env file → fallback env var
def _get_groq_key() -> str:
    # 1. Try Streamlit secrets (Streamlit Cloud deployment)
    try:
        import streamlit as st
        return st.secrets["GROQ_API_KEY"]
    except Exception:
        pass
    # 2. Try environment variable / .env file
    key = os.getenv("GROQ_API_KEY", "")
    if key:
        return key
    raise RuntimeError(
        "GROQ_API_KEY not found. Set it in Streamlit secrets, a .env file, or as an env var."
    )

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
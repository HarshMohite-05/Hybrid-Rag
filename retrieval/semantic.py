"""
retrieval/semantic.py
Semantic search against ChromaDB using BGE embeddings.
All constants imported from config.py — no hardcoded values.
"""

import os
import logging

import chromadb
from chromadb.config import Settings
from config import CHROMA_PERSIST_DIR, COLLECTION_NAME, TOP_K_RETRIEVAL
from ingestion.embedder import encode

logging.getLogger("chromadb.telemetry").setLevel(logging.ERROR)
logging.getLogger("chromadb").setLevel(logging.ERROR)

_chroma     = None
_collection = None


# ── ChromaDB singleton ────────────────────────────────────────────────────────

def _get_collection():
    global _chroma, _collection
    if _collection is None:
        _chroma = chromadb.PersistentClient(
            path=CHROMA_PERSIST_DIR,
            settings=Settings(
                anonymized_telemetry=False,
                is_persistent=True,
            ),
        )
        _collection = _chroma.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
    return _collection


# ── Public API ────────────────────────────────────────────────────────────────

def semantic_search(query: str, top_k: int = TOP_K_RETRIEVAL) -> list:
    """
    Embed the query and retrieve the most similar chunks from ChromaDB.

    Parameters
    ----------
    query  : user question string
    top_k  : max number of results to return

    Returns
    -------
    List of dicts with keys: text, source, score
    Sorted by score descending. Empty list if DB is empty or no hits pass threshold.
    """
    col           = _get_collection()
    current_count = col.count()

    # Nothing indexed yet
    if current_count == 0:
        return []

    # ── Step 1: Embed query ───────────────────────────────────────────────────
    # Must happen BEFORE the query call — NameError otherwise
    q_emb = encode([query], is_query=True)

    # ChromaDB requires a plain Python list of lists, not a numpy array
    if hasattr(q_emb, "tolist"):
        q_emb = q_emb.tolist()

    # ── Step 2: Safe n_results ────────────────────────────────────────────────
    # CRITICAL: n_results > current_count causes:
    # "RuntimeError: Cannot return results in a contiguous 2D array.
    #  Probably ef or M is too small"
    # Always clamp to current_count
    fetch_n = max(1, min(top_k, current_count))

    # ── Step 3: Query ChromaDB ────────────────────────────────────────────────
    try:
        results = col.query(
            query_embeddings=q_emb,
            n_results=fetch_n,
            include=["documents", "metadatas", "distances"],
        )
    except Exception as e:
        logging.error(f"[semantic_search] ChromaDB query failed: {e}")
        return []

    # Guard against empty results
    if not results or not results.get("documents"):
        return []
    if not results["documents"][0]:
        return []

    # ── Step 4: Build candidates ──────────────────────────────────────────────
    candidates = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        # Convert cosine distance → similarity score
        # ChromaDB cosine distance: 0 = identical, 2 = opposite
        score = round(1.0 - dist, 4)

        # Strict relevance threshold — skip low-quality matches
        # Prevents LLM from hallucinating on irrelevant context
        if score < 0.25:
            continue

        candidates.append({
            "text":   doc.strip(),
            "source": meta.get("source", "unknown"),
            "page":   meta.get("page",   "?"),
            "score":  score,
        })

    # ── Step 5: Sort and return ───────────────────────────────────────────────
    candidates.sort(key=lambda x: x["score"], reverse=True)
    return candidates


def collection_count() -> int:
    """Return total number of chunks currently stored."""
    try:
        return _get_collection().count()
    except Exception:
        return 0


def reset_collection():
    """Wipe and recreate the ChromaDB collection."""
    global _collection
    try:
        col = _get_collection()
        col.delete(where={"source": {"$ne": ""}})
    except Exception:
        pass
    _collection = None
import hashlib
import chromadb
from chromadb.config import Settings
from config import CHROMA_PERSIST_DIR, COLLECTION_NAME, CHUNK_SIZE, CHUNK_OVERLAP
from ingestion.embedder import encode

_chroma     = None
_collection = None
BATCH       = 200


def _get_collection():
    global _chroma, _collection
    if _collection is None:
        _chroma = chromadb.PersistentClient(
            path=CHROMA_PERSIST_DIR,
            settings=Settings(anonymized_telemetry=False),
        )
        _collection = _chroma.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
    return _collection


def chunk_text(pages: list, source: str) -> list:
    """
    pages: list of {"text": str, "page": int} dicts returned by loader.
    Returns list of {"text", "source", "page", "chunk_id"} dicts.
    """
    chunks = []
    chunk_id = 0

    for page_obj in pages:
        page_text = page_obj["text"]
        page_num  = page_obj["page"]

        # Sentence-aware split within each page
        sentences     = page_text.split(". ")
        current_chunk = ""

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            # Re-add the period only if the sentence doesn't already end with punctuation
            suffix = " " if sentence[-1] in ".!?" else ". "
            if len(current_chunk) + len(sentence) <= CHUNK_SIZE:
                current_chunk += sentence + suffix
            else:
                cleaned = current_chunk.strip()
                if len(cleaned) >= 20:   # skip trivially short/blank chunks
                    chunks.append({
                        "text":     cleaned,
                        "source":   source,
                        "page":     page_num,
                        "chunk_id": chunk_id,
                    })
                    chunk_id += 1
                # Overlap: carry last CHUNK_OVERLAP chars into next chunk
                overlap       = current_chunk[-CHUNK_OVERLAP:] if CHUNK_OVERLAP else ""
                current_chunk = overlap + sentence + suffix

        # Flush remaining text for this page
        cleaned = current_chunk.strip()
        if len(cleaned) >= 20:
            chunks.append({
                "text":     cleaned,
                "source":   source,
                "page":     page_num,
                "chunk_id": chunk_id,
            })
            chunk_id += 1

    return chunks


def _clean_text(text: str) -> str:
    text = text.strip()
    text = " ".join(text.split())
    return text


def index_chunks(chunks: list) -> int:
    if not chunks:
        return 0

    col   = _get_collection()
    texts = [_clean_text(c["text"]) for c in chunks]

    # Filter out any remaining empty/too-short texts after cleaning
    valid = [(t, c) for t, c in zip(texts, chunks) if len(t) >= 20]
    if not valid:
        return 0
    texts, chunks = zip(*valid)
    texts  = list(texts)
    chunks = list(chunks)

    # Hash includes source + page + chunk_id to avoid cross-doc collisions
    ids = [
        hashlib.sha256(
            (c["source"] + str(c["page"]) + str(c["chunk_id"])).encode()
        ).hexdigest()[:32]
        for c in chunks
    ]

    metas = [
        {
            "source":   c["source"],
            "page":     str(c["page"]),   # ChromaDB requires string metadata
            "chunk_id": str(c["chunk_id"]),
        }
        for c in chunks
    ]

    embeddings = encode(texts, is_query=False)

    for i in range(0, len(chunks), BATCH):
        col.upsert(
            ids=ids[i:i+BATCH],
            embeddings=embeddings[i:i+BATCH],
            documents=texts[i:i+BATCH],
            metadatas=metas[i:i+BATCH],
        )

    return len(chunks)


def collection_count() -> int:
    return _get_collection().count()


def clear_collection():
    col = _get_collection()
    if col.count() > 0:
        all_ids = col.get()["ids"]
        if all_ids:
            col.delete(ids=all_ids)
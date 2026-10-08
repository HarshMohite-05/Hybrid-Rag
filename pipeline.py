import logging
logging.getLogger('chromadb.telemetry').setLevel(logging.ERROR)

from ingestion.loader import load_and_clean
from ingestion.indexer import chunk_text, index_chunks, collection_count, clear_collection
from retrieval.semantic import semantic_search
from reranker.rerank import rerank
from generation.generator import stream_generate, generate


def run_ingestion(file_obj, filename):
    pages  = load_and_clean(file_obj, filename)   # list of {text, page}
    chunks = chunk_text(pages, source=filename)    # list of {text, source, page, chunk_id}
    return index_chunks(chunks)


def run_query_stream(query, history=None):
    candidates = semantic_search(query)
    if not candidates:
        yield "The uploaded documents do not contain enough information to answer this question."
        return
    top_chunks = rerank(query, candidates)
    yield from stream_generate(query, top_chunks, history or [])


def run_query(query):
    candidates = semantic_search(query)
    if not candidates:
        return "The uploaded documents do not contain enough information to answer this question.", []
    top_chunks = rerank(query, candidates)
    return generate(query, top_chunks), top_chunks


def get_sources(query):
    candidates = semantic_search(query)
    if not candidates:
        return []
    return rerank(query, candidates)


def db_count():
    return collection_count()


def reset_db():
    clear_collection()

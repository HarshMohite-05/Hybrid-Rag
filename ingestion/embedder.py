from sentence_transformers import SentenceTransformer
from config import EMBED_MODEL, EMBED_DEVICE, BGE_QUERY_PREFIX

_embedder = None


def get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(EMBED_MODEL, device=EMBED_DEVICE)
    return _embedder


def encode(texts: list, is_query: bool = False) -> list:
    model = get_embedder()
    if is_query:
        texts = [BGE_QUERY_PREFIX + t for t in texts]
    return model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=False,
        batch_size=32,
    ).tolist()
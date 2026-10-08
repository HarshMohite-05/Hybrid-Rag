from sentence_transformers import CrossEncoder
from config import RERANK_MODEL, TOP_K_RERANK

_model = None


def _get_model() -> CrossEncoder:
    global _model
    if _model is None:
        _model = CrossEncoder(RERANK_MODEL)
    return _model


def rerank(query: str, candidates: list) -> list:
    if not candidates:
        return []
    model = _get_model()
    pairs = [(query, c["text"]) for c in candidates]
    scores = model.predict(pairs)
    ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
    return [{**cand, "rerank_score": float(score)} for cand, score in ranked[:TOP_K_RERANK]]
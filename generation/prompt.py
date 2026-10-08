from typing import Dict, List

SYSTEM = """\
You are a precise document Q&A assistant. Your answers must be grounded ONLY in the context excerpts provided.

Rules:
- Answer specifically using facts, names, numbers, and details from the context — not general knowledge.
- Cite every claim with [SourceName, chunk N] inline, e.g. [internship_report.pdf, chunk 2].
- If multiple chunks support the answer, cite all of them.
- Structure your answer clearly: use short paragraphs or bullet points when listing multiple points.
- Do NOT pad with generic filler or background knowledge.
- Do NOT repeat or rephrase the question.
- If the context genuinely lacks the answer, say: "The document does not contain information about this."
"""

FEW_SHOT_Q = """\
Context:
[chunk 1] (resume.pdf, page 1)
John worked at Acme Corp from 2021-2023 as a Data Engineer. He built ETL pipelines using Apache Spark and reduced data processing time by 40%.

[chunk 2] (resume.pdf, page 2)
At Acme Corp, John also mentored 3 junior engineers and introduced automated testing that caught 95% of bugs before production.

Question: What did John do at Acme Corp?"""

FEW_SHOT_A = """\
At Acme Corp (2021–2023), John worked as a Data Engineer where he built ETL pipelines using Apache Spark, achieving a **40% reduction in data processing time** [resume.pdf, chunk 1].

Beyond engineering, he mentored 3 junior engineers and introduced automated testing that caught **95% of bugs before production** [resume.pdf, chunk 2]."""


def _build_context(chunks):
    if not chunks:
        return "No data available."
    parts = []
    for i, c in enumerate(chunks, 1):
        text = c.get("text", "").strip()
        src  = c.get("source", "unknown")
        page = c.get("page", "?")
        if text:
            header = f"[chunk {i}] ({src}, page {page})"
            parts.append(header + "\n" + text)
    return "\n\n".join(parts)


def build_prompt(query, chunks, history=None):
    context  = _build_context(chunks)
    user_msg = (
        "Context:\n"
        + context
        + "\n\nQuestion: "
        + query
    )
    messages = [
        {"role": "system",    "content": SYSTEM},
        {"role": "user",      "content": FEW_SHOT_Q},
        {"role": "assistant", "content": FEW_SHOT_A},
    ]
    if history:
        messages.extend(history[-4:])
    messages.append({"role": "user", "content": user_msg})
    return messages


def build_prompt_with_history(query, chunks, history):
    return build_prompt(query, chunks, history)

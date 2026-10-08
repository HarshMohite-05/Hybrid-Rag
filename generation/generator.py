from groq import Groq
from config import GROQ_MODEL, GROQ_MAX_TOKENS, GROQ_TEMPERATURE, _get_groq_key
from generation.prompt import build_prompt, build_prompt_with_history

_client = None
_cached_key = None


def _get_client() -> Groq:
    global _client, _cached_key
    key = _get_groq_key()
    if _client is None or _cached_key != key:
        _client = Groq(api_key=key)
        _cached_key = key
    return _client


def _clean_output(text: str) -> str:
    if not text:
        return "No response generated."

    # remove unwanted spacing / artifacts
    text = text.strip()

    # optional: remove repeated newlines
    text = "\n".join(line.strip() for line in text.splitlines() if line.strip())

    return text


def stream_generate(query: str, chunks: list, history: list = None):
    client = _get_client()
    msgs = build_prompt_with_history(query, chunks, history or [])

    stream = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=msgs,
        max_tokens=GROQ_MAX_TOKENS,
        temperature=GROQ_TEMPERATURE,
        stream=True,
    )

    for chunk in stream:
        try:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta
        except Exception:
            continue


def generate(query: str, chunks: list) -> str:
    client = _get_client()
    msgs = build_prompt(query, chunks)

    try:
        resp = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=msgs,
            max_tokens=GROQ_MAX_TOKENS,
            temperature=GROQ_TEMPERATURE,
            stream=False,
        )

        content = resp.choices[0].message.content
        return _clean_output(content)

    except Exception as e:
        return f"Error generating response: {str(e)}"
# Hybrid RAG Studio 🔍

A dynamic Retrieval-Augmented Generation (RAG) app built with Streamlit, ChromaDB, BGE embeddings, and Groq LLMs.

## Features
- 📄 Upload PDF, CSV, TXT, DOCX, XLSX files
- 🔍 Semantic search with `BAAI/bge-base-en-v1.5` embeddings
- ⚡ Re-ranking with `cross-encoder/ms-marco-MiniLM-L-6-v2`
- 🤖 LLM answers via Groq API (`openai/gpt-oss-20b`)
- 💬 Multi-turn conversation with history

## Local Setup

```bash
git clone https://github.com/HarshMohite-05/Hybrid-Rag
cd Hybrid-Rag
pip install -r requirements.txt
```

Create a `.env` file:
```
GROQ_API_KEY=your_groq_api_key_here
```

Run:
```bash
streamlit run app.py
```

## Streamlit Cloud Deployment

1. Fork/push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io) → New app
3. Select this repo, branch `main`, file `app.py`
4. In **Settings → Secrets**, add:
```toml
GROQ_API_KEY = "your_groq_api_key_here"
```
5. Deploy — models download automatically on first run (~500MB)

## Tech Stack
| Component | Technology |
|-----------|-----------|
| Frontend | Streamlit |
| Embeddings | BAAI/bge-base-en-v1.5 |
| Re-ranker | cross-encoder/ms-marco-MiniLM-L-6-v2 |
| Vector DB | ChromaDB |
| LLM | Groq (openai/gpt-oss-20b) |
| PDF parsing | pdfplumber |

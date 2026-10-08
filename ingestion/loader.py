import re
import pandas as pd
from pathlib import Path


def load_and_clean(file_obj, filename: str):
    """
    Returns a list of dicts: [{"text": str, "page": int_or_str}, ...]
    For non-PDF formats a single entry is returned with page="1".
    """
    ext = Path(filename).suffix.lower()
    if ext == ".pdf":
        return _read_pdf(file_obj)
    elif ext == ".csv":
        return [{"text": _clean(_read_csv(file_obj)), "page": 1}]
    elif ext == ".txt":
        return [{"text": _clean(_read_txt(file_obj)), "page": 1}]
    elif ext == ".docx":
        return [{"text": _clean(_read_docx(file_obj)), "page": 1}]
    elif ext in (".xlsx", ".xls"):
        return [{"text": _clean(_read_excel(file_obj)), "page": 1}]
    else:
        raise ValueError("Unsupported: " + ext)


def _read_pdf(file_obj) -> list:
    """Returns list of {text, page} dicts — one per PDF page."""
    import pdfplumber
    pages = []
    with pdfplumber.open(file_obj) as pdf:
        for i, page in enumerate(pdf.pages, 1):
            t = page.extract_text()
            if t and t.strip():
                pages.append({"text": _clean(t), "page": i})
    return pages


def _read_csv(file_obj) -> str:
    df = pd.read_csv(file_obj)
    return df.to_string(index=False)


def _read_txt(file_obj) -> str:
    raw = file_obj.read()
    return raw.decode("utf-8", errors="ignore") if isinstance(raw, bytes) else raw


def _read_docx(file_obj) -> str:
    from docx import Document
    doc = Document(file_obj)
    return "\n".join(p.text for p in doc.paragraphs if p.text.strip())


def _read_excel(file_obj) -> str:
    xl = pd.ExcelFile(file_obj)
    parts = []
    for sheet in xl.sheet_names:
        df = xl.parse(sheet)
        parts.append("[Sheet: " + sheet + "]\n" + df.to_string(index=False))
    return "\n\n".join(parts)


def _clean(text: str) -> str:
    text = re.sub(r'\r\n', '\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    return text.strip()
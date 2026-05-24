import os
from typing import List
import google.generativeai as genai

_MODEL = "models/gemini-embedding-001"
_configured = False


def _get_api_key() -> str:
    key = os.environ.get("GEMINI_API_KEY")
    if key:
        return key
    try:
        import streamlit as st
        return st.secrets["GEMINI_API_KEY"]
    except Exception:
        pass
    raise RuntimeError(
        "GEMINI_API_KEY not found. Set it as an environment variable "
        "or add it to Streamlit secrets."
    )


def _configure():
    global _configured
    if not _configured:
        genai.configure(api_key=_get_api_key())
        _configured = True


def embed_texts(texts: List[str]) -> List[List[float]]:
    """Embed a list of document texts for storage."""
    _configure()
    embeddings = []
    for text in texts:
        response = genai.embed_content(
            model=_MODEL,
            content=text,
            task_type="retrieval_document",
        )
        embeddings.append(response["embedding"])
    return embeddings


def embed_query(query: str) -> List[float]:
    """Embed a query string for retrieval."""
    _configure()
    response = genai.embed_content(
        model=_MODEL,
        content=query,
        task_type="retrieval_query",
    )
    return response["embedding"]

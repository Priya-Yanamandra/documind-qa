import os
from typing import List
from google import genai

_MODEL = "models/gemini-embedding-001"
_client = None


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY environment variable is not set.")
        _client = genai.Client(api_key=api_key)
    return _client


def embed_texts(texts: List[str]) -> List[List[float]]:
    """Embed a list of texts. Returns list of embedding vectors."""
    client = _get_client()
    embeddings = []
    for text in texts:
        result = client.models.embed_content(
            model=_MODEL,
            contents=text,
        )
        embeddings.append(result.embeddings[0].values)
    return embeddings


def embed_query(query: str) -> List[float]:
    """Embed a single query string."""
    client = _get_client()
    result = client.models.embed_content(
        model=_MODEL,
        contents=query,
    )
    return result.embeddings[0].values

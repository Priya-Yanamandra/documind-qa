import os
from typing import List, Dict, Tuple
import google.generativeai as genai
from google.genai import types

_GENERATION_MODEL = "models/gemini-2.5-flash"
_client = None


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is not set.")
        _client = genai.Client(api_key=api_key)
    return _client


def build_prompt(
    question: str,
    context_chunks: List[str],
    history: List[Dict[str, str]],
) -> str:
    history_text = ""
    if history:
        pairs = []
        msgs = history[-6:]  # last 3 Q&A pairs
        i = 0
        while i < len(msgs) - 1:
            if msgs[i]["role"] == "user" and msgs[i + 1]["role"] == "assistant":
                pairs.append(f"User: {msgs[i]['content']}\nAssistant: {msgs[i+1]['content']}")
                i += 2
            else:
                i += 1
        if pairs:
            history_text = "\n\n".join(pairs)

    context_text = "\n\n---\n\n".join(context_chunks)

    prompt = f"""You are a helpful document Q&A assistant. Answer the question based ONLY on the provided document context.
If the answer is not found in the context, say "I couldn't find that information in the document."
Be concise, accurate, and cite which part of the document supports your answer when possible.

DOCUMENT CONTEXT:
{context_text}
"""
    if history_text:
        prompt += f"\nCONVERSATION HISTORY:\n{history_text}\n"

    prompt += f"\nQUESTION: {question}\n\nANSWER:"
    return prompt


def generate_answer(
    question: str,
    context_chunks: List[str],
    history: List[Dict[str, str]],
) -> Tuple[str, str]:
    """Returns (answer, prompt_used)."""
    client = _get_client()
    prompt = build_prompt(question, context_chunks, history)
    response = client.models.generate_content(
        model=_GENERATION_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.2,
            max_output_tokens=1024,
        ),
    )
    answer = response.text.strip()
    return answer, prompt

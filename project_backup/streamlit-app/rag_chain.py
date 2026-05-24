import os
from typing import List, Dict, Tuple
import google.generativeai as genai

_GENERATION_MODEL = "gemini-2.5-flash"
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


def build_prompt(
    question: str,
    context_chunks: List[str],
    history: List[Dict[str, str]],
) -> str:
    history_text = ""
    if history:
        pairs = []
        msgs = history[-6:]
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
    _configure()
    prompt = build_prompt(question, context_chunks, history)
    model = genai.GenerativeModel(_GENERATION_MODEL)
    response = model.generate_content(
        prompt,
        generation_config=genai.types.GenerationConfig(
            temperature=0.2,
            max_output_tokens=1024,
        ),
    )
    answer = response.text.strip()
    return answer, prompt

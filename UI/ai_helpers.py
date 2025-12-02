# ai_helpers.py — robust, example-friendly answers with chat history
import os
from typing import Dict, List, Optional

try:
    import streamlit as st  # optional, only for st.secrets
except Exception:
    st = None

from openai import OpenAI, APIError, RateLimitError

# --- Get API Key ---
def _get_openai_key() -> str:
    key = os.getenv("OPENAI_API_KEY", "")
    if not key and st is not None:
        try:
            key = st.secrets.get("OPENAI_API_KEY", "")
        except Exception:
            pass
    return key


_client: Optional[OpenAI] = None

def _client_once() -> OpenAI:
    global _client
    if _client is None:
        key = _get_openai_key()
        if not key:
            raise RuntimeError("OPENAI_API_KEY not set (env or .streamlit/secrets.toml).")
        _client = OpenAI(api_key=key)
    return _client


# --- Context builder ---
def _gather_context(
    chapter: str,
    chapter_summaries: Dict[str, str],
    chapter_questions: Dict[str, List[str]],
    chapter_answers: Dict[str, List[str]],
    chat_history: Optional[List[Dict[str, str]]] = None,
    max_chars: int = 8000,
) -> str:
    """
    Build a compact context block from the selected chapter and optional chat history.
    """
    summary = (chapter_summaries.get(chapter, "") or "").strip()
    qs = chapter_questions.get(chapter, []) or []
    ans = chapter_answers.get(chapter, []) or []

    # Pair top Q/A safely
    pairs: List[str] = []
    for q, a in zip(qs, ans):
        q = (q or "").strip()
        a = (a or "").strip()
        if q and a:
            pairs.append("Q: " + q + "\nA: " + a)

    qa_block = "\n\n".join(pairs).strip()

    pieces: List[str] = []
    if summary:
        pieces.append("Chapter Summary:\n" + summary)
    if qa_block:
        pieces.append("Sample Q&A:\n" + qa_block)

    # Include recent chat context (optional)
    if chat_history:
        # keep only recent, short messages
        recent = chat_history[-8:]
        # flatten to simple text to avoid structured tokens in context
        chat_lines = []
        for m in recent:
            role = m.get("role", "user")
            content = (m.get("content", "") or "").strip()
            if content:
                chat_lines.append(f"{role}: {content}")
        if chat_lines:
            pieces.append("Recent Conversation:\n" + "\n".join(chat_lines))

    context = "\n\n".join(pieces).strip()
    if len(context) > max_chars:
        context = context[:max_chars] + "…"
    return context


# --- Main function ---
def answer_with_ai(
    user_q: str,
    chapter: str,
    chapter_summaries: Dict[str, str],
    chapter_questions: Dict[str, List[str]],
    chapter_answers: Dict[str, List[str]],
    system_preamble: str = (
        "You are an intelligent, friendly textbook tutor. "
        "Use the chapter content as your main guide, but you may also use general world knowledge "
        "to provide simple real-world examples or analogies. "
        "Explain concepts clearly and make them easy to understand. "
        "If the context does not contain the answer, say you’re unsure and suggest where to look."
    ),
    temperature: float = 0.5,
    model: str = "gpt-4o-mini",
    chat_history: Optional[List[Dict[str, str]]] = None,
    max_tokens: int = 500,
) -> str:
    """
    Returns a short, chapter-grounded but example-rich answer using the OpenAI Chat API.
    """
    user_q = (user_q or "").strip()
    if not user_q:
        return "Please ask a question."

    context = _gather_context(
        chapter, chapter_summaries, chapter_questions, chapter_answers, chat_history
    )

    user_prompt = (
        "Below is relevant textbook content and any previous discussion.\n"
        "Use this to answer the student's question helpfully and with clarity. "
        "Prefer short, concrete examples.\n\n"
        "Context:\n" + context + "\n\n"
        "Question: " + user_q
    )

    try:
        client = _client_once()
        resp = client.chat.completions.create(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system_preamble},
                {"role": "user", "content": user_prompt},
            ],
        )
        return (resp.choices[0].message.content or "").strip()
    except RateLimitError:
        return "I’m hitting request limits right now. Please try again in a few seconds."
    except APIError as e:
        return f"API error: {e}"
    except Exception as e:
        return f"Sorry — I couldn't generate an answer right now ({e})."

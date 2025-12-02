import os
import json
import uuid
import re
import random
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import joblib
from scipy import sparse

# ----------------- PATHS / STORAGE -----------------
STORE_DIR = "store"
os.makedirs(STORE_DIR, exist_ok=True)

META_PATH = os.path.join(STORE_DIR, "meta.json")
VECTORIZER_PATH = os.path.join(STORE_DIR, "tfidf_vectorizer.pkl")
MATRIX_PATH = os.path.join(STORE_DIR, "tfidf_matrix.npz")


# ----------------- META HELPERS -----------------


def _load_meta() -> Dict[str, Any]:
    """
    Metadata file holds chunk info (book_id, chapter_id, page, text, etc.).
    We keep chunks in the same order as rows in the TF-IDF matrix.
    """
    if os.path.exists(META_PATH):
        with open(META_PATH, "r") as f:
            return json.load(f)
    return {"docs": [], "chunks": []}


def _save_meta(meta: Dict[str, Any]):
    with open(META_PATH, "w") as f:
        json.dump(meta, f, indent=2)


# ----------------- TF-IDF INDEX HELPERS -----------------


def _save_tfidf(vectorizer: TfidfVectorizer, X):
    joblib.dump(vectorizer, VECTORIZER_PATH)
    sparse.save_npz(MATRIX_PATH, X)


def _load_tfidf():
    if not (os.path.exists(VECTORIZER_PATH) and os.path.exists(MATRIX_PATH)):
        return None, None
    vectorizer = joblib.load(VECTORIZER_PATH)
    X = sparse.load_npz(MATRIX_PATH)
    return vectorizer, X


def _rebuild_tfidf_index():
    """
    Rebuild TF-IDF index from meta["chunks"].
    Called after ingesting new documents.
    """
    meta = _load_meta()
    texts = [c["text"] for c in meta.get("chunks", [])]
    if not texts:
        # No data yet
        return None, None

    # Simple, robust vectorizer – no heavy models, no tokens
    vectorizer = TfidfVectorizer(
        max_features=50000,
        ngram_range=(1, 2),
        stop_words="english",
    )
    X = vectorizer.fit_transform(texts)
    _save_tfidf(vectorizer, X)
    return vectorizer, X


# ----------------- DOCUMENT INGESTION -----------------


def add_document(
    chunks_with_pages: List[Tuple[str, int]],
    book_title: str,
) -> Tuple[str, int, List[str]]:
    """
    Store chunks in meta.json and rebuild TF-IDF index.
    No FAISS, no LLM. This is used by /ingest.
    """
    meta = _load_meta()

    book_id = str(uuid.uuid4())[:8]
    chapters = set()
    start_id = len(meta["chunks"])

    for i, (text, page) in enumerate(chunks_with_pages):
        chapter_id = f"{book_id}_ch_{(page - 1) // 10:02d}"
        chapters.add(chapter_id)
        chunk_id = start_id + i
        meta["chunks"].append(
            {
                "id": chunk_id,
                "book_id": book_id,
                "book_title": book_title or "Untitled Book",
                "chapter_id": chapter_id,
                "chapter_title": chapter_id,
                "page": page,
                "text": text,
            }
        )

    meta["docs"].append(
        {
            "book_id": book_id,
            "book_title": book_title,
            "n_chunks": len(chunks_with_pages),
        }
    )

    _save_meta(meta)

    # Rebuild TF-IDF index on all chunks (still cheap for a few books)
    _rebuild_tfidf_index()

    return book_id, len(chunks_with_pages), sorted(list(chapters))


def list_books() -> Dict[str, Any]:
    """
    Aggregate chunks -> books and chapter list for /books.
    """
    meta = _load_meta()
    books: Dict[str, Dict[str, Any]] = {}
    for c in meta.get("chunks", []):
        b = c["book_id"]
        if b not in books:
            books[b] = {"book_title": c["book_title"], "chapters": set()}
        books[b]["chapters"].add(c["chapter_id"])
    return {
        b: {"book_title": v["book_title"], "chapters": sorted(list(v["chapters"]))}
        for b, v in books.items()
    }


# ----------------- SEARCH + CONTEXT BUILDING -----------------


def _ensure_index():
    """
    Make sure TF-IDF index exists. If not, rebuild it.
    """
    vectorizer, X = _load_tfidf()
    if vectorizer is None or X is None:
        vectorizer, X = _rebuild_tfidf_index()
    return vectorizer, X


def _search(
    q: str,
    k: int = 5,
    scope: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    Vector search using TF-IDF + cosine similarity.
    No OpenAI, no local transformer.
    """
    meta = _load_meta()
    chunks = meta.get("chunks", [])
    if not chunks:
        return []

    vectorizer, X = _ensure_index()
    if vectorizer is None or X is None:
        return []

    q_vec = vectorizer.transform([q])  # shape: [1, n_features]
    sims = linear_kernel(q_vec, X).flatten()  # cosine similarity

    n = sims.shape[0]
    if n == 0:
        return []

    fetch_n = min(max(80, k), n)
    top_idx = np.argsort(sims)[::-1][:fetch_n]

    results: List[Dict[str, Any]] = []
    for idx in top_idx:
        idx = int(idx)
        if idx >= len(chunks):
            continue
        c = chunks[idx]

        if scope:
            book_ids = scope.get("book_ids") or None
            chapter_ids = scope.get("chapter_ids") or None
            if book_ids and c["book_id"] not in book_ids:
                continue
            if chapter_ids and c["chapter_id"] not in chapter_ids:
                continue

        results.append({"chunk": c, "score": float(sims[idx])})

    return results[:k]


def _build_context(hits: List[Dict[str, Any]]) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Turn hit list into a context string + citation metadata.
    """
    ctx_lines: List[str] = []
    cites: List[Dict[str, Any]] = []
    for h in hits:
        c, score = h["chunk"], h["score"]
        ctx_lines.append(
            f"[book {c['book_id']} | {c['chapter_id']} | p.{c['page']}] {c['text']}"
        )
        cites.append(
            {
                "chunk_id": c["id"],
                "page": c["page"],
                "book_id": c["book_id"],
                "chapter_id": c["chapter_id"],
                "score": score,
            }
        )
    return "\n".join(ctx_lines), cites


# ----------------- PUBLIC API: ANSWER / STORY / CASE -----------------


def _extract_sentences(text: str, max_sentences: int = 6) -> str:
    """
    Simple sentence splitter to avoid super long answers.
    """
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    sentences = [s.strip() for s in sentences if s.strip()]
    return " ".join(sentences[:max_sentences])


def answer(
    query: str,
    k: int = 5,
    scope: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Core RAG answer used by /ask.
    No LLM: we return a cleaned excerpt from the most relevant chunk(s).
    """
    hits = _search(query, k=k, scope=scope)
    if not hits:
        return {
            "answer": "I don't have indexed content in this scope. Try ingesting a PDF or widening filters.",
            "citations": [],
        }

    # Very small threshold; TF-IDF sims are often small but non-zero.
    sim_th = float(os.getenv("SIMILARITY_THRESHOLD", "0.02"))
    best_score = max(h["score"] for h in hits)
    if best_score < sim_th:
        return {
            "answer": "Not enough relevant context in the selected scope.",
            "citations": [],
        }

    # Take the best chunk and show a trimmed excerpt
    main_chunk = hits[0]["chunk"]
    excerpt = _extract_sentences(main_chunk["text"], max_sentences=6)

    ctx, cites = _build_context(hits)
    answer_text = (
        "Here is the most relevant explanation I found in your textbook:\n\n"
        f"{excerpt}\n\n"
        "These lines come from the most similar section based on your question."
    )

    return {"answer": answer_text, "citations": cites}


def story(
    scope: Optional[Dict[str, Any]] = None,
    chapter_id: Optional[str] = None,
    max_words: int = 160,
    seed_question: Optional[str] = None,
) -> str:
    """
    Build a simple story-like paragraph grounded in a few top chunks.
    Still no LLM – just templated text + excerpts.
    """
    q = seed_question or "key narrative concepts from this chapter"
    if chapter_id:
        scope = scope or {}
        scope = {
            "book_ids": scope.get("book_ids") if scope else None,
            "chapter_ids": [chapter_id],
        }

    hits = _search(q, k=3, scope=scope)
    if not hits:
        return "No chapter context available to craft a story."

    paragraphs = []
    for h in hits:
        text = _extract_sentences(h["chunk"]["text"], max_sentences=3)
        if text:
            paragraphs.append(text)

    base = " ".join(paragraphs)
    # Very crude length control
    words = base.split()
    if len(words) > max_words:
        base = " ".join(words[:max_words]) + "..."

    return (
        "Here is a short story-like explanation based on your textbook chapter:\n\n"
        f"{base}\n\n"
        "Try to relate this description to an example from your own experience."
    )


def business_case(
    scope: Optional[Dict[str, Any]] = None,
    chapter_id: Optional[str] = None,
    seed_question: Optional[str] = None,
) -> str:
    """
    Build a short workplace-style case using textbook excerpts.
    No LLM, just templates and snippets.
    """
    q = seed_question or "workplace scenario from this chapter"
    if chapter_id:
        scope = scope or {}
        scope = {
            "book_ids": scope.get("book_ids") if scope else None,
            "chapter_ids": [chapter_id],
        }

    hits = _search(q, k=2, scope=scope)
    if not hits:
        return "No chapter context to build a business case."

    company = random.choice(
        ["AlphaCorp", "Beta Analytics", "Gamma Systems", "Delta Insights", "NovaTech"]
    )

    snippet = _extract_sentences(hits[0]["chunk"]["text"], max_sentences=4)

    case_text = f"""
{company} is facing a practical problem that directly relates to concepts from your textbook.

The team has been struggling with the following situation:
{snippet}

Your task:
1. Identify which concept(s) from the chapter could help this company.
2. Explain how applying those ideas would improve their process or decisions.
3. Suggest one concrete action the team should take next.
""".strip()

    return case_text


# ----------------- MCQ GENERATION (NO LLM) -----------------

_STOPWORDS = {
    "the",
    "and",
    "or",
    "for",
    "with",
    "from",
    "that",
    "this",
    "these",
    "those",
    "into",
    "onto",
    "about",
    "after",
    "before",
    "because",
    "such",
    "many",
    "most",
    "some",
    "other",
    "where",
    "when",
    "which",
    "while",
    "using",
    "used",
    "also",
    "very",
    "more",
    "less",
    "data",
    "value",
    "values",
    "table",
    "tables",
    "chapter",
    "example",
    "examples",
}


def _extract_keywords(text: str, max_keywords: int = 6) -> List[str]:
    """
    Very simple keyword extractor:
    - take alphabetic words of length >= 5
    - skip common stopwords
    - return unique lowercase forms in order
    """
    words = re.findall(r"[A-Za-z][A-Za-z\-]{4,}", text)
    seen = set()
    keywords: List[str] = []
    for w in words:
        base = w.strip(".,;:!?").lower()
        if base in _STOPWORDS:
            continue
        if base not in seen:
            seen.add(base)
            keywords.append(base)
        if len(keywords) >= max_keywords:
            break
    return keywords


def quiz(
    scope: Optional[Dict[str, Any]] = None,
    chapter_id: Optional[str] = None,
    n_mcq: int = 5,
):
    """
    Build n_mcq multiple-choice questions directly from textbook chunks.

    Logic:
    - Use TF-IDF search to get relevant chunks.
    - Split them into sentences.
    - Pick one keyword per sentence.
    - Make a cloze question by blanking that keyword.
    - Use other keywords as distractors.
    """
    if chapter_id:
        scope = scope or {}
        scope = {
            "book_ids": scope.get("book_ids") if scope else None,
            "chapter_ids": [chapter_id],
        }

    hits = _search("key concepts and definitions", k=max(40, n_mcq * 6), scope=scope)
    if not hits:
        return []

    sentences: List[Tuple[str, List[str]]] = []
    all_keywords: List[str] = []

    for h in hits:
        text = h["chunk"]["text"]
        for sent in re.split(r"(?<=[.!?])\s+", text):
            s = sent.strip()
            if len(s.split()) < 8:
                continue
            kws = _extract_keywords(s)
            if not kws:
                continue
            sentences.append((s, kws))
            all_keywords.extend(kws)

    if not sentences or not all_keywords:
        return []

    all_keywords = list(dict.fromkeys(all_keywords))  # unique, preserve order

    questions = []
    used_answers = set()
    random.shuffle(sentences)

    for sent, kws in sentences:
        answer = None
        for kw in kws:
            if kw not in used_answers:
                answer = kw
                break
        if not answer:
            continue

        pattern = re.compile(rf"\b{re.escape(answer)}\b", re.IGNORECASE)
        if not pattern.search(sent):
            continue

        question_text = pattern.sub("_____", sent, count=1)
        used_answers.add(answer)

        distractors = [kw for kw in all_keywords if kw != answer]
        random.shuffle(distractors)
        distractors = distractors[:3]
        if len(distractors) < 3:
            continue

        options_list = distractors + [answer]
        random.shuffle(options_list)
        labels = ["A", "B", "C", "D"]
        options = {lbl: opt for lbl, opt in zip(labels, options_list)}
        correct_label = labels[options_list.index(answer)]

        questions.append(
            {
                "question": question_text,
                "options": options,
                "correct": correct_label,
                "evidence": sent,
            }
        )

        if len(questions) >= n_mcq:
            break

    return questions

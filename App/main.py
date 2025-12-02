# App/main.py

import os
from typing import Optional

from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from dotenv import load_dotenv

from .models import (
    IngestResponse,
    AskRequest,
    AskResponse,
    StoryRequest,
    CaseRequest,
    QuizRequest,
)
from .ingest import extract_text_with_pages, recursive_split
from .rag import add_document, answer, list_books, story, business_case, quiz

# --------------------------------------------------------------------
# Load environment variables (for API keys etc.)
# --------------------------------------------------------------------
load_dotenv()

# --------------------------------------------------------------------
# Create FastAPI app
# --------------------------------------------------------------------
app = FastAPI(
    title="Textbook Tutor (Multi-Book)",
    version="0.1.0",
    description="Backend API for multi-book RAG: ingest PDFs, list books, ask questions, stories, cases, quizzes.",
)

# --------------------------------------------------------------------
# Simple health check
# --------------------------------------------------------------------
@app.get("/health")
async def health():
    return {"status": "ok"}


# --------------------------------------------------------------------
# List all books currently in the RAG store
# --------------------------------------------------------------------
@app.get("/books")
async def books():
    """
    Return whatever `list_books()` provides, e.g.
    {
        "book_1": {
            "book_title": "Some Book",
            "chapters": ["1", "2", "3"]
        },
        ...
    }
    """
    return list_books()


# --------------------------------------------------------------------
# Ingest a new PDF
# --------------------------------------------------------------------
@app.post("/ingest", response_model=IngestResponse)
async def ingest(
    file: UploadFile = File(...),
    book_title: Optional[str] = Form(default="Untitled Book"),
):
    # 1) Basic validation
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Please upload a PDF file")

    # 2) Save temp file
    raw = await file.read()
    os.makedirs("data", exist_ok=True)
    tmp_path = os.path.join("data", file.filename)
    with open(tmp_path, "wb") as f:
        f.write(raw)

    # 3) Extract pages: List[(page_no, text)]
    pages = extract_text_with_pages(tmp_path)

    # 4) Chunk long pages using recursive_split
    chunks_with_pages = []
    for page_no, text in pages:
        # skip empty pages
        if not text or not text.strip():
            continue

        if len(text) > 1200:
            for ch in recursive_split(text, 900, 120):
                chunks_with_pages.append((ch, page_no))
        else:
            chunks_with_pages.append((text, page_no))

    # 5) Add to vector store / index via rag.add_document
    book_id, n, chapters = add_document(chunks_with_pages, book_title=book_title)

    return IngestResponse(book_id=book_id, n_chunks=n, chapters=chapters)


# --------------------------------------------------------------------
# Ask a question over selected books/chapters
# --------------------------------------------------------------------
@app.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest):
    scope = req.scope.dict() if req.scope else None
    out = answer(req.query, k=req.k, scope=scope)
    # `out` should be a dict with keys matching AskResponse fields
    return AskResponse(**out)


# --------------------------------------------------------------------
# Generate a short story grounded in selected scope
# --------------------------------------------------------------------
@app.post("/story")
async def mk_story(req: StoryRequest):
    scope = req.scope.dict() if req.scope else None
    story_text = story(
        scope=scope,
        chapter_id=req.chapter_id,
        max_words=req.max_words,
        seed_question=req.seed_question,
    )
    return {"story": story_text}


# --------------------------------------------------------------------
# Generate a business-style case
# --------------------------------------------------------------------
@app.post("/case")
async def mk_case(req: CaseRequest):
    scope = req.scope.dict() if req.scope else None
    case_text = business_case(
        scope=scope,
        chapter_id=req.chapter_id,
        seed_question=req.seed_question,
    )
    return {"case": case_text}


# --------------------------------------------------------------------
# Generate quiz items (MCQs) for the selected scope
# --------------------------------------------------------------------
@app.post("/quiz")
async def mk_quiz(req: QuizRequest):
    scope = req.scope.dict() if req.scope else None
    items = quiz(
        scope=scope,
        chapter_id=req.chapter_id,
        n_mcq=req.n_mcq,
    )
    return {"items": items}

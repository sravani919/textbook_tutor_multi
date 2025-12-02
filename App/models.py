from pydantic import BaseModel, Field
from typing import List, Optional

class IngestResponse(BaseModel):
    book_id: str
    n_chunks: int
    chapters: List[str]

class AskScope(BaseModel):
    book_ids: Optional[List[str]] = None
    chapter_ids: Optional[List[str]] = None

class AskRequest(BaseModel):
    query: str = Field(..., examples=["Explain gradient descent in simple terms"])
    k: int = 5
    scope: Optional[AskScope] = None

class Citation(BaseModel):
    chunk_id: int
    page: int
    book_id: str
    chapter_id: str
    score: float

class AskResponse(BaseModel):
    answer: str
    citations: List[Citation]

class StoryRequest(BaseModel):
    chapter_id: Optional[str] = None
    scope: Optional[AskScope] = None
    max_words: int = 160
    seed_question: Optional[str] = None

class CaseRequest(BaseModel):
    chapter_id: Optional[str] = None
    scope: Optional[AskScope] = None
    seed_question: Optional[str] = None

class QuizRequest(BaseModel):
    chapter_id: Optional[str] = None
    scope: Optional[AskScope] = None
    n_mcq: int = 5

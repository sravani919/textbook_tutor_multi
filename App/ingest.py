import fitz  # PyMuPDF
from typing import List, Tuple
import re

def extract_text_with_pages(pdf_path: str) -> List[Tuple[int, str]]:
    doc = fitz.open(pdf_path)
    results = []
    for i, page in enumerate(doc):
        text = page.get_text("text")
        text = re.sub(r"\s+", " ", text).strip()
        if text:
            results.append((i + 1, text))
    return results

def recursive_split(text: str, max_chars: int = 900, overlap: int = 120) -> List[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + max_chars)
        chunks.append(text[start:end])
        start = max(end - overlap, end)
    return chunks

"""
processor.py
Cleans raw paper text and splits it into chunks suitable for embedding.
Prefers a paper's full text (when available) over its abstract, so
chunks carry as much real content as the source legally/technically allows.
"""

import re
from bs4 import BeautifulSoup
from pydantic import BaseModel
from typing import List
from app.data.europepmc_fetcher import Paper


class Chunk(BaseModel):
    """A single piece of text ready for embedding, traceable back to its source paper."""
    paper_id: str
    paper_title: str
    source: str
    chunk_index: int
    text: str
    is_india_affiliated: bool = False
    published: str = "unknown"
    is_full_text: bool = False  # True if this chunk came from the full paper, not just its abstract


def clean_text(raw_text: str) -> str:
    soup = BeautifulSoup(raw_text, "html.parser")
    text = soup.get_text()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def chunk_text(text: str, max_words: int = 150) -> List[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text)

    chunks = []
    current_chunk = []
    current_word_count = 0

    for sentence in sentences:
        word_count = len(sentence.split())

        if current_word_count + word_count > max_words and current_chunk:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            current_word_count = 0

        current_chunk.append(sentence)
        current_word_count += word_count

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


def process_paper(paper: Paper, max_words: int = 150) -> List[Chunk]:
    # Prefer full text when it's actually available (open-access fetched
    # papers that successfully returned full text), otherwise fall back
    # to the abstract. User-uploaded PDFs already have their COMPLETE
    # extracted text sitting in `summary` (upload_handler puts the whole
    # document there), so we treat those as full text too.
    fetched_full_text = getattr(paper, "full_text", None)
    is_full_text = bool(fetched_full_text) or (paper.source == "user_upload")
    source_text = fetched_full_text or paper.summary

    cleaned = clean_text(source_text)
    text_chunks = chunk_text(cleaned, max_words=max_words)

    return [
        Chunk(
            paper_id=paper.id,
            paper_title=paper.title,
            source=paper.source,
            chunk_index=i,
            text=chunk,
            is_india_affiliated=paper.is_india_affiliated,
            published=paper.published,
            is_full_text=is_full_text
        )
        for i, chunk in enumerate(text_chunks)
    ]


if __name__ == "__main__":
    from app.data.fetcher import fetch_papers

    papers = fetch_papers("tuberculosis drug resistance India", max_results=2, india_only=True)

    for paper in papers:
        print(f"\n=== {paper.title} ===")
        chunks = process_paper(paper)
        print(f"Split into {len(chunks)} chunk(s), full_text={chunks[0].is_full_text if chunks else 'N/A'}")
        for c in chunks:
            print(f"\n  [Chunk {c.chunk_index}] Published: {c.published}")
            print(" ", c.text[:150], "...")
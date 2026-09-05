"""
processor.py
Cleans raw paper text and splits it into chunks suitable for embedding.

Why chunking matters: embedding models work best on smaller, focused
pieces of text (a few sentences to a paragraph). A whole abstract is
usually fine as one chunk, but full-text papers (later) need splitting.
This module handles both cases with the same logic.
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


def clean_text(raw_text: str) -> str:
    """
    Remove HTML/XML tags and normalize whitespace from raw text.
    """
    # Strip any HTML/XML tags (some abstracts contain <i>, <sub>, etc.)
    soup = BeautifulSoup(raw_text, "html.parser")
    text = soup.get_text()

    # Collapse multiple spaces/newlines into single spaces
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def chunk_text(text: str, max_words: int = 150) -> List[str]:
    """
    Split text into chunks of roughly `max_words` words, breaking on
    sentence boundaries so we don't cut a sentence in half.

    Args:
        text: cleaned text to split
        max_words: target max words per chunk

    Returns:
        A list of text chunks.
    """
    # Simple sentence split (good enough for abstracts; we can upgrade
    # to a proper NLP sentence splitter later if needed)
    sentences = re.split(r"(?<=[.!?])\s+", text)

    chunks = []
    current_chunk = []
    current_word_count = 0

    for sentence in sentences:
        word_count = len(sentence.split())

        if current_word_count + word_count > max_words and current_chunk:
            # Save the current chunk and start a new one
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            current_word_count = 0

        current_chunk.append(sentence)
        current_word_count += word_count

    # Add the last remaining chunk
    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


def process_paper(paper: Paper, max_words: int = 150) -> List[Chunk]:
    """
    Clean and chunk a single paper's abstract/text into Chunk objects.

    Args:
        paper: a Paper object (from europepmc_fetcher or arxiv_fetcher)
        max_words: target words per chunk

    Returns:
        A list of Chunk objects, each traceable back to the source paper.
    """
    cleaned = clean_text(paper.summary)
    text_chunks = chunk_text(cleaned, max_words=max_words)

    return [
        Chunk(
            paper_id=paper.id,
            paper_title=paper.title,
            source=paper.source,
            chunk_index=i,
            text=chunk
        )
        for i, chunk in enumerate(text_chunks)
    ]


if __name__ == "__main__":
    from app.data.fetcher import fetch_papers

    papers = fetch_papers("tuberculosis drug resistance India", max_results=2, india_only=True)

    for paper in papers:
        print(f"\n=== {paper.title} ===")
        chunks = process_paper(paper)
        print(f"Split into {len(chunks)} chunk(s)")
        for c in chunks:
            print(f"\n  [Chunk {c.chunk_index}] ({len(c.text.split())} words)")
            print(" ", c.text[:150], "...")
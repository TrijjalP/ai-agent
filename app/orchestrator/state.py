"""
state.py
Defines the shared state object that flows through all agents.
Every agent reads from this and adds its own piece to it.
"""

from typing import TypedDict, List, Optional


class ResearchState(TypedDict, total=False):
    # --- Input (set at the start) ---
    topic: str                    # e.g. "tuberculosis drug resistance in India"
    india_only: bool              # whether to prioritize India-affiliated research
    year_from: Optional[int]
    year_to: Optional[int]

    # --- Filled in by Retrieval Agent ---
    retrieved_chunks: List[dict]  # relevant chunks pulled from ChromaDB (with metadata)

    # --- Filled in later agents (placeholders for now) ---
    extracted_facts: List[dict]
    identified_gaps: List[dict]
    hypotheses: List[dict]
    critiques: List[dict]
    ranked_hypotheses: List[dict]
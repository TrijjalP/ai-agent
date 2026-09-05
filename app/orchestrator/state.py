"""
state.py
Defines the shared state object that flows through all agents.
Every agent reads from this and adds its own piece to it.
"""

from typing import TypedDict, List, Optional


class ResearchState(TypedDict, total=False):
    # --- Input (set at the start) ---
    topic: str                    # e.g. "AI in biomedical research"
    india_only: bool              # whether to prioritize India-affiliated research
    year_from: Optional[int]
    year_to: Optional[int]
    gap_focus: str                # user's instruction for what kind of gap analysis they want
                                   # e.g. "compare India vs other countries"
                                   # e.g. "general gaps in current research"
                                   # e.g. "2021 onwards, India vs globally"
    score_weights: dict            # user-defined weights for critique scoring, e.g.
                                    # {"feasibility_score": 0.3, "novelty_score": 0.2,
                                    #  "evidence_grounding_score": 0.3, "specificity_score": 0.2}
                                    # must sum to 1.0; defaults to equal weights if not provided
    uploaded_file_paths: List[str]  # local paths to up to 4 user-uploaded PDF/txt files
    # --- Filled in by Retrieval Agent ---
    retrieved_chunks: List[dict]

    # --- Filled in by Extraction Agent ---
    extracted_facts: List[dict]

    # --- Filled in by Gap Detection Agent ---
    identified_gaps: List[dict]

    # --- Placeholders for later agents ---
    hypotheses: List[dict]
    critiques: List[dict]
    ranked_hypotheses: List[dict]
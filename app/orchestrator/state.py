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
    gap_focus: str                # user's instruction for what kind of comparison/gap
                                   # analysis they want, e.g. "compare methodology and
                                   # conclusions across these papers"
    score_weights: dict            # user-defined weights for critique scoring, e.g.
                                    # {"feasibility_score": 0.3, "novelty_score": 0.2,
                                    #  "evidence_grounding_score": 0.3, "specificity_score": 0.2}
                                    # must sum to 1.0; defaults to equal weights if not provided
    uploaded_file_paths: List[str]  # local paths to up to 4 user-uploaded PDF/txt files

    # --- Filled in by Retrieval Agent ---
    retrieved_chunks: List[dict]   # flattened chunks belonging to the 4 selected papers
    selected_papers: list           # the 4 (or fewer) working-set papers with their chunks
    candidate_papers: list          # remaining matching papers, for the swap UI
    confirmed_paper_ids: list        # set by the API when the user confirms/overrides the selection

    # --- Filled in by Comparison Agent (replaces Extraction + Gap Detection) ---
    paper_summaries: List[dict]     # per paper: {paper_id, paper_title, methodology_summary, conclusion_summary}
    identified_gaps: List[dict]     # differences/under-explored angles between the 4 papers
                                     # (same shape as before: gap_description, gap_type, supporting_papers)
    recommended_paper: dict          # {paper_id, paper_title, rationale} - the "best paper to use" pick

    # --- Filled in by Hypothesis Generation Agent ---
    hypotheses: List[dict]

    # --- Filled in by Critique Agent ---
    critiques: List[dict]

    # --- Filled in by Ranking (plain code step) ---
    ranked_hypotheses: List[dict]

    # --- Filled in by Retrieval Agent's topic guard ---
    is_valid_topic: bool
    validation_message: str
"""
ranking_agent.py
The Ranking Agent: merges each hypothesis with its critique data, sorts
them by overall soundness score, and generates a short rationale
explaining the ranking — especially why the top hypothesis was ranked
highest. This produces the final, UI-ready structure for the whole pipeline.
"""

from app.utils.llm_client import call_llm
from app.orchestrator.state import ResearchState


RATIONALE_PROMPT_TEMPLATE = """You are summarizing why a research hypothesis was ranked #1 among several candidates.

Top hypothesis: "{hypothesis}"
Scores: Feasibility {feasibility}/10, Novelty {novelty}/10, Evidence Grounding {grounding}/10, Specificity {specificity}/10
Overall score: {overall}/10

In 2-3 short sentences, explain why this hypothesis ranked highest compared to the others, referencing its strongest scoring criteria.
Be concise — under 60 words total. Respond with plain text only, no JSON, no markdown.
"""


def merge_hypothesis_and_critique(hypotheses: list, critiques: list, retrieved_chunks: list) -> list:
    """
    Merges each hypothesis with its corresponding critique data, and attaches
    a clear list of source papers (with origin: fetched vs user-uploaded)
    that the hypothesis's grounding evidence maps back to.
    """
    # Build a lookup: paper_title -> source type, from the actual retrieved chunks
    title_to_source = {}
    for chunk in retrieved_chunks:
        title_to_source[chunk["paper_title"]] = chunk["source"]

    merged = []
    for hyp, crit in zip(hypotheses, critiques):
        grounding_titles = hyp.get("grounding_evidence", [])

        source_breakdown = []
        for title in grounding_titles:
            origin = title_to_source.get(title, "unknown")
            origin_label = "User-uploaded" if origin == "user_upload" else (
                "arXiv" if origin == "arxiv" else "Europe PMC (fetched)"
            )
            source_breakdown.append({"title": title, "origin": origin_label})

        merged.append({
            "hypothesis": hyp.get("hypothesis", ""),
            "addresses_gap": hyp.get("addresses_gap", ""),
            "novelty_justification": hyp.get("novelty_justification", ""),
            "grounding_evidence": grounding_titles,
            "source_breakdown": source_breakdown,
            "feasibility_score": crit.get("feasibility_score", 0),
            "novelty_score": crit.get("novelty_score", 0),
            "evidence_grounding_score": crit.get("evidence_grounding_score", 0),
            "specificity_score": crit.get("specificity_score", 0),
            "overall_soundness_score": crit.get("overall_soundness_score", 0),
            "weaknesses": crit.get("weaknesses", []),
            "existing_evidence_relation": crit.get("existing_evidence_relation", "")
        })
    return merged


def generate_ranking_rationale(top_hypothesis: dict) -> str:
    """
    Asks the LLM for a short, human-readable explanation of why the
    top-ranked hypothesis earned the highest score. Defensively filters
    out any sentence that looks like a hallucinated new instruction
    (small local models occasionally drift into inventing an unrelated
    task partway through, even within the first few sentences).
    """
    prompt = RATIONALE_PROMPT_TEMPLATE.format(
        hypothesis=top_hypothesis["hypothesis"],
        feasibility=top_hypothesis["feasibility_score"],
        novelty=top_hypothesis["novelty_score"],
        grounding=top_hypothesis["evidence_grounding_score"],
        specificity=top_hypothesis["specificity_score"],
        overall=top_hypothesis["overall_soundness_score"]
    )

    rationale = call_llm(prompt, temperature=0.3, max_tokens=80)
    rationale = rationale.strip()

    import re
    sentences = re.split(r"(?<=[.!?])\s+", rationale)

    # Drop any sentence that looks like the model drifted into a new
    # instruction/task rather than continuing to explain the ranking.
    instruction_like_starts = (
        "you are", "you're", "your task", "evaluate the following",
        "consider the following", "please evaluate"
    )

    clean_sentences = []
    for s in sentences:
        s_lower = s.strip().lower()
        if any(s_lower.startswith(phrase) for phrase in instruction_like_starts):
            break  # stop entirely once drift is detected, don't include anything after either
        clean_sentences.append(s)

    cleaned = " ".join(clean_sentences[:2]).strip()  # cap at 2 sentences even if all clean

    return cleaned if cleaned else "Ranked highest based on overall weighted score across all evaluation criteria."


def run_ranking_agent(state: ResearchState) -> ResearchState:
    """
    Ranking Agent's job:
    1. Merge hypotheses with their critiques and source origin breakdown
    2. Sort by overall_soundness_score, highest first
    3. Assign a rank number to each
    4. Generate a short rationale for why the #1 hypothesis ranked highest

    Returns:
        Updated state with 'ranked_hypotheses' populated (final output for the UI).
    """
    hypotheses = state.get("hypotheses", [])
    critiques = state.get("critiques", [])
    retrieved_chunks = state.get("retrieved_chunks", [])

    print(f"[Ranking Agent] Merging {len(hypotheses)} hypotheses with their critiques...")

    if not hypotheses or not critiques:
        print("[Ranking Agent] No hypotheses/critiques available to rank.")
        state["ranked_hypotheses"] = []
        return state

    merged = merge_hypothesis_and_critique(hypotheses, critiques, retrieved_chunks)

    print("[Ranking Agent] Sorting by overall soundness score...")
    ranked = sorted(merged, key=lambda h: h["overall_soundness_score"], reverse=True)

    for i, hyp in enumerate(ranked, start=1):
        hyp["rank"] = i

    print("[Ranking Agent] Generating rationale for top-ranked hypothesis...")
    if ranked:
        ranked[0]["ranking_rationale"] = generate_ranking_rationale(ranked[0])
        for hyp in ranked[1:]:
            hyp["ranking_rationale"] = None

    print(f"[Ranking Agent] Done. Ranked {len(ranked)} hypothesis/hypotheses.")

    state["ranked_hypotheses"] = ranked
    return state


if __name__ == "__main__":
    from app.agents.retrieval_agent import run_retrieval_agent
    from app.agents.extraction_agent import run_extraction_agent
    from app.agents.gap_detection_agent import run_gap_detection_agent
    from app.agents.hypothesis_generation_agent import run_hypothesis_generation_agent
    from app.agents.hypothesis_critique_agent import run_hypothesis_critique_agent

    custom_weights = {
        "feasibility_score": 0.35,
        "novelty_score": 0.15,
        "evidence_grounding_score": 0.35,
        "specificity_score": 0.15
    }

    initial_state: ResearchState = {
        "topic": "AI in biomedical research",
        "india_only": False,
        "year_from": 2021,
        "year_to": 2026,
        "gap_focus": "Compare AI in biomedical research in India versus other countries, focused on research from 2021 onwards",
        "score_weights": custom_weights
    }

    state = run_retrieval_agent(initial_state, n_results=6)
    state = run_extraction_agent(state)
    state = run_gap_detection_agent(state)
    state = run_hypothesis_generation_agent(state)
    state = run_hypothesis_critique_agent(state)
    state = run_ranking_agent(state)

    print("\n=== FINAL RANKED HYPOTHESES ===")
    for hyp in state["ranked_hypotheses"]:
        print(f"\n--- Rank #{hyp['rank']} (Score: {hyp['overall_soundness_score']}/10) ---")
        print("Hypothesis:", hyp["hypothesis"])
        print("Addresses gap:", hyp["addresses_gap"])
"""
gap_detection_agent.py
The Gap Detection Agent: looks across ALL extracted facts together (not one
chunk at a time) and identifies gaps, contradictions, or under-explored
areas in the current literature — guided by the user's own framing of what
kind of comparison/gap analysis they want (general, India vs global,
time-based, etc.)
"""

import json
from app.utils.llm_client import call_llm
from app.orchestrator.state import ResearchState


GAP_DETECTION_PROMPT_TEMPLATE = """You are a senior biomedical research analyst. Below is a set of facts extracted from multiple research papers, each tagged with its source paper, publication date, and whether the research is India-affiliated.

The user wants the following type of analysis:
"{gap_focus}"

Facts:
{facts_block}

Based on the facts above and the user's requested focus, identify 3 to 5 concrete research gaps.
For each gap, explain:
- what is missing, under-explored, or contradictory
- which papers/findings support this observation (use exact paper titles)
- whether this gap relates to a comparison the user asked for (e.g. India vs global, or a time period), if relevant

Respond ONLY with a valid JSON array in this exact format, with no extra text before or after:
[
  {{
    "gap_description": "clear description of the gap",
    "supporting_papers": ["exact paper title 1", "exact paper title 2"],
    "gap_type": "short label, e.g. 'India vs Global', 'Temporal', 'Methodological', 'General'"
  }}
]
"""


def format_facts_for_prompt(facts: list) -> str:
    """Turns the list of extracted fact dicts into readable text for the LLM prompt."""
    lines = []
    for i, fact in enumerate(facts, start=1):
        region_tag = "India-affiliated" if fact.get("is_india_affiliated") else "General/Global"
        lines.append(
            f"{i}. Paper: \"{fact['paper_title']}\" "
            f"(Published: {fact.get('published', 'unknown')}, {region_tag})\n"
            f"   Finding: {fact['key_finding']}\n"
            f"   Entities: {', '.join(fact.get('entities', [])) or 'none'}\n"
            f"   Statistics: {', '.join(fact.get('statistics', [])) or 'none'}"
        )
    return "\n\n".join(lines)


def run_gap_detection_agent(state: ResearchState) -> ResearchState:
    """
    Gap Detection Agent's job:
    1. Take all extracted facts from state['extracted_facts']
    2. Format them for the LLM along with the user's gap_focus instruction
    3. Ask the LLM to identify concrete, traceable research gaps

    Returns:
        Updated state with 'identified_gaps' populated.
    """
    facts = state.get("extracted_facts", [])
    gap_focus = state.get("gap_focus", "General gaps in current research")

    print(f"[Gap Detection Agent] Analyzing {len(facts)} facts with focus: '{gap_focus}'")

    if not facts:
        print("[Gap Detection Agent] No facts available to analyze.")
        state["identified_gaps"] = []
        return state

    facts_block = format_facts_for_prompt(facts)
    prompt = GAP_DETECTION_PROMPT_TEMPLATE.format(gap_focus=gap_focus, facts_block=facts_block)

    raw_response = call_llm(prompt, temperature=0.4)

    try:
        cleaned = raw_response.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`")
            cleaned = cleaned.replace("json", "", 1).strip()

        gaps = json.loads(cleaned)
        if not isinstance(gaps, list):
            raise ValueError("Expected a JSON list")
    except (json.JSONDecodeError, ValueError, AttributeError):
        print("[Gap Detection Agent] Warning: could not parse LLM output as JSON. Using fallback.")
        gaps = [{
            "gap_description": raw_response[:300] if raw_response else "Could not identify gaps",
            "supporting_papers": [],
            "gap_type": "Unparsed"
        }]

    print(f"[Gap Detection Agent] Identified {len(gaps)} gap(s).")

    state["identified_gaps"] = gaps
    return state


if __name__ == "__main__":
    from app.agents.retrieval_agent import run_retrieval_agent
    from app.agents.extraction_agent import run_extraction_agent

    initial_state: ResearchState = {
        "topic": "AI in biomedical research",
        "india_only": False,
        "year_from": 2021,
        "year_to": 2026,
        "gap_focus": "Compare AI in biomedical research in India versus other countries, focused on research from 2021 onwards"
    }

    state = run_retrieval_agent(initial_state, n_results=6)
    state = run_extraction_agent(state)
    state = run_gap_detection_agent(state)

    print("\n=== Identified Gaps ===")
    for i, gap in enumerate(state["identified_gaps"], start=1):
        print(f"\n--- Gap {i} ({gap.get('gap_type', 'Unknown')}) ---")
        print("Description:", gap["gap_description"])
        print("Supporting papers:", gap.get("supporting_papers", []))
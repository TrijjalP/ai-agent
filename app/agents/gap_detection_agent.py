"""
gap_detection_agent.py
The Gap Detection Agent: looks across ALL extracted facts together (not one
chunk at a time) and identifies gaps, contradictions, or under-explored
areas in the current literature — guided by the user's own framing of what
kind of comparison/gap analysis they want (general, India vs global,
time-based, etc.)
"""

from app.utils.llm_json import call_llm_for_json
from app.orchestrator.state import ResearchState


GAP_DETECTION_PROMPT_TEMPLATE = """You are a senior biomedical research analyst. Below is a set of facts extracted from multiple research papers, each numbered, tagged with its source paper, publication date, and whether the research is India-affiliated.

The user wants the following type of analysis:
"{gap_focus}"

Facts:
{facts_block}

Identify 3 to 5 concrete research gaps. Write COMPLETE descriptions - do not truncate.
For each gap:
- gap_description: 20-35 words, clearly explaining what is missing or under-explored.
- supporting_paper_numbers: list of paper NUMBERS (integers, e.g. [1, 3]) from the facts above that support this gap. Do NOT write paper titles, only numbers.
- gap_type: one short label (e.g. 'India vs Global', 'Temporal', 'Methodological', 'General').

Respond ONLY with a valid, complete JSON array in this exact format, with no extra text before or after:
[
  {{
    "gap_description": "complete 20-35 word description here",
    "supporting_paper_numbers": [1, 2],
    "gap_type": "label"
  }}
]
"""


def format_facts_for_prompt(facts: list) -> str:
    """Turns the list of extracted fact dicts into readable, numbered text for the LLM prompt."""
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
    2. Format them (numbered) for the LLM along with the user's gap_focus instruction
    3. Ask the LLM to identify concrete gaps, referencing supporting facts by NUMBER
    4. Programmatically resolve those numbers to exact real paper titles ourselves,
       guaranteeing accurate traceability regardless of how the LLM phrases things

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

    gaps, raw_response = call_llm_for_json(prompt, temperature=0.4, max_retries=3, agent_name="Gap Detection Agent")

    if not gaps or not isinstance(gaps, list):
        print("[Gap Detection Agent] Warning: could not parse LLM output as JSON. Using fallback.")
        gaps = [{
            "gap_description": raw_response[:300] if raw_response else "Could not identify gaps",
            "supporting_papers": [],
            "gap_type": "Unparsed"
        }]
    else:
        # Resolve paper NUMBERS to exact real titles ourselves, rather than
        # trusting the LLM to reproduce titles verbatim (which previously
        # caused mangled text like "Paper 3: '...'" that broke downstream
        # source attribution matching in the Ranking Agent).
        for gap in gaps:
            numbers = gap.get("supporting_paper_numbers", [])
            resolved_titles = []
            for num in numbers:
                if isinstance(num, str) and num.strip().isdigit():
                    num = int(num.strip())
                if isinstance(num, int) and 1 <= num <= len(facts):
                    resolved_titles.append(facts[num - 1]["paper_title"])
            gap["supporting_papers"] = resolved_titles

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
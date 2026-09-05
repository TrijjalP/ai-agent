"""
hypothesis_generation_agent.py
The Hypothesis Generation Agent: takes identified research gaps (grounded in
extracted facts) and generates novel, testable research hypotheses to
address them. Each hypothesis explicitly states which gap it targets and
why it's novel, so downstream agents (Critique, Ranking) and the final UI
have rich, traceable content to work with.
"""

import json
# from app.utils.json_parser import extract_json
# from app.utils.llm_client import call_llm
from app.utils.llm_json import call_llm_for_json
from app.orchestrator.state import ResearchState


HYPOTHESIS_PROMPT_TEMPLATE = """You are a biomedical research scientist generating new, testable research hypotheses.

The research topic is: "{topic}"

Here are the identified gaps in current research:
{gaps_block}

Here is a summary of existing findings from the literature (for grounding, so your hypotheses are realistic and build on real evidence, not speculation):
{facts_block}

For EACH gap listed above, propose ONE novel, testable research hypothesis that could help address it.

Respond ONLY with a valid JSON array in this exact format, with no extra text before or after:
[
  {{
    "hypothesis": "a clear, specific, testable hypothesis statement",
    "addresses_gap": "short restatement of which gap this targets",
    "novelty_justification": "1-2 sentences explaining what makes this hypothesis new/different from existing research",
    "grounding_evidence": ["exact paper title(s) this hypothesis builds on"]
  }}
]
"""


def format_gaps_for_prompt(gaps: list) -> str:
    lines = []
    for i, gap in enumerate(gaps, start=1):
        lines.append(
            f"{i}. [{gap.get('gap_type', 'General')}] {gap['gap_description']}\n"
            f"   Supporting papers: {', '.join(gap.get('supporting_papers', [])) or 'none listed'}"
        )
    return "\n\n".join(lines)


def format_facts_for_prompt(facts: list) -> str:
    lines = []
    for i, fact in enumerate(facts, start=1):
        lines.append(f"{i}. \"{fact['paper_title']}\": {fact['key_finding']}")
    return "\n".join(lines)


def run_hypothesis_generation_agent(state: ResearchState) -> ResearchState:
    """
    Hypothesis Generation Agent's job:
    1. Take identified gaps + extracted facts from state
    2. Ask the LLM to propose one testable hypothesis per gap
    3. Require inline novelty justification and grounding evidence

    Returns:
        Updated state with 'hypotheses' populated.
    """
    gaps = state.get("identified_gaps", [])
    facts = state.get("extracted_facts", [])
    topic = state.get("topic", "the given research area")

    print(f"[Hypothesis Generation Agent] Generating hypotheses for {len(gaps)} gap(s)...")

    if not gaps:
        print("[Hypothesis Generation Agent] No gaps available to generate hypotheses from.")
        state["hypotheses"] = []
        return state

    gaps_block = format_gaps_for_prompt(gaps)
    facts_block = format_facts_for_prompt(facts)

    prompt = HYPOTHESIS_PROMPT_TEMPLATE.format(
        topic=topic,
        gaps_block=gaps_block,
        facts_block=facts_block
    )

    hypotheses, raw_response = call_llm_for_json(prompt, temperature=0.6, agent_name="Hypothesis Generation Agent")

    if not hypotheses or not isinstance(hypotheses, list):
        print("[Hypothesis Generation Agent] Warning: could not parse LLM output as JSON. Using fallback.")
        hypotheses = [{
            "hypothesis": raw_response[:300] if raw_response else "Could not generate hypothesis",
            "addresses_gap": "Unknown",
            "novelty_justification": "N/A (parsing failed)",
            "grounding_evidence": []
        }]

    print(f"[Hypothesis Generation Agent] Generated {len(hypotheses)} hypothesis/hypotheses.")

    state["hypotheses"] = hypotheses
    return state


if __name__ == "__main__":
    from app.agents.retrieval_agent import run_retrieval_agent
    from app.agents.extraction_agent import run_extraction_agent
    from app.agents.gap_detection_agent import run_gap_detection_agent

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
    state = run_hypothesis_generation_agent(state)

    print("\n=== Generated Hypotheses ===")
    for i, hyp in enumerate(state["hypotheses"], start=1):
        print(f"\n--- Hypothesis {i} ---")
        print("Statement:", hyp["hypothesis"])
        print("Addresses gap:", hyp["addresses_gap"])
        print("Novelty:", hyp["novelty_justification"])
        print("Grounded in:", hyp.get("grounding_evidence", []))
"""
hypothesis_generation_agent.py
The Hypothesis Generation Agent: takes identified research gaps (grounded in
extracted facts) and generates novel, testable research hypotheses to
address them. Each hypothesis explicitly states which gap it targets and
why it's novel, so downstream agents (Critique, Ranking) and the final UI
have rich, traceable content to work with.
"""

from app.utils.llm_json import call_llm_for_json
from app.orchestrator.state import ResearchState


HYPOTHESIS_PROMPT_TEMPLATE = """You are a biomedical research scientist generating new, testable research hypotheses.

The research topic is: "{topic}"

Here are the identified gaps in current research, each numbered:
{gaps_block}

Here is a summary of existing findings from the literature, each numbered (for grounding, so your hypotheses are realistic and build on real evidence, not speculation):
{facts_block}

For EACH gap listed above, propose ONE novel, testable research hypothesis that could help address it.
Reference which gap each hypothesis addresses using its NUMBER only (not a restatement of the text).
Reference which facts support your hypothesis using their NUMBERS only (not titles).

Respond ONLY with a valid JSON array in this exact format, with no extra text before or after:
[
  {{
    "hypothesis": "a clear, specific, testable hypothesis statement",
    "addresses_gap_number": 1,
    "novelty_justification": "1-2 sentences explaining what makes this hypothesis new/different from existing research",
    "grounding_evidence_numbers": [1, 2]
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
    2. Ask the LLM to propose one testable hypothesis per gap, referencing
       gaps AND grounding facts by number only (more reliable than trusting
       the LLM to restate gap/paper text accurately)
    3. Programmatically attach the FULL original gap_description and exact
       real paper titles ourselves, guaranteeing every hypothesis always
       shows detailed, accurately traceable data regardless of how the
       LLM phrases things

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

    hypotheses, raw_response = call_llm_for_json(prompt, temperature=0.6, max_retries=3, max_tokens=3072, agent_name="Hypothesis Generation Agent")

    if not hypotheses or not isinstance(hypotheses, list):
        print("[Hypothesis Generation Agent] Warning: could not parse LLM output as JSON. Using fallback.")
        hypotheses = [{
            "hypothesis": raw_response[:300] if raw_response else "Could not generate hypothesis",
            "addresses_gap": "Unknown",
            "gap_type": "General",
            "novelty_justification": "N/A (parsing failed)",
            "grounding_evidence": []
        }]
    else:
        # Programmatically attach the FULL, original gap description AND
        # exact real paper titles ourselves, rather than trusting the LLM
        # to restate them consistently. This guarantees every hypothesis
        # always shows detailed, traceable data regardless of how the LLM
        # phrases its own response.
        for hyp in hypotheses:
            gap_num = hyp.get("addresses_gap_number")
            if isinstance(gap_num, str) and gap_num.strip().isdigit():
                gap_num = int(gap_num.strip())

            if isinstance(gap_num, int) and 1 <= gap_num <= len(gaps):
                matched_gap = gaps[gap_num - 1]
                hyp["addresses_gap"] = matched_gap.get("gap_description", "Unknown")
                hyp["gap_type"] = matched_gap.get("gap_type", "General")
            else:
                hyp["addresses_gap"] = "Unknown"
                hyp["gap_type"] = "General"

            # Resolve grounding evidence numbers to exact real paper titles
            evidence_numbers = hyp.get("grounding_evidence_numbers", [])
            resolved_evidence = []
            for num in evidence_numbers:
                if isinstance(num, str) and num.strip().isdigit():
                    num = int(num.strip())
                if isinstance(num, int) and 1 <= num <= len(facts):
                    resolved_evidence.append(facts[num - 1]["paper_title"])
            hyp["grounding_evidence"] = resolved_evidence

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
        print("Gap type:", hyp.get("gap_type", "General"))
        print("Novelty:", hyp["novelty_justification"])
        print("Grounded in:", hyp.get("grounding_evidence", []))
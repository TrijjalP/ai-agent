"""
hypothesis_critique_agent.py
The Hypothesis Critique Agent: takes each generated hypothesis and
critically evaluates it using an explicit 4-criteria rubric (Feasibility,
Novelty, Evidence Grounding, Specificity). Each criterion is scored 0-10
by the LLM, and we COMPUTE the overall soundness score ourselves as a
weighted average.

The weighting is USER-CONFIGURABLE via state['score_weights'], since the
"right" balance between these criteria genuinely depends on the research
context (e.g. a clinical feasibility study may value Feasibility highly,
while an exploratory/theoretical research review may value Novelty more).
If the user doesn't provide weights, we default to equal weighting (25%
each) as a neutral, defensible baseline.
"""

# from app.utils.llm_client import call_llm
# from app.utils.json_parser import extract_json
from app.utils.llm_json import call_llm_for_json
from app.orchestrator.state import ResearchState


DEFAULT_SCORE_WEIGHTS = {
    "feasibility_score": 0.25,
    "novelty_score": 0.25,
    "evidence_grounding_score": 0.25,
    "specificity_score": 0.25
}


CRITIQUE_PROMPT_TEMPLATE = """You are a skeptical, rigorous peer reviewer evaluating a proposed biomedical research hypothesis.

Hypothesis to evaluate:
"{hypothesis}"

This hypothesis claims to address the following gap:
"{addresses_gap}"

It is grounded in these existing findings:
{facts_block}

Score this hypothesis on FOUR criteria, each from 0 (very poor) to 10 (excellent):
1. Feasibility: can this realistically be tested with available data/methods?
2. Novelty: does it meaningfully differ from existing research?
3. Evidence grounding: is it well-supported by the facts above, not speculative?
4. Specificity: is it precise and falsifiable, not vague or overly broad?

Also list weaknesses and describe how existing evidence relates to it.

Be extremely concise. Keep each weakness under 12 words. Provide exactly 2 weaknesses.
Keep existing_evidence_relation under 20 words.

Respond ONLY with a valid JSON object in this exact format, with no extra text before or after:
{{
  "feasibility_score": 0-10,
  "novelty_score": 0-10,
  "evidence_grounding_score": 0-10,
  "specificity_score": 0-10,
  "weaknesses": ["short weakness 1", "short weakness 2"],
  "existing_evidence_relation": "one short sentence, under 20 words"
}}
"""


def format_facts_for_prompt(facts: list) -> str:
    lines = []
    for fact in facts:
        lines.append(f"- \"{fact['paper_title']}\": {fact['key_finding']}")
    return "\n".join(lines) if lines else "No supporting facts available."


def validate_weights(weights: dict) -> dict:
    """
    Validates user-supplied weights. Falls back to equal weighting if
    weights are missing, incomplete, or don't sum to approximately 1.0.
    """
    if not weights:
        return DEFAULT_SCORE_WEIGHTS

    required_keys = set(DEFAULT_SCORE_WEIGHTS.keys())
    if set(weights.keys()) != required_keys:
        print("[Hypothesis Critique Agent] Warning: score_weights missing required keys. Using defaults.")
        return DEFAULT_SCORE_WEIGHTS

    total = sum(weights.values())
    if not (0.99 <= total <= 1.01):  # allow tiny floating point tolerance
        print(f"[Hypothesis Critique Agent] Warning: score_weights sum to {total}, not 1.0. Using defaults.")
        return DEFAULT_SCORE_WEIGHTS

    return weights


def compute_overall_score(scores: dict, weights: dict) -> float:
    """
    Computes a weighted overall soundness score (0-10) from the 4 sub-scores,
    using the given (user-defined or default) weights.
    """
    total = 0.0
    for key, weight in weights.items():
        total += scores.get(key, 0) * weight
    return round(total, 2)


def critique_single_hypothesis(hypothesis: dict, facts: list, weights: dict, max_retries: int = 1) -> dict:
    facts_block = format_facts_for_prompt(facts)

    prompt = CRITIQUE_PROMPT_TEMPLATE.format(
        hypothesis=hypothesis.get("hypothesis", ""),
        addresses_gap=hypothesis.get("addresses_gap", ""),
        facts_block=facts_block
    )

    parsed, raw_response = call_llm_for_json(prompt, temperature=0.3, agent_name="Hypothesis Critique Agent")

    if parsed and isinstance(parsed, dict):
        sub_scores = {
            "feasibility_score": parsed.get("feasibility_score", 0),
            "novelty_score": parsed.get("novelty_score", 0),
            "evidence_grounding_score": parsed.get("evidence_grounding_score", 0),
            "specificity_score": parsed.get("specificity_score", 0)
        }
        overall_score = compute_overall_score(sub_scores, weights)

        return {
            "feasibility_score": sub_scores["feasibility_score"],
            "novelty_score": sub_scores["novelty_score"],
            "evidence_grounding_score": sub_scores["evidence_grounding_score"],
            "specificity_score": sub_scores["specificity_score"],
            "overall_soundness_score": overall_score,
            "weaknesses": parsed.get("weaknesses", []),
            "existing_evidence_relation": parsed.get("existing_evidence_relation", "")
        }
    else:
        print("[Hypothesis Critique Agent] Warning: could not parse LLM output as JSON. Using fallback.")
        return {
            "feasibility_score": 0,
            "novelty_score": 0,
            "evidence_grounding_score": 0,
            "specificity_score": 0,
            "overall_soundness_score": 0,
            "weaknesses": ["Could not parse critique"],
            "existing_evidence_relation": raw_response[:200] if raw_response else "N/A"
        }


def run_hypothesis_critique_agent(state: ResearchState) -> ResearchState:
    """
    Hypothesis Critique Agent's job:
    1. Take each hypothesis from state['hypotheses']
    2. Critically evaluate it against 4 explicit criteria
    3. Compute a weighted overall soundness score using user-defined
       (or default equal) weights from state['score_weights']

    Returns:
        Updated state with 'critiques' populated.
    """
    hypotheses = state.get("hypotheses", [])
    facts = state.get("extracted_facts", [])
    weights = validate_weights(state.get("score_weights"))

    print(f"[Hypothesis Critique Agent] Critiquing {len(hypotheses)} hypothesis/hypotheses...")
    print(f"[Hypothesis Critique Agent] Using score weights: {weights}")

    critiques = []
    for i, hyp in enumerate(hypotheses, start=1):
        print(f"[Hypothesis Critique Agent] Evaluating hypothesis {i}/{len(hypotheses)}...")
        critique = critique_single_hypothesis(hyp, facts, weights)
        critiques.append({
            "hypothesis": hyp.get("hypothesis", ""),
            **critique
        })

    print(f"[Hypothesis Critique Agent] Done. Critiqued {len(critiques)} hypothesis/hypotheses.")

    state["critiques"] = critiques
    return state


if __name__ == "__main__":
    from app.agents.retrieval_agent import run_retrieval_agent
    from app.agents.extraction_agent import run_extraction_agent
    from app.agents.gap_detection_agent import run_gap_detection_agent
    from app.agents.hypothesis_generation_agent import run_hypothesis_generation_agent

    # Example: user decides Feasibility and Evidence Grounding matter most
    # for THIS particular research context (clinical, India-focused topic)
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

    print("\n=== Hypothesis Critiques ===")
    for i, crit in enumerate(state["critiques"], start=1):
        print(f"\n--- Critique {i} ---")
        print("Hypothesis:", crit["hypothesis"][:100], "...")
        print(f"  Feasibility: {crit['feasibility_score']}/10")
        print(f"  Novelty: {crit['novelty_score']}/10")
        print(f"  Evidence grounding: {crit['evidence_grounding_score']}/10")
        print(f"  Specificity: {crit['specificity_score']}/10")
        print(f"  OVERALL SOUNDNESS: {crit['overall_soundness_score']}/10")
        print("  Weaknesses:", crit["weaknesses"])
        print("  Existing evidence relation:", crit["existing_evidence_relation"])